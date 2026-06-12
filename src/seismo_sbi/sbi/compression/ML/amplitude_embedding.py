"""Per-station amplitude → transformer-token embedding.

Lifts the per-station peak amplitude out of the waveform channel (where the encoder buries
it as a single scalar) and turns it into a **full-width token embedding** that is added to
each station token *before* the cross-station attention. Inter-station amplitude ratio is a
first-class moment-tensor observable (relative-radiation-pattern focal-mechanism methods
invert exactly this), so it must reach the attention robustly rather than as one channel
that LayerNorm and the additive positional embeddings can swamp.

Representation (config ``mode``)
--------------------------------
``array_relative`` (default)
    For each event the token feature is ``log A_i - ref(event)``, where ``ref`` is the
    mean/median ``log`` amplitude over the event's **valid** stations. Since
    ``log A_i = log M0 + log(radiation_i · spreading_i)``, subtracting the per-event
    reference cancels the shared ``log M0`` ⇒ an O(1), magnitude-invariant radiation pattern
    (ideal Random-Fourier-Feature input, computed per-event ⇒ no batch coupling). The
    deterministic ``ref`` (≈ ``log M0``) is embedded separately and returned as a **global**
    vector so absolute moment keeps a clean path to the flow.
``absolute``
    The token feature is the absolute ``log A_i``, standardised by a running standardiser so
    the RFF sees ~unit scale; the network disentangles ``M0`` from mechanism itself. No global
    vector.

Granularity (config ``per_component``): one peak per station (default) or one peak per
component (``K = C``). Within-station component ratios + polarity are already preserved by the
encoder's shared ``(C, T)`` normalisation, so per-station is the minimal physics-targeted fix.

Robustness (both opt-in, array_relative)
----------------------------------------
``distance_correction``
    Subtract a learnable geometric-spreading/attenuation trend ``g(log distance)`` before forming
    the reference, so neither the radiation-pattern token nor the M0 proxy is biased by the
    array's distance distribution. Requires per-station distance ⇒ source-location conditioning.
``snr_weighting``
    Down-weight noise-dominated stations when forming the per-event reference, via a rough,
    self-contained peak-to-(low-percentile-floor) SNR proxy gated by a learnable sigmoid — so
    low-magnitude / low-SNR events do not let noise floors bias the reference.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fourier_features import RunningStandardizer, ScalarFourierEmbedding
from .station_encoders import station_amplitudes

_CONFIG_KEYS = {
    "mode", "per_component", "reference", "num_freqs", "sigma", "learnable_freqs", "scale",
    "distance_correction", "snr_weighting", "snr_floor_quantile", "distance_mlp_hidden",
}


class DistanceDetrend(nn.Module):
    """Remove the geometric-spreading / attenuation amplitude trend with distance.

    Models ``log`` amplitude's distance dependence as ``g(log d) = alpha * z + mlp(z)`` where
    ``z`` is the running-standardised ``log`` distance: a learnable spreading-exponent term plus
    a small MLP residual for attenuation/dispersion curvature. Zero-initialised (``alpha = 0``,
    MLP final layer zero) so it starts as the identity (no correction) and *learns* the de-trend,
    keeping early training close to the un-corrected reference. Subtracting ``g`` before forming
    the array reference removes the distance bias from both the token feature and the M0 proxy.
    """

    def __init__(self, hidden: int = 16, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = float(eps)
        self.standardizer = RunningStandardizer(1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, distance: torch.Tensor, mask=None) -> torch.Tensor:
        # distance: (B, N, 1) epicentral distance → g: (B, N, 1)
        log_d = distance.clamp_min(self.eps).log()
        z = self.standardizer(log_d, mask=mask)
        return self.alpha * z + self.mlp(z)


class AmplitudeTokenEmbedding(nn.Module):
    """Embed per-station amplitude into ``d_model`` station tokens (+ optional global vector).

    ``forward(x, mask) -> (token_emb (B, N, d_model), global_emb (B, d_model) | None)``.
    ``x`` is the unpacked ``(B, N, C, T)`` seismogram batch; ``mask`` is the ``(B, N)`` station
    validity (True = real) for variable-station configs, or ``None`` (fixed full array).
    """

    def __init__(
        self,
        d_model: int,
        num_components: int = 1,
        *,
        mode: str = "array_relative",
        per_component: bool = False,
        reference: str = "mean",
        num_freqs: int = 16,
        sigma: float = 1.0,
        learnable_freqs: bool = False,
        scale: float = 1.0,
        distance_correction: bool = False,
        snr_weighting: bool = False,
        snr_floor_quantile: float = 0.2,
        distance_mlp_hidden: int = 16,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if mode not in ("array_relative", "absolute"):
            raise ValueError(f"mode must be 'array_relative' or 'absolute', got '{mode}'.")
        if reference not in ("mean", "median"):
            raise ValueError(f"reference must be 'mean' or 'median', got '{reference}'.")
        self.mode = mode
        self.per_component = bool(per_component)
        self.reference = reference
        self.eps = float(eps)
        self.K = num_components if self.per_component else 1

        # Distance de-trend (needs per-station distance ⇒ source-relative geometry at forward).
        self.distance_correction = bool(distance_correction)
        self.detrend = (
            DistanceDetrend(hidden=distance_mlp_hidden, eps=self.eps)
            if self.distance_correction else None
        )
        # SNR-weighted reference: down-weight noise-dominated stations when forming the per-event
        # reference (array_relative only). Learnable soft gate on a rough peak-to-floor ratio.
        self.snr_weighting = bool(snr_weighting)
        self.snr_floor_quantile = float(snr_floor_quantile)
        if not (0.0 < self.snr_floor_quantile < 1.0):
            raise ValueError("snr_floor_quantile must be in (0, 1).")
        if self.snr_weighting:
            self.snr_gate_slope = nn.Parameter(torch.zeros(1))        # softplus(0) ≈ 0.69
            self.snr_gate_bias = nn.Parameter(torch.full((1,), 3.0))  # ≈ pure-noise peak/floor

        if mode == "array_relative":
            # Token feature is already centred per-event ⇒ fixed standardisation (center 0,
            # configurable spread). The reference is an absolute scalar with an unknown
            # offset ⇒ running standardiser.
            self.token_embed = ScalarFourierEmbedding(
                self.K, d_model, num_freqs=num_freqs, sigma=sigma,
                learnable_freqs=learnable_freqs, standardize="fixed",
                center=0.0, scale=scale, seed=0,
            )
            self.ref_embed = ScalarFourierEmbedding(
                self.K, d_model, num_freqs=num_freqs, sigma=sigma,
                learnable_freqs=learnable_freqs, standardize="running", seed=1,
            )
        else:  # absolute
            self.token_embed = ScalarFourierEmbedding(
                self.K, d_model, num_freqs=num_freqs, sigma=sigma,
                learnable_freqs=learnable_freqs, standardize="running", seed=0,
            )
            self.ref_embed = None

    @classmethod
    def from_config(
        cls, d_model: int, num_components: int, cfg: Dict[str, Any]
    ) -> "AmplitudeTokenEmbedding":
        """Build from the ``amplitude_embedding`` config dict, rejecting unknown keys."""
        unknown = set(cfg) - _CONFIG_KEYS
        if unknown:
            raise ValueError(
                f"Unknown amplitude_embedding keys {sorted(unknown)}; "
                f"valid: {sorted(_CONFIG_KEYS)}"
            )
        return cls(d_model, num_components, **cfg)

    @property
    def uses_distance(self) -> bool:
        """Whether forward() needs per-station distance (⇒ a source location must be available)."""
        return self.distance_correction

    def _snr_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Per-station soft reliability weight from a rough, self-contained SNR proxy.

        ``s_i = log(peak_i) - log(floor_i)`` with ``peak`` the max |x| and ``floor`` a LOW
        percentile of |x| over (components, time). A low quantile (not the median) is used so the
        floor still lands in the quiet samples even when much of the trace is event signal —
        a deliberately rough but robust noise-level estimate. A learnable sigmoid turns it into a
        weight in (0, 1) that suppresses noise-dominated stations in the reference.
        """
        B, N, C, T = x.shape
        ax = x.abs().reshape(B, N, C * T)
        peak = ax.amax(dim=-1).clamp_min(self.eps)                                   # (B, N)
        # torch.quantile requires float/double — under bf16 autocast ax is bf16, so compute
        # the floor in fp32 (an amplitude statistic; precision here is immaterial) then cast
        # back. Newer torch rejects bf16 outright; torch 2.0 tolerated it silently.
        floor = torch.quantile(
            ax.float(), self.snr_floor_quantile, dim=-1
        ).to(ax.dtype).clamp_min(self.eps)
        s = peak.log() - floor.log()                                                 # (B, N)
        slope = F.softplus(self.snr_gate_slope)                                      # > 0
        w = torch.sigmoid(slope * (s - self.snr_gate_bias))                          # (B, N)
        return w.unsqueeze(-1)                                                        # (B, N, 1)

    def _reference(
        self, c: torch.Tensor, weight: Optional[torch.Tensor], use_median: bool,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Per-event reference over stations. ``c``: (B,N,K) → (B,1,K).

        ``weight`` (B,N,1, non-negative) gives a weighted mean (station validity × SNR). When
        ``use_median`` is set (median reference, no SNR weighting) a masked median is used instead.
        """
        if use_median:
            if mask is None:
                return c.median(dim=1, keepdim=True).values
            masked = c.masked_fill(~mask.unsqueeze(-1), float("nan"))
            ref = torch.nanmedian(masked, dim=1, keepdim=True).values
            return torch.nan_to_num(ref, nan=0.0)                # guard fully-padded events
        if weight is None:
            return c.mean(dim=1, keepdim=True)
        denom = weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (c * weight).sum(dim=1, keepdim=True) / denom

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
        distance: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        log_amp = station_amplitudes(x, per_component=self.per_component, eps=self.eps)  # (B,N,K)

        # Distance de-trend (remove geometric spreading) when enabled AND distance supplied.
        if self.detrend is not None and distance is not None:
            c = log_amp - self.detrend(distance, mask=mask)      # (B,N,1) broadcasts over K
        else:
            c = log_amp

        if self.mode == "array_relative":
            # Reference weights: station validity AND (optionally) SNR reliability.
            mask_f = mask.unsqueeze(-1).to(c.dtype) if mask is not None else None
            if self.snr_weighting:
                w = self._snr_weights(x)                         # (B,N,1)
                weight = w if mask_f is None else w * mask_f
            else:
                weight = mask_f
            use_median = (self.reference == "median") and not self.snr_weighting
            ref = self._reference(c, None if use_median else weight, use_median, mask)  # (B,1,K)
            token_feat = c - ref                                 # (B, N, K)
            token_emb = self.token_embed(token_feat)             # (B, N, d_model)
            global_emb = self.ref_embed(ref.squeeze(1))          # (B, d_model)
        else:  # absolute — running-standardise the (de-trended) log-amp, masking padded stations
            token_emb = self.token_embed(c, mask=mask)           # (B, N, d_model)
            global_emb = None

        if mask is not None:
            token_emb = token_emb * mask.unsqueeze(-1).to(token_emb.dtype)
        return token_emb, global_emb
