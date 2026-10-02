"""Random-Fourier-Feature positional encoding of station geometry.

A token-index sinusoid tuned for indices up to ~1e4 leaves most channels near-constant over a
regional array, so geometry is encoded here by a Gaussian random-Fourier map with a learnable
scale. ``coords_kind='relative'`` encodes source-relative epicentral distance and azimuth, the
azimuth entering as ``(cos, sin)`` so the map is periodic across the branch cut; ``'absolute'``
encodes station latitude and longitude. ``include_depth`` adds source depth, on which take-off
angle and hence first-motion polarity depend. Features are standardised to O(1) before the map.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from seismo_sbi.sbi.npe.networks.fourier_features import ScalarFourierEmbedding

# Keys of the ``positional_encoding`` block read here; ``mode`` and ``inject_every_layer`` are
# read by the transformer but validated with the rest.
_MODULE_KEYS = {"include_depth", "num_freqs", "sigma", "learnable_freqs", "standardize"}
_CONFIG_KEYS = _MODULE_KEYS | {"mode", "inject_every_layer"}


class FourierStationPositionalEncoding(nn.Module):
    """Encode station geometry into ``d_model`` positional tokens with Random Fourier Features.

    ``forward(coords, depth, mask) -> (B, N, d_model)``. ``coords`` is ``(B, N, 2)`` — either
    source-relative ``(distance, azimuth)`` (``coords_kind="relative"``) or absolute
    ``(lat, lon)`` (``coords_kind="absolute"``). ``depth`` is the per-sample source depth
    ``(B, 1)`` (required iff ``include_depth``), broadcast across stations. ``mask`` is the
    ``(B, N)`` station validity (True = real) for variable-station configs, or ``None``.
    """

    def __init__(
        self,
        d_model: int,
        coords_kind: str = "relative",
        *,
        include_depth: bool = False,
        num_freqs: int = 16,
        sigma: float = 1.0,
        learnable_freqs: bool = False,
        standardize: str = "running",
        seed: int = 2,
    ) -> None:
        super().__init__()
        if coords_kind not in ("relative", "absolute"):
            raise ValueError(
                f"coords_kind must be 'relative' or 'absolute', got '{coords_kind}'."
            )
        self.coords_kind = coords_kind
        self.include_depth = bool(include_depth)
        # relative ⇒ (distance, cos az, sin az); absolute ⇒ (coord0, coord1); +1 for depth.
        in_dim = (3 if coords_kind == "relative" else 2) + (1 if self.include_depth else 0)
        self.in_dim = in_dim
        self.embed = ScalarFourierEmbedding(
            in_dim, d_model, num_freqs=num_freqs, sigma=sigma,
            learnable_freqs=learnable_freqs, standardize=standardize, seed=seed,
        )

    @classmethod
    def from_config(
        cls, d_model: int, coords_kind: str, cfg: Dict[str, Any]
    ) -> "FourierStationPositionalEncoding":
        """Build from the ``positional_encoding`` config dict, rejecting unknown keys.

        ``mode`` / ``inject_every_layer`` are accepted (consumed by the transformer) but ignored
        here; only the module-level keys are forwarded to the constructor.
        """
        unknown = set(cfg) - _CONFIG_KEYS
        if unknown:
            raise ValueError(
                f"Unknown positional_encoding keys {sorted(unknown)}; "
                f"valid: {sorted(_CONFIG_KEYS)}"
            )
        kwargs = {k: v for k, v in cfg.items() if k in _MODULE_KEYS}
        return cls(d_model, coords_kind, **kwargs)

    def _features(
        self, coords: torch.Tensor, depth: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Build the RFF input feature vector ``(B, N, in_dim)`` from raw geometry.

        Relative mode lifts azimuth to ``(cos, sin)`` so the encoding is periodic; ``atan2``
        of all-zero (padded / at-source) coordinates is 0, so ``cos/sin`` stay finite.
        """
        if self.coords_kind == "relative":
            dist = coords[..., 0:1]
            az = coords[..., 1:2]
            feat = torch.cat([dist, torch.cos(az), torch.sin(az)], dim=-1)  # (B, N, 3)
        else:
            feat = coords  # (B, N, 2) absolute (lat, lon)

        if self.include_depth:
            if depth is None:
                raise ValueError(
                    "include_depth=True but no source depth was provided to the positional "
                    "encoding (need a conditioned model with n_cond >= 3)."
                )
            B, N = feat.shape[0], feat.shape[1]
            d = depth.reshape(B, 1, 1).expand(B, N, 1).to(feat.dtype)
            feat = torch.cat([feat, d], dim=-1)
        return feat

    def forward(
        self,
        coords: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = self._features(coords, depth)
        out = self.embed(feat, mask=mask)                # (B, N, d_model)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)  # zero padded stations
        return out
