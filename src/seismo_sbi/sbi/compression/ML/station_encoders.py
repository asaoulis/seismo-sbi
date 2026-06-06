"""Per-station encoder registry and interface.

Every per-station encoder is an ``nn.Module`` that maps a batch of flattened
station traces ``x:(B*N, C, T)`` to a sequence of token embeddings
``(B*N, L, D)``.  The transformer aggregator receives these tokens and folds
in spatial / cross-station information.

Interface contract
------------------
Each registered encoder **must**:

1. Accept ``__init__(num_seismic_components, input_length, d_model, **encoder_config)``.
   ``d_model`` is the transformer width; encoders need **not** output ``D == d_model``
   — ``SeismogramTransformer`` will add an ``nn.Linear(D, d_model)`` projection
   (``nn.Identity`` when equal).
2. Implement ``forward(x: Tensor) -> Tensor`` with
   ``x:(B*N, C, T)`` → output ``(B*N, L, D)``.
3. Expose scalar attributes ``.output_length`` (L) and ``.output_dim`` (D)
   immediately after ``__init__`` (i.e. without a forward pass).

Selection
---------
Set ``model_config["station_encoder"] = "<name>"`` (default ``"cnn"``).
Pass encoder-specific kwargs via ``model_config["encoder_config"]`` (dict, optional).
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Shared amplitude primitives (single-sourced so cnn/pno/tcn don't duplicate)
# ---------------------------------------------------------------------------
#
# Every per-station encoder max-abs-normalises a trace over (C, T) and appends a
# broadcast log-amplitude channel. These helpers define that operation in ONE place so
# the three encoders stay numerically identical, and so the transformer-level amplitude
# embedding (amplitude_embedding.py) can reuse the SAME max-abs definition.

_AMP_EPS = 1e-12


def normalize_trace(x: torch.Tensor, eps: float = _AMP_EPS):
    """Per-trace max-abs scaling over the (component, time) axes.

    ``x``: ``(B*N, C, T)``. Returns ``(x_norm, max_val)`` where ``max_val`` is the
    ``(B*N, 1, 1)`` clamped peak amplitude (shared across components ⇒ within-station
    component ratios and polarity are preserved in ``x_norm``).
    """
    max_val = x.abs().amax(dim=(1, 2), keepdim=True).clamp_min(eps)
    return x / max_val, max_val


def log_amp_channel(max_val: torch.Tensor, length: int, eps: float = _AMP_EPS) -> torch.Tensor:
    """Broadcast ``log(max_val)`` into a ``(B*N, 1, length)`` feature channel."""
    log_amp = max_val.clamp_min(eps).log()          # (B*N, 1, 1)
    return log_amp.expand(-1, -1, length)           # (B*N, 1, length)


def station_amplitudes(
    x: torch.Tensor, per_component: bool = False, eps: float = _AMP_EPS
) -> torch.Tensor:
    """Per-station log-amplitude features for the transformer-level embedding.

    ``x``: ``(B, N, C, T)``. Returns ``(B, N, K)`` log-amplitudes, with ``K == 1``
    (per-station peak, max over components and time) or ``K == C`` (per-component peak,
    max over time) when ``per_component`` is set. Uses the same max-abs definition as
    :func:`normalize_trace`.
    """
    if per_component:
        amp = x.abs().amax(dim=3)                    # (B, N, C)
    else:
        amp = x.abs().amax(dim=(2, 3)).unsqueeze(-1)  # (B, N, 1)
    return amp.clamp_min(eps).log()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PER_STATION_ENCODER_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_encoder(name: str):
    """Class decorator that registers an encoder class under *name*."""
    def _deco(cls):
        PER_STATION_ENCODER_REGISTRY[name] = cls
        return cls
    return _deco


def build_station_encoder(
    name: str,
    *,
    num_seismic_components: int,
    input_length: int,
    d_model: int,
    **encoder_config: Any,
) -> nn.Module:
    """Instantiate the encoder registered under *name*.

    Parameters
    ----------
    name:
        Key into ``PER_STATION_ENCODER_REGISTRY`` (e.g. ``"cnn"``, ``"pno"``, ``"tcn"``).
    num_seismic_components:
        Number of seismic channels C (e.g. 1 for Z-only, 3 for ZNE).
    input_length:
        Per-trace sample count T.
    d_model:
        Transformer width.  Encoders may produce a different width D; the
        calling code will project if necessary.
    **encoder_config:
        Forwarded verbatim to the encoder ``__init__``.
    """
    if name not in PER_STATION_ENCODER_REGISTRY:
        raise KeyError(
            f"Unknown station encoder '{name}'. "
            f"Registered: {sorted(PER_STATION_ENCODER_REGISTRY)}"
        )
    cls = PER_STATION_ENCODER_REGISTRY[name]
    return cls(
        num_seismic_components=num_seismic_components,
        input_length=input_length,
        d_model=d_model,
        **encoder_config,
    )


# ---------------------------------------------------------------------------
# CNNEncoder — thin adapter around SeismicTraceCNN (backward-compatible default)
# ---------------------------------------------------------------------------

@register_encoder("cnn")
class CNNEncoder(nn.Module):
    """Default per-station encoder: wraps ``SeismicTraceCNN`` unchanged.

    The CNN outputs ``(B*N, D, L)`` (channels-first); this adapter permutes
    to the canonical ``(B*N, L, D)`` token layout.

    All encoder_config kwargs are forwarded to ``SeismicTraceCNN`` so any
    existing CNN customisation (conv_channels, conv_kernels, etc.) continues
    to work.
    """

    def __init__(
        self,
        num_seismic_components: int,
        input_length: int,
        d_model: int,
        **encoder_config: Any,
    ) -> None:
        super().__init__()
        # Import here to avoid circular imports at module load time
        from .cnn_feature_extractor import SeismicTraceCNN

        # Match the original construction exactly: the old SeismogramTransformer
        # passed `cnn_output_dim=d_model` → `final_layer=d_model`. In SeismicTraceCNN
        # the default last conv has `final_layer - 1` channels and a log-amplitude
        # channel is appended, so `output_channels = final_layer = d_model`. Passing
        # d_model here therefore reproduces the current numerics and keeps the
        # downstream projection an Identity (output_dim == d_model).
        self._cnn = SeismicTraceCNN(
            num_seismic_components,
            input_length=input_length,
            final_layer=d_model,
            **encoder_config,
        )

        self.output_length: int = self._cnn.output_length
        self.output_dim: int = self._cnn.output_channels  # = d_model after +1 amp channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*N, C, T)  →  (B*N, L, D)"""
        # CNN returns (B*N, D, L) — permute to (B*N, L, D)
        feats = self._cnn(x)                        # (B*N, D, L)
        return feats.permute(0, 2, 1).contiguous()  # (B*N, L, D)


# ---------------------------------------------------------------------------
# FNO primitives (vendored — no new dependency)
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """Fourier-domain conv: rfft → complex weight multiply (lowest *modes*) → irfft.

    Operates on channels-first tensors ``(B, C_in, T)`` → ``(B, C_out, T)``.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        # Complex weights for the lowest *modes* Fourier coefficients
        scale = 1.0 / (in_channels * out_channels) ** 0.5
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        # rfft along the time axis
        x_ft = torch.fft.rfft(x, dim=-1)                   # (B, C_in, T//2+1)
        m = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1],
                             dtype=x_ft.dtype, device=x.device)
        # Einstein sum over in-channels and modes
        out_ft[:, :, :m] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :m], self.weights[:, :, :m]
        )
        return torch.fft.irfft(out_ft, n=T, dim=-1)         # (B, C_out, T)


class FNOBlock(nn.Module):
    """One FNO layer: SpectralConv1d + pointwise skip-Conv1d + activation + residual + norm."""

    def __init__(
        self,
        channels: int,
        modes: int,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.spec = SpectralConv1d(channels, channels, modes)
        self.skip = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.InstanceNorm1d(channels, affine=True)
        self.act = _get_act(activation)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.act(self.spec(x) + self.skip(x))
        h = self.drop(h)
        h = self.norm(h + x)   # residual
        return h


def _get_act(name: str) -> nn.Module:
    name = (name or "gelu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    return nn.GELU()


def _compute_conv_output_length(L: int, kernel: int, stride: int, padding: int = 0) -> int:
    return (L + 2 * padding - (kernel - 1) - 1) // stride + 1


# ---------------------------------------------------------------------------
# PhaseNeuralOperatorEncoder  ("pno")
# ---------------------------------------------------------------------------

@register_encoder("pno")
class PhaseNeuralOperatorEncoder(nn.Module):
    """Per-station Phase Neural Operator encoder (FNO-style, 1-D in time).

    Maps ``x:(B*N, C, T)`` → ``(B*N, L, D)``.

    Architecture
    ------------
    1. Amplitude normalisation + log-amplitude feature channel (mirrors ``SeismicTraceCNN``).
    2. Lifting Conv1d: ``C+1 → width``.
    3. ``n_blocks`` ``FNOBlock`` layers over the time axis.
    4. Optional strided Conv1d downsampling so ``L = T // downsample``.
    5. Projection Conv1d: ``width → out_dim``.
    6. Permute to ``(B*N, L, D=out_dim)``.

    Cross-station coupling is intentionally **not** reproduced here — the
    axial-transformer aggregator already plays that role.

    Config keys (``model_config["encoder_config"]``)
    -------------------------------------------------
    width       int   Intermediate channel count (default 32).
    modes       int   Fourier modes kept per FNO block (default 16).
    n_blocks    int   Number of FNO blocks (default 4).
    out_dim     int   Output feature dim D (default == d_model).
    downsample  int   Temporal stride for downsampling step (default 4).
    activation  str   Activation name: "gelu"/"relu"/"silu" (default "gelu").
    dropout     float Dropout rate inside FNO blocks (default 0.0).
    """

    _EPS = 1e-12

    def __init__(
        self,
        num_seismic_components: int,
        input_length: int,
        d_model: int,
        *,
        width: int = 32,
        modes: int = 16,
        n_blocks: int = 4,
        out_dim: int = None,
        downsample: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        out_dim = out_dim if out_dim is not None else d_model

        # Lifting: C+1 → width  (+1 for log-amplitude channel)
        self.lift = nn.Conv1d(num_seismic_components + 1, width, kernel_size=1)

        # FNO blocks
        self.fno_blocks = nn.Sequential(
            *[FNOBlock(width, modes, activation=activation, dropout=dropout)
              for _ in range(n_blocks)]
        )

        # Optional downsampling
        if downsample > 1:
            self.downsample = nn.Conv1d(width, width, kernel_size=downsample, stride=downsample)
            L_out = _compute_conv_output_length(input_length, downsample, downsample)
        else:
            self.downsample = nn.Identity()
            L_out = input_length

        # Projection to output dim
        self.proj = nn.Conv1d(width, out_dim, kernel_size=1)

        self.output_length: int = L_out
        self.output_dim: int = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*N, C, T)  →  (B*N, L, D)"""
        # Safe per-trace amplitude normalisation + appended log-amplitude channel.
        x_norm, max_val = normalize_trace(x, self._EPS)
        log_amp = log_amp_channel(max_val, x.shape[-1], self._EPS)  # (B*N, 1, T)
        x_in = torch.cat([x_norm, log_amp], dim=1)                  # (B*N, C+1, T)

        h = self.lift(x_in)           # (B*N, width, T)
        h = self.fno_blocks(h)        # (B*N, width, T)
        h = self.downsample(h)        # (B*N, width, L)
        h = self.proj(h)              # (B*N, D, L)
        return h.permute(0, 2, 1).contiguous()   # (B*N, L, D)


# ---------------------------------------------------------------------------
# DilatedTCNEncoder  ("tcn")
# ---------------------------------------------------------------------------

class _GLUBlock(nn.Module):
    """One dilated WaveNet-style block: dilated Conv1d → GLU → residual.

    Uses symmetric (same) padding so the output length equals the input length.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        # Outputs 2*channels so GLU can split in half
        self.conv = nn.Conv1d(
            channels, 2 * channels, kernel_size=kernel_size,
            dilation=dilation, padding=pad,
        )
        self.norm = nn.InstanceNorm1d(channels, affine=True)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)                          # (B, 2C, T)
        h, gate = h.chunk(2, dim=1)               # (B, C, T) each
        h = h * torch.sigmoid(gate)               # GLU
        h = self.drop(h)
        h = self.norm(h + x)                      # residual + norm
        return h


@register_encoder("tcn")
class DilatedTCNEncoder(nn.Module):
    """Per-station dilated Temporal Convolutional Network (WaveNet-style) encoder.

    Maps ``x:(B*N, C, T)`` → ``(B*N, L, D)``.

    Architecture
    ------------
    1. Amplitude normalisation + log-amplitude feature channel.
    2. Lifting Conv1d: ``C+1 → channels``.
    3. ``n_blocks`` ``_GLUBlock`` layers with exponentially growing dilation
       ``(1, 2, 4, …, 2^(n_blocks-1))`` — receptive field doubles each block.
    4. Optional strided Conv1d downsampling: ``L = T // downsample``.
    5. Projection Conv1d: ``channels → out_dim``.
    6. Permute to ``(B*N, L, D=out_dim)``.

    Config keys (``model_config["encoder_config"]``)
    -------------------------------------------------
    channels      int   Intermediate channel count (default 32).
    n_blocks      int   Number of GLU blocks (default 4).
    kernel_size   int   Dilated conv kernel width (default 3).
    dilation_base int   Dilation base (default 2; block k gets dilation_base^k).
    dropout       float Dropout rate (default 0.0).
    out_dim       int   Output feature dim D (default == d_model).
    downsample    int   Temporal stride for downsampling step (default 4).
    """

    _EPS = 1e-12

    def __init__(
        self,
        num_seismic_components: int,
        input_length: int,
        d_model: int,
        *,
        channels: int = 32,
        n_blocks: int = 4,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.0,
        out_dim: int = None,
        downsample: int = 4,
    ) -> None:
        super().__init__()
        out_dim = out_dim if out_dim is not None else d_model

        # Symmetric padding pad = dilation*(k-1)//2 only preserves the trace length for an
        # ODD kernel; an even kernel produces length T-1 and breaks the residual add in
        # _GLUBlock. Reject it up front with a clear message.
        if kernel_size % 2 == 0:
            raise ValueError(
                f"DilatedTCNEncoder requires an odd kernel_size to preserve length; got {kernel_size}."
            )

        # Lifting: C+1 → channels
        self.lift = nn.Conv1d(num_seismic_components + 1, channels, kernel_size=1)

        # Dilated GLU blocks
        self.blocks = nn.Sequential(
            *[_GLUBlock(channels, kernel_size, dilation=dilation_base ** i, dropout=dropout)
              for i in range(n_blocks)]
        )

        # Optional downsampling
        if downsample > 1:
            self.downsample = nn.Conv1d(channels, channels, kernel_size=downsample, stride=downsample)
            L_out = _compute_conv_output_length(input_length, downsample, downsample)
        else:
            self.downsample = nn.Identity()
            L_out = input_length

        # Projection to output dim
        self.proj = nn.Conv1d(channels, out_dim, kernel_size=1)

        self.output_length: int = L_out
        self.output_dim: int = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*N, C, T)  →  (B*N, L, D)"""
        # Safe per-trace amplitude normalisation + appended log-amplitude channel.
        x_norm, max_val = normalize_trace(x, self._EPS)
        log_amp = log_amp_channel(max_val, x.shape[-1], self._EPS)  # (B*N, 1, T)
        x_in = torch.cat([x_norm, log_amp], dim=1)                  # (B*N, C+1, T)

        h = self.lift(x_in)     # (B*N, channels, T)
        h = self.blocks(h)      # (B*N, channels, T)
        h = self.downsample(h)  # (B*N, channels, L)
        h = self.proj(h)        # (B*N, D, L)
        return h.permute(0, 2, 1).contiguous()  # (B*N, L, D)
