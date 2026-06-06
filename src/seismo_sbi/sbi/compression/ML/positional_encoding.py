"""Random-Fourier-Feature station positional encoding (review §3.2).

The legacy station positional encodings (``station_position_embedding`` /
``_station_position_embedding_batched`` in ``axial_transformer.py``) apply the Vaswani
``10000^(-2i/d)`` token-index sinusoid directly to geographic coordinates (degrees, or
source-relative distance/azimuth in radians). That frequency ladder is tuned for integer
token indices up to ~10⁴, so on a regional array most embedding channels are near-constant
and there is no learnable scale matched to the array's extent. This module replaces it with a
**well-scaled Random Fourier Feature** map built on the shared primitives in
``fourier_features.py`` (the same design the per-station amplitude embedding uses, review §3.1).

What it encodes (config ``coords_kind``)
----------------------------------------
``relative`` (the training default path)
    Source-relative geometry ``(epicentral distance, azimuth)`` from
    :func:`relative_station_geometry`. **Azimuth is encoded as ``(cos az, sin az)`` before the
    RFF and the raw angle is never fed in**, so the encoding is periodic — 11° and 350° map to
    nearby points and the embedding is continuous across the ±180°/0–360° branch cut by
    construction. A random-Fourier linear mix ``b·cos az + c·sin az = R·cos(az − φ)`` stays
    periodic, and per-feature affine standardisation preserves it.
``absolute``
    Raw ``(lat, lon)`` station coordinates, RFF-encoded with a scale matched to the data.

``include_depth`` additionally RFF-encodes the **source depth** (broadcast across stations) —
take-off angle, hence first-motion polarity, depends on depth, but depth never reached the
station geometry before. Take-off-angle proxy itself is out of scope here.

Scaling discipline is inherited from :class:`ScalarFourierEmbedding`: the heterogeneous
feature vector (distance in radians ~O(1), ``cos/sin`` az in ``[-1, 1]``, depth in km) is
brought to O(1) per-feature by a running standardiser before the Gaussian RFF, with a raw
standardised pass-through guaranteeing a non-vanishing gradient path.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .fourier_features import ScalarFourierEmbedding

# Keys accepted in the ``positional_encoding`` config block. ``mode`` (fourier|sinusoidal) and
# ``inject_every_layer`` are consumed by the transformer, not this module, but are listed here so
# the single validator covers the whole block.
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
