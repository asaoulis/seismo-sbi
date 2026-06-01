"""Source-location conditioning for the ML compressor.

Models can be *given* a source location (e.g. ``(latitude, longitude, depth)`` or
Cartesian ``(x, y, z)``) as scalar conditioning.  This module provides the pure,
self-contained pieces used by :class:`SeismogramTransformer`:

* :func:`relative_station_geometry` — per-(source, station) epicentral distance and
  azimuth, used as **source-relative** positional embeddings (an alternative to the
  current absolute lat/lon station embeddings).
* :class:`SourceConditioner` — maps a raw source-coordinate vector to a learned
  source embedding ``(B, d_cond)``.
* :class:`FiLM` — feature-wise linear modulation generated from the source embedding.
* :func:`pack_context` / :func:`unpack_context` — fold the source vector into the
  single context tensor that nflows requires, and recover it inside the model.

Design contracts
----------------
* The packed conditioning vector carries **RAW (unscaled)** coordinates, so
  :func:`relative_station_geometry` is computed on true geometry.  ``SourceConditioner``
  normalises internally before its MLP.
* All of this is **opt-in**: with no conditioning the model never imports these paths
  (``n_cond == 0`` ⇒ the 4-D context path is unchanged).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def relative_station_geometry(
    source: torch.Tensor,
    station_coords: torch.Tensor,
    coord_mode: str = "geographic",
) -> torch.Tensor:
    """Per-(source, station) distance & azimuth.

    Parameters
    ----------
    source:
        ``(B, >=2)`` source coordinates.  Geographic: ``(lat, lon, [depth, ...])`` in
        **degrees**.  Cartesian: ``(x, y, [z, ...])``.  Only the first two entries are
        used (depth does not affect epicentral geometry).
    station_coords:
        ``(N, 2)`` station coordinates — ``(lat, lon)`` (geographic) or ``(x, y)``
        (cartesian).  Shared across the batch.
    coord_mode:
        ``"geographic"`` (great-circle central angle in radians + initial bearing) or
        ``"cartesian"`` (in-plane Euclidean distance + ``atan2`` azimuth).

    Returns
    -------
    ``(B, N, 2)`` tensor of ``(distance, azimuth)``.  Azimuth is in radians in
    ``(-pi, pi]``; geographic distance is the central angle in radians ``[0, pi]``.
    """
    if source.dim() == 1:
        source = source.unsqueeze(0)
    B = source.shape[0]
    N = station_coords.shape[0]
    station_coords = station_coords.to(source.dtype).to(source.device)

    src = source[:, :2]                                   # (B, 2)
    sta = station_coords.unsqueeze(0).expand(B, N, 2)     # (B, N, 2)

    if coord_mode == "geographic":
        lat1 = torch.deg2rad(src[:, 0]).unsqueeze(1)      # (B, 1)
        lon1 = torch.deg2rad(src[:, 1]).unsqueeze(1)      # (B, 1)
        lat2 = torch.deg2rad(sta[:, :, 0])                # (B, N)
        lon2 = torch.deg2rad(sta[:, :, 1])                # (B, N)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        # Haversine central angle.
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        dist = 2 * torch.asin(torch.sqrt(a.clamp(0.0, 1.0)))            # (B, N), radians
        # Initial bearing from source to station.
        az = torch.atan2(
            torch.sin(dlon) * torch.cos(lat2),
            torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon),
        )                                                              # (B, N), radians
    elif coord_mode == "cartesian":
        dx = sta[:, :, 0] - src[:, 0:1]
        dy = sta[:, :, 1] - src[:, 1:2]
        dist = torch.sqrt(dx ** 2 + dy ** 2)
        az = torch.atan2(dy, dx)
    else:
        raise ValueError(f"Unknown coord_mode '{coord_mode}'. Use 'geographic' or 'cartesian'.")

    return torch.stack([dist, az], dim=-1)                # (B, N, 2)


# ---------------------------------------------------------------------------
# Source embedding
# ---------------------------------------------------------------------------

class SourceConditioner(nn.Module):
    """Map a raw source-coordinate vector ``(B, n_cond)`` to an embedding ``(B, d_cond)``.

    A small MLP with an optional Fourier-feature expansion of the (internally
    normalised) coordinates.  ``coord_mode`` only affects the internal normalisation
    constants; the raw coordinates are expected as input so that callers can also use
    them for :func:`relative_station_geometry`.
    """

    def __init__(
        self,
        n_cond: int,
        d_cond: int,
        coord_mode: str = "geographic",
        n_fourier: int = 0,
        hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        if n_cond <= 0:
            raise ValueError("SourceConditioner requires n_cond > 0")
        self.n_cond = n_cond
        self.d_cond = d_cond
        self.coord_mode = coord_mode
        self.n_fourier = int(n_fourier)

        # Per-coordinate normalisation so the MLP sees O(1) inputs. Geographic coords are
        # roughly lat∈[-90,90], lon∈[-180,180], depth in km; a fixed scale is enough — the
        # MLP learns the rest. Cartesian: assume already O(1)..O(1e2); scale lightly.
        if coord_mode == "geographic":
            base = torch.tensor([90.0, 180.0] + [100.0] * max(0, n_cond - 2))[:n_cond]
        else:
            base = torch.ones(n_cond) * 100.0
        self.register_buffer("_norm", base)

        feat_dim = n_cond * (1 + 2 * self.n_fourier)
        hidden = hidden or max(d_cond, 2 * feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_cond),
        )

    def _features(self, source: torch.Tensor) -> torch.Tensor:
        x = source / self._norm.to(source.dtype)
        feats = [x]
        for k in range(self.n_fourier):
            freq = 2.0 ** k * math.pi
            feats.append(torch.sin(freq * x))
            feats.append(torch.cos(freq * x))
        return torch.cat(feats, dim=-1)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        if source.dim() == 1:
            source = source.unsqueeze(0)
        return self.mlp(self._features(source))


class FiLM(nn.Module):
    """Feature-wise linear modulation generated from a conditioning embedding.

    Produces ``(γ, β)`` from the source embedding and returns ``x * (1 + γ) + β``.
    Initialised so that at start ``γ ≈ 0, β ≈ 0`` (identity modulation), which keeps the
    conditioned model close to the unconditioned one at the beginning of training.
    """

    def __init__(self, d_cond: int, d_model: int) -> None:
        super().__init__()
        self.to_film = nn.Linear(d_cond, 2 * d_model)
        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_film.bias)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model); cond: (B, d_cond). Broadcasts cond over the middle dims."""
        gamma, beta = self.to_film(cond).chunk(2, dim=-1)     # each (B, d_model)
        # Reshape (B, d_model) → (B, 1, ..., d_model) to broadcast over x's middle dims.
        extra = x.dim() - 2
        view = (x.shape[0],) + (1,) * extra + (self.d_model,)
        gamma = gamma.view(view)
        beta = beta.view(view)
        return x * (1 + gamma) + beta


# ---------------------------------------------------------------------------
# Context packing
# ---------------------------------------------------------------------------

def pack_context(seismograms: torch.Tensor, source_vec: torch.Tensor) -> torch.Tensor:
    """Flatten seismograms and append the (raw) source vector along the last dim.

    ``seismograms``: ``(..., N, C, T)`` or ``(..., N*C*T)``; ``source_vec``: ``(..., n_cond)``.
    Returns ``(..., N*C*T + n_cond)`` — the single tensor nflows passes to the embedding net.
    """
    lead = seismograms.shape[: -3] if seismograms.dim() >= 3 else seismograms.shape[:-1]
    flat = seismograms.reshape(*lead, -1)
    return torch.cat([flat, source_vec], dim=-1)


def unpack_context(
    ctx: torch.Tensor,
    n_stations: int,
    n_components: int,
    trace_length: int,
    n_cond: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Inverse of :func:`pack_context`.

    Returns ``(seismograms (B,N,C,T), source_vec (B,n_cond) | None)``.  If ``ctx`` is
    already 4-D (the unconditioned path) it is returned unchanged with ``source_vec=None``.
    """
    if ctx.dim() == 4:
        return ctx, None
    B = ctx.shape[0]
    nct = n_stations * n_components * trace_length
    # Fail loudly on a length mismatch rather than silently slicing the source vector
    # out of the seismogram tail (e.g. trace-length / n_components / n_cond drift).
    expected = nct + n_cond
    if ctx.shape[1] != expected:
        raise ValueError(
            f"Packed context width {ctx.shape[1]} != expected {expected} "
            f"(n_stations={n_stations} * n_components={n_components} * trace_length={trace_length} "
            f"+ n_cond={n_cond}). Check that the model's trace_length/components/n_cond match the data."
        )
    seis = ctx[:, :nct].reshape(B, n_stations, n_components, trace_length)
    source_vec = ctx[:, nct:] if n_cond > 0 else None
    return seis, source_vec
