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

import numpy as np
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
        ``(N, 2)`` station coordinates shared across the batch, **or** ``(B, N, 2)``
        per-sample coordinates (used for variable-station configurations where each
        sample carries its own — possibly padded — station set).  ``(lat, lon)``
        (geographic) or ``(x, y)`` (cartesian).
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
    station_coords = station_coords.to(source.dtype).to(source.device)

    src = source[:, :2]                                   # (B, 2)
    # Accept either shared (N, 2) coords (broadcast over the batch) or per-sample
    # (B, N, 2) coords (variable-station configurations carry their own coords).
    if station_coords.dim() == 3:
        sta = station_coords                              # (B, N, 2)
    else:
        N = station_coords.shape[0]
        sta = station_coords.unsqueeze(0).expand(B, N, 2)  # (B, N, 2)

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


# ---------------------------------------------------------------------------
# Variable-station context packing
# ---------------------------------------------------------------------------
#
# For variable station configurations the single nflows ``context`` tensor must carry,
# per sample: the (padded) seismograms, the per-sample station coordinates, a validity
# mask marking real vs padded stations, and (optionally) the source vector.  Within a
# batch every sample is padded to a common ``max_N`` (done in the collate), so the batch
# is rectangular; ``max_N`` may differ across batches.  The model recovers ``max_N`` from
# the context width because everything else (n_components, trace_length, n_cond) is known:
#
#     width  W = max_N * (n_components * trace_length + 2 + 1) + n_cond
#     max_N    = (W - n_cond) // (n_components * trace_length + 3)
#
# Layout per sample (flattened, in order):
#     [ seismograms (max_N*C*T) | coords (max_N*2) | mask (max_N) | source_vec (n_cond)? ]

def pack_variable_context(
    seismograms: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    source_vec: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pack one (already padded) variable-station sample into a flat context vector.

    ``seismograms``: ``(N, C, T)``; ``coords``: ``(N, 2)``; ``mask``: ``(N,)`` (True=real
    station); ``source_vec``: ``(n_cond,)`` or ``None``.  Returns a 1-D tensor.
    """
    parts = [
        seismograms.reshape(-1),
        coords.reshape(-1),
        mask.to(seismograms.dtype).reshape(-1),
    ]
    if source_vec is not None:
        parts.append(source_vec.reshape(-1).to(seismograms.dtype))
    return torch.cat(parts, dim=-1)


def pack_subset_observation(
    stacked_data,
    station_coords,
    present_mask=None,
    source_vec=None,
) -> torch.Tensor:
    """Pack a single (subset) station configuration into the 2-D variable-station context
    a ``variable_stations=True`` model expects at inference time.

    This is the single-sample inference counterpart of
    :func:`...dataloading.variable_station_collate`: a lone observation defines its own
    ``N`` so no padding is needed, and the returned tensor has a leading batch dim of 1.

    Parameters
    ----------
    stacked_data : array-like ``(N, C, T)``
        Seismograms for the ``N`` selected stations, in the SAME order as ``station_coords``.
    station_coords : array-like ``(N, 2)``
        ``(latitude, longitude)`` of those stations.
    present_mask : array-like ``(N,)`` of bool, optional
        ``False`` marks an absent/padded station. Defaults to all-present.
    source_vec : array-like ``(n_cond,)``, optional
        Raw source vector for conditioned / relative-coord models. ``None`` ⇒ unconditioned.

    Returns
    -------
    torch.Tensor ``(1, W)`` — ready to hand to the embedding net's ``forward``/``embed``.
    """
    seis = torch.as_tensor(np.asarray(stacked_data), dtype=torch.float32)
    if seis.dim() != 3:
        raise ValueError(f"stacked_data must be (N, C, T); got shape {tuple(seis.shape)}")
    n_stations = seis.shape[0]

    coords = torch.as_tensor(np.asarray(station_coords), dtype=torch.float32)
    if coords.shape != (n_stations, 2):
        raise ValueError(
            f"station_coords must be (N, 2) matching stacked_data N={n_stations}; "
            f"got {tuple(coords.shape)}."
        )

    if present_mask is None:
        mask = torch.ones(n_stations, dtype=torch.bool)
    else:
        mask = torch.as_tensor(np.asarray(present_mask)).reshape(-1).to(torch.bool)
        if mask.shape[0] != n_stations:
            raise ValueError(
                f"present_mask must have length N={n_stations}; got {mask.shape[0]}."
            )

    sv = None if source_vec is None else torch.as_tensor(
        np.asarray(source_vec), dtype=torch.float32).reshape(-1)

    return pack_variable_context(seis, coords, mask, sv).unsqueeze(0)


def unpack_variable_context(
    ctx: torch.Tensor,
    n_components: int,
    trace_length: int,
    n_cond: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Inverse of :func:`pack_variable_context` for a batched context ``(B, W)``.

    Returns ``(seismograms (B,N,C,T), coords (B,N,2), mask (B,N) bool,
    source_vec (B,n_cond) | None)``.  ``N`` (== batch ``max_N``) is recovered from ``W``.
    """
    B, W = ctx.shape
    per_station = n_components * trace_length + 2 + 1
    payload = W - n_cond
    if payload <= 0 or payload % per_station != 0:
        raise ValueError(
            f"Variable-station context width {W} is inconsistent with "
            f"n_components={n_components}, trace_length={trace_length}, n_cond={n_cond} "
            f"(payload {payload} not divisible by per-station size {per_station})."
        )
    N = payload // per_station

    nct = N * n_components * trace_length
    # Split the four contiguous segments in one call (sizes sum to W, matching pack order).
    seis_flat, coords_flat, mask_flat, source_vec = torch.split(ctx, [nct, N * 2, N, n_cond], dim=1)
    seis = seis_flat.reshape(B, N, n_components, trace_length)
    coords = coords_flat.reshape(B, N, 2)
    mask = mask_flat > 0.5
    return seis, coords, mask, (source_vec if n_cond > 0 else None)
