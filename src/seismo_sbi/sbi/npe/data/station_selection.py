"""The stations a variable-station NPE training sample keeps.

:class:`StationSubsampler` draws a random station subset per sample, :func:`select_stations`
applies that draw or a fixed per-sample mask to one sample, and
:func:`variable_station_collate` pads the ragged samples of a batch into the one flat context
tensor the embedding network unpacks.
"""

import numpy as np
import torch

from seismo_sbi.sbi.npe.source_conditioning import pack_context, pack_variable_context


class StationSubsampler:
    """Randomly select a subset of station indices to emulate variable station configs.

    Each draw samples a *keep fraction* from a configurable distribution and keeps that
    fraction of the master station set (with a ``min_stations`` floor), returning the
    sorted kept indices so the canonical station ordering is preserved.  Uses the global
    numpy RNG, which the DataLoader's ``_seed_worker`` re-seeds per worker for
    reproducible, per-worker-distinct augmentation.

    Parameters
    ----------
    keep_fraction:
        A fixed fraction (``float``) or a ``(low, high)`` range drawn uniformly per sample.
        Default ``(0.5, 1.0)``.
    min_stations:
        Lower bound on the number of kept stations (also clamped to the available count).
    """

    def __init__(self, keep_fraction=(0.5, 1.0), min_stations: int = 1):
        self.keep_fraction = keep_fraction if keep_fraction is not None else (0.5, 1.0)
        self.min_stations = int(min_stations)

    def _draw_fraction(self) -> float:
        kf = self.keep_fraction
        if isinstance(kf, (int, float)):
            return float(kf)
        return float(np.random.uniform(*kf))

    def __call__(self, num_stations: int, available: np.ndarray = None) -> np.ndarray:
        """Draw kept station indices.

        available:
            Optional boolean mask over the master station axis. When given, the draw is
            restricted to stations flagged available -- used with an incomplete real-noise
            window so a station without noise is never kept (it would otherwise enter the
            model as exactly-zero data). The keep FRACTION is still drawn from the
            configured distribution and applied to the available count, so the dropout
            distribution is unchanged in shape; only its support shrinks.
        """
        if available is None:
            pool = np.arange(num_stations)
        else:
            pool = np.flatnonzero(np.asarray(available, dtype=bool)[:num_stations])
            if pool.size == 0:
                raise ValueError("StationSubsampler: no stations available for this sample")
        n_keep = int(round(self._draw_fraction() * pool.size))
        n_keep = min(pool.size, max(min(self.min_stations, pool.size), n_keep))
        return np.sort(np.random.choice(pool, size=n_keep, replace=False))


def select_stations(theta, x, source_vec, noise_present, *, item_mask, station_subsampler,
                    station_coords, torch_dtype):
    """The returned training item for one sample.

    ``x`` is ``(n_stations, n_components, n_samples)`` and ``station_coords`` the master
    ``(n_stations, 2)`` coordinates in the same order. With ``item_mask``, a ``(keep_indices,
    zero_channels)`` pair, or a ``station_subsampler``, the item is ``(theta, (x_sub, coords_sub,
    source_vec))``; otherwise it is ``(theta, x)`` with the source vector packed into ``x``.
    ``noise_present`` marks the stations the noise window carried, or is None.
    """
    if item_mask is not None:
        keep, zero_channels = item_mask
        for (si, ci) in (zero_channels or ()):
            x[si, ci, :] = 0.0
        keep = np.asarray(keep, dtype=int)
        x_sub = x[keep]
        coords_sub = torch.as_tensor(
            station_coords[keep], dtype=torch_dtype)
        return theta, (x_sub, coords_sub, source_vec)

    if station_subsampler is not None:
        num_stations = x.shape[0]
        keep = station_subsampler(num_stations, available=noise_present)
        x_sub = x[keep]                                            # (N_sub, C, T)
        coords_sub = torch.as_tensor(
            station_coords[keep], dtype=torch_dtype
        )                                                          # (N_sub, 2)
        return theta, (x_sub, coords_sub, source_vec)

    if noise_present is not None and not noise_present.all():
        raise RuntimeError(
            "RealNoiseSampler(allow_incomplete=True) produced a window missing "
            f"{int((~noise_present).sum())} station(s), but this dataset has no "
            "station_subsampler to mask them out -- they would enter the model as "
            "exactly-zero data. Use incomplete noise windows only with "
            "variable-station training."
        )

    if source_vec is not None:
        x = pack_context(x, source_vec)
    return theta, x


def variable_station_collate(batch):
    """Collate variable-station samples into a ragged-aware batch.

    Each item is ``(theta (D,), (x (N_i,C,T), coords (N_i,2), source_vec|None))``.  Pads
    every sample to the batch's ``max_N`` with zeros, builds a boolean validity mask
    ``(B, max_N)`` (True=real station), and packs each into the single flat context tensor
    the embedding net unpacks. Returns ``(theta (B,D), context (B,W))``.
    """
    thetas, samples = zip(*batch)
    xs, coords, source_vecs = zip(*samples)

    max_N = max(x.shape[0] for x in xs)
    C, T = xs[0].shape[1], xs[0].shape[2]
    dtype = xs[0].dtype
    has_source = source_vecs[0] is not None

    packed = []
    for x, crd, sv in zip(xs, coords, source_vecs):
        n = x.shape[0]
        x_pad = x.new_zeros((max_N, C, T)); x_pad[:n] = x
        crd_pad = crd.new_zeros((max_N, 2)); crd_pad[:n] = crd
        mask = torch.zeros(max_N, dtype=torch.bool); mask[:n] = True
        packed.append(pack_variable_context(x_pad, crd_pad, mask, sv if has_source else None))

    context = torch.stack(packed, dim=0)
    theta = torch.stack([torch.as_tensor(t, dtype=dtype) for t in thetas], dim=0)
    return theta, context
