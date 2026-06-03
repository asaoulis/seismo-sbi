"""
station_dropout.py
==================
Reusable utilities for evaluating a variable-station NPE under station selection /
fractional dropout. Groups with the rest of the variable-station machinery
(:mod:`source_conditioning` for packing, :mod:`dataloading.StationSubsampler` for the
training-time subsampling this mirrors at inference).

Two concerns:

* :class:`StationConfig` + :func:`make_dropout_configs` / :func:`config_from_kept` --
  choosing *which* stations a config keeps (pure, no torch).
* :func:`sample_station_dropout_ensemble` -- drawing the posterior for each config via the
  packed-subset inference path (the single shared loop both eval scripts use).

The plotting of the resulting ensemble lives in
:mod:`seismo_sbi.plotting.evaluation` (``plot_ensemble_lune_kde`` etc.).
"""
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


@dataclass
class StationConfig:
    """One named station configuration.

    ``keep`` are sorted integer indices into the master station-name list; ``kept`` /
    ``dropped`` are the corresponding names. ``n`` is the number of kept stations.
    """

    label: str
    keep: np.ndarray
    kept: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(len(self.keep))

    def as_dict(self) -> dict:
        """JSON-friendly view (used for the reproducibility ``station_configs.json``)."""
        return {
            "label": self.label,
            "n": self.n,
            "keep_indices": [int(i) for i in self.keep],
            "kept": list(self.kept),
            "dropped": list(self.dropped),
        }


def config_from_kept(station_names: Sequence[str], kept_names: Sequence[str],
                     label: str) -> StationConfig:
    """Build a :class:`StationConfig` keeping exactly ``kept_names`` (order = master order)."""
    names = list(station_names)
    keep_set = set(kept_names)
    keep = np.array([i for i, nm in enumerate(names) if nm in keep_set], dtype=int)
    kept = [names[i] for i in keep]
    dropped = [nm for nm in names if nm not in keep_set]
    return StationConfig(label=label, keep=keep, kept=kept, dropped=dropped)


def make_dropout_configs(station_names: Sequence[str], *, keep_fraction: float = 0.6,
                         n_subsets: int = 4, min_stations: int = 3, seed: int = 0,
                         include_full: bool = True) -> List[StationConfig]:
    """Build an ordered list of station configs for a dropout ensemble.

    The full master set first (if ``include_full``), then ``n_subsets`` **distinct**
    seeded random subsets, each keeping ``round(keep_fraction * N)`` stations (clamped to
    ``[min_stations, N - 1]`` so a subset always drops at least one station). Warns if too
    few distinct subsets are feasible for the requested ``n_subsets``.

    Mirrors the training-time :class:`dataloading.StationSubsampler` selection (which draws a
    *range* of fractions per sample); here a single ``keep_fraction`` is used for a clean,
    reproducible evaluation grid.
    """
    names = list(station_names)
    n_stations = len(names)
    if n_stations == 0:
        raise ValueError("station_names is empty.")
    full_idx = np.arange(n_stations)

    configs: List[StationConfig] = []
    if include_full:
        configs.append(StationConfig(
            label=f"all (N={n_stations})", keep=full_idx,
            kept=list(names), dropped=[]))

    k = int(round(keep_fraction * n_stations))
    k = max(min_stations, min(k, n_stations))
    if k >= n_stations:
        k = n_stations - 1  # a subset must drop at least one station

    rng = np.random.default_rng(seed)
    seen = {frozenset(full_idx.tolist())}
    n_made = 0
    attempts = 0
    while n_made < n_subsets and attempts < 1000:
        attempts += 1
        if k < 1:
            break
        keep = np.sort(rng.choice(n_stations, size=k, replace=False))
        key = frozenset(keep.tolist())
        if key in seen:
            continue
        seen.add(key)
        n_made += 1
        kept = [names[i] for i in keep]
        dropped = [names[i] for i in full_idx if i not in set(keep.tolist())]
        configs.append(StationConfig(
            label=f"rand {n_made} (N={k})", keep=keep, kept=kept, dropped=dropped))

    if n_made < n_subsets:
        print(f"WARNING: make_dropout_configs produced {n_made} distinct subsets "
              f"(requested {n_subsets}); N={n_stations}, k={k} limits the choices.")
    return configs


def sample_station_dropout_ensemble(posterior, obs, coords, configs: Sequence[StationConfig],
                                    data_scaler, *, num_samples: int, device=None,
                                    event_name: str = "", source_vec=None):
    """Sample the variable-station posterior for each station config.

    For each config, physically subset the observation rows + coords, pack with
    :func:`source_conditioning.pack_subset_observation` (the inference mirror of the
    training collate), sample the posterior and map back to physical units.

    Parameters
    ----------
    posterior : sbi DirectPosterior (or any object with ``.sample((n,), x)``).
    obs : array ``(N, C, T)`` -- the full observation, rows aligned to the master station list.
    coords : array ``(N, 2)`` -- station ``(lat, lon)`` in the same order.
    configs : sequence of :class:`StationConfig`.
    data_scaler : the training-time scaler; ``inverse_transform`` maps samples to physical MT.
    num_samples : posterior samples per config.
    device : torch device string; defaults to cuda if available.
    event_name : optional prefix for the per-config ``InversionResult.event_name``.
    source_vec : array ``(n_cond,)``, optional. Raw source-conditioning vector (e.g. the event's
        ``[latitude, longitude, depth]`` in ``ml_conditioning.param_map`` order) required by a
        **conditioned** model at inference. ``None`` (default) ⇒ unconditioned model; the source
        vector is shared across all station configs of one event (same source, different stations).

    Returns
    -------
    (ensemble, results) : ``OrderedDict[label -> InversionData]`` and ``list[InversionResult]``
        (the latter ready to pickle as ``(None, None, results)``).
    """
    import torch
    from seismo_sbi.sbi.compression.ML.source_conditioning import pack_subset_observation
    from seismo_sbi.sbi.types.results import InversionData, InversionResult, InversionConfig

    obs = np.asarray(obs)
    coords = np.asarray(coords, dtype=float)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ensemble = OrderedDict()
    results = []
    for c in configs:
        ctx = pack_subset_observation(
            obs[c.keep], coords[c.keep], source_vec=source_vec).to(device)   # (1, W)
        samples = posterior.sample((num_samples,), ctx, show_progress_bars=False)
        phys = data_scaler.inverse_transform(np.asarray(samples.cpu().numpy()))  # (num_samples, 6)
        inv = InversionData(theta0=None, samples=phys, data_scaler=data_scaler)
        ensemble[c.label] = inv
        results.append(InversionResult(
            event_name=f"{event_name}:{c.label}" if event_name else c.label,
            inversion_data=inv,
            inversion_config=InversionConfig(
                train_noise="", test_noise="real", inversion_method="ml_compressor"),
        ))
    return ensemble, results
