"""Posterior samples of a variable-station model for chosen station subsets.

:func:`robust_posterior_sample` and its batched form draw samples inside the prior box.
:class:`StationConfig`, :func:`make_dropout_configs` and :func:`config_from_kept` choose the
stations a configuration keeps, as the training-time
:class:`~seismo_sbi.sbi.npe.data.station_selection.StationSubsampler` does, and
:func:`sample_station_dropout_ensemble` draws the posterior for each configuration; the figures
live in :mod:`seismo_sbi.plotting.evaluation`.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from seismo_sbi.sbi.types.results import InversionData, InversionResult, InversionConfig


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

    Mirrors the training-time :class:`~seismo_sbi.sbi.npe.data.station_selection.StationSubsampler` selection (which draws a
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


def _prior_bounds(prior):
    """(low, high) tensors for a BoxUniform prior, else (None, None)."""
    base = getattr(prior, "base_dist", prior)
    return getattr(base, "low", None), getattr(base, "high", None)


def _flow_sample(estimator, ctx, n):
    """Draw ``n`` samples from the (nflows) posterior estimator conditioned on ``ctx``,
    bypassing sbi's prior-rejection.  Robust to the context kwarg name."""
    import torch
    with torch.no_grad():
        try:
            s = estimator.sample(n, context=ctx)
        except TypeError:
            s = estimator.sample((n,), condition=ctx)
    return s.reshape(-1, s.shape[-1])


def robust_posterior_sample(posterior, ctx, num_samples, *, oversample=4, max_factor=64):
    """Sample a sbi ``DirectPosterior`` WITHOUT the leakage-rejection hang.

    sbi 0.21's ``DirectPosterior.sample`` rejects flow draws outside the prior box to
    correct for leakage; if the trained flow puts ~all mass outside the prior for an
    out-of-distribution observation, the acceptance rate is ~0% and the rejection loop
    never terminates (the ``Only 0.000% proposal samples are accepted`` warning that
    stalled some catalogue events).

    This draws directly from the underlying flow in bounded batches, keeps the
    in-prior-box samples (identical to sbi's rejection for the healthy, high-acceptance
    case), and — only when the posterior leaks so badly it cannot fill the request
    within ``max_factor`` × ``num_samples`` candidates — tops up with flow samples
    *clipped* to the prior box so the call always returns ``num_samples`` and never
    hangs.  Returns a torch tensor ``(num_samples, dim)`` on the estimator's device.
    """
    import torch
    est = getattr(posterior, "posterior_estimator", None)
    if est is None:
        # Duck-typed posterior (no sbi flow internals to bypass): the plain sample
        # path IS the whole contract — leakage rejection only exists on DirectPosterior.
        return posterior.sample((num_samples,), ctx, show_progress_bars=False)
    prior = getattr(posterior, "_prior", None) or getattr(posterior, "prior", None)
    lo, hi = _prior_bounds(prior)
    batch = max(int(num_samples * oversample), int(num_samples))
    cap = int(num_samples * max_factor)
    collected, n_have, total, last = [], 0, 0, None
    while n_have < num_samples and total < cap:
        s = _flow_sample(est, ctx, batch)
        last, total = s, total + s.shape[0]
        if lo is not None and hi is not None:
            acc = s[((s >= lo) & (s <= hi)).all(dim=-1)]
        else:
            acc = s
        if acc.shape[0]:
            collected.append(acc)
            n_have += acc.shape[0]
        elif n_have == 0 and total >= num_samples * 8:
            break          # clearly leaking -> stop early, fall through to clip
    if n_have >= num_samples:
        return torch.cat(collected)[:num_samples]
    # leakage fallback: pad with clipped flow samples (best-effort, never hang/short)
    need = num_samples - n_have
    filler = last if last is not None else _flow_sample(est, ctx, need)
    if lo is not None and hi is not None:
        filler = torch.max(torch.min(filler, hi), lo)
    return torch.cat(collected + [filler[:need]])[:num_samples]


def sample_station_dropout_ensemble(posterior, obs, coords, configs: Sequence[StationConfig],
                                    data_scaler, *, num_samples: int, device=None,
                                    event_name: str = "", source_vec=None):
    """Sample the variable-station posterior for each station config.

    For each config, physically subset the observation rows + coords, pack with
    :func:`~seismo_sbi.sbi.npe.source_conditioning.pack_subset_observation` (the inference mirror of the
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
    from seismo_sbi.sbi.npe.source_conditioning import pack_subset_observation

    obs = np.asarray(obs)
    coords = np.asarray(coords, dtype=float)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ensemble = OrderedDict()
    results = []
    for c in configs:
        ctx = pack_subset_observation(
            obs[c.keep], coords[c.keep], source_vec=source_vec).to(device)   # (1, W)
        samples = robust_posterior_sample(posterior, ctx, num_samples)
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


# --- Batched variable-station inference: pads like variable_station_collate, so an item
# batched is bit-comparable to the same item packed alone. ---

def pack_subset_batch(items, device=None):
    """Pack many ``(seismograms, coords, source_vec)`` subsets into one ``(B, W)`` context.

    ``items`` is a sequence of ``(obs (N_i, C, T), coords (N_i, 2), source_vec | None)``;
    the ``N_i`` may differ.  Every item is zero-padded to ``max_N`` with a False mask
    entry, mirroring :func:`~seismo_sbi.sbi.npe.data.station_selection.variable_station_collate`, then flattened by
    :func:`~seismo_sbi.sbi.npe.source_conditioning.pack_variable_context`.

    Returns a ``(B, W)`` float32 tensor on ``device`` (default: the current CUDA device
    if available, else CPU).
    """
    import torch
    from seismo_sbi.sbi.npe.source_conditioning import pack_variable_context

    if not len(items):
        raise ValueError("pack_subset_batch: empty item list")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seis = [torch.as_tensor(np.asarray(x), dtype=torch.float32) for x, _, _ in items]
    crds = [torch.as_tensor(np.asarray(c), dtype=torch.float32) for _, c, _ in items]
    for i, (s, c) in enumerate(zip(seis, crds)):
        if s.dim() != 3:
            raise ValueError(f"item {i}: seismograms must be (N, C, T), got {tuple(s.shape)}")
        if c.shape != (s.shape[0], 2):
            raise ValueError(f"item {i}: coords must be (N, 2) matching N={s.shape[0]}, "
                             f"got {tuple(c.shape)}")
    max_n = max(s.shape[0] for s in seis)
    C, T = seis[0].shape[1], seis[0].shape[2]

    packed = []
    for s, c, (_, _, sv) in zip(seis, crds, items):
        n = s.shape[0]
        s_pad = s.new_zeros((max_n, C, T)); s_pad[:n] = s
        c_pad = c.new_zeros((max_n, 2)); c_pad[:n] = c
        mask = torch.zeros(max_n, dtype=torch.bool); mask[:n] = True
        svt = None if sv is None else torch.as_tensor(np.asarray(sv), dtype=torch.float32)
        packed.append(pack_variable_context(s_pad, c_pad, mask, svt))
    return torch.stack(packed, dim=0).to(device)


#: Rows (context x samples) a single nflows LU triangular solve will accept on CUDA.
#: ``cublasStrsmBatched`` raises ``CUBLAS_STATUS_NOT_SUPPORTED`` at 2**19 rows, so the
#: auto-chunking below stays a comfortable factor under the observed cliff.
TRSM_SAFE_ROWS = 2 ** 18


def flow_sample_chunked(est, ctx, num_samples, *, chunk=None):
    """Draw ``(B, num_samples, D)`` from an nflows estimator, embedding the context ONCE.

    Why this exists instead of ``est.sample(n, context=ctx)``:

    * the nflows LU transform's batched triangular solve fails with
      ``CUBLAS_STATUS_NOT_SUPPORTED`` once ``B * num_samples`` exceeds 2**19 rows, which
      caps the usable context batch far below the GPU's memory;
    * ``nflows.distributions.Distribution.sample(..., batch_size=...)``, which looks like
      the fix, is broken for the CONDITIONAL case — it concatenates the chunks (``torch.cat``) along
      dim 0, which is the *context* dimension when a context is given, so the returned
      tensor is mis-shaped. It also re-runs the embedding net per chunk.

    Chunking the SAMPLE dimension while reusing one embedding decouples the encoder batch
    from that cuBLAS limit. Falls back to ``est.sample`` when the estimator does not expose
    the nflows internals (any non-nflows / stub estimator).
    """
    import torch
    from nflows.utils import torchutils

    needed = (getattr(est, "_embedding_net", None), getattr(est, "_distribution", None),
              getattr(est, "_transform", None))
    if any(x is None for x in needed):
        with torch.no_grad():
            return est.sample(num_samples, context=ctx)
    embed, dist, transform = needed
    if chunk is None or chunk >= num_samples:
        with torch.no_grad():
            return est.sample(num_samples, context=ctx)

    with torch.no_grad():
        emb = embed(ctx)                                        # (B, E) — computed ONCE
        parts, drawn = [], 0
        while drawn < num_samples:
            m = min(chunk, num_samples - drawn)
            noise = dist.sample(m, context=emb)                 # (B, m, D)
            flat = torchutils.merge_leading_dims(noise, num_dims=2)
            rep = torchutils.repeat_rows(emb, num_reps=m)
            out, _ = transform.inverse(flat, context=rep)
            parts.append(torchutils.split_leading_dim(out, shape=[-1, m]))
            drawn += m
        return torch.cat(parts, dim=1)                          # concat along SAMPLES


def robust_posterior_sample_batched(posterior, ctx, num_samples, *, oversample=2,
                                    max_rounds=6, flow_chunk=None):
    """Batched counterpart of :func:`robust_posterior_sample`: ``(B, W) -> (B, n, D)``.

    Draws ``num_samples * oversample`` candidates for the whole batch in ONE flow call,
    keeps each row's first ``num_samples`` in-prior-box draws, and re-draws only for the
    rows that came up short (up to ``max_rounds``).  Rows that still cannot be filled —
    the pathological leakage case ``robust_posterior_sample`` exists to survive — are
    topped up with samples *clipped* into the prior box, so the call always returns a
    full ``(B, num_samples, D)`` tensor and never hangs.

    ``flow_chunk`` caps how many draws per row go through the flow at once (see
    :func:`flow_sample_chunked`); ``None`` sends them all in one call.

    Equivalent in distribution to calling :func:`robust_posterior_sample` per row; it is
    NOT sample-identical (one shared RNG stream for the batch).
    """
    import torch
    est = getattr(posterior, "posterior_estimator", None)
    if est is None:
        return torch.stack([posterior.sample((num_samples,), c.unsqueeze(0),
                                             show_progress_bars=False) for c in ctx])
    prior = getattr(posterior, "_prior", None) or getattr(posterior, "prior", None)
    lo, hi = _prior_bounds(prior)
    B = ctx.shape[0]
    out, filled = None, torch.zeros(B, dtype=torch.long)
    todo = torch.arange(B)
    last, last_todo = None, None
    for _ in range(max_rounds):
        if todo.numel() == 0:
            break
        s = flow_sample_chunked(est, ctx[todo], int(num_samples * oversample),
                                chunk=flow_chunk)                              # (b, m, D)
        if out is None:
            out = s.new_zeros((B, num_samples, s.shape[-1]))
        last, last_todo = s, todo
        for j, i in enumerate(todo.tolist()):
            cand = s[j]
            if lo is not None and hi is not None:
                cand = cand[((cand >= lo) & (cand <= hi)).all(dim=-1)]
            take = min(int(cand.shape[0]), num_samples - int(filled[i]))
            if take > 0:
                out[i, filled[i]:filled[i] + take] = cand[:take]
                filled[i] += take
        todo = torch.nonzero(filled < num_samples, as_tuple=False).reshape(-1)
    if out is None:                                   # max_rounds == 0
        raise RuntimeError("robust_posterior_sample_batched: no sampling round ran")
    if todo.numel():                                  # leakage fallback: clip
        # ``last`` rows follow the previous round's todo, so look each row up by its position there.
        row_of = {int(v): j for j, v in enumerate(last_todo.tolist())}
        for i in todo.tolist():
            filler = last[row_of.get(int(i), last.shape[0] - 1)]
            if lo is not None and hi is not None:
                filler = torch.max(torch.min(filler, hi), lo)
            need = num_samples - int(filled[i])
            out[i, filled[i]:] = filler[:need]
            filled[i] = num_samples
    return out


def sample_subsets_batched(posterior, items, data_scaler, *, num_samples: int,
                           device=None, max_batch: int = 64, oversample: float = 2,
                           flow_chunk=None, progress=None):
    """Sample the variable-station posterior for MANY subsets, ``max_batch`` at a time.

    ``items``: sequence of ``(obs (N_i, C, T), coords (N_i, 2), source_vec | None)`` — one
    entry per (event, station-subset) to invert.  Returns a list of ``(num_samples, D)``
    numpy arrays in PHYSICAL units (``data_scaler.inverse_transform`` applied), aligned
    with ``items``.

    ``max_batch`` trades GPU memory for throughput; peak memory is dominated by the
    ``max_batch * num_samples * oversample`` flow draws and their repeated context
    embedding.  ``flow_chunk`` keeps each flow call under the nflows/cuBLAS batched-solve
    limit so ``max_batch`` can be raised past it (see :func:`flow_sample_chunked`).
    ``progress`` is an optional ``callable(n_done, n_total)``.
    """
    out = []
    n = len(items)
    for start in range(0, n, max_batch):
        chunk = items[start:start + max_batch]
        ctx = pack_subset_batch(chunk, device=device)
        fc = flow_chunk if flow_chunk is not None else max(1, TRSM_SAFE_ROWS // len(chunk))
        s = robust_posterior_sample_batched(posterior, ctx, num_samples,
                                            oversample=oversample, flow_chunk=fc)
        arr = s.detach().cpu().numpy()
        for row in arr:
            out.append(data_scaler.inverse_transform(row) if data_scaler is not None else row)
        del ctx, s
        if progress is not None:
            progress(min(start + max_batch, n), n)
    return out
