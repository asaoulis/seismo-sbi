"""Calibration and accuracy metrics for a trained model's validation posteriors.

:func:`tarp_coverage` computes TARP expected coverage; :func:`compute_evaluation_metrics` turns a
validation result into per-parameter bias, width, coverage and figures of merit, including the
derived gamma, delta, Mw, strike, dip and rake; :func:`write_run_metrics` and
:func:`scan_run_metrics` store and collect them per run; :func:`spread_stats` summarises the
source-type spread of one posterior.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

METRICS_FILENAME = "evaluation_metrics.json"
METRICS_SCHEMA_VERSION = 1
# Names of the 6 moment-tensor components (pipeline up-south-east convention).
_MT_NAMES = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]
# Derived source-type / orientation quantities and the subsets used for the
# figure-of-merit sqrt(det C).
_DERIVED_NAMES = ["gamma", "delta", "Mw", "strike", "dip", "rake"]
_FOM_SUBSETS = {
    "delta_gamma": ("derived", [1, 0]),     # δ, γ
    "sdr": ("derived", [3, 4, 5]),          # strike, dip, rake
    "full_mt": ("mt", list(range(6))),      # all 6 MT components
}


def spread_stats(mt_samples) -> Dict[str, float]:
    """Median + 68% interval width of gamma/delta/Mw for an (N, 6) MT sample set."""
    gamma, delta, mw = _gamma_delta_mw(np.asarray(mt_samples))

    def med_width(a):
        a = np.asarray(a, dtype=float)
        return float(np.median(a)), float(np.percentile(a, 84) - np.percentile(a, 16))

    g_med, g_w = med_width(gamma)
    d_med, d_w = med_width(delta)
    m_med, m_w = med_width(mw)
    return {
        "gamma_deg_median": g_med, "gamma_deg_width68": g_w,
        "delta_deg_median": d_med, "delta_deg_width68": d_w,
        "Mw_median": m_med, "Mw_width68": m_w,
    }


def tarp_coverage(samples_per_sim: np.ndarray, theta_true: np.ndarray,
                  num_bootstrap: int = 100, references: str = "random",
                  metric: str = "euclidean", seed: Optional[int] = 0,
                  num_alpha_bins: Optional[int] = None):
    """
    Compute TARP expected-coverage with bootstrap error bands.

    Parameters
    ----------
    samples_per_sim : np.ndarray, shape (n_posterior_samples, n_sims, n_dims)
        Posterior samples for each validation simulation.
    theta_true : np.ndarray, shape (n_sims, n_dims)
        Ground-truth parameter vector for each simulation.

    Returns
    -------
    (ecp_bootstrap, alpha) : tuple
        ``ecp_bootstrap`` shape ``(num_bootstrap, num_alpha)``, ``alpha`` shape
        ``(num_alpha,)`` — exactly the form ``plotting.coverage.plot_coverage``
        expects as a coverage_dict value.
    """
    from tarp import get_tarp_coverage

    samples_per_sim = np.asarray(samples_per_sim)
    theta_true = np.asarray(theta_true)
    if samples_per_sim.ndim != 3:
        raise ValueError(
            f"samples_per_sim must be (n_samples, n_sims, n_dims), got {samples_per_sim.shape}")
    if theta_true.shape[0] != samples_per_sim.shape[1]:
        raise ValueError(
            f"theta_true n_sims ({theta_true.shape[0]}) != samples n_sims "
            f"({samples_per_sim.shape[1]})")

    # The default bin count is ``n_sims // 10``, which is zero for a small validation run,
    # so it is floored to keep a coarse coverage curve.
    n_sims = samples_per_sim.shape[1]
    if num_alpha_bins is None:
        num_alpha_bins = max(2, n_sims // 10)

    ecp, alpha = get_tarp_coverage(
        samples_per_sim, theta_true, references=references, metric=metric,
        num_alpha_bins=num_alpha_bins, num_bootstrap=num_bootstrap,
        norm=True, bootstrap=True, seed=seed,
    )
    return ecp, alpha


def _gamma_delta_mw(mt_samples: np.ndarray):
    """Vectorised (gamma_deg, delta_deg) and per-sample Mw for (N,6) MT samples."""
    from seismo_sbi.moment_tensor.lune_angles import mts6_to_gamma_delta
    from seismo_sbi.moment_tensor.decomposition import get_MW_and_epsilon
    gamma, delta = mts6_to_gamma_delta(mt_samples)
    mw = np.array([get_MW_and_epsilon(s)[0] for s in mt_samples])
    return gamma, delta, mw


def _derived_matrix(mt_samples: np.ndarray) -> np.ndarray:
    """
    Map ``(N, 6)`` moment-tensor vectors to ``(N, 6)`` derived quantities
    ``(gamma, delta, Mw, strike, dip, rake)`` — degrees for the angles. Reuses the
    project's lune / pyrocko converters; the nodal-plane ambiguity is resolved by
    consistently taking the first plane (as ``MomentTensorReparametrised`` does).
    """
    from seismo_sbi.moment_tensor.decomposition import get_nodal_planes

    mt_samples = np.asarray(mt_samples, dtype=float)
    gamma, delta, mw = _gamma_delta_mw(mt_samples)
    out = np.empty((len(mt_samples), 6), dtype=float)
    for i, mt in enumerate(mt_samples):
        sdr = get_nodal_planes(mt)[0]
        out[i] = [gamma[i], delta[i], mw[i], float(sdr[0]), float(sdr[1]), float(sdr[2])]
    return out


def _per_param_stats(truth: np.ndarray, samples: np.ndarray) -> Dict[str, list]:
    """
    Per-parameter summary across the validation set.

    ``truth`` shape ``(n_sims, n_dims)``; ``samples`` shape
    ``(n_samples, n_sims, n_dims)``. All values are lists over the n_dims, each the
    mean over the validation examples (except ``rmse`` which is a root-mean-square).
    Returns keys: ``std, bias, mae, rmse, ci68, ci90, iqr``.
    """
    truth = np.asarray(truth, dtype=float)
    samples = np.asarray(samples, dtype=float)
    n_sims, n_dims = truth.shape

    med = np.median(samples, axis=0)                                   # (n_sims, n_dims)
    std = np.std(samples, axis=0)                                      # (n_sims, n_dims)
    err = med - truth                                                  # signed error

    def width(lo_q, hi_q):
        lo = np.percentile(samples, lo_q, axis=0)
        hi = np.percentile(samples, hi_q, axis=0)
        return hi - lo                                                 # (n_sims, n_dims)

    return {
        "std": std.mean(axis=0).tolist(),
        "bias": err.mean(axis=0).tolist(),
        "mae": np.abs(err).mean(axis=0).tolist(),
        "rmse": np.sqrt((err ** 2).mean(axis=0)).tolist(),
        "ci68": width(16, 84).mean(axis=0).tolist(),
        "ci90": width(5, 95).mean(axis=0).tolist(),
        "iqr": width(25, 75).mean(axis=0).tolist(),
    }


def _named(stats: Dict[str, list], names: List[str]) -> Dict[str, Dict[str, float]]:
    """Transpose ``{metric: [per-dim]}`` to ``{param_name: {metric: value}}``."""
    return {name: {m: float(stats[m][k]) for m in stats}
            for k, name in enumerate(names)}


def _empirical_coverage(truth: np.ndarray, samples: np.ndarray, q: float) -> float:
    """
    Mean (over params) fraction of examples whose truth falls inside the central
    ``q`` credible interval of the marginal posterior. ``q`` e.g. 0.68 / 0.90.
    """
    lo_q, hi_q = (1 - q) / 2 * 100, (1 - (1 - q) / 2) * 100
    lo = np.percentile(samples, lo_q, axis=0)                          # (n_sims, n_dims)
    hi = np.percentile(samples, hi_q, axis=0)
    inside = (truth >= lo) & (truth <= hi)                             # (n_sims, n_dims)
    return float(inside.mean())


def _mean_sqrt_det_cov(samples_subset: np.ndarray) -> float:
    """
    Mean over examples of ``sqrt(det Cov)`` of the posterior on a parameter subset.
    ``samples_subset`` shape ``(n_samples, n_sims, k)``. A 1-D subset reduces to the
    mean posterior std. Non-positive determinants (numerical / degenerate) floored.
    """
    n_samples, n_sims, k = samples_subset.shape
    vals = np.empty(n_sims)
    for j in range(n_sims):
        if k == 1:
            vals[j] = np.std(samples_subset[:, j, 0])
        else:
            cov = np.cov(samples_subset[:, j, :].T)
            vals[j] = np.sqrt(max(np.linalg.det(cov), 0.0))
    return float(vals.mean())


def compute_evaluation_metrics(val: dict, ecp=None, alpha=None,
                               derived_max_samples: int = 400) -> dict:
    """
    Post-process a validation-inference result (the dict returned by
    ``evaluate_validation_set``) into concrete per-run performance metrics.

    Computes per-parameter spread / bias / error in **moment-tensor space** and, when
    the inference is the 6-component moment tensor, in the **derived space**
    (γ, δ, Mw, strike, dip, rake); the ``sqrt(det C)`` figure-of-merit for the
    {δ, γ}, {strike, dip, rake} and full-MT subsets; empirical credible-interval
    coverage; and (if a TARP curve is supplied) its calibration error.

    Parameters
    ----------
    val : dict
        Must contain ``theta_phys`` (n_sims, n_dims) and ``samples_phys``
        (n_samples, n_sims, n_dims). Derived-space metrics + the δγ/sdr FoM are only
        computed when ``n_dims == 6``.
    ecp, alpha : optional
        TARP coverage outputs (``ecp`` shape (n_bootstrap, n_alpha), ``alpha`` shape
        (n_alpha,)) — used for the calibration-error scalar.
    derived_max_samples : int
        Cap on posterior draws per example used for the (pyrocko-heavy) derived-space
        metrics. MT-space metrics + coverage always use all draws.
    """
    theta = np.asarray(val["theta_phys"], dtype=float)
    samples = np.asarray(val["samples_phys"], dtype=float)
    n_samples, n_sims, n_dims = samples.shape

    metrics: dict = {
        "meta": {"n_val": int(n_sims), "num_samples": int(n_samples), "n_dims": int(n_dims)},
        "mt_space": {},
        "derived": {},
        "figure_of_merit": {},
        "coverage": {},
    }

    # --- moment-tensor-space per-parameter metrics --------------------------- #
    mt_names = _MT_NAMES if n_dims == 6 else [f"theta_{i}" for i in range(n_dims)]
    metrics["mt_space"] = _named(_per_param_stats(theta, samples), mt_names)

    # --- coverage (cheap, marginal credible intervals in MT space) ----------- #
    metrics["coverage"]["emp_68"] = _empirical_coverage(theta, samples, 0.68)
    metrics["coverage"]["emp_90"] = _empirical_coverage(theta, samples, 0.90)
    if ecp is not None and alpha is not None:
        ecp = np.asarray(ecp, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        ecp_mean = ecp.mean(axis=0) if ecp.ndim == 2 else ecp
        ecp_std = ecp.std(axis=0) if ecp.ndim == 2 else np.zeros_like(ecp_mean)
        metrics["coverage"]["tarp_calibration_error"] = float(
            np.trapz(np.abs(ecp_mean - alpha), alpha))
        # Persist the actual coverage curve + bootstrap CI bands so it can be
        # re-plotted and overlaid across runs without re-running inference.
        metrics["coverage"]["tarp_curve"] = {
            "alpha": alpha.tolist(),
            "ecp_mean": ecp_mean.tolist(),
            "ecp_std": ecp_std.tolist(),
        }

    # --- figure of merit (full MT always; δγ/sdr need the derived space) ----- #
    metrics["figure_of_merit"]["full_mt"] = _mean_sqrt_det_cov(samples[:, :, :6]) \
        if n_dims >= 6 else _mean_sqrt_det_cov(samples)

    if n_dims == 6:
        # Subsample draws for the pyrocko-per-sample derived conversion.
        if derived_max_samples and n_samples > derived_max_samples:
            sel = np.linspace(0, n_samples - 1, derived_max_samples).astype(int)
        else:
            sel = np.arange(n_samples)

        derived_truth = _derived_matrix(theta)                          # (n_sims, 6)
        derived_samples = np.empty((len(sel), n_sims, 6))
        for j in range(n_sims):
            derived_samples[:, j, :] = _derived_matrix(samples[sel, j, :])

        metrics["derived"] = _named(
            _per_param_stats(derived_truth, derived_samples), _DERIVED_NAMES)

        for name, (space, cols) in _FOM_SUBSETS.items():
            if space == "derived":
                metrics["figure_of_merit"][name] = _mean_sqrt_det_cov(
                    derived_samples[:, :, cols])
        # scalar Mw "FoM" is just its mean posterior std
        metrics["figure_of_merit"]["Mw"] = float(metrics["derived"]["Mw"]["std"])

    return metrics


def write_run_metrics(out_dir, run_label: str, config_path, metrics: dict) -> Path:
    """
    Dump ``evaluation_metrics.json`` into a run's artifacts dir in a self-describing,
    diffable format. Returns the written path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "run_label": run_label,
        "config": str(config_path),
        "timestamp": time.time(),
        "metrics": metrics,
    }
    path = out_dir / METRICS_FILENAME
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def scan_run_metrics(runs_root, exclude=None, max_runs: int = 12) -> Dict[str, dict]:
    """
    Discover per-run metrics files for cross-run comparison.

    Globs ``<runs_root>/*/artifacts/evaluation_metrics.json``, drops unreadable /
    empty files and any label in ``exclude``, dedupes by ``run_label`` (newest
    ``timestamp`` wins) and keeps the newest ``max_runs`` (safety cap so the
    comparison can't blow up). Returns ``{run_label: payload}``.
    """
    runs_root = Path(runs_root)
    exclude = set(exclude or [])
    best: Dict[str, dict] = {}
    for p in runs_root.glob(f"*/artifacts/{METRICS_FILENAME}"):
        try:
            payload = json.load(open(p))
        except Exception:
            continue
        if not payload.get("metrics"):
            continue
        label = payload.get("run_label") or p.parent.parent.name
        if label in exclude:
            continue
        ts = payload.get("timestamp", p.stat().st_mtime)
        payload.setdefault("timestamp", ts)
        if label not in best or ts >= best[label].get("timestamp", 0):
            best[label] = payload
    # newest max_runs by timestamp
    ordered = sorted(best.items(), key=lambda kv: kv[1].get("timestamp", 0), reverse=True)
    return dict(ordered[:max_runs])
