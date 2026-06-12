"""
evaluation.py
=============
General, run-agnostic helpers for the **final evaluation stage** of a continuity /
training run: load inversion result pickles, overlay a freshly trained model's
posterior against the frozen gold-standard inversions, and produce calibration /
recovery diagnostics (TARP coverage, per-parameter recovery scatter).

This module is intentionally **import-light** at the top level: the heavy plotting
dependencies (basemap, pyrocko, chainconsumer, scienceplots, torch) and ``tarp``
are imported lazily inside the functions that need them, so importing this module
(e.g. during the fast unit-test gate) does not pull the whole plotting stack.

The recovery/lune/chainconsumer plots are thin wrappers over the existing
``seismo_sbi.plotting`` code (``SBIPipelinePlotter`` / ``PosteriorPlotter``); the
data-assembly helpers (``load_inversion_pkl``, ``discover_ml_runs``,
``build_recovery_dict``) and ``tarp_coverage`` are dependency-light and unit-tested.
"""
from __future__ import annotations

import glob
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

METRICS_FILENAME = "evaluation_metrics.json"
METRICS_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Pickle / run discovery
# --------------------------------------------------------------------------- #
def load_inversion_pkl(path) -> List:
    """
    Load an inversion results pickle and return its list of ``InversionResult``.

    Handles both shapes produced in the repo:
      - ``(job_data, job_results, inversion_results)`` from ``event_inversion.py``
      - ``(None, None, [InversionResult(...)])`` from ``run_ml_inversion.py``
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not (isinstance(data, (tuple, list)) and len(data) == 3):
        raise ValueError(
            f"{path}: expected a 3-tuple (job_data, job_results, inversion_results), "
            f"got {type(data)}"
        )
    inversion_results = data[2]
    return list(inversion_results) if inversion_results else []


def results_by_method(results: List) -> Dict[str, object]:
    """Map ``inversion_config.inversion_method`` -> ``InversionResult`` (last wins)."""
    return {r.inversion_config.inversion_method: r for r in results}


def discover_ml_runs(pipeline_outputs_dir) -> Dict[str, Path]:
    """
    Discover ML inversion pickles produced by the continuity pipeline.

    Globs ``<pipeline_outputs_dir>/continuity_ml_*/inversion_results_ml.pkl`` and
    returns ``{run_label: pkl_path}`` where ``run_label`` is the directory suffix
    after ``continuity_ml_`` (e.g. ``train_cnn_nonuisance``).
    """
    base = Path(pipeline_outputs_dir)
    out: Dict[str, Path] = {}
    for pkl in sorted(base.glob("continuity_ml_*/inversion_results_ml.pkl")):
        label = pkl.parent.name[len("continuity_ml_"):]
        out[label] = pkl
    return out


# --------------------------------------------------------------------------- #
# Recovery dict (gold standard + ML runs) for lune / chainconsumer plots
# --------------------------------------------------------------------------- #
# Friendly labels for the standard gold-standard methods.
_METHOD_LABELS = {
    "theory_optimal_score": "Optimal Score",
    "optimal_score": "Optimal Score",
    "gaussian_likelihood_theory_optimal_score": "Gaussian",
    "gaussian_likelihood_optimal_score": "Gaussian",
    "ml_compressor": "NPE ML",
}


def _pretty_method(method: str) -> str:
    return _METHOD_LABELS.get(method, method)


def build_recovery_dict(
    gold_results: List,
    ml_runs: Optional[Dict[str, List]] = None,
    gold_methods=("theory_optimal_score", "gaussian_likelihood_theory_optimal_score"),
) -> "dict":
    """
    Assemble an ordered ``{label: InversionData}`` dict for overlaying on the same
    lune / chainconsumer axes.

    The first entry is the gold-standard score-compression result (so its ``theta0``
    — the gold MLE — becomes the reference/truth line in the chainconsumer plots).
    Then the Gaussian-likelihood gold result, then each ML run.

    Parameters
    ----------
    gold_results : list of InversionResult
        From the gold-standard ``inversion_results.pkl``.
    ml_runs : dict[label -> list of InversionResult], optional
        One entry per trained run to overlay (e.g. ``{"train_cnn_nonuisance": [...]}``).
    gold_methods : tuple
        Which gold methods to include, in order. Missing methods are skipped.
    """
    recovery = {}
    gold_map = results_by_method(gold_results) if gold_results else {}

    for method in gold_methods:
        if method in gold_map:
            recovery[_pretty_method(method)] = gold_map[method].inversion_data

    if ml_runs:
        multiple = len(ml_runs) > 1
        for label, results in ml_runs.items():
            ml_map = results_by_method(results)
            ml_res = ml_map.get("ml_compressor")
            if ml_res is None and results:
                ml_res = results[0]  # ML pkls carry a single result
            if ml_res is None:
                continue
            key = f"NPE ML ({label})" if multiple else "NPE ML"
            recovery[key] = ml_res.inversion_data
    return recovery


# --------------------------------------------------------------------------- #
# Lune recovery plot (with ISO/CLVD/DC decomposition beachballs)
# --------------------------------------------------------------------------- #
def add_decomposition_beachballs(ax, theta0_mt, posterior_plotter, color="salmon"):
    """
    Add scaled ISO / CLVD / DC beachballs + percentages to the left of a lune axis,
    in axes coordinates. ``theta0_mt`` is a 6-component moment tensor in the pipeline
    (up-south-east) convention. Adapted from the project's reference snippet.
    """
    import numpy as np
    from pyrocko import moment_tensor as pmt
    from seismo_sbi.plotting.distributions import create_matrix
    from seismo_sbi.plotting.rocko_beachball_patch import plot_beachball_on_axes

    mt_matrix = create_matrix(posterior_plotter.convert_mt_convention(theta0_mt))
    mt_rocko = pmt.MomentTensor(m_up_south_east=mt_matrix)
    res = mt_rocko.standard_decomposition()

    _, rat_iso, m_iso = res[0]
    _, rat_dc, m_dc = res[1]
    _, rat_clvd, m_clvd = res[2]
    p_iso, p_dc, p_clvd = rat_iso * 100, rat_dc * 100, rat_clvd * 100

    max_rat = max(rat_iso, rat_dc, rat_clvd) or 1.0
    d_iso = (rat_iso / max_rat) * 1.0
    d_dc = ((rat_dc / max_rat) * 1.0) ** (1 / 4)
    d_clvd = ((rat_clvd / max_rat) * 1.0) ** (1 / 4)

    bb_size = 0.22
    x_pos = -0.3
    rect_iso = [x_pos, 0.85 - bb_size / 2, bb_size, bb_size]
    rect_clvd = [x_pos, 0.65 - bb_size / 2, bb_size, bb_size]
    rect_dc = [x_pos, 0.45 - bb_size / 2, bb_size, bb_size]

    ax_iso = ax.inset_axes(rect_iso)
    ax_clvd = ax.inset_axes(rect_clvd)
    ax_dc = ax.inset_axes(rect_dc)
    for sax in (ax_iso, ax_dc, ax_clvd):
        sax.set_axis_off()
        sax.set_xlim(-1.2, 1.2)
        sax.set_ylim(-1.2, 1.2)
        sax.set_aspect("equal")

    if abs(rat_iso) > 1e-9:
        plot_beachball_on_axes(ax_iso, np.array(m_iso), 0, 0, diameter=d_iso,
                               color_t=color, linewidth=1)
    if abs(rat_clvd) > 1e-9:
        plot_beachball_on_axes(ax_clvd, np.array(m_clvd), 0, 0, diameter=d_clvd,
                               color_t=color, linewidth=1)
    if abs(rat_dc) > 1e-9:
        plot_beachball_on_axes(ax_dc, np.array(m_dc), 0, 0, diameter=d_dc,
                               color_t=color, linewidth=1)

    ax_iso.text(1.0, 0.3, f"ISO\n{p_iso:.1f}%", transform=ax_iso.transAxes,
                va="center", ha="left", fontsize=16)
    ax_clvd.text(1.0, 0.3, f"CLVD\n{p_clvd:.1f}%", transform=ax_clvd.transAxes,
                 va="center", ha="left", fontsize=16)
    ax_dc.text(1.0, 0.3, f"DC\n{p_dc:.1f}%", transform=ax_dc.transAxes,
               va="center", ha="left", fontsize=16)


def _lune_zoom_limits(gamma_deg=35, pad_frac=0.05):
    """Hammer-projection crop limits for a γ in [-gamma_deg, gamma_deg] lune view."""
    from mpl_toolkits.basemap import Basemap
    bm = Basemap(projection="hammer", lon_0=0)
    x_min, _ = bm(-gamma_deg, 0)
    x_max, _ = bm(gamma_deg, 0)
    _, y_s = bm(0, 0)
    _, y_n = bm(0, 90)
    y_h = y_n - y_s
    return (x_min, x_max, y_s - y_h * pad_frac, y_n + y_h * pad_frac)


def plot_recovery_lune(recovery_dict, plotter, figsave=None, num_samples=2500,
                       zoom=True, decomposition=True, reference_label=None,
                       extra_references=None, reference_name=None):
    """
    Overlay the recovery_dict ensembles on a single Tape & Tape lune (KDE contours),
    optionally cropped to ±35° γ with ISO/CLVD/DC decomposition beachballs for the
    reference (gold) solution. Reuses ``PosteriorPlotter.plot_lunes_kde``.

    ``extra_references`` (optional) maps ``label -> MT 6-vector`` for additional published
    reference solutions, overlaid as distinct scatter markers (e.g. extra catalogues).
    ``reference_name`` (optional) is the legend label for the primary (gold diamond) reference
    — e.g. "Zahradník"; ``reference_label`` still selects which ensemble's theta0 feeds the
    decomposition beachballs.
    """
    import matplotlib.pyplot as plt

    pp = plotter.posterior_plotter
    fig, ax = plt.subplots(figsize=(12, 12))
    pp.plot_lunes_kde(recovery_dict, ax=ax, plot_beachballs=False,
                      num_samples=num_samples, plot_inset=False, show=False, legend=True,
                      extra_references=extra_references, reference_label=reference_name)
    if zoom:
        x_min, x_max, y_min, y_max = _lune_zoom_limits()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    if decomposition:
        # reference = first ensemble's theta0 (gold MLE), unless overridden
        label = reference_label or next(iter(recovery_dict))
        theta0_vec = recovery_dict[label][0]
        if theta0_vec is not None:
            inputs = plotter.parameters.vector_to_simulation_inputs(
                theta0_vec, only_theta_fiducial=True)
            add_decomposition_beachballs(ax, inputs["moment_tensor"], pp, color="salmon")

    if figsave is not None:
        Path(figsave).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figsave, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figsave


# --------------------------------------------------------------------------- #
# Station-config ensemble overlay (variable-station dropout evaluation)
# --------------------------------------------------------------------------- #
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


def plot_ensemble_lune_kde(ensemble_dict, plotter, figsave=None, *,
                           plot_beachballs=False, legend=True, num_samples=2500,
                           extra_references=None, reference_label=None,
                           primary_reference=None):
    """Full-lune KDE overlay of several labelled posteriors, with a per-config legend.

    Wraps the house ``plotter.posterior_plotter.plot_lunes_kde`` (whole lune in shot, no
    crop), forwarding the per-config legend it now builds in-house. ``ensemble_dict`` maps
    ``label -> InversionData``; contour colours follow dict order (matched by the legend).
    ``extra_references`` (optional) maps ``label -> MT 6-vector`` for additional reference
    overlays drawn as distinct scatter markers. ``primary_reference`` + ``reference_label``
    draw and label the primary (gold diamond) reference — needed here because dropout-ensemble
    configs carry no theta0 truth.
    """
    import matplotlib.pyplot as plt

    pp = plotter.posterior_plotter
    fig, ax = plt.subplots(figsize=(12, 12))
    pp.plot_lunes_kde(ensemble_dict, ax=ax, plot_beachballs=plot_beachballs,
                      num_samples=num_samples, plot_inset=False, show=False,
                      legend=legend, legend_title="config\n(solid 68%, dashed 95% HPD)",
                      extra_references=extra_references, reference_label=reference_label,
                      primary_reference=primary_reference)
    if figsave is not None:
        Path(figsave).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figsave, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figsave


def plot_ensemble_spread_summary(configs, ensemble_dict, figsave=None):
    """Posterior spread vs station config / count for a station-dropout ensemble.

    ``configs`` is a sequence of ``StationConfig`` (provides ``.label`` / ``.n``);
    ``ensemble_dict`` maps ``label -> InversionData``. Left panel: per-config 68%
    gamma/delta widths (bars); right panel: the same widths vs station count.
    """
    import matplotlib
    matplotlib.rcParams["text.usetex"] = False  # ChainConsumer may have left usetex on
    import matplotlib.pyplot as plt

    labels = [c.label for c in configs]
    ns = np.array([c.n for c in configs], dtype=float)
    stats = {c.label: spread_stats(ensemble_dict[c.label].samples) for c in configs}
    g_w = np.array([stats[c.label]["gamma_deg_width68"] for c in configs])
    d_w = np.array([stats[c.label]["delta_deg_width68"] for c in configs])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(configs))
    w = 0.38
    ax0.bar(x - w / 2, g_w, w, label=r"$\gamma$ 68% width", color="cornflowerblue")
    ax0.bar(x + w / 2, d_w, w, label=r"$\delta$ 68% width", color="indianred")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=30, ha="right")
    ax0.set_ylabel("posterior 68% interval width (deg)")
    ax0.set_title("Posterior spread per station config")
    for xi, c in zip(x, configs):
        ax0.annotate(f"N={c.n}", (xi, 0), xytext=(0, 2), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, color="dimgray")
    ax0.legend()

    ax1.scatter(ns, g_w, color="cornflowerblue", label=r"$\gamma$", s=60, zorder=3)
    ax1.scatter(ns, d_w, color="indianred", label=r"$\delta$", s=60, zorder=3)
    for xi, gi, di, lab in zip(ns, g_w, d_w, labels):
        ax1.annotate(lab, (xi, max(gi, di)), xytext=(4, 4),
                     textcoords="offset points", fontsize=7, color="dimgray")
    ax1.set_xlabel("number of stations used")
    ax1.set_ylabel("posterior 68% interval width (deg)")
    ax1.set_title("Spread vs station count")
    ax1.legend()

    fig.tight_layout()
    if figsave is not None:
        Path(figsave).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figsave, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return figsave


# --------------------------------------------------------------------------- #
# TARP coverage
# --------------------------------------------------------------------------- #
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

    # tarp defaults num_alpha_bins to n_sims // 10, which is 0 for small n_sims
    # (and then matplotlib/np raise "`bins` must be positive"). Floor it so small
    # validation runs still produce a (coarse) coverage curve.
    n_sims = samples_per_sim.shape[1]
    if num_alpha_bins is None:
        num_alpha_bins = max(2, n_sims // 10)

    ecp, alpha = get_tarp_coverage(
        samples_per_sim, theta_true, references=references, metric=metric,
        num_alpha_bins=num_alpha_bins, num_bootstrap=num_bootstrap,
        norm=True, bootstrap=True, seed=seed,
    )
    return ecp, alpha


# --------------------------------------------------------------------------- #
# Per-parameter recovery scatter (true vs recovered, source-type quantities)
# --------------------------------------------------------------------------- #
def _gamma_delta_mw(mt_samples: np.ndarray):
    """Vectorised (gamma_deg, delta_deg) and per-sample Mw for (N,6) MT samples."""
    from seismo_sbi.plotting.lune import mts6_to_gamma_delta
    from seismo_sbi.plotting.distributions import get_MW_and_epsilon
    gamma, delta = mts6_to_gamma_delta(mt_samples)
    mw = np.array([get_MW_and_epsilon(s)[0] for s in mt_samples])
    return gamma, delta, mw


def plot_recovery_scatter(theta_true: np.ndarray, samples_per_sim: np.ndarray,
                          figsave=None):
    """
    True-vs-recovered scatter for the source-type quantities γ, δ, Mw across the
    validation set. For each sim the posterior median is plotted with 68% CI error
    bars against the truth; the diagonal is the perfect-recovery line.

    ``theta_true`` shape ``(n_sims, 6)``; ``samples_per_sim`` shape
    ``(n_samples, n_sims, 6)`` — moment-tensor parametrisation.
    """
    import matplotlib.pyplot as plt

    theta_true = np.asarray(theta_true)
    samples_per_sim = np.asarray(samples_per_sim)
    n_sims = theta_true.shape[0]

    tg, td, tmw = _gamma_delta_mw(theta_true)

    med = np.zeros((n_sims, 3))
    lo = np.zeros((n_sims, 3))
    hi = np.zeros((n_sims, 3))
    for j in range(n_sims):
        g, d, mw = _gamma_delta_mw(samples_per_sim[:, j, :])
        for k, arr in enumerate((g, d, mw)):
            med[j, k] = np.median(arr)
            lo[j, k] = np.percentile(arr, 16)
            hi[j, k] = np.percentile(arr, 84)

    truths = np.stack([tg, td, tmw], axis=1)
    labels = [r"$\gamma$ (°)", r"$\delta$ (°)", r"$M_w$"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for k, ax in enumerate(axes):
        yerr = np.abs(np.stack([med[:, k] - lo[:, k], hi[:, k] - med[:, k]]))
        ax.errorbar(truths[:, k], med[:, k], yerr=yerr, fmt="o", ms=4,
                    color="cornflowerblue", ecolor="lightgray", alpha=0.8,
                    capsize=2, label="posterior median ± 68%")
        lim_lo = min(truths[:, k].min(), lo[:, k].min())
        lim_hi = max(truths[:, k].max(), hi[:, k].max())
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, label="truth")
        ax.set_xlabel(f"true {labels[k]}")
        ax.set_ylabel(f"recovered {labels[k]}")
        ax.set_aspect("equal", adjustable="datalim")
        if k == 0:
            ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if figsave is not None:
        Path(figsave).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figsave, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figsave


# --------------------------------------------------------------------------- #
# Quantitative evaluation metrics (post-processing of validation inference)
# --------------------------------------------------------------------------- #
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


def _derived_matrix(mt_samples: np.ndarray) -> np.ndarray:
    """
    Map ``(N, 6)`` moment-tensor vectors to ``(N, 6)`` derived quantities
    ``(gamma, delta, Mw, strike, dip, rake)`` — degrees for the angles. Reuses the
    project's lune / pyrocko converters; the nodal-plane ambiguity is resolved by
    consistently taking the first plane (as ``MomentTensorReparametrised`` does).
    """
    from seismo_sbi.plotting.distributions import get_nodal_planes

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


# --------------------------------------------------------------------------- #
# Per-run metrics file + cross-run comparison
# --------------------------------------------------------------------------- #
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


# (display label [mathtext-safe], filename slug, dotted path into the metrics dict,
# lower-is-better?) for the headline bars. The display label uses mathtext for greek so
# it renders whether or not matplotlib's usetex is active; the slug keeps figure
# filenames ASCII-clean and independent of the (mathtext) label.
_HEADLINE_SPECS = [
    (r"FoM $\delta\gamma$", "FoM_delta_gamma", ("figure_of_merit", "delta_gamma"), True),
    ("FoM strike/dip/rake", "FoM_sdr", ("figure_of_merit", "sdr"), True),
    ("FoM full MT", "FoM_full_MT", ("figure_of_merit", "full_mt"), True),
    (r"MAE $\gamma$", "MAE_gamma", ("derived", "gamma", "mae"), True),
    (r"MAE $\delta$", "MAE_delta", ("derived", "delta", "mae"), True),
    ("RMSE Mw", "RMSE_Mw", ("derived", "Mw", "rmse"), True),
    ("Coverage 68pct", "Coverage_68pct", ("coverage", "emp_68"), False),
    ("Coverage 90pct", "Coverage_90pct", ("coverage", "emp_90"), False),
    ("TARP calib err", "TARP_calib_err", ("coverage", "tarp_calibration_error"), True),
]


def _dig(d: dict, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, (int, float)) else None


def plot_cross_run_comparison(this_label: str, all_metrics: Dict[str, dict],
                              out_dir, specs=None) -> Dict[str, str]:
    """
    For each headline metric build a bar chart of every run that reports it, with
    ``this_label`` highlighted. ``all_metrics`` maps ``run_label -> payload`` (as from
    ``scan_run_metrics``; values may also be bare metrics dicts). Writes one PNG per
    metric into ``out_dir/cross_run`` and returns ``{metric_label: path}``.
    """
    import matplotlib.pyplot as plt

    specs = specs or _HEADLINE_SPECS
    out_dir = Path(out_dir) / "cross_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # normalise payloads to metrics dicts
    metrics_by_run = {
        label: (payload.get("metrics", payload) if isinstance(payload, dict) else {})
        for label, payload in all_metrics.items()
    }

    figures: Dict[str, str] = {}
    # Force usetex off: an upstream SBIPipelinePlotter may have enabled scienceplots'
    # text.usetex globally, which then chokes on run labels / greek glyphs here.
    with plt.rc_context({"text.usetex": False}):
        for label, slug, path, lower_better in specs:
            vals = {run: _dig(m, path) for run, m in metrics_by_run.items()}
            vals = {run: v for run, v in vals.items() if v is not None}
            if len(vals) < 1:
                continue
            runs = list(vals.keys())
            colors = ["crimson" if r == this_label else "lightsteelblue" for r in runs]
            fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(runs)), 4.5))
            ax.bar(range(len(runs)), [vals[r] for r in runs], color=colors)
            ax.set_xticks(range(len(runs)))
            ax.set_xticklabels(runs, rotation=30, ha="right", fontsize=8)
            arrow = "lower is better" if lower_better else "higher is better"
            ax.set_title(f"{label}   ({arrow})")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fpath = out_dir / f"cross_run_{slug}.png"
            fig.savefig(fpath, dpi=150, bbox_inches="tight")
            plt.close(fig)
            figures[label] = str(fpath)
    return figures


def plot_cross_run_tarp(this_label: str, all_metrics: Dict[str, dict], out_dir,
                        n_std: float = 1.0) -> Optional[str]:
    """
    Overlay every run's stored TARP expected-coverage curve (mean ± ``n_std``·σ
    bootstrap band) on one axis with the diagonal calibration line, ``this_label``
    drawn bold on top. Reads ``metrics.coverage.tarp_curve`` (written by
    ``compute_evaluation_metrics``). Writes ``out_dir/cross_run/cross_run_tarp_curves.png``
    and returns its path, or ``None`` if no run stored a curve.
    """
    import matplotlib.pyplot as plt

    metrics_by_run = {
        label: (payload.get("metrics", payload) if isinstance(payload, dict) else {})
        for label, payload in all_metrics.items()
    }
    curves = {label: m.get("coverage", {}).get("tarp_curve")
              for label, m in metrics_by_run.items()}
    curves = {label: c for label, c in curves.items() if c}
    if not curves:
        return None

    out_dir = Path(out_dir) / "cross_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # usetex off (see plot_cross_run_comparison); mathtext for the +/- sigma band label.
    with plt.rc_context({"text.usetex": False}):
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.plot([0, 1], [0, 1], ls="--", color="k", label="Calibrated", zorder=20)
        cmap = plt.get_cmap("tab10")
        for i, (label, c) in enumerate(sorted(curves.items())):
            alpha = np.asarray(c["alpha"])
            mean = np.asarray(c["ecp_mean"])
            std = np.asarray(c.get("ecp_std", np.zeros_like(mean)))
            is_this = (label == this_label)
            color = "crimson" if is_this else cmap(i % 10)
            ax.plot(alpha, mean, color=color, lw=2.5 if is_this else 1.5,
                    zorder=15 if is_this else 5, label=label)
            ax.fill_between(alpha, mean - n_std * std, mean + n_std * std,
                            color=color, alpha=0.18, zorder=2)
        ax.set_xlabel("Credibility level")
        ax.set_ylabel("Expected coverage")
        ax.set_title(rf"TARP coverage - cross-run (band: $\pm{n_std:g}\sigma$)")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fpath = out_dir / "cross_run_tarp_curves.png"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return str(fpath)
