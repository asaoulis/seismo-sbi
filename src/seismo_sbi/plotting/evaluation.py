"""Figures and diagnostics for the final evaluation stage of a training run.

Loads inversion result pickles, overlays a freshly trained model's posterior against frozen
reference inversions, and draws the calibration and recovery figures: cross-run TARP curves and
per-parameter recovery scatter. The metrics they show are computed in
:mod:`seismo_sbi.evaluation.posterior_metrics`.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from seismo_sbi.moment_tensor.conventions import create_matrix
from seismo_sbi.evaluation.posterior_metrics import _gamma_delta_mw, spread_stats


# Pickle / run discovery
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


# Recovery dict (gold standard + ML runs) for lune / chainconsumer plots
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


# Lune recovery plot (with ISO/CLVD/DC decomposition beachballs)
def add_decomposition_beachballs(ax, theta0_mt, color="salmon"):
    """
    Add scaled ISO / CLVD / DC beachballs + percentages to the left of a lune axis,
    in axes coordinates. ``theta0_mt`` is a 6-component moment tensor in the pipeline
    (up-south-east) convention. Adapted from the project's reference snippet.
    """
    from pyrocko import moment_tensor as pmt
    from seismo_sbi.plotting.rocko_beachball_patch import plot_beachball_on_axes

    mt_matrix = create_matrix(theta0_mt)
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
    — e.g. a catalogue's name; ``reference_label`` still selects which ensemble's theta0 feeds the
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
            add_decomposition_beachballs(ax, inputs["moment_tensor"], color="salmon")

    if figsave is not None:
        Path(figsave).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figsave, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figsave


# Station-config ensemble overlay (variable-station dropout evaluation)


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


# Per-parameter recovery scatter (true vs recovered, source-type quantities)


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


# Per-run metrics file + cross-run comparison


#: ``(display label, filename slug, dotted path into the metrics dict, lower is better)`` for
#: the headline bars. Labels use mathtext so they render with or without usetex.
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
