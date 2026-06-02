"""Diagnostic figures for data QA (obs-vs-synthetic overlay, station scorecard,
time-shift before/after). Moved from the Santorini ``qa_forward_check.py`` and adapted
to the QA dataclasses. Plotting is intentionally not in the test gate — the verdict/shift
data is the contract; these are diagnostics.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .alignment import ShiftResult
from .metrics import TraceDescriptor, TraceMetrics
from .policy import QAThresholds, StationVerdict, VERDICT_COLORS, VERDICT_LABELS


def _verdict_legend(fig, *, loc, ncol, fontsize):
    handles = [Patch(facecolor=VERDICT_COLORS[k], label=VERDICT_LABELS[k])
               for k in VERDICT_COLORS]
    fig.legend(handles=handles, loc=loc, ncol=ncol, fontsize=fontsize)


def plot_overlay_grid(
    obs2d: np.ndarray,
    syn2d: np.ndarray,
    traces: List[TraceDescriptor],
    metrics: List[TraceMetrics],
    sampling_rate: float,
    title: str,
    out_png,
    verdicts: Optional[Dict[str, StationVerdict]] = None,
) -> None:
    """Trace overlay grid (rows = stations by distance, cols = Z/E/N), each panel
    optionally tinted by the station verdict."""
    dist_by_sta = {m.station: m.dist_km for m in metrics}
    sta_order = sorted(dist_by_sta, key=dist_by_sta.get)
    comp_idx = {"Z": 0, "E": 1, "N": 2}
    pos_lut = {(d.station, d.component): i for i, d in enumerate(traces)}
    metric_lut = {(m.station, m.component): m for m in metrics}
    t = np.arange(obs2d.shape[1]) / sampling_rate

    nrow = len(sta_order)
    fig, axes = plt.subplots(nrow, 3, figsize=(13, 1.5 * nrow), squeeze=False)
    for ri, sta in enumerate(sta_order):
        v = verdicts.get(sta).verdict if (verdicts and sta in verdicts) else None
        for comp, ci in comp_idx.items():
            ax = axes[ri][ci]
            if (sta, comp) in pos_lut:
                i = pos_lut[(sta, comp)]
                ax.plot(t, obs2d[i], "k", lw=0.7)
                ax.plot(t, syn2d[i], "r", lw=0.7, alpha=0.8)
                m = metric_lut[(sta, comp)]
                ax.set_title(f"{sta}.{comp} VR={m.vr:.2f} aVR={m.aligned_vr:.2f} "
                             f"lag={m.best_lag_samples}", fontsize=6.5)
            if v:
                ax.set_facecolor(VERDICT_COLORS[v] + "22")  # light tint
            ax.set_yticks([])
            ax.tick_params(labelsize=6)
            if ri == 0 and ci == 0:
                ax.legend(["obs", "syn"], fontsize=6, loc="upper right")
    if verdicts:
        _verdict_legend(fig, loc="upper right", ncol=len(VERDICT_COLORS), fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def plot_station_scorecard(
    verdicts: Dict[str, StationVerdict],
    event: str,
    out_png,
    thresholds: QAThresholds,
) -> None:
    """Four-panel scorecard: amp ratio (log), Z xcorr, aligned VR, azimuth/distance
    polar map — each coloured by verdict."""
    stas = sorted(verdicts, key=lambda s: verdicts[s].summary.dist_km)
    cols = [VERDICT_COLORS[verdicts[s].verdict] for s in stas]
    x = np.arange(len(stas))
    fig = plt.figure(figsize=(15, 9))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.bar(x, [verdicts[s].summary.median_amp_ratio for s in stas], color=cols)
    ax1.axhline(1.0, color="gray", lw=0.8)
    ax1.axhline(thresholds.amp_hi, color="red", ls="--", lw=0.8,
                label=f"drop > {thresholds.amp_hi:g}")
    ax1.axhline(thresholds.amp_lo, color="red", ls="--", lw=0.8)
    ax1.set_yscale("log")
    ax1.set_title("median obs/syn peak amplitude ratio")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stas, rotation=90, fontsize=7)
    ax1.legend(fontsize=7)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.bar(x, [verdicts[s].summary.xcorr_Z for s in stas], color=cols)
    ax2.axhline(thresholds.xcorr_drop, color="red", ls="--", lw=0.8,
                label=f"drop < {thresholds.xcorr_drop:g}")
    ax2.set_ylim(0, 1)
    ax2.set_title("Z-component max cross-correlation (after best lag)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(stas, rotation=90, fontsize=7)
    ax2.legend(fontsize=7)

    ax3 = fig.add_subplot(2, 2, 3)
    avr = [float(np.median(list(verdicts[s].summary.aligned_vr.values()))) for s in stas]
    ax3.bar(x, avr, color=cols)
    ax3.axhline(0.0, color="gray", lw=0.8)
    ax3.set_title("median aligned variance reduction (per station)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(stas, rotation=90, fontsize=7)

    ax4 = fig.add_subplot(2, 2, 4, projection="polar")
    ax4.set_theta_zero_location("N")
    ax4.set_theta_direction(-1)
    for s in stas:
        summ = verdicts[s].summary
        az = np.deg2rad(summ.azimuth)
        ax4.scatter(az, summ.dist_km, s=80,
                    color=VERDICT_COLORS[verdicts[s].verdict], edgecolor="k", zorder=3)
        ax4.annotate(s, (az, summ.dist_km), fontsize=6)
    ax4.set_title("station azimuth / distance (km)", pad=18)

    _verdict_legend(fig, loc="upper center", ncol=len(VERDICT_COLORS), fontsize=9)
    fig.suptitle(f"{event}: station QA scorecard (reference MT @ fixed location)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_shift_before_after(
    shift_results: Dict[str, ShiftResult],
    sta_order: List[str],
    event: str,
    out_png,
) -> None:
    """Per-station VR before vs after the optimal static shift (labels = shift)."""
    stas = [s for s in sta_order if s in shift_results]
    x = np.arange(len(stas))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.2, [shift_results[s].vr_before for s in stas], width=0.4,
           label="VR before (no shift)", color="gray")
    ax.bar(x + 0.2, [shift_results[s].vr_after for s in stas], width=0.4,
           label="VR after (optimal shift)", color="#2ca02c")
    for i, s in enumerate(stas):
        sh = shift_results[s].shift
        if sh:
            ax.annotate(f"{sh:+d}", (i + 0.2, shift_results[s].vr_after),
                        ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(stas, rotation=90, fontsize=8)
    ax.set_ylabel("station aligned variance reduction")
    ax.set_title(f"{event}: per-station time-shift optimisation (labels = shift in samples)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
