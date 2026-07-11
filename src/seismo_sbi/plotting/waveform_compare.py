"""Simple observed-vs-synthetic waveform comparison plots — a record section and per-station
3-component overlays. Array-in, figure-out: no pipeline / NPE object needed.

Crystallised from the 2026-07 physics-debug investigation (the record section + overlay figures
that diagnosed the 60 s window offset). Kept deliberately minimal and self-contained.

Inputs everywhere:
  obs, syn : ``(N_stations, C, T)`` stacked arrays, SAME station+component order.
  station_names : list length N.
  coords : ``(N, 2)`` (lat, lon) of the stations, same order.
  event_location : ``(lat, lon, depth_km)``.
  sampling_rate : Hz.

xcorr + best-lag annotations come from :func:`seismo_sbi.data_quality.align_best_lag`
(``+lag`` delays the synthetic). Components are assumed ordered ``Z, E, N`` (the pipeline default).

IMPORTANT for a meaningful comparison (both were real bugs in this project):
  * forward-model the synthetic at the event's TRUE location
    (``NpeBackend.forward_synthetic(mt6, stations, source_vec=[lat, lon, depth])``) — an unpinned
    forward model samples a random prior location and the moveout is meaningless; and
  * build the observed catalogue with ``--pre_event_window 60`` — every synthetic places the
    origin at t = +60 s (``SyntheticsPreprocessing``), so the observation window must too.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

COMPONENTS = ("Z", "E", "N")


def _dist_km(ev_lat, ev_lon, sta_lat, sta_lon) -> float:
    from obspy.geodetics.base import gps2dist_azimuth
    return gps2dist_azimuth(ev_lat, ev_lon, sta_lat, sta_lon)[0] / 1000.0


def _shift(x, lag: int):
    """Delay (``+lag``) / advance (``-lag``) a trace by an integer number of samples, zero-padded."""
    from seismo_sbi.instaseis_simulator.utils import shift_1d_with_padding
    return shift_1d_with_padding(np.asarray(x, float), int(lag))


def _xcorr_lag(obs1d, syn1d, max_lag: int):
    from seismo_sbi.data_quality import align_best_lag
    return align_best_lag(np.asarray(obs1d, float), np.asarray(syn1d, float), max_lag)


def _apply_align(o, s, align: str, max_lag: int, sr: float):
    """Return (syn_aligned, label) for the requested alignment mode."""
    if align == "window60":
        return _shift(s, -int(round(60 * sr))), "syn+60s"
    if align == "best":
        _, lag = _xcorr_lag(o, s, max_lag)
        return _shift(s, lag), f"lag={lag/sr:+.0f}s"
    return np.asarray(s, float), "raw"


def record_section(obs, syn, station_names, coords, event_location, *, sampling_rate=1.0,
                   component="Z", reduction_velocity=None, align="none", max_lag=60,
                   figname=None, gain=0.4, color_obs="black", color_syn="#b87333", ax=None):
    """One-component record section: obs (black) + syn (colour), one trace per station offset by
    its epicentral distance (km). ``reduction_velocity`` (km/s), if given, plots at reduced time
    ``t - dist/v_red`` so a moving-out phase lines up. ``align`` in {'none','window60','best'}.
    Each trace is peak-normalised (by its own obs) and scaled by ``gain`` in distance units.
    """
    import matplotlib.pyplot as plt
    obs = np.asarray(obs, float); syn = np.asarray(syn, float)
    ci = COMPONENTS.index(component)
    ev_lat, ev_lon, _ = event_location
    dists = np.array([_dist_km(ev_lat, ev_lon, coords[i][0], coords[i][1])
                      for i in range(len(station_names))])
    order = np.argsort(dists)
    T = obs.shape[2]
    t = np.arange(T) / float(sampling_rate)
    span = (dists.max() - dists.min()) or 1.0
    g = gain * span / max(len(order), 1)            # trace half-amplitude in distance units

    own = ax is not None
    if not own:
        fig, ax = plt.subplots(figsize=(9, 10))
    for i in order:
        o = obs[i, ci]; s = syn[i, ci]
        s, lbl = _apply_align(o, s, align, max_lag, sampling_rate)
        # normalise each trace by the max of BOTH obs and syn so neither channel (nor a quiet /
        # dead observation) blows the record section up.
        norm = max(float(np.max(np.abs(o))), float(np.max(np.abs(s)))) or 1.0
        tt = t - (dists[i] / reduction_velocity if reduction_velocity else 0.0)
        ax.plot(tt, dists[i] + g * o / norm, color=color_obs, lw=0.8, zorder=3)
        ax.plot(tt, dists[i] + g * s / norm, color=color_syn, lw=0.8, alpha=0.9, zorder=2)
        ax.text(t[0] - (dists[i] / reduction_velocity if reduction_velocity else 0.0),
                dists[i], f" {station_names[i]}", va="center", ha="right", fontsize=7)
    xlab = "reduced time  t − Δ/v  [s]" if reduction_velocity else "time [s]"
    ax.set(xlabel=xlab, ylabel="epicentral distance [km]",
           title=f"Record section — {component}  (obs=black, syn={color_syn}; {align})")
    ax.grid(alpha=0.2)
    if not own:
        _finish(fig, figname)


def station_overlays(obs, syn, station_names, coords, event_location, *, sampling_rate=1.0,
                     align="none", max_lag=60, per_trace=True, stations=None,
                     figname=None, color_syn="#1f4e79"):
    """Grid of per-station rows × 3 component columns (Z/E/N). obs (black) vs syn (colour) with the
    zero-lag xcorr (and, for ``align='best'``, the best lag) annotated in every panel title.

    ``align``: 'none' (raw), 'window60' (advance syn 60 s to undo the build offset), or 'best'
    (per-trace best-lag). ``per_trace`` normalises each panel by its own obs peak.
    ``stations``: optional subset (list of names) to keep the figure small.
    """
    import matplotlib.pyplot as plt
    obs = np.asarray(obs, float); syn = np.asarray(syn, float)
    idx = [station_names.index(s) for s in stations] if stations else list(range(len(station_names)))
    T = obs.shape[2]
    t = np.arange(T) / float(sampling_rate)
    n = len(idx)
    fig, axes = plt.subplots(n, 3, figsize=(13, 1.7 * n + 1), squeeze=False)
    global_max = np.max(np.abs(obs)) or 1.0
    for r, i in enumerate(idx):
        for c in range(3):
            ax = axes[r][c]
            o = obs[i, c]; s = syn[i, c]
            s_al, lbl = _apply_align(o, s, align, max_lag, sampling_rate)
            xc0, _ = _xcorr_lag(o, s_al, 0)                 # xcorr at the chosen alignment
            norm = (np.max(np.abs(o)) or 1.0) if per_trace else global_max
            ax.plot(t, o / norm, color="black", lw=0.9, zorder=3)
            ax.plot(t, s_al / norm, color=color_syn, lw=0.9, alpha=0.9, zorder=2)
            ax.set_title(f"{station_names[i]} {COMPONENTS[c]}  xc={xc0:+.2f} ({lbl})", fontsize=8)
            ax.tick_params(labelsize=7)
            if r < n - 1:
                ax.set_xticklabels([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    axes[-1][1].set_xlabel("time [s]")
    fig.suptitle(f"obs (black) vs synthetic ({color_syn}) — align={align}, "
                 f"{'per-trace' if per_trace else 'global'} norm", y=1.002)
    _finish(fig, figname)


def _finish(fig, figname):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    if figname is not None:
        fig.savefig(figname, dpi=140, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
