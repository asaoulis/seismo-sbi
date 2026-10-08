"""Observed-versus-synthetic waveforms of many stations as one moveout record section.

Arrays in, figures out; no pipeline or posterior object is needed. ``obs`` and ``syn`` are
``(n_stations, n_components, n_samples)`` in the same station and component order, ``coords`` is
``(n_stations, 2)`` of latitude and longitude, ``event_location`` is ``(lat, lon, depth_km)`` and
``sampling_rate_hz`` is in Hz. Components are assumed ordered Z, E, N. Cross-correlation and
best-lag annotations come from :func:`seismo_sbi.data_quality.metrics.align_best_lag`, where a positive
lag delays the synthetic. Forward-model the synthetic at the event's true location.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

COMPONENTS = ("Z", "E", "N")


def _dist_km(ev_lat, ev_lon, sta_lat, sta_lon) -> float:
    from obspy.geodetics.base import gps2dist_azimuth
    return gps2dist_azimuth(ev_lat, ev_lon, sta_lat, sta_lon)[0] / 1000.0


def _xcorr_lag(obs1d, syn1d, max_lag: int):
    from seismo_sbi.data_quality.metrics import align_best_lag
    return align_best_lag(np.asarray(obs1d, float), np.asarray(syn1d, float), max_lag)


def _finish(fig, figname):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    if figname is not None:
        fig.savefig(figname, dpi=140, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig


# --- Moveout record section: many stations x 3 components, deterministic overlays and ensembles ---

#: Overlay colours, in assignment order. Deliberately distinct in hue AND lightness so the
#: figure survives greyscale printing; the first is the "our solution" colour.
OVERLAY_COLORS = ("#b3446c", "#1f6fb4", "#2e8b57", "#d1701a", "#6a4c93")
OBS_COLOR = "black"
DROPPED_COLOR = "#b0b0b0"


def _azimuth_deg(ev_lat, ev_lon, sta_lat, sta_lon) -> float:
    from obspy.geodetics.base import gps2dist_azimuth
    return gps2dist_azimuth(ev_lat, ev_lon, sta_lat, sta_lon)[1]


def _norm_table(obs, overlay_cubes, mode):
    """``(N, C)`` divisor array for the requested normalisation.

    Every mode normalises by the max of BOTH obs and the overlays, so an overlay that
    overshoots is visibly clipped-looking rather than silently rescaling the observation.
    """
    stack = [np.abs(obs)] + [np.abs(c) for c in overlay_cubes]
    peak = np.max([s.max(axis=-1) for s in stack], axis=0)      # (N, C)
    if mode == "per_trace":
        table = peak
    elif mode == "per_station":
        table = np.repeat(peak.max(axis=1, keepdims=True), peak.shape[1], axis=1)
    elif mode == "global":
        table = np.full_like(peak, peak.max())
    else:
        raise ValueError(f"unknown normalise={mode!r}")
    return np.where(table > 0, table, 1.0)


def _window_slice(n_t, sr, window, arrival_s):
    """``(i0, i1, t_origin)`` for one trace. ``t_origin`` is subtracted from the time axis."""
    if window is None:
        return 0, n_t, 0.0
    if isinstance(window, (tuple, list)) and window and window[0] == "arrival":
        pre, post = float(window[1]), float(window[2])
        if arrival_s is None:
            return 0, n_t, 0.0
        i0 = max(0, int(round((arrival_s - pre) * sr)))
        i1 = min(n_t, int(round((arrival_s + post) * sr)))
        if i1 - i0 < 2:
            return 0, n_t, 0.0
        return i0, i1, arrival_s
    t0, t1 = float(window[0]), float(window[1])
    return max(0, int(round(t0 * sr))), min(n_t, int(round(t1 * sr))), 0.0


def moveout_record_section(
    obs,
    overlays=None,
    station_names=None,
    coords=None,
    event_location=None,
    *,
    components=COMPONENTS,
    sampling_rate=1.0,
    order_by="distance",
    y_scale="rank",
    normalise="per_station",
    layout="panels",
    window=None,
    reduction_velocity=None,
    arrivals=None,
    ensemble_style="band",
    quantiles=(0.05, 0.95),
    max_lines=30,
    channel_mask=None,
    annotate=("xcorr",),
    gain=0.42,
    colors=None,
    title=None,
    figname=None,
    ax=None,
    seed=0,
):
    """Record section where **every trace is individually legible**.

    The failure mode this replaces: one global amplitude scale over 45 traces leaves the
    quiet stations as flat lines. Here each trace (or each station) carries its own scale,
    and the physical peak stays recoverable from the annotation.

    Parameters
    ----------
    obs : ``(N, C, T)``
        Observed waveforms, station-major, in the receivers' master order.
    overlays : dict, optional
        ``label -> array``. A ``(N, C, T)`` array is drawn as a single line (a best-fit or
        reference-catalogue synthetic); a ``(S, N, C, T)`` array is an ensemble drawn per
        ``ensemble_style``. Insertion order sets colour assignment.
    station_names, coords, event_location
        Length-``N`` names, ``(N, 2)`` (lat, lon), and ``(lat, lon, depth_km)``.
    order_by : {'distance', 'azimuth', 'arrival', 'none'}
        Vertical ordering. ``'azimuth'`` is the radiation-pattern view — the most
        diagnostic ordering for an ISO-vs-DC decision.
    y_scale : {'rank', 'true'}
        ``'rank'`` spaces stations uniformly (readable when a cluster dominates);
        ``'true'`` puts them at their real distance/azimuth (honest moveout, but crowds).
    normalise : {'per_station', 'per_trace', 'global'}
        ``'per_station'`` keeps the relative size of Z/E/N within a station meaningful
        while still letting distant stations be seen — usually the right compromise.
    layout : {'panels', 'interleaved'}
        ``'panels'`` gives one column per component; ``'interleaved'`` stacks all three
        under each station in a single axis.
    window : None | (t0, t1) | ('arrival', pre_s, post_s)
        Time crop. The arrival form needs ``arrivals`` and aligns each trace on its own
        predicted arrival (t = 0).
    arrivals : dict, optional
        ``station -> arrival time (s)``, used by ``order_by='arrival'`` and the arrival
        window.
    ensemble_style : {'band', 'spaghetti', 'band+best'}
        How ``(S, N, C, T)`` overlays are drawn. ``'band'`` = median + inter-quantile fill.
    channel_mask : ``(N, C)`` bool, optional
        ``False`` marks a QA-dropped channel: drawn greyed, tagged, and excluded from the
        annotations (the NPE never saw it, so its misfit is not evidence of anything).
    annotate : tuple
        Any of ``'xcorr'`` (zero-lag, vs the first overlay), ``'peak'`` (physical obs peak),
        ``'ratio'`` (syn/obs peak).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    obs = np.asarray(obs, float)
    if obs.ndim != 3:
        raise ValueError(f"obs must be (N, C, T), got {obs.shape}")
    n_sta, n_comp, n_t = obs.shape
    components = list(components)[:n_comp]
    station_names = list(station_names if station_names is not None
                         else [f"S{i:02d}" for i in range(n_sta)])
    overlays = OrderedDict(overlays or {})
    sr = float(sampling_rate)

    # --- geometry -----------------------------------------------------------------
    if coords is not None and event_location is not None:
        ev_lat, ev_lon = float(event_location[0]), float(event_location[1])
        dists = np.array([_dist_km(ev_lat, ev_lon, coords[i][0], coords[i][1])
                          for i in range(n_sta)])
        azis = np.array([_azimuth_deg(ev_lat, ev_lon, coords[i][0], coords[i][1])
                         for i in range(n_sta)])
    else:
        dists = np.arange(n_sta, dtype=float)
        azis = np.zeros(n_sta)
    arr = np.array([float(arrivals.get(s, np.nan)) if arrivals else np.nan
                    for s in station_names])

    key = {"distance": dists, "azimuth": azis, "arrival": arr}.get(order_by)
    if key is None or not np.isfinite(key).all():
        key = dists
    order = np.argsort(key) if order_by != "none" else np.arange(n_sta)

    if y_scale == "true":
        base = key.astype(float)
        span = float(np.ptp(base)) or 1.0
        step = span / max(n_sta - 1, 1)
        ylabel = {"azimuth": "azimuth [°]"}.get(order_by, "epicentral distance [km]")
    else:
        base = np.empty(n_sta, float)
        base[order] = np.arange(n_sta, dtype=float)
        step, span = 1.0, float(n_sta)
        ylabel = ""
    half = gain * step

    # --- normalisation ------------------------------------------------------------
    det = OrderedDict((k, np.asarray(v, float)) for k, v in overlays.items()
                      if np.asarray(v).ndim == 3)
    ens = OrderedDict((k, np.asarray(v, float)) for k, v in overlays.items()
                      if np.asarray(v).ndim == 4)
    norm_inputs = list(det.values()) + [e.mean(axis=0) for e in ens.values()]
    norm = _norm_table(obs, norm_inputs, normalise)

    mask = (np.ones((n_sta, n_comp), bool) if channel_mask is None
            else np.asarray(channel_mask, bool))

    palette = list(colors or OVERLAY_COLORS)
    color_of = {lbl: palette[i % len(palette)] for i, lbl in enumerate(overlays)}
    rng = np.random.default_rng(seed)

    # --- axes ---------------------------------------------------------------------
    interleaved = layout == "interleaved"
    n_axes = 1 if interleaved else n_comp
    if ax is not None:
        axes, fig, own = [ax], ax.figure, True
    else:
        height = max(4.0, 0.40 * n_sta * (n_comp if interleaved else 1) + 2.0)
        fig, axarr = plt.subplots(1, n_axes, figsize=(6.0 * n_axes if not interleaved else 11.0,
                                                      height),
                                  sharey=True, squeeze=False)
        axes, own = list(axarr[0]), False

    comp_off = np.linspace(0.30, -0.30, n_comp) * step if interleaved else np.zeros(n_comp)

    for ci, comp in enumerate(components):
        axis = axes[0] if interleaved else axes[ci]
        for i in order:
            y0 = base[i] + comp_off[ci]
            i0, i1, t_org = _window_slice(n_t, sr, window, arr[i] if np.isfinite(arr[i]) else None)
            red = (dists[i] / reduction_velocity) if reduction_velocity else 0.0
            t = np.arange(i0, i1) / sr - t_org - red
            d = norm[i, ci]
            alive = bool(mask[i, ci])

            for lbl, cube in ens.items():
                block = cube[:, i, ci, i0:i1] / d
                col = color_of[lbl]
                if ensemble_style == "spaghetti":
                    k = min(max_lines, block.shape[0])
                    for s in rng.choice(block.shape[0], k, replace=False):
                        axis.plot(t, y0 + half * block[s], color=col, lw=0.5,
                                  alpha=0.25, zorder=2)
                else:
                    lo, hi = np.quantile(block, quantiles, axis=0)
                    med = np.median(block, axis=0)
                    axis.fill_between(t, y0 + half * lo, y0 + half * hi, color=col,
                                      alpha=0.30, lw=0, zorder=2)
                    axis.plot(t, y0 + half * med, color=col, lw=0.9, alpha=0.95, zorder=3)
                    if ensemble_style == "band+best":
                        # member 0 is the best-fitting one: `select_best_synthetics`
                        # returns the ensemble already ordered by the selection metric.
                        axis.plot(t, y0 + half * block[0], color=col, lw=1.0,
                                  alpha=1.0, ls="--", zorder=4)
            for lbl, cube in det.items():
                axis.plot(t, y0 + half * cube[i, ci, i0:i1] / d, color=color_of[lbl],
                          lw=1.0, alpha=0.95, zorder=4)
            axis.plot(t, y0 + half * obs[i, ci, i0:i1] / d,
                      color=OBS_COLOR if alive else DROPPED_COLOR,
                      lw=1.0 if alive else 0.8, alpha=1.0 if alive else 0.55, zorder=5)

            if not alive:
                axis.text(t[-1], y0, " QA", va="center", ha="left", fontsize=6,
                          color=DROPPED_COLOR, style="italic")
            elif annotate:
                axis.text(t[-1], y0, " " + _annot(obs[i, ci], det, ens, i, ci, annotate),
                          va="center", ha="left", fontsize=5.6, color="#555555")

        axis.set_title(comp if not interleaved else "  ".join(components),
                       fontsize=10, fontweight="bold")
        axis.set_xlabel("time − arrival [s]" if (window and window[0] == "arrival")
                        else ("reduced time  t − Δ/v  [s]" if reduction_velocity else "time [s]"))
        axis.grid(axis="x", alpha=0.15)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.tick_params(left=False)

    # station labels on the leftmost axis only
    labels = [f"{station_names[i]}  {dists[i]:.0f}km {azis[i]:.0f}°" for i in range(n_sta)]
    axes[0].set_yticks(base)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].set_ylabel(ylabel)
    axes[0].set_ylim(base.min() - 1.2 * step, base.max() + 1.2 * step)
    if y_scale == "rank":
        axes[0].invert_yaxis()          # nearest station at the top

    handles = [Line2D([], [], color=OBS_COLOR, lw=1.2, label="observed")]
    handles += [Line2D([], [], color=color_of[l], lw=1.2, label=l) for l in overlays]
    if channel_mask is not None and not mask.all():
        handles.append(Line2D([], [], color=DROPPED_COLOR, lw=1.2, label="QA-dropped"))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    if title:
        fig.suptitle(f"{title}\n", fontsize=11, y=1.045)
    if own:
        return fig
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return _finish(fig, figname)


def _annot(o, det, ens, i, ci, annotate):
    """Compact per-trace annotation string (xcorr vs the first overlay, peak, ratio)."""
    ref = None
    if det:
        ref = next(iter(det.values()))[i, ci]
    elif ens:
        ref = np.median(next(iter(ens.values()))[:, i, ci], axis=0)
    bits = []
    if "xcorr" in annotate and ref is not None:
        xc, _ = _xcorr_lag(o, ref, 0)
        bits.append(f"xc{xc:+.2f}")
    if "peak" in annotate:
        bits.append(f"{np.max(np.abs(o)):.1e}")
    if "ratio" in annotate and ref is not None:
        po = float(np.max(np.abs(o))) or 1.0
        bits.append(f"r{float(np.max(np.abs(ref)))/po:.2f}")
    return " ".join(bits)
