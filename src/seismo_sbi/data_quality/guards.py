"""Quality guards that need no forward model, and the calibrated gate composition over them.

Each guard catches a distinct failure mode of a recorded trace without reference to a synthetic.
Verdicts are per component and a component that fails is zero-filled individually; a station is
dropped only when no component survives. Event-level contamination is a flag, never a silent
drop. Everything is array in, array out, with no pipeline and no file handles except the
explicit ``read_noise_sigma`` helper.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .policy import (
    ComponentVerdict,
    QAThresholds,
    component_verdicts,
    event_contamination,
    sigma_outlier_verdicts,
)


def data_qa_thresholds(level: str = "minimal", **overrides) -> QAThresholds:
    """Calibrated QA presets, checked against forward models of reference moment tensors.

    Every gate judges the misfit given the expected signal, so an expected-low-signal trace
    (nodal, distant or a small event) is always kept. Levels:

    * ``"minimal"``: the gates that each catch a failure no other gate sees, checked by eye.
      Dead (signal predicted at 5 sigma or more, observed below 10 % of it, or below 25 % with
      xcorr below 0.1), sigma outlier (pre-event sigma above 50 times the network median: a
      broken channel), excess (observed energy above 25 times the signal-plus-noise budget:
      glitches and interloping events). The classical fit gates are switched off.
    * ``"full"``: minimal plus the conditional fit gates. Where a signal is clearly expected
      (snr_syn >= 5) and observed (snr_sig >= 2), a trace drops if its best-lag xcorr is below
      0.2 or its amplitude ratio leaves [0.1, 5] (coherent gain errors). It drops about 1 %
      more of a clean population and is the recommended preset.
    """
    kw = dict(enable_snr_gates=True, snr_dead_ratio=0.1, snr_dead_min_syn=5.0,
              snr_dead_unrecog_ratio=0.25, xcorr_dead=0.1,
              sigma_rel_max=50.0,
              enable_snr_excess=True, snr_excess_factor=5.0,
              # classical station-level gates neutralised (per-component QA only)
              xcorr_drop=0.0, amp_hi=1e12, amp_lo=0.0, enable_ppc_drops=False)
    if level == "full":
        kw.update(conditional_fit_gates=True, snr_fit_min_syn=5.0, snr_fit_sig_min=2.0,
                  xcorr_drop=0.2, amp_lo=0.1, amp_hi=5.0)
    elif level != "minimal":
        raise ValueError(f"unknown QA preset level: {level!r}")
    kw.update(overrides)
    return QAThresholds(**kw)


def read_noise_sigma(event_h5, stations, components) -> Dict[Tuple[str, str], float]:
    """Pre-event noise std sigma per (station, component) from the h5 ``/misc`` group.

    ``/misc/<sta>/<Z|1|2>`` is the pre-event autocorrelation; lag-0 = pre-event
    mean(x^2) = sigma^2. Returns sigma = sqrt(that). Missing/degenerate entries are
    omitted (the SNR gate then treats them as dead channels).
    """
    import h5py

    from seismo_sbi.simulators.simulation_io import component_alias
    out: Dict[Tuple[str, str], float] = {}
    with h5py.File(event_h5, "r") as f:
        misc = f.get("misc")
        if misc is None:
            return out
        for sta in stations:
            g = misc.get(sta)
            if g is None:
                continue
            for comp in components:
                ds = g.get(component_alias(comp))
                if ds is None:
                    continue
                arr = np.asarray(ds)
                var = float(arr.flat[0]) if arr.size else float("nan")  # lag-0
                if np.isfinite(var) and var > 0:
                    out[(sta, comp)] = float(np.sqrt(var))
    return out


def obs_dead_components(obs, present, components, *, rel_floor=0.02, abs_floor=1e-12):
    """Dead channels found from the observation alone, with no forward model.

    A channel is dead if its event-window RMS is below ``abs_floor`` (a flat line or a dead
    sensor) or below ``rel_floor`` times the per-component median RMS across the present
    stations (a channel reading about zero while its peers record the event). It needs no
    synthetic, so unlike the SNR and fit gates it does not depend on the forward model.
    Returns ``{(station, component): 'drop-dead'}``.
    """
    obs = np.asarray(obs)                       # (Np, C, T)
    Np, C, _ = obs.shape
    rms = np.sqrt(np.mean(obs ** 2, axis=2))    # (Np, C)
    dead = {}
    for c in range(C):
        col = rms[:, c]
        pos = col[col > 0]
        med = float(np.median(pos)) if pos.size else 0.0
        for si in range(Np):
            if col[si] < abs_floor or (med > 0 and col[si] < rel_floor * med):
                dead[(present[si], components[c])] = "drop-dead"
    return dead


def neighbour_window_flag(origin_time, duration_s, catalogue_times, pre_s=120.0,
                          catalogue_mags=None, event_mag=None, delta_mag=None):
    """Whether another catalogue event contaminates this event's window: a flag, never a drop.

    Another catalogue event with origin inside ``[origin - pre_s, origin + duration_s]``
    puts its wavetrain (or, before the origin, its coda) into this event's window. A
    pre-window neighbour also corrupts the pre-event noise-sigma estimates and so blinds the
    metric gates, which the metric-based contamination flag cannot detect.
    ``catalogue_times`` holds the origin datetimes of the other events.

    In a dense swarm an absolute flag saturates, since nearly every window contains some
    micro-event. With ``catalogue_mags`` (parallel to the times), ``event_mag`` and
    ``delta_mag``, only neighbours with ``mag >= event_mag - delta_mag`` (moment within about
    10^(1.5 delta_mag) of the event's) count. The offset and magnitude of the nearest counting
    neighbour are reported, with ``nearest_any_s`` for the nearest of any size.

    Returns ``{"neighbour_in_window": bool, "nearest_neighbour_s": float,
    "nearest_neighbour_mag": float | nan, "nearest_any_s": float}``.
    """
    relative = (catalogue_mags is not None and event_mag is not None
                and delta_mag is not None)
    mags = list(catalogue_mags) if catalogue_mags is not None else None
    best = float("inf")          # nearest qualifying neighbour
    best_mag = float("nan")
    best_any = float("inf")      # nearest neighbour of any size
    for i, t in enumerate(catalogue_times):
        dt = (t - origin_time).total_seconds()
        if abs(dt) < 1e-6:
            continue
        if abs(dt) < abs(best_any):
            best_any = dt
        if relative and mags[i] < event_mag - delta_mag:
            continue
        if abs(dt) < abs(best):
            best = dt
            best_mag = float(mags[i]) if mags is not None else float("nan")
    return {"neighbour_in_window": bool(best != float("inf")
                                        and -pre_s <= best <= duration_s),
            "nearest_neighbour_s": best,
            "nearest_neighbour_mag": best_mag,
            "nearest_any_s": best_any}


def compose_component_qa(metrics, snr, present: List[str], components: List[str],
                         thresholds: QAThresholds, *,
                         obs=None, blocklist=(),
                         min_stations: int = 5, min_fraction: float = 0.25,
                         contaminated_action: str = "warn"):
    """The calibrated per-component QA of one event.

    The layers, in order: per-trace gate verdicts (``component_verdicts``, SNR gates first),
    the cross-station sigma-outlier check, the ``blocklist`` of persistently bad channels
    (data about the deployment, never restored by the keep floor), and the dead-channel check on
    the observation alone (needs ``obs`` shaped (n_present, n_components, n_samples)). A station
    drops only when none of its components survives.

    Event-level contamination is computed as a flag. With ``contaminated_action="warn"`` (the
    default, chosen by comparing posteriors both ways) a flagged window keeps every channel
    except the health drops that do not use sigma (blocklist and dead observation): dropping
    most of a contaminated window's traces makes the posterior worse, and the sigma-based gates
    cannot be trusted there, since a neighbour's coda corrupts the pre-event sigma. ``"drop"``
    applies all gates as usual. Either way the flag is the result to report.

    Keep floor: if fewer than ``max(min_stations, ceil(min_fraction * len(present)))`` stations
    survive, stations are restored in order of observed signal SNR, not all at once (which would
    restore the noise-only traces the gates removed). Blocklisted channels are never restored.

    Returns ``(comp_map, dropped, component_drops, event_flags)`` where ``comp_map`` is
    ``{station: [kept components]}``, ``dropped`` is ``{station: verdict}`` for fully
    dropped stations, ``component_drops`` is ``{(station, component): verdict}`` and
    ``event_flags`` is the contamination-diagnostics dict.
    """
    comps = list(components)
    cv = component_verdicts(metrics, thresholds, snr_metrics=snr)
    # sigma outliers need the other stations, so they are applied here, not in the per-trace gate
    for (sta, comp), verdict in sigma_outlier_verdicts(
            snr or [], thresholds, metrics=metrics).items():
        old = cv.get(sta, {}).get(comp)
        if old is not None and old.is_kept:
            cv[sta][comp] = ComponentVerdict(sta, comp, verdict,
                                             old.max_xcorr, old.amp_ratio_obs_syn)
    for (sta, comp) in blocklist:
        old = cv.get(sta, {}).get(comp)
        if old is not None:
            cv[sta][comp] = ComponentVerdict(sta, comp, "drop-blocklist",
                                             old.max_xcorr, old.amp_ratio_obs_syn)

    obs_dead = {}
    if obs is not None:
        obs3d = np.asarray(obs).reshape(len(present), len(comps), -1)
        obs_dead = obs_dead_components(obs3d, present, comps)
        for (sta, comp) in obs_dead:
            old = cv.get(sta, {}).get(comp)
            if old is not None and old.is_kept:
                cv[sta][comp] = ComponentVerdict(sta, comp, "drop-dead",
                                                 old.max_xcorr, old.amp_ratio_obs_syn)

    # per-component collapse: keep surviving channels; a station drops only when none survive
    component_drops = {(s, c): v.verdict for s, d in cv.items()
                       for c, v in d.items() if v.is_dropped}

    def _collapse(drops):
        comp_map: Dict[str, List[str]] = {}
        dropped: Dict[str, str] = {}
        for sta in present:
            kept_c = [c for c in comps
                      if cv.get(sta, {}).get(c) is not None and (sta, c) not in drops]
            if kept_c:
                comp_map[sta] = kept_c
            else:
                vs = [drops.get((sta, c)) for c in comps if (sta, c) in drops]
                dropped[sta] = vs[0] if vs else "drop"
        return comp_map, dropped

    comp_map, dropped = _collapse(component_drops)

    event_flags = event_contamination(metrics, snr or [], cv, exclude=tuple(blocklist))
    if event_flags.get("contaminated") and contaminated_action == "warn":
        keep_drops = {k: v for k, v in component_drops.items() if v == "drop-blocklist"}
        for key in obs_dead:
            keep_drops.setdefault(key, "drop-dead")
        component_drops = keep_drops
        comp_map, dropped = _collapse(component_drops)

    # keep floor: restore stations in order of observed debiased signal SNR
    floor = max(min_stations, int(np.ceil(min_fraction * len(present))))
    if len(comp_map) < floor:
        snr_by_sta: Dict[str, list] = {}
        for s in (snr or []):
            snr_by_sta.setdefault(s.station, []).append(s.snr_sig)
        rank = sorted(present, key=lambda st: -max(snr_by_sta.get(st, [0.0])))
        block = set(blocklist)
        for st in rank:
            if len(comp_map) >= floor:
                break
            if st not in comp_map:
                restore = [c for c in comps if (st, c) not in block]   # never un-blocklist
                if restore:
                    comp_map[st] = restore
                    dropped.pop(st, None)

    return comp_map, dropped, component_drops, event_flags
