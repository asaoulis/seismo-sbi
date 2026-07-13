"""Model-free QA guards + the calibrated 2026-07 gate composition.

Promoted from the F-net Japan live-inference worker (personal-page
``fnet_monitor/qa.py``) so every deployment (Japan live path, Santorini Lomax
catalogue, future regions) shares one calibrated implementation. The evidence
base is the 62-event pe60 calibration (see the ``qa_calibration/FINDINGS.md``
artifact of the ``personal-page/testing-and-inference-prep`` task): each gate
catches a distinct, eyeball-confirmed failure mode, per-COMPONENT verdicts are
zero-filled individually, a station drops only when NO component survives, and
event-level contamination is a FLAG, never a silent drop.

This module stays array-in/array-out (no pipeline, no h5 handles except the
explicit ``read_noise_sigma`` helper) so it is unit-testable like the rest of
``seismo_sbi.data_quality``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .policy import (
    ComponentVerdict,
    QAThresholds,
    component_verdicts,
    event_contamination,
    sigma_outlier_verdicts,
)

# Trace component (Z/E/N, receiver order) -> h5 /misc key (Z/1/2, the E->1 N->2 rename).
_MISC_KEY = {"Z": "Z", "E": "1", "N": "2", "1": "1", "2": "2"}


def data_qa_thresholds(level: str = "minimal", **overrides) -> QAThresholds:
    """CALIBRATED QA presets (2026-07, 62-event pe60 catalogue vs F-net reference-MT
    forward models).

    Every gate encodes *misfit conditional on expected signal* — an expected-low-signal
    trace (nodal / distant / small event) is always KEPT. Levels:

    * ``"minimal"`` — the ESSENTIAL gates only (each catches a distinct, eyeball-confirmed
      failure mode that no other gate sees):
        DEAD (signal predicted >=5 sigma, observed <10% of it, or <25% with xcorr<0.1),
        SIGMA-OUTLIER (pre-event sigma >50x network median: broken channel),
        EXCESS (obs energy >25x the signal+noise budget: glitches / interloper events).
      Classical fit gates are neutralised.
    * ``"full"`` — minimal + the CONDITIONAL FIT gates: where signal is clearly expected
      (snr_syn>=5) AND observed (snr_sig>=2), drop if best-lag xcorr<0.2 or the amplitude
      ratio leaves [0.1, 5] (catches coherent gain errors). ~1% extra drops on the clean
      population. The recommended production preset.
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
                ds = g.get(_MISC_KEY.get(comp, comp))
                if ds is None:
                    continue
                arr = np.asarray(ds)
                var = float(arr.flat[0]) if arr.size else float("nan")  # lag-0
                if np.isfinite(var) and var > 0:
                    out[(sta, comp)] = float(np.sqrt(var))
    return out


def obs_dead_components(obs, present, components, *, rel_floor=0.02, abs_floor=1e-12):
    """Model-INDEPENDENT dead-channel detection from the observation alone.

    A channel is dead if its event-window RMS is < ``abs_floor`` (flatline / dead
    sensor) OR < ``rel_floor`` * the per-component MEDIAN RMS across present stations
    (a gross amplitude outlier reading ~0 while its peers see the event). Needs no
    synthetic, so unlike the SNR/fit gates it is robust to model quality — the
    reliable "obviously broken" basic QA. Returns ``{(station, component): 'drop-dead'}``.
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
    """Catalogue-neighbour contamination check (a FLAG, never a silent drop).

    Another catalogue event with origin inside ``[origin - pre_s, origin + duration_s]``
    puts its wavetrain (or, before the origin, its coda) into this event's window; a
    pre-window neighbour also corrupts the pre-event noise-sigma estimates (silently
    blinding the metric gates), which is why this check is essential even alongside the
    metric-based contamination flag. ``catalogue_times``: iterable of datetimes of OTHER
    events.

    In a dense swarm an absolute flag saturates (nearly every window contains *some*
    micro-event), so a MAGNITUDE-RELATIVE criterion is available: pass parallel
    ``catalogue_mags`` plus the analysed event's ``event_mag`` and a ``delta_mag`` and
    only neighbours with ``mag >= event_mag - delta_mag`` (moment within ~10^(1.5*delta)
    of the event's) qualify for the flag. The nearest QUALIFYING neighbour's offset and
    magnitude are reported (plus ``nearest_any_s`` for context).

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
    """The calibrated 2026-07 per-component QA composition (pure, backend-free).

    Layers, in order: per-trace gate verdicts (``component_verdicts``, SNR gates first),
    the cross-station sigma-outlier health check, the persistent-bad-channel
    ``blocklist`` (deployment DATA, never restored by the keep-floor), and the
    model-independent obs-only dead guard (needs ``obs`` shaped (Np, C, T)). A station
    drops only when NO component survives.

    Event-level contamination is computed as a FLAG; with ``contaminated_action="warn"``
    (A/B-calibrated default) a flagged window keeps every channel except the
    sigma-INDEPENDENT health drops (blocklist + obs-dead) — mass-dropping a contaminated
    window's traces makes the posterior worse, and the sigma-based gates are themselves
    untrustworthy there (a neighbour's coda corrupts the pre-event sigma). ``"drop"``
    applies all gates as usual. Either way the flag is the product; display it.

    Keep-floor: if fewer than ``max(min_stations, ceil(min_fraction * len(present)))``
    stations survive, RANK-FILL by observed signal SNR (never revert-all — that would
    re-inject the pure-noise traces the gates just removed); blocklisted channels are
    never restored.

    Returns ``(comp_map, dropped, component_drops, event_flags)`` where ``comp_map`` is
    ``{station: [kept components]}``, ``dropped`` is ``{station: verdict}`` for fully
    dropped stations, ``component_drops`` is ``{(station, component): verdict}`` and
    ``event_flags`` is the contamination-diagnostics dict.
    """
    comps = list(components)
    cv = component_verdicts(metrics, thresholds, snr_metrics=snr)
    # model-free sigma-outlier channel health (needs the cross-station context, so it is
    # applied here rather than inside the per-trace gate)
    for (sta, comp), verdict in sigma_outlier_verdicts(
            snr or [], thresholds, metrics=metrics).items():
        old = cv.get(sta, {}).get(comp)
        if old is not None and old.is_kept:
            cv[sta][comp] = ComponentVerdict(sta, comp, verdict,
                                             old.max_xcorr, old.amp_ratio_obs_syn)
    # persistent-bad-channel blocklist (deployment data, not a gate)
    for (sta, comp) in blocklist:
        old = cv.get(sta, {}).get(comp)
        if old is not None:
            cv[sta][comp] = ComponentVerdict(sta, comp, "drop-blocklist",
                                             old.max_xcorr, old.amp_ratio_obs_syn)

    # model-INDEPENDENT dead-channel guard (basic QA, robust to a bad first guess)
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

    # event-level contamination diagnostics (FLAGS, never silent drops)
    event_flags = event_contamination(metrics, snr or [], cv, exclude=tuple(blocklist))
    if event_flags.get("contaminated") and contaminated_action == "warn":
        keep_drops = {k: v for k, v in component_drops.items() if v == "drop-blocklist"}
        for key in obs_dead:
            keep_drops.setdefault(key, "drop-dead")
        component_drops = keep_drops
        comp_map, dropped = _collapse(component_drops)

    # keep-floor: RANK-FILL by observed debiased signal SNR (not revert-all).
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
