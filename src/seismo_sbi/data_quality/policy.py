"""Per-station keep / time-shift / drop decision policy.

Pure, deterministic logic over :class:`TraceMetrics`. The classical gates
(Z-component coherence, gross amplitude, timing) are ported verbatim from the
Santorini ``qa_forward_check.py``; an additional, *lag-aligned multi-component
coherence* gate (a principled generalisation of the Z-only coherence gate that uses
all components and is therefore robust to timing errors) is folded in, controllable
via :class:`QAThresholds`. The confounded zero-lag PPC metrics
(``corr_misfit``/``envelope_misfit``/``reduced_chi2``) are carried for *fidelity
reporting* and are only drop gates if their thresholds are explicitly set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .metrics import TraceMetrics, SNRMetrics

# Verdict vocabulary (single source of truth, used by plotting too).
VERDICT_COLORS = {
    "keep": "#2ca02c",        # green
    "time-shift": "#ff7f0e",  # orange
    "drop-amp": "#d62728",    # red
    "drop-corr": "#9467bd",   # purple
    "drop-fit": "#8c564b",    # brown (PPC aligned-coherence gate)
    "drop-snr-dead": "#000000",    # black  (dead / flatlined channel)
    "drop-snr-noise": "#7f7f7f",   # grey   (RETIRED 2026-07: below-noise is a KEEP, kept for legacy plots)
    "drop-snr-excess": "#e377c2",  # pink   (obs energy exceeds signal+noise budget)
    "drop-snr-noisy": "#bcbd22",   # olive  (pre-event noise sigma is a gross network outlier)
}
VERDICT_LABELS = {
    "keep": "keep",
    "time-shift": "keep + time-shift",
    "drop-amp": "DROP (bad amplitude)",
    "drop-corr": "DROP (incoherent, Z)",
    "drop-fit": "DROP (incoherent, all comp.)",
    "drop-snr-dead": "DROP (dead / no signal)",
    "drop-snr-noise": "DROP (below noise floor)",
    "drop-snr-excess": "DROP (excess energy)",
    "drop-snr-noisy": "DROP (noise floor outlier)",
}
# SNR drop verdicts, most-severe first (for the station-collapse tie-break).
_SNR_DROP_ORDER = ("drop-snr-dead", "drop-snr-noisy", "drop-snr-excess", "drop-snr-noise")

KEPT_VERDICTS = ("keep", "time-shift")


@dataclass(frozen=True)
class QAThresholds:
    """Decision thresholds. Defaults replicate the original Santorini constants.

    The classical gates judge coherence first (an incoherent trace is useless), then
    gross amplitude (a response/units error the MT scale cannot absorb), then timing
    (fixable with a static shift). ``enable_ppc_drops`` activates the extra aligned
    multi-component coherence gate; the zero-lag misfit gates stay off unless their
    thresholds are set.
    """

    xcorr_drop: float = 0.50       # Z-component max xcorr below this -> drop-corr
    amp_hi: float = 20.0           # median obs/syn peak ratio above this -> drop-amp
    amp_lo: float = 0.05           # ... or below this -> drop-amp
    shift_min: int = 2             # |best lag Z| at/above this -> recommend a shift
    xcorr_shift_ok: float = 0.55   # ... but only if the Z xcorr is at least this

    # PPC-derived gates. The aligned multi-component coherence gate is ON by default
    # (lag-robust, so it never punishes a station that merely needs a time shift); the
    # zero-lag misfit gates are confounded by timing/amplitude and stay OFF (None).
    enable_ppc_drops: bool = True
    coherence_drop: float = 0.45         # mean aligned xcorr (all comp.) below -> drop-fit
    corr_misfit_drop: Optional[float] = None
    envelope_misfit_drop: Optional[float] = None

    # Pre-event-noise SNR gates (need per-trace :class:`SNRMetrics`). DEFAULT OFF, so every
    # existing caller + golden regression is byte-identical; a caller opts in explicitly and
    # EVERY gate below has its own independent switch. Thresholds were calibrated on the
    # 62-event F-net Japan pe60 catalogue against F-net reference-MT forward models
    # (2026-07, see the qa_calibration FINDINGS): because a 1-D forward model over-predicts
    # amplitude ~1.4-2x, thresholds are in "synthetic units" and deliberately coarse
    # (sigma^2 itself carries ~20% estimation error).
    #
    # GOVERNING PRINCIPLE: drop a trace only when the synthetic-vs-observation mismatch
    # indicates a DATA-QUALITY problem or extreme mismodelling — never merely because the
    # absolute SNR is low. An expected-low-signal trace (nodal / distant station, small
    # event) is uninformative, not bad, and is KEPT. The old G2 "below-noise" gate
    # (drop if snr_syn < snr_syn_min) dropped exactly those traces and was RETIRED 2026-07;
    # ``snr_syn_min`` is kept only for constructor compatibility and is no longer read.
    enable_snr_gates: bool = False  # arms the DEAD gate (+ invalid-sigma dead routing)
    snr_syn_min: float = 2.0        # RETIRED (was G2); field kept for API compatibility
    snr_dead_ratio: float = 0.1     # G1: drop if debiased obs signal < this * predicted...
    snr_dead_min_syn: float = 5.0   # ...but only when the signal SHOULD be clearly visible
    snr_sigma_floor: float = 0.0    # sigma <= this (or non-finite) => dead channel
    # G1u "unrecognisable" OR-branch of the dead gate (None = off): also dead when the
    # observed energy is marginal (< this * snr_syn) AND the best-lag xcorr is < xcorr_dead
    # (event-scale glitch peaks defeat a peak-amplitude test; energy+coherence do not).
    snr_dead_unrecog_ratio: Optional[float] = None   # calibrated value: 0.25
    xcorr_dead: float = 0.1
    # G3 (excess-energy / glitch / interloper event): obs whole-window energy exceeds the
    # signal+noise budget. Deliberately NOT conditioned on snr_syn — an overlapping event at
    # an expected-quiet station is exactly what it must catch. OPT-IN (needs a well-scaled
    # synthetic). Calibrated factor 5 (energy 25x; clean-population q99.5 is ~13).
    enable_snr_excess: bool = False
    snr_excess_factor: float = 5.0  # G3: drop if obs energy > this^2 * (syn energy + noise)
    # CONDITIONAL FIT GATES (opt-in): re-scope the classical xcorr/amplitude gates of
    # :func:`decide_component` to fire ONLY where a signal is clearly expected
    # (snr_syn >= snr_fit_min_syn) AND actually observed (snr_sig >= snr_fit_sig_min) — so
    # a trace is never dropped for failing to correlate with noise. Below those levels the
    # classical gates are BYPASSED (keep). Requires SNRMetrics; no effect otherwise.
    conditional_fit_gates: bool = False
    snr_fit_min_syn: float = 5.0
    snr_fit_sig_min: float = 2.0
    # SIGMA-OUTLIER channel-health gate (see :func:`sigma_outlier_verdicts`; None = off):
    # a channel whose pre-event noise sigma is > sigma_rel_max * the network median (same
    # component, same event) is broken/garbage-dominated (YMZ Z ~1400x, KSN Z ~83x; healthy
    # transients reach ~10-30x). Escape hatch: a pre-window spike can inflate sigma on an
    # otherwise-good trace, so a trace that visibly matches the synthetic
    # (xcorr >= escape_xcorr with a sane amplitude ratio) is kept.
    sigma_rel_max: Optional[float] = None            # calibrated value: 50.0
    sigma_outlier_escape_xcorr: float = 0.4
    sigma_outlier_escape_amp: tuple = (0.1, 10.0)


@dataclass(frozen=True)
class StationSummary:
    """Per-station collapse of its traces' metrics."""

    dist_km: float
    azimuth: float
    vr: Dict[str, float]
    aligned_vr: Dict[str, float]
    lag_Z: int
    xcorr_Z: float
    median_amp_ratio: float
    aligned_coherence: float           # mean max_xcorr across the station's components
    corr_misfit: Optional[float] = None
    envelope_misfit: Optional[float] = None
    reduced_chi2: Optional[float] = None


@dataclass(frozen=True)
class StationVerdict:
    """A station's verdict plus the summary that justified it."""

    verdict: str
    suggested_shift: int
    summary: StationSummary

    @property
    def is_kept(self) -> bool:
        return self.verdict in KEPT_VERDICTS

    @property
    def is_dropped(self) -> bool:
        return self.verdict.startswith("drop")


def summarise_station(
    metrics: List[TraceMetrics],
    ppc: Optional[dict] = None,
) -> StationSummary:
    """Collapse one station's per-trace metrics into a :class:`StationSummary`.

    ``ppc`` optionally supplies the obs/syn-derived fidelity fields
    (``corr_misfit``/``envelope_misfit``/``reduced_chi2``) for reporting.
    """
    by_c = {m.component: m for m in metrics}
    z = by_c.get("Z", metrics[0])
    ppc = ppc or {}
    return StationSummary(
        dist_km=metrics[0].dist_km,
        azimuth=metrics[0].azimuth,
        vr={m.component: m.vr for m in metrics},
        aligned_vr={m.component: m.aligned_vr for m in metrics},
        lag_Z=z.best_lag_samples,
        xcorr_Z=z.max_xcorr,
        median_amp_ratio=float(np.median([m.amp_ratio_obs_syn for m in metrics])),
        aligned_coherence=float(np.mean([m.max_xcorr for m in metrics])),
        corr_misfit=ppc.get("corr_misfit"),
        envelope_misfit=ppc.get("envelope_misfit"),
        reduced_chi2=ppc.get("reduced_chi2"),
    )


def _ppc_drop(summary: StationSummary, t: QAThresholds) -> bool:
    """Whether the (enabled) PPC fidelity gates flag this station for dropping."""
    if not t.enable_ppc_drops:
        return False
    if summary.aligned_coherence < t.coherence_drop:
        return True
    if (t.corr_misfit_drop is not None and summary.corr_misfit is not None
            and summary.corr_misfit > t.corr_misfit_drop):
        return True
    if (t.envelope_misfit_drop is not None and summary.envelope_misfit is not None
            and summary.envelope_misfit > t.envelope_misfit_drop):
        return True
    return False


def decide_station(summary: StationSummary, thresholds: QAThresholds) -> StationVerdict:
    """Apply the gates in priority order: Z-coherence -> amplitude -> PPC fidelity ->
    timing -> keep."""
    s, t = summary, thresholds
    if s.xcorr_Z < t.xcorr_drop:
        verdict = "drop-corr"
    elif s.median_amp_ratio > t.amp_hi or s.median_amp_ratio < t.amp_lo:
        verdict = "drop-amp"
    elif _ppc_drop(s, t):
        verdict = "drop-fit"
    elif abs(s.lag_Z) >= t.shift_min and s.xcorr_Z >= t.xcorr_shift_ok:
        verdict = "time-shift"
    else:
        verdict = "keep"
    return StationVerdict(verdict, s.lag_Z if verdict == "time-shift" else 0, s)


def summarise_event(
    metrics: List[TraceMetrics],
    thresholds: QAThresholds,
    ppc: Optional[Dict[str, dict]] = None,
) -> Dict[str, StationVerdict]:
    """Group per-trace metrics by station and produce one verdict per station.

    ``ppc`` maps ``station -> {corr_misfit, envelope_misfit, reduced_chi2}``.
    """
    ppc = ppc or {}
    by_sta: Dict[str, List[TraceMetrics]] = {}
    for m in metrics:
        by_sta.setdefault(m.station, []).append(m)
    return {
        sta: decide_station(summarise_station(ms, ppc.get(sta)), thresholds)
        for sta, ms in by_sta.items()
    }


# ---------------------------------------------------------------------------
# Per-COMPONENT verdicts — finer-grained QA that drops individual dodgy channels
# (zero-filled at load, exactly like the ``component_dropout`` nuisance) rather
# than the whole station. The station-level policy above is unchanged; component
# verdicts are additive and consumed alongside it (see
# ``serialization.components_from_verdicts`` ``component_verdicts=`` argument).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComponentVerdict:
    """A single (station, component) keep/drop verdict."""

    station: str
    component: str
    verdict: str                 # "keep" | "drop-corr" | "drop-amp"
    max_xcorr: float
    amp_ratio_obs_syn: float

    @property
    def is_kept(self) -> bool:
        return self.verdict == "keep"

    @property
    def is_dropped(self) -> bool:
        return self.verdict.startswith("drop")


def _snr_component_gate(snr: Optional[SNRMetrics], t: QAThresholds,
                        m: Optional[TraceMetrics] = None) -> Optional[str]:
    """SNR drop verdict for one trace, or None. Order: dead -> excess.

    Fable-designed, noise-aware gates (only armed when ``enable_snr_gates``), each encoding
    *misfit conditional on expected signal* — never "low absolute SNR":

    * **dead (G1)** — the signal SHOULD be clearly visible (``snr_syn >= snr_dead_min_syn``)
      yet the noise-debiased observed signal is essentially absent
      (``snr_sig < snr_dead_ratio * snr_syn``). Catches a flatlined channel (obs/syn ~1e-3)
      while sparing the benign ~0.5 1-D amplitude misfit (RMS-ratio separation ~1e2-1e3),
      because it only fires when the prediction is strong.
    * **unrecognisable (G1u, opt-in via ``snr_dead_unrecog_ratio``)** — the observed energy
      is marginal (``snr_sig < snr_dead_unrecog_ratio * snr_syn``) AND nothing resembling
      the predicted waveform is present at any lag (``max_xcorr < xcorr_dead``). Requires
      the trace's :class:`TraceMetrics` ``m``; skipped when ``m`` is None.
    * **excess (G3, opt-in via ``enable_snr_excess``)** — whole-window observed energy
      exceeds the signal+noise budget
      ``snr_obs_full^2 > snr_excess_factor^2 * (snr_syn_full^2 + 1)``: an overlapping event,
      transient or glitch the MT cannot explain. Never fires when obs <= syn, and is NOT
      conditioned on ``snr_syn`` (an interloper at an expected-quiet station must fire it).

    The old **G2 below-noise** drop (``snr_syn < snr_syn_min``) was RETIRED (2026-07): an
    expected-low-signal trace is uninformative, not bad — it is KEPT.
    """
    if not t.enable_snr_gates or snr is None:
        return None
    if not np.isfinite(snr.sigma) or snr.sigma <= t.snr_sigma_floor:
        return "drop-snr-dead"
    if snr.snr_syn >= t.snr_dead_min_syn:
        if snr.snr_sig < t.snr_dead_ratio * snr.snr_syn:
            return "drop-snr-dead"
        if (t.snr_dead_unrecog_ratio is not None and m is not None
                and snr.snr_sig < t.snr_dead_unrecog_ratio * snr.snr_syn
                and m.max_xcorr < t.xcorr_dead):
            return "drop-snr-dead"
    if t.enable_snr_excess and \
            snr.snr_obs_full ** 2 > (t.snr_excess_factor ** 2) * (snr.snr_syn_full ** 2 + 1.0):
        return "drop-snr-excess"
    return None


def decide_component(m: TraceMetrics, thresholds: QAThresholds,
                     snr: Optional[SNRMetrics] = None) -> ComponentVerdict:
    """Keep/drop ONE trace. When ``enable_snr_gates`` and an :class:`SNRMetrics` is supplied,
    the noise-aware SNR gates run FIRST (a below-noise trace's coherence/amplitude are
    meaningless — it should be labelled "no signal", not "incoherent"); otherwise the
    classical coherence-then-amplitude gates apply, exactly as before.

    With ``conditional_fit_gates`` (opt-in, needs ``snr``) the classical gates fire only
    where a signal is clearly expected (``snr_syn >= snr_fit_min_syn``) AND clearly observed
    (``snr_sig >= snr_fit_sig_min``); anywhere below, the trace is KEPT — an
    expected-low-signal trace must never be dropped for failing to correlate with noise.

    With ``snr=None`` OR ``enable_snr_gates=False`` this is byte-identical to the legacy
    behaviour (so all existing callers + golden regressions are unaffected)."""
    t = thresholds
    snr_verdict = _snr_component_gate(snr, t, m)
    if snr_verdict is not None:
        verdict = snr_verdict
    elif (t.conditional_fit_gates and snr is not None
          and np.isfinite(snr.sigma) and snr.sigma > t.snr_sigma_floor):
        if snr.snr_syn < t.snr_fit_min_syn or snr.snr_sig < t.snr_fit_sig_min:
            verdict = "keep"                       # signal not expected / not observed
        elif m.max_xcorr < t.xcorr_drop:
            verdict = "drop-corr"
        elif m.amp_ratio_obs_syn > t.amp_hi or m.amp_ratio_obs_syn < t.amp_lo:
            verdict = "drop-amp"
        else:
            verdict = "keep"
    elif m.max_xcorr < t.xcorr_drop:
        verdict = "drop-corr"
    elif m.amp_ratio_obs_syn > t.amp_hi or m.amp_ratio_obs_syn < t.amp_lo:
        verdict = "drop-amp"
    else:
        verdict = "keep"
    return ComponentVerdict(m.station, m.component, verdict,
                            m.max_xcorr, m.amp_ratio_obs_syn)


def component_verdicts(
    metrics: List[TraceMetrics],
    thresholds: QAThresholds,
    snr_metrics: Optional[List[SNRMetrics]] = None,
) -> Dict[str, Dict[str, ComponentVerdict]]:
    """``{station: {component: ComponentVerdict}}`` from per-trace metrics.

    ``snr_metrics`` (optional) attaches the pre-event-noise SNR gates per (station,
    component); ``None`` reproduces the classical per-component verdicts exactly.
    """
    snr_lookup = {(s.station, s.component): s for s in (snr_metrics or [])}
    out: Dict[str, Dict[str, ComponentVerdict]] = {}
    for m in metrics:
        out.setdefault(m.station, {})[m.component] = decide_component(
            m, thresholds, snr=snr_lookup.get((m.station, m.component)))
    return out


def sigma_outlier_verdicts(
    snr_metrics: List[SNRMetrics],
    thresholds: QAThresholds,
    metrics: Optional[List[TraceMetrics]] = None,
) -> Dict[tuple, str]:
    """Model-FREE channel-health gate: ``{(station, component): "drop-snr-noisy"}`` for
    every trace whose pre-event noise sigma is > ``sigma_rel_max`` x the network MEDIAN
    sigma of the same component (across the stations of this event).

    A channel this far above its peers is broken or garbage-dominated (F-net calibration:
    YMZ Z ~1400x, KSN Z noisy days ~83x; healthy transients stay ~10-30x) — its own sigma
    "explains" the garbage, so the SNR gates cannot see it; only the cross-station
    comparison can. No-op ({}) unless ``thresholds.sigma_rel_max`` is set (opt-in).

    Escape hatch: a single pre-window spike can inflate sigma on an otherwise-good trace
    (observed during calibration: OKW N with xcorr 0.64). If ``metrics`` is supplied, a
    trace that visibly matches the synthetic (``max_xcorr >= sigma_outlier_escape_xcorr``
    with an amplitude ratio inside ``sigma_outlier_escape_amp``) is spared.
    """
    t = thresholds
    if t.sigma_rel_max is None:
        return {}
    by_comp: Dict[str, list] = {}
    for s in snr_metrics:
        if np.isfinite(s.sigma) and s.sigma > 0:
            by_comp.setdefault(s.component, []).append(s.sigma)
    med = {c: float(np.median(v)) for c, v in by_comp.items() if v}
    fit = {(m.station, m.component): m for m in (metrics or [])}
    out: Dict[tuple, str] = {}
    for s in snr_metrics:
        m0 = med.get(s.component, 0.0)
        if not (np.isfinite(s.sigma) and s.sigma > 0) or m0 <= 0:
            continue
        if s.sigma <= t.sigma_rel_max * m0:
            continue
        m = fit.get((s.station, s.component))
        lo, hi = t.sigma_outlier_escape_amp
        if m is not None and m.max_xcorr >= t.sigma_outlier_escape_xcorr \
                and lo <= m.amp_ratio_obs_syn <= hi:
            continue                                   # waveform visibly matches: spared
        out[(s.station, s.component)] = "drop-snr-noisy"
    return out


def event_contamination(
    metrics: List[TraceMetrics],
    snr_metrics: List[SNRMetrics],
    verdicts: Dict[str, Dict[str, ComponentVerdict]],
    *,
    exclude: tuple = (),
    snr_expected_min: float = 5.0,
    min_expected: int = 8,
    frac_hard: float = 0.4,
    frac_soft: float = 0.2,
    med_xcorr_max: float = 0.35,
) -> Dict[str, float]:
    """EVENT-level contamination diagnostic (a FLAG, never a silent drop).

    An overlapping earthquake inside the observation window corrupts many normally-good
    traces at once (F-net calibration: two windows with catalogue neighbours 98 s / 152 s
    away, plus two interlopers BELOW the catalogue threshold). Statistic: among the traces
    where signal is clearly expected (``snr_syn >= snr_expected_min``), excluding the
    ``exclude``-d (station, component) pairs (persistent-bad blocklist), the event is
    flagged when the dropped fraction >= ``frac_hard``, or >= ``frac_soft`` with the median
    best-lag xcorr < ``med_xcorr_max`` (clean events sit at ~0.4-0.6; the F-net interloper
    windows at 0.13-0.28). Needs at least ``min_expected`` such traces, else the statistic
    is meaningless (flagged=False).

    Returns ``{"contaminated": 0/1, "n_expected": n, "frac_expected_dropped": f,
    "median_xcorr_expected": x}``.
    """
    snr_by = {(s.station, s.component): s for s in snr_metrics}
    excl = set(exclude)
    fracs, xcs = [], []
    for m in metrics:
        key = (m.station, m.component)
        if key in excl:
            continue
        s = snr_by.get(key)
        if s is None or not (np.isfinite(s.sigma) and s.sigma > 0):
            continue
        if s.snr_syn < snr_expected_min:
            continue
        v = verdicts.get(m.station, {}).get(m.component)
        fracs.append(1.0 if (v is not None and v.is_dropped) else 0.0)
        xcs.append(m.max_xcorr)
    n = len(fracs)
    if n < min_expected:
        return {"contaminated": 0.0, "n_expected": float(n),
                "frac_expected_dropped": float("nan"), "median_xcorr_expected": float("nan")}
    frac = float(np.mean(fracs))
    med_xc = float(np.median(xcs))
    flag = frac >= frac_hard or (frac >= frac_soft and med_xc < med_xcorr_max)
    return {"contaminated": float(flag), "n_expected": float(n),
            "frac_expected_dropped": frac, "median_xcorr_expected": med_xc}


def snr_station_drop(component_verdicts_for_station: Dict[str, ComponentVerdict]) -> Optional[str]:
    """Whole-station SNR-collapse verdict, or None.

    Mirrors the Z-primacy philosophy of :func:`decide_station`: a station is SNR-dropped
    when its Z channel fails an SNR gate OR >= 2 of its components do. Returns the most
    severe SNR drop verdict (dead > excess > noise) among the failures.

    NOTE (2026-07 F-net calibration): for per-component NPE inputs prefer NOT collapsing —
    YMZ's Z was broken for weeks while its horizontals stayed healthy, so Z-primacy throws
    away good data. Kept unchanged for existing callers; the F-net worker no longer uses it.
    """
    dropped = {c: v.verdict for c, v in component_verdicts_for_station.items()
               if v.verdict.startswith("drop-snr")}
    if not dropped:
        return None
    if "Z" in dropped or len(dropped) >= 2:
        for verdict in _SNR_DROP_ORDER:
            if verdict in dropped.values():
                return verdict
    return None
