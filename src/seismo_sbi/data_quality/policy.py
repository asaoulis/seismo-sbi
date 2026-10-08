"""Per-station keep, time-shift or drop decisions.

Pure deterministic logic over :class:`TraceMetrics`, controlled by :class:`QAThresholds`. The
gates are vertical-component coherence, gross amplitude, timing, and a lag-aligned
multi-component coherence gate that generalises the vertical-only one and is therefore robust to
timing error. The zero-lag posterior-predictive metrics (``corr_misfit``, ``envelope_misfit``,
``reduced_chi2``) are carried for fidelity reporting and only become drop gates if their
thresholds are set explicitly.
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
    "drop-snr-noise": "#7f7f7f",   # grey   (retired verdict: below-noise is a KEEP; old plots use it)
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
KEPT_VERDICTS = ("keep", "time-shift")


@dataclass(frozen=True)
class QAThresholds:
    """Decision thresholds, with the library's defaults.

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

    # The aligned multi-component coherence gate is on by default because it is lag-robust;
    # the zero-lag misfit gates are confounded by timing and amplitude, so they stay off.
    enable_ppc_drops: bool = True
    coherence_drop: float = 0.45         # mean aligned xcorr (all comp.) below -> drop-fit
    corr_misfit_drop: Optional[float] = None
    envelope_misfit_drop: Optional[float] = None

    # SNR gates against the pre-event window, each off by default. A trace is dropped only for a
    # data problem or extreme mismodelling, never for low signal.
    enable_snr_gates: bool = False  # arms the DEAD gate (+ invalid-sigma dead routing)
    snr_dead_ratio: float = 0.1     # G1: drop if debiased obs signal < this * predicted...
    snr_dead_min_syn: float = 5.0   # ...but only when the signal SHOULD be clearly visible
    snr_sigma_floor: float = 0.0    # sigma <= this (or non-finite) => dead channel
    # The unrecognisable branch of the dead gate: also dead when the observed energy is
    # marginal and the best-lag correlation is low, which a peak-amplitude test alone misses.
    snr_dead_unrecog_ratio: Optional[float] = None   # calibrated value: 0.25
    xcorr_dead: float = 0.1
    # Excess energy: the observed whole-window energy exceeds the signal-plus-noise budget.
    # Not conditioned on the synthetic, since an overlapping event at a quiet station is the case.
    enable_snr_excess: bool = False
    snr_excess_factor: float = 5.0  # G3: drop if obs energy > this^2 * (syn energy + noise)
    # Re-scope the correlation and amplitude gates to fire only where a signal is both expected
    # and observed, so a trace is never dropped for failing to correlate with noise.
    conditional_fit_gates: bool = False
    snr_fit_min_syn: float = 5.0
    snr_fit_sig_min: float = 2.0
    # Pre-event noise width above this multiple of the network median (same component) means a
    # broken channel; a trace that visibly matches the synthetic is kept anyway.
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


# Per-component verdicts drop individual channels, zero-filled at load, rather than the whole
# station; they are additive to the station-level policy above, not a replacement for it.

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

    There is no below-noise drop: an expected-low-signal trace is uninformative, not bad, and
    is KEPT.
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
        else:
            verdict = _coherence_amplitude_verdict(m, t)
    else:
        verdict = _coherence_amplitude_verdict(m, t)
    return ComponentVerdict(m.station, m.component, verdict,
                            m.max_xcorr, m.amp_ratio_obs_syn)


def _coherence_amplitude_verdict(m: TraceMetrics, t: QAThresholds) -> str:
    """The classical gate for one trace: incoherent, then gross amplitude error, else keep."""
    if m.max_xcorr < t.xcorr_drop:
        return "drop-corr"
    if m.amp_ratio_obs_syn > t.amp_hi or m.amp_ratio_obs_syn < t.amp_lo:
        return "drop-amp"
    return "keep"


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

    A channel this far above its peers is broken or garbage-dominated (broken channels have
    been seen at ~80-1400x; healthy transients stay ~10-30x) — its own sigma
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
    traces at once, including interlopers below the catalogue's magnitude threshold.
    Statistic: among the traces
    where signal is clearly expected (``snr_syn >= snr_expected_min``), excluding the
    ``exclude``-d (station, component) pairs (persistent-bad blocklist), the event is
    flagged when the dropped fraction >= ``frac_hard``, or >= ``frac_soft`` with the median
    best-lag xcorr < ``med_xcorr_max`` (clean events sit at ~0.4-0.6, contaminated windows
    at ~0.1-0.3). Needs at least ``min_expected`` such traces, else the statistic
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
