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

from .metrics import TraceMetrics

# Verdict vocabulary (single source of truth, used by plotting too).
VERDICT_COLORS = {
    "keep": "#2ca02c",        # green
    "time-shift": "#ff7f0e",  # orange
    "drop-amp": "#d62728",    # red
    "drop-corr": "#9467bd",   # purple
    "drop-fit": "#8c564b",    # brown (PPC aligned-coherence gate)
}
VERDICT_LABELS = {
    "keep": "keep",
    "time-shift": "keep + time-shift",
    "drop-amp": "DROP (bad amplitude)",
    "drop-corr": "DROP (incoherent, Z)",
    "drop-fit": "DROP (incoherent, all comp.)",
}

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
