"""Cross-event station reliability statistics.

Roll up many events' :class:`QAArtifacts` into per-station reliability, the driver for
catalogue-scale station filtering ("station X is dropped in 80% of events -> blacklist").
Pure and deterministic, so it is unit-testable without any forward model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .serialization import QAArtifacts


@dataclass(frozen=True)
class StationReliability:
    """Aggregate QA behaviour of one station across a catalogue of events."""

    station: str
    n_events_present: int
    n_events_dropped: int
    drop_rate: float
    median_amp_ratio: float
    median_xcorr_Z: float
    median_abs_shift: float


def _median(values: List[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def station_reliability(artifacts: List[QAArtifacts]) -> Dict[str, StationReliability]:
    """Aggregate per-station statistics over a list of per-event artifacts.

    Uses each event's ``present`` verdict records (``verdict``, ``xcorr_Z``,
    ``median_amp_ratio``) and ``time_shifts``. Stations are keyed by code; a station
    counts as "present" in an event if it appears in that event's ``present`` map.
    """
    present_count: Dict[str, int] = {}
    dropped_count: Dict[str, int] = {}
    amp: Dict[str, List[float]] = {}
    xcorr: Dict[str, List[float]] = {}
    abs_shift: Dict[str, List[float]] = {}

    for art in artifacts:
        for sta, rec in art.present.items():
            present_count[sta] = present_count.get(sta, 0) + 1
            if str(rec.get("verdict", "")).startswith("drop"):
                dropped_count[sta] = dropped_count.get(sta, 0) + 1
            if rec.get("median_amp_ratio") is not None:
                amp.setdefault(sta, []).append(float(rec["median_amp_ratio"]))
            if rec.get("xcorr_Z") is not None:
                xcorr.setdefault(sta, []).append(float(rec["xcorr_Z"]))
            abs_shift.setdefault(sta, []).append(abs(float(art.time_shifts.get(sta, 0))))

    out: Dict[str, StationReliability] = {}
    for sta, n_present in present_count.items():
        n_drop = dropped_count.get(sta, 0)
        out[sta] = StationReliability(
            station=sta,
            n_events_present=n_present,
            n_events_dropped=n_drop,
            drop_rate=n_drop / n_present if n_present else float("nan"),
            median_amp_ratio=_median(amp.get(sta, [])),
            median_xcorr_Z=_median(xcorr.get(sta, [])),
            median_abs_shift=_median(abs_shift.get(sta, [])),
        )
    return out
