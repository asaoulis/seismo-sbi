"""Cross-correlation static time-shift estimation.

Per station, pick the integer sample shift that maximises the station's aligned
variance reduction *jointly* across its components. The search is bounded to
``|shift| <= max_shift`` to avoid cycle-skips: at long periods a shift beyond ~half
the dominant period locks onto a spurious secondary correlation peak, and static
Earth-model corrections should be a few seconds, not tens.

Ported from ``qa_forward_check.run_optimize_shifts``; uses the canonical
:func:`shift_1d_with_padding` (``+shift`` delays the synthetic, matching
``apply_station_time_shifts``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from seismo_sbi.instaseis_simulator.utils import shift_1d_with_padding

from .metrics import TraceDescriptor, variance_reduction


@dataclass(frozen=True)
class ShiftResult:
    """Outcome of the per-station shift search."""

    shift: int
    vr_before: float
    vr_after: float


def optimise_station_shift(
    obs2d_sta: np.ndarray,
    syn2d_sta: np.ndarray,
    max_shift: int,
) -> ShiftResult:
    """Best joint-component integer shift for one station.

    ``obs2d_sta`` / ``syn2d_sta`` have shape ``(n_components, trace_length)``. Returns
    the shift maximising the station's aligned variance reduction, with the before/after
    VR (``vr_before`` is the no-shift VR; ``vr_after >= vr_before`` always).
    """
    o = np.asarray(obs2d_sta)
    s = np.asarray(syn2d_sta)
    den = np.sum(o ** 2)
    best_lag, best_avr = 0, -np.inf
    for lag in range(-max_shift, max_shift + 1):
        ssh = np.stack([shift_1d_with_padding(s[k], lag) for k in range(s.shape[0])])
        avr = 1.0 - np.sum((o - ssh) ** 2) / den if den > 0 else -np.inf
        if avr > best_avr:
            best_avr, best_lag = avr, lag
    return ShiftResult(int(best_lag),
                       variance_reduction(o.flatten(), s.flatten()),
                       float(best_avr))


def optimise_event_shifts(
    obs2d: np.ndarray,
    syn2d: np.ndarray,
    traces: List[TraceDescriptor],
    max_shift: int,
) -> Dict[str, ShiftResult]:
    """Run :func:`optimise_station_shift` for every station in ``traces``."""
    idx: Dict[str, List[int]] = {}
    for i, d in enumerate(traces):
        idx.setdefault(d.station, []).append(i)
    return {sta: optimise_station_shift(obs2d[ii], syn2d[ii], max_shift)
            for sta, ii in idx.items()}


def nonzero_shifts(results: Dict[str, ShiftResult]) -> Dict[str, int]:
    """Extract the ``{station: shift}`` map of non-zero shifts (what gets written to
    ``time_shifts.json``)."""
    return {sta: r.shift for sta, r in results.items() if r.shift != 0}
