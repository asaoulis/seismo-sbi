"""Unit tests for seismo_sbi.data_quality.aggregate.station_reliability."""
import pytest

from seismo_sbi.data_quality.aggregate import station_reliability
from seismo_sbi.data_quality.serialization import QAArtifacts


def _art(event, present, time_shifts=None):
    return QAArtifacts(event=event, present=present, time_shifts=time_shifts or {})


def _rec(verdict, xcorr_Z, amp):
    return {"verdict": verdict, "xcorr_Z": xcorr_Z, "median_amp_ratio": amp}


def test_station_reliability_basic():
    arts = [
        _art("E1", {"GOOD": _rec("keep", 0.9, 1.0), "BAD": _rec("drop-amp", 0.8, 100.0)},
             time_shifts={"GOOD": 2}),
        _art("E2", {"GOOD": _rec("time-shift", 0.7, 1.2), "BAD": _rec("drop-corr", 0.3, 80.0)},
             time_shifts={"GOOD": -4}),
        _art("E3", {"GOOD": _rec("keep", 0.8, 0.8)}),  # BAD absent here
    ]
    rel = station_reliability(arts)

    good = rel["GOOD"]
    assert good.n_events_present == 3
    assert good.n_events_dropped == 0
    assert good.drop_rate == 0.0
    assert good.median_xcorr_Z == pytest.approx(0.8)
    assert good.median_amp_ratio == pytest.approx(1.0)
    # |shifts| = {2, 4, 0} -> median 2
    assert good.median_abs_shift == pytest.approx(2.0)

    bad = rel["BAD"]
    assert bad.n_events_present == 2
    assert bad.n_events_dropped == 2
    assert bad.drop_rate == pytest.approx(1.0)
    assert bad.median_amp_ratio == pytest.approx(90.0)


def test_station_reliability_empty():
    assert station_reliability([]) == {}
