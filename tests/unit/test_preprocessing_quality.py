"""Unit tests for the catalogue preprocessing quality gate.

Covers the drop-bad-keep-good ``partition_window_quality`` (added so a single
dead/zero channel removes just its station rather than the whole event/noise
window) and confirms the all-or-nothing ``check_window_quality`` behaviour is
unchanged by the shared-kernel refactor.
"""
from datetime import timedelta

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from seismo_sbi.data_handling.preprocessing.quality import (
    check_window_quality, partition_window_quality)

SR = 1.0
NPTS = 64
DURATION = timedelta(seconds=NPTS)


def _trace(station, channel, data):
    tr = Trace(data=np.asarray(data, dtype=float))
    tr.stats.station = station
    tr.stats.channel = channel
    tr.stats.sampling_rate = SR
    tr.stats.starttime = UTCDateTime(0)
    return tr


def _good(station):
    rng = np.random.default_rng(abs(hash(station)) % (2**32))
    return [_trace(station, c, rng.standard_normal(NPTS)) for c in ("HHZ", "HHN", "HHE")]


def _station_with_zero_component(station):
    rng = np.random.default_rng(0)
    return [
        _trace(station, "HHZ", rng.standard_normal(NPTS)),
        _trace(station, "HHN", rng.standard_normal(NPTS)),
        _trace(station, "HHE", np.zeros(NPTS)),  # dead channel
    ]


def test_partition_keeps_good_drops_bad():
    stream = Stream(_good("AAA") + _station_with_zero_component("BBB") + _good("CCC"))
    kept, dropped = partition_window_quality(stream, ["AAA", "BBB", "CCC"], SR, DURATION)
    assert kept == ["AAA", "CCC"]
    assert [s for s, _ in dropped] == ["BBB"]
    assert "all samples are zero" in dropped[0][1]


def test_partition_all_good():
    stream = Stream(_good("AAA") + _good("CCC"))
    kept, dropped = partition_window_quality(stream, ["AAA", "CCC"], SR, DURATION)
    assert kept == ["AAA", "CCC"]
    assert dropped == []


def test_partition_all_bad_returns_empty_kept():
    stream = Stream(_station_with_zero_component("BBB"))
    kept, dropped = partition_window_quality(stream, ["BBB"], SR, DURATION)
    assert kept == []
    assert len(dropped) == 1


def test_partition_preserves_receiver_order():
    stream = Stream(_good("CCC") + _good("AAA") + _good("BBB"))
    kept, _ = partition_window_quality(stream, ["AAA", "BBB", "CCC"], SR, DURATION)
    assert kept == ["AAA", "BBB", "CCC"]


def test_partition_missing_station_dropped():
    stream = Stream(_good("AAA"))
    kept, dropped = partition_window_quality(stream, ["AAA", "ZZZ"], SR, DURATION)
    assert kept == ["AAA"]
    assert dropped == [("ZZZ", "station 'ZZZ' has no traces")]


def test_check_window_quality_unchanged_first_failure():
    # All-or-nothing gate still fails on the first bad trace with the same reason.
    stream = Stream(_good("AAA") + _station_with_zero_component("BBB"))
    ok, reason = check_window_quality(stream, ["AAA", "BBB"], SR, DURATION)
    assert ok is False
    assert reason == "BBB.HHE: all samples are zero"


def test_check_window_quality_all_good():
    stream = Stream(_good("AAA") + _good("BBB"))
    ok, reason = check_window_quality(stream, ["AAA", "BBB"], SR, DURATION)
    assert ok is True
    assert reason == ""
