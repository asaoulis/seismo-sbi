"""Unit tests for seismo_sbi.data_quality.alignment."""
import numpy as np
import pytest

from seismo_sbi.data_quality.alignment import (
    nonzero_shifts, optimise_event_shifts, optimise_station_shift)
from seismo_sbi.data_quality.metrics import TraceDescriptor
from seismo_sbi.instaseis_simulator.utils import shift_1d_with_padding


def _wave(n=128, seed=0):
    rng = np.random.default_rng(seed)
    return np.sin(np.linspace(0, 6 * np.pi, n)) + 0.05 * rng.standard_normal(n)


@pytest.mark.parametrize("k", [-4, -1, 2, 6])
def test_recovers_known_station_shift(k):
    syn = np.stack([_wave(seed=1), _wave(seed=2)])
    obs = np.stack([shift_1d_with_padding(syn[i], k) for i in range(2)])
    res = optimise_station_shift(obs, syn, max_shift=10)
    assert res.shift == k
    assert res.vr_after > res.vr_before
    assert res.vr_after == pytest.approx(1.0, abs=1e-9)


def test_shift_clamped_to_cap():
    syn = np.stack([_wave(seed=3)])
    obs = np.stack([shift_1d_with_padding(syn[0], 9)])
    res = optimise_station_shift(obs, syn, max_shift=4)  # true shift 9 > cap
    assert abs(res.shift) <= 4


def test_zero_shift_when_already_aligned():
    syn = np.stack([_wave(seed=4)])
    res = optimise_station_shift(syn.copy(), syn, max_shift=5)
    assert res.shift == 0
    assert res.vr_before == pytest.approx(res.vr_after)


def test_optimise_event_and_nonzero_filter():
    traces = [TraceDescriptor("AAA", "Z", 0, 0), TraceDescriptor("AAA", "E", 0, 0),
              TraceDescriptor("BBB", "Z", 0, 0)]
    base = _wave(seed=5)
    obs = np.stack([shift_1d_with_padding(base, 3), shift_1d_with_padding(base, 3), base])
    syn = np.stack([base, base, base])
    results = optimise_event_shifts(obs, syn, traces, max_shift=8)
    assert results["AAA"].shift == 3
    assert results["BBB"].shift == 0
    # the writer keeps only non-zero shifts
    assert nonzero_shifts(results) == {"AAA": 3}
