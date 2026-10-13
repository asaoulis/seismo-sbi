"""Resampling keeps the input's time grid: no lag, exact start time and spacing."""
import numpy as np
from obspy import Stream, Trace, UTCDateTime

from seismo_sbi.data_handling.preprocessing.processing import resample_on_input_grid

PERIOD_S = 28.0
START = UTCDateTime(2025, 1, 3, 4, 5, 25)


def test_odd_length_sine_resamples_without_lag():
    times_s = np.arange(3441) * 0.5
    trace = Trace(np.sin(2 * np.pi * times_s / PERIOD_S), header={"sampling_rate": 2.0, "starttime": START})
    resampled = resample_on_input_grid(Stream([trace]), 1.0)[0]
    assert resampled.stats.starttime == START
    assert resampled.stats.delta == 1.0
    assert resampled.stats.npts == 1720
    interior = slice(100, 1620)
    expected = np.sin(2 * np.pi * np.arange(resampled.stats.npts) / PERIOD_S)
    np.testing.assert_allclose(resampled.data[interior], expected[interior], atol=0.01)
