"""Synthetics come out on a sample grid anchored at the origin minus the pre-event pad."""
import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from seismo_sbi.simulators.instaseis.querier import SYNTHETICS_PRE_EVENT_PAD_S, SyntheticsPreprocessing

ORIGIN = UTCDateTime(2020, 1, 1)
BAND = {"type": "bandpass", "freqmin": 0.06, "freqmax": 0.12, "corners": 4, "zerophase": False}
#: Sample intervals (s) of Instaseis databases in use, and two that divide the pad exactly.
DATABASE_DT_S = [1.2376, 2.3464, 1.2026, 2.41866, 0.5, 1.0]


def recording(sampling_rate_hz, database_dt_s, duration_s=210.0):
    """A band-limited transient starting at the origin, with content below the database Nyquist."""
    freqs_hz = np.arange(0.002, 0.95 * 0.5 / database_dt_s, 0.002)
    phases = np.random.default_rng(1).uniform(0, 2 * np.pi, len(freqs_hz))
    time_s = np.arange(0, duration_s, 1 / sampling_rate_hz)
    data = np.sin(2 * np.pi * np.outer(time_s, freqs_hz) + phases).sum(1) * np.exp(-((time_s - 120) / 60) ** 2)
    trace = Trace(data)
    trace.stats.sampling_rate, trace.stats.starttime = sampling_rate_hz, ORIGIN
    return Stream([trace])


def at_one_hertz(stream):
    """The querier's last step: Lanczos interpolation to 1 Hz from the stream's first sample."""
    return stream.interpolate(1.0, starttime=stream[0].stats.starttime, npts=201, method="lanczos", a=20)[0].data


@pytest.mark.parametrize("database_dt_s", DATABASE_DT_S)
def test_the_synthetic_grid_starts_exactly_one_pad_before_the_origin(database_dt_s):
    processed = SyntheticsPreprocessing({"filter": BAND, "sampling_rate": 1.0})(
        recording(1 / database_dt_s, database_dt_s))
    assert abs(processed[0].stats.starttime - (ORIGIN - SYNTHETICS_PRE_EVENT_PAD_S)) < 1e-6
