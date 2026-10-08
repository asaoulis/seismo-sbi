"""Synthetics see the observed data's filter, on a grid anchored at the origin minus the pad."""
import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from seismo_sbi.simulators.instaseis.querier import SyntheticsPreprocessing
from seismo_sbi.simulators.simulation_io import SYNTHETICS_PRE_EVENT_PAD_S
from seismo_sbi.simulators.spectral_filter import filter_and_shift

ORIGIN = UTCDateTime(2020, 1, 1)
BAND = {"type": "bandpass", "freqmin": 0.06, "freqmax": 0.12, "corners": 4, "zerophase": False}
#: Sample intervals (s) of Instaseis databases in use, and two that divide the pad exactly.
DATABASE_DT_S = [1.2376, 2.3464, 1.2026, 2.41866, 0.5, 1.0]
#: (database dt in s, filter rate in Hz, band in Hz) of the studies' configurations, and a 0.3 s database.
STUDY_FILTERS = [(1.2376, 5.0, (0.06, 0.2)), (1.2376, 5.0, (0.06, 0.12)), (2.3464, 2.0, (0.02, 0.05)),
                 (1.2026, 100.0, (0.03, 0.08)), (2.41866, 100.0, (0.02, 0.04)), (0.3, 100.0, (0.03, 0.08))]


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
    processed = SyntheticsPreprocessing({"filter": BAND, "sampling_rate": 1.0, "filter_sampling_rate": 5.0})(
        recording(1 / database_dt_s, database_dt_s))
    assert abs(processed[0].stats.starttime - (ORIGIN - SYNTHETICS_PRE_EVENT_PAD_S)) < 1e-6


@pytest.mark.parametrize("database_dt_s", [1.2376, 2.3464, 1.2026])
def test_synthetics_filtered_at_the_data_rate_match_the_data_path(database_dt_s):
    data_rate_hz = 5.0
    data = recording(data_rate_hz, database_dt_s)
    data.filter(**BAND)
    observed = at_one_hertz(data)
    synthetic = at_one_hertz(SyntheticsPreprocessing(
        {"filter": BAND, "sampling_rate": 1.0, "filter_sampling_rate": data_rate_hz})(
        recording(1 / database_dt_s, database_dt_s)))
    pad = int(SYNTHETICS_PRE_EVENT_PAD_S)
    observed, synthetic = observed[40:140], synthetic[pad + 40:pad + 140]
    assert np.linalg.norm(synthetic - observed) / np.linalg.norm(observed) < 1e-3


def upsampled_then_filtered(stream, processing):
    """The earlier path: Lanczos-interpolate to the filter rate on the anchored grid, filter there."""
    start, end = stream[0].stats.starttime, stream[0].stats.endtime
    length = (end - start) * processing["sampling_rate"]
    stream = stream.trim(start - length * 0.3, end + length * 0.3, pad=True, fill_value=0).taper(0.05, "cosine")
    anchor, dt = start - SYNTHETICS_PRE_EVENT_PAD_S, 1.0 / processing["filter_sampling_rate"]
    first_sample = anchor + np.ceil((stream[0].stats.starttime - anchor) / dt - 1e-9) * dt
    stream = stream.interpolate(processing["filter_sampling_rate"], method="lanczos", a=20, starttime=first_sample)
    stream = stream.filter(**processing["filter"]).trim(anchor, end - length * 0.1)
    return stream.trim(anchor, end - SYNTHETICS_PRE_EVENT_PAD_S, pad=True, fill_value=0)


@pytest.mark.parametrize("database_dt_s, filter_rate_hz, band_hz", STUDY_FILTERS)
def test_the_spectral_filter_matches_upsampling_to_the_filter_rate(database_dt_s, filter_rate_hz, band_hz):
    band = dict(BAND, freqmin=band_hz[0], freqmax=band_hz[1])
    processing = {"filter": band, "sampling_rate": 1.0, "filter_sampling_rate": filter_rate_hz}
    spectral = at_one_hertz(SyntheticsPreprocessing(processing)(recording(1 / database_dt_s, database_dt_s)))
    upsampled = at_one_hertz(upsampled_then_filtered(recording(1 / database_dt_s, database_dt_s), processing))
    assert np.linalg.norm(spectral - upsampled) / np.linalg.norm(upsampled) < 1e-3


@pytest.mark.parametrize("zerophase", [False, True])
@pytest.mark.parametrize("spike_index", [0, 299])
def test_at_its_own_rate_the_spectral_filter_is_obspys_filter_without_wrap_around(zerophase, spike_index):
    band = dict(BAND, freqmin=0.02, freqmax=0.05, zerophase=zerophase)
    spike = np.zeros(300)
    spike[spike_index] = 1.0
    obspy_filtered = Trace(spike.copy()).filter(**band).data
    peak_response = np.abs(Trace(np.roll(spike, -spike_index)).filter(**band).data).max()
    spectral = filter_and_shift(spike, 1.0, band, 1.0)
    np.testing.assert_allclose(spectral, obspy_filtered, atol=1e-8 * peak_response)
