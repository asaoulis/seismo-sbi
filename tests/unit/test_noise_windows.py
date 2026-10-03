"""Noise windows cut from a continuous Stream feed the noise sampler and the covariance estimator; a
noise pool is screened for windows holding earthquakes."""
import numpy as np
from obspy import Stream, Trace, UTCDateTime

from seismo_sbi.data_handling.preprocessing.noise_windows import noise_windows_from_stream, quiet_window_mask
from seismo_sbi.sbi.noises.covariance_estimator import EmpiricalCovarianceEstimator
from seismo_sbi.sbi.noises.real_noise import RealNoiseSampler
from seismo_sbi.simulators.receivers import Receivers

START = UTCDateTime(2020, 1, 1)
DURATION_S = 3 * 3600
EVENT = (START + 5000, START + 5200)


def continuous_stream():
    """Two stations of unit noise at 1 Hz; a large event at ``EVENT``; BBB starts an hour late."""
    rng = np.random.default_rng(5)
    stream = Stream()
    for station, offset_s in (("AAA", 0), ("BBB", 3600)):
        for channel in ("BHZ", "BHE", "BHN"):
            data = rng.normal(size=DURATION_S - offset_s)
            data[int(EVENT[0] - START) - offset_s:int(EVENT[1] - START) - offset_s] = 1e6
            stream.append(Trace(data=data, header={"network": "XX", "station": station, "channel": channel,
                                                   "sampling_rate": 1.0, "starttime": START + offset_s}))
    return stream


def receivers():
    return Receivers.from_arrays(["AAA", "BBB"], ["XX", "XX"], [0.0, 1.0], [0.0, 1.0])


def test_noise_windows_avoid_the_event_and_mark_the_late_station_absent():
    noise_windows, present = noise_windows_from_stream(continuous_stream(), receivers(), 200.0, 1.0,
                                                       avoid_windows_utc=[EVENT], buffer_s=120.0, step_s=300.0)

    assert noise_windows.shape == (len(present), 2 * 3 * 201)
    assert np.abs(noise_windows).max() < 100.0
    assert present[:, 0].all() and not present[:, 1].all() and present[:, 1].any()
    assert not noise_windows[~present[:, 1], 603:].any()


def test_noise_windows_feed_the_sampler_and_the_estimator():
    noise_windows, present = noise_windows_from_stream(continuous_stream(), receivers(), 200.0, 1.0,
                                                       avoid_windows_utc=[EVENT], buffer_s=120.0, step_s=300.0)

    noise_vector, present_mask = RealNoiseSampler.from_windows(noise_windows, receivers(), "ZEN", present=present)()
    assert noise_vector.shape == (1206,) and present_mask[0]
    covariances = EmpiricalCovarianceEstimator(None, receivers(), "ZEN", covariance_exp_tapering=False,
                                               verbose=False).estimate_from_windows(noise_windows, present)
    assert abs(covariances["BBB"]["2"][0] - 1.0) < 0.1


def test_a_window_loud_at_one_station_is_screened_out():
    vertical_rms = np.random.default_rng(2).uniform(0.8, 1.2, size=(50, 4)) * [1.0, 10.0, 0.1, 3.0]
    vertical_rms[7, 2] *= 6.0

    keep = quiet_window_mask(vertical_rms)

    assert not keep[7] and keep.sum() == 49
