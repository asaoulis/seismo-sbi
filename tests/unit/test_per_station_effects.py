"""Amplitude and time-shift errors configured per station: a ``{station: value}`` map with a
``"default"`` entry, alongside the single value every station shares."""
import numpy as np
import pytest

from seismo_sbi.nuisance_effects.amplitude_effect import AmplitudeErrorEffect
from seismo_sbi.nuisance_effects.post_processing import build_post_processing_chain
from seismo_sbi.nuisance_effects.seismogram_effect import station_value
from seismo_sbi.nuisance_effects.time_shift_effect import TimeShiftErrorEffect

TRACES = {station: {"Z": np.ones(64)} for station in ("NOISY", "QUIET", "OTHER")}


def test_station_value_reads_a_scalar_a_named_station_and_the_default():
    assert station_value(0.3, "ANY", 1.0) == 0.3
    assert station_value({"NOISY": 2.0, "default": 0.5}, "NOISY", 1.0) == 2.0
    assert station_value({"NOISY": 2.0, "default": 0.5}, "OTHER", 1.0) == 0.5
    assert station_value({"NOISY": 2.0}, "OTHER", 1.0) == 1.0


def test_an_amplitude_range_per_station_scales_each_station_within_its_own_range():
    effect = AmplitudeErrorEffect(scale_range={"NOISY": [3.0, 4.0], "default": [0.9, 1.1]})
    np.random.seed(0)
    out = effect(TRACES, None, amplitude_error=1.0)
    assert 3.0 <= out["NOISY"]["Z"][0] <= 4.0
    assert all(0.9 <= out[station]["Z"][0] <= 1.1 for station in ("QUIET", "OTHER"))


def test_a_lognormal_width_per_station_leaves_a_zero_width_station_untouched():
    effect = AmplitudeErrorEffect(distribution="lognormal", log_sigma_dex={"QUIET": 0.0, "default": 0.5},
                                  always_on=True)
    np.random.seed(1)
    out = effect(TRACES, None, amplitude_error=1.0)
    np.testing.assert_array_equal(out["QUIET"]["Z"], TRACES["QUIET"]["Z"])
    assert not np.allclose(out["NOISY"]["Z"], TRACES["NOISY"]["Z"])


def test_a_negative_lognormal_width_in_a_map_is_refused():
    with pytest.raises(ValueError, match="log_sigma_dex"):
        AmplitudeErrorEffect(distribution="lognormal", log_sigma_dex={"NOISY": -0.1})


def test_a_time_shift_width_per_station_shifts_only_the_stations_given_one():
    trace = np.sin(np.linspace(0.0, 6.0, 64))
    traces = {station: {"Z": trace} for station in TRACES}
    effect = TimeShiftErrorEffect(1.0, gaussian_sigma={"NOISY": 3.0, "default": 0.0})
    np.random.seed(2)
    out = effect(traces, None, time_shift_error=1.0)
    np.testing.assert_allclose(out["QUIET"]["Z"], trace, atol=1e-12)
    assert not np.allclose(out["NOISY"]["Z"], trace)


def test_a_per_station_map_from_a_configuration_reaches_the_effect():
    chain = build_post_processing_chain(
        ["amplitude_error", "time_shift_error"],
        {"amplitude_error": {"scale_range": {"NOISY": [3.0, 4.0], "default": [1.0, 1.0]}},
         "time_shift_error": {"sampling_rate": 1.0, "gaussian_sigma": {"NOISY": 2.0, "default": 0.5}}})
    amplitude, time_shift = chain.effects
    assert amplitude._scale_low == {"NOISY": 3.0, "default": 1.0}
    assert time_shift._sigma == {"NOISY": 2.0, "default": 0.5}
