"""Characterization tests for Simulator.run_simulation() and apply_station_time_shifts().

These tests lock down the existing dispatch behaviour before the nuisance-parameter
refactor so we can confirm nothing breaks.  They use a lightweight MockSimulator that
avoids any real forward-model (Instaseis / CPS) calls.
"""

import numpy as np
import pytest

from seismo_sbi.instaseis_simulator.simulator import Simulator
from seismo_sbi.instaseis_simulator.wrapper import GenericPointSource, GeneralMomentTensor, SimpleMomentTensor
from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
from seismo_sbi.instaseis_simulator.utils import shift_1d_with_padding, apply_station_time_shifts
from seismo_sbi.sbi.configuration import InvalidConfiguration

TRACE_LEN = 40


# ---------------------------------------------------------------------------
# Minimal mock simulator
# ---------------------------------------------------------------------------

class MockSimulator(Simulator):
    """Concrete Simulator returning canned seismograms; records the last call."""

    def __init__(self, receivers, trace_len=TRACE_LEN):
        super().__init__(
            components=["Z"],
            receivers=receivers,
            seismogram_duration_in_s=trace_len,
            synthetics_processing={
                "sampling_rate": 1.0,
                "filter": {"type": "bandpass", "freqmin": 0.01, "freqmax": 0.1},
            },
        )
        self._trace_len = trace_len
        self.last_source = None
        self.last_kwargs = {}

    def generic_point_source_simulation(self, source: GenericPointSource, **kwargs):
        self.last_source = source
        self.last_kwargs = dict(kwargs)
        return {
            rec.station_name: {comp: np.ones(self._trace_len) for comp in rec.components}
            for rec in self.receivers.iterate()
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def one_station_receivers():
    r = Receiver(latitude=0.0, longitude=0.0, network="XX", station_name="STA1", components=["Z"])
    return Receivers(receivers=[r])


@pytest.fixture
def mock_sim(one_station_receivers):
    return MockSimulator(one_station_receivers)


@pytest.fixture
def two_station_receivers():
    r1 = Receiver(0.0,  0.0, "XX", "STA1", ["Z"], time_shift=0)
    r2 = Receiver(1.0, 10.0, "XX", "STA2", ["Z"], time_shift=5)
    return Receivers(receivers=[r1, r2])


# Minimal source_parameters dicts
_SOURCE_LOC = [37.6, -118.9, 5.0, 0.0]
_MT_PARAMS   = {"source_location": _SOURCE_LOC, "moment_tensor": [1e14] * 6}
_MAG_PARAMS  = {"source_location": _SOURCE_LOC, "earthquake_magnitude": [1e14]}


# ===========================================================================
# shift_1d_with_padding
# ===========================================================================

class TestShift1dWithPadding:

    def test_zero_shift_is_identity(self):
        x = np.arange(10, dtype=float)
        out = shift_1d_with_padding(x, shift=0)
        assert np.array_equal(out, x)

    def test_zero_shift_returns_copy(self):
        x = np.ones(10)
        out = shift_1d_with_padding(x, shift=0)
        out[0] = 99.0
        assert x[0] != 99.0, "shift=0 should return a copy, not a view"

    def test_positive_shift_delays_trace(self):
        """Positive shift pads zeros at the front and discards the tail."""
        x = np.ones(10)
        out = shift_1d_with_padding(x, shift=3)
        assert np.all(out[:3] == 0.0), "First 3 samples should be zero after delay"
        assert np.all(out[3:] == 1.0), "Remaining samples should be 1.0"

    def test_negative_shift_advances_trace(self):
        """Negative shift discards the head and pads zeros at the tail."""
        x = np.ones(10)
        out = shift_1d_with_padding(x, shift=-4)
        assert np.all(out[:-4] == 1.0), "First 6 samples should be 1.0"
        assert np.all(out[-4:] == 0.0), "Last 4 samples should be zero after advance"

    def test_output_length_unchanged(self):
        x = np.arange(15, dtype=float)
        for s in [-7, -1, 0, 1, 7]:
            assert len(shift_1d_with_padding(x, s)) == 15


# ===========================================================================
# apply_station_time_shifts
# ===========================================================================

class TestApplyStationTimeShifts:

    def test_zero_shift_returns_same_values(self, one_station_receivers):
        seismo_map = {"STA1": {"Z": np.ones(TRACE_LEN)}}
        shifted = apply_station_time_shifts(one_station_receivers, seismo_map)
        assert np.array_equal(shifted["STA1"]["Z"], np.ones(TRACE_LEN))

    def test_returns_new_dict_not_in_place(self, one_station_receivers):
        seismo_map = {"STA1": {"Z": np.ones(TRACE_LEN)}}
        shifted = apply_station_time_shifts(one_station_receivers, seismo_map)
        assert shifted is not seismo_map

    def test_positive_receiver_shift_delays_trace(self):
        r = Receiver(0.0, 0.0, "XX", "STA1", ["Z"], time_shift=5)
        receivers = Receivers(receivers=[r])
        seismo_map = {"STA1": {"Z": np.ones(TRACE_LEN)}}
        shifted = apply_station_time_shifts(receivers, seismo_map)
        assert np.all(shifted["STA1"]["Z"][:5] == 0.0)
        assert np.all(shifted["STA1"]["Z"][5:] == 1.0)

    def test_zero_shift_station_unchanged_multi_station(self, two_station_receivers):
        """STA1 has shift=0; its trace must not change."""
        seismo_map = {
            "STA1": {"Z": np.arange(TRACE_LEN, dtype=float)},
            "STA2": {"Z": np.arange(TRACE_LEN, dtype=float)},
        }
        shifted = apply_station_time_shifts(two_station_receivers, seismo_map)
        assert np.array_equal(shifted["STA1"]["Z"], seismo_map["STA1"]["Z"])

    def test_nonzero_shift_station_changed_multi_station(self, two_station_receivers):
        """STA2 has shift=5; its trace must differ from the original."""
        seismo_map = {
            "STA1": {"Z": np.ones(TRACE_LEN)},
            "STA2": {"Z": np.ones(TRACE_LEN)},
        }
        shifted = apply_station_time_shifts(two_station_receivers, seismo_map)
        assert not np.array_equal(shifted["STA2"]["Z"], seismo_map["STA2"]["Z"])


# ===========================================================================
# Simulator.run_simulation() — parameter dispatch
# ===========================================================================

class TestRunSimulationDispatch:

    def test_moment_tensor_params_create_general_moment_tensor(self, mock_sim):
        source, _ = mock_sim.run_simulation(dict(_MT_PARAMS))
        assert isinstance(source.moment_tensor, GeneralMomentTensor)

    def test_earthquake_magnitude_creates_simple_moment_tensor(self, mock_sim):
        source, _ = mock_sim.run_simulation(dict(_MAG_PARAMS))
        assert isinstance(source.moment_tensor, SimpleMomentTensor)

    def test_source_location_unpacked_correctly(self, mock_sim):
        source, _ = mock_sim.run_simulation(dict(_MT_PARAMS))
        assert source.source_location.latitude == pytest.approx(37.6)
        assert source.source_location.longitude == pytest.approx(-118.9)
        assert source.source_location.depth == pytest.approx(5.0)
        assert source.source_location.time_shift == pytest.approx(0.0)

    def test_returns_tuple_of_source_and_seismograms(self, mock_sim):
        result = mock_sim.run_simulation(dict(_MT_PARAMS))
        assert isinstance(result, tuple) and len(result) == 2

    def test_seismograms_map_has_correct_stations(self, mock_sim, one_station_receivers):
        _, seismograms = mock_sim.run_simulation(dict(_MT_PARAMS))
        assert set(seismograms.keys()) == {"STA1"}

    def test_seismograms_map_has_correct_components(self, mock_sim):
        _, seismograms = mock_sim.run_simulation(dict(_MT_PARAMS))
        assert "Z" in seismograms["STA1"]

    def test_seismogram_length_matches_trace_len(self, mock_sim):
        _, seismograms = mock_sim.run_simulation(dict(_MT_PARAMS))
        assert len(seismograms["STA1"]["Z"]) == TRACE_LEN

    def test_velocity_model_passed_as_kwarg(self, mock_sim):
        """velocity_model in source_parameters must be forwarded as a kwarg."""
        vmodel = object()  # sentinel — identity check
        params = dict(_MT_PARAMS)
        params["velocity_model"] = vmodel
        mock_sim.run_simulation(params)
        assert mock_sim.last_kwargs.get("velocity_model") is vmodel

    def test_no_velocity_model_gives_none_kwarg(self, mock_sim):
        mock_sim.run_simulation(dict(_MT_PARAMS))
        assert mock_sim.last_kwargs.get("velocity_model") is None

    def test_missing_moment_tensor_and_magnitude_raises(self, mock_sim):
        params = {"source_location": _SOURCE_LOC}
        with pytest.raises(InvalidConfiguration):
            mock_sim.run_simulation(params)

    def test_missing_source_location_raises(self, mock_sim):
        params = {"moment_tensor": [1e14] * 6}
        with pytest.raises(KeyError):
            mock_sim.run_simulation(params)

    def test_extra_unknown_keys_are_silently_accepted(self, mock_sim):
        """Extra keys (future post-processing params) must not raise errors."""
        params = dict(_MT_PARAMS)
        params["amplitude_error"] = 0.9
        params["some_future_param"] = 42
        # Must not raise
        mock_sim.run_simulation(params)

    def test_run_simulation_does_not_mutate_source_parameters(self, mock_sim):
        params = dict(_MT_PARAMS)
        params["velocity_model"] = None
        original_keys = set(params.keys())

        mock_sim.run_simulation(params)

        assert set(params.keys()) == original_keys, (
            "run_simulation() must not remove keys from the caller's dict"
        )
