"""Selecting which real event an evaluation loads."""

import inspect
from types import SimpleNamespace

import pytest

from seismo_sbi.evaluation.inference import load_real_observation


def test_job_name_has_no_default():
    """The event key comes from the caller, so the library names no study's event."""
    assert (inspect.signature(load_real_observation).parameters["job_name"].default
            is inspect.Parameter.empty)


def test_an_unknown_job_name_lists_the_available_events():
    config = SimpleNamespace(real_event_jobs={"event1": "/events/event1.h5"})
    with pytest.raises(KeyError, match="event1"):
        load_real_observation(config, None, "not_an_event")


def test_load_observation_undoes_the_receiver_time_shifts(tmp_path):
    import numpy as np

    from seismo_sbi.evaluation.inference import load_observation
    from seismo_sbi.simulators.receivers import Receiver, Receivers
    from seismo_sbi.simulators.simulation_io import SimulationSaver

    receivers = Receivers(receivers=[Receiver(37.0, -118.0, "XX", "AAA", ["Z"]),
                                     Receiver(38.0, -119.0, "XX", "BBB", ["Z"])])
    traces = {"AAA": np.arange(1.0, 9.0), "BBB": np.arange(11.0, 19.0)}
    path = tmp_path / "event.h5"
    SimulationSaver(output_data={station: {"Z": trace} for station, trace in traces.items()}).dump_data_as_hdf5(path)

    observation = load_observation(path, receivers, "Z", time_shifts={"AAA": 2})

    assert observation.shape == (2, 1, 8)
    np.testing.assert_array_equal(observation[0, 0], np.r_[traces["AAA"][2:], 0.0, 0.0])
    np.testing.assert_array_equal(observation[1, 0], traces["BBB"])
    assert [receiver.time_shift for receiver in receivers] == [-2, 0]
