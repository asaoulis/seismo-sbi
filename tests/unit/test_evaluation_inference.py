"""Loading a real event or event file for evaluation, finding a run's checkpoint directory and rebuilding its posterior."""

import inspect
from pathlib import Path
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


def test_resolve_ckpt_dir_finds_the_nested_run_directory(tmp_path):
    from seismo_sbi.evaluation.inference import resolve_ckpt_dir

    run_directory = tmp_path / "model" / "run_1"
    run_directory.mkdir(parents=True)
    (run_directory / "model_meta.json").write_text("{}")
    checkpoint_only = tmp_path / "staged" / "run_2" / "checkpoints"
    checkpoint_only.mkdir(parents=True)
    (checkpoint_only / "best_model-epoch=3.ckpt").touch()

    assert resolve_ckpt_dir(tmp_path / "model") == run_directory
    assert resolve_ckpt_dir(run_directory) == run_directory
    assert resolve_ckpt_dir(tmp_path / "staged") == checkpoint_only.parent
    with pytest.raises(FileNotFoundError):
        resolve_ckpt_dir(tmp_path / "missing")


LV2_CHECKPOINTS = Path(__file__).resolve().parents[2] / "examples" / "ml-checkpoints"
LV2_CHECKPOINT = LV2_CHECKPOINTS / "checkpoints" / "best_model-LV2.ckpt"


@pytest.mark.requires_data
@pytest.mark.skipif(not LV2_CHECKPOINT.is_file() or LV2_CHECKPOINT.stat().st_size < 1_000_000,
                    reason="needs the LV2 checkpoint (git lfs pull)")
def test_build_ml_posterior_from_the_lv2_checkpoint():
    import numpy as np
    import torch

    from seismo_sbi.evaluation.inference import build_ml_posterior
    from seismo_sbi.simulators.receivers import Receivers

    receivers = Receivers.from_station_file(str(LV2_CHECKPOINTS.parent / "configs" / "stations.txt"))
    pipeline = SimpleNamespace(data_manager=SimpleNamespace(data_loader=SimpleNamespace(components="ZEN")),
                               simulation_parameters=SimpleNamespace(receivers=receivers), trace_length=200)

    posterior = build_ml_posterior(LV2_CHECKPOINTS, pipeline)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    observation = torch.as_tensor(np.random.default_rng(0).normal(scale=1e-6, size=(1, 5, 3, 200)),
                                  dtype=torch.float32).to(device)
    samples = posterior.sample((50,), observation, show_progress_bars=False).cpu().numpy()

    assert samples.shape == (50, 6) and np.all(np.isfinite(samples))


@pytest.mark.requires_data
@pytest.mark.skipif(not LV2_CHECKPOINT.is_file() or LV2_CHECKPOINT.stat().st_size < 1_000_000,
                    reason="needs the LV2 checkpoint (git lfs pull)")
def test_load_trained_posterior_samples_as_build_ml_posterior_does(monkeypatch):
    import numpy as np
    import torch

    from seismo_sbi.evaluation.inference import build_ml_posterior, load_trained_posterior

    monkeypatch.chdir(LV2_CHECKPOINTS.parent)
    trained = load_trained_posterior("configs/LV2_continuity.yaml", LV2_CHECKPOINTS)
    pipeline = SimpleNamespace(data_manager=trained.pipeline.data_manager,
                               simulation_parameters=trained.pipeline.simulation_parameters,
                               trace_length=200)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    observation = torch.as_tensor(np.random.default_rng(0).normal(scale=1e-6, size=(1, 5, 3, 200)),
                                  dtype=torch.float32).to(device)
    draws = []
    for posterior in (trained.posterior, build_ml_posterior(LV2_CHECKPOINTS, pipeline)):
        torch.manual_seed(0)
        draws.append(posterior.sample((50,), observation, show_progress_bars=False).cpu().numpy())

    np.testing.assert_array_equal(draws[0], draws[1])
    assert trained.data_scaler.inverse_transform(draws[0]).shape == (50, 6)
