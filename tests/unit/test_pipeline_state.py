"""The state a pipeline declares, and the receiver time shifts it keeps from its configuration."""
from types import SimpleNamespace

import h5py
import numpy as np

import pytest

from seismo_sbi.sbi import pipeline as pipeline_module
from seismo_sbi.sbi.pipeline import SingleEventPipeline
from seismo_sbi.sbi.types.parameters import ModelParameters, PipelineParameters, SimulationParameters
from seismo_sbi.simulators.receivers import Receiver, Receivers
from seismo_sbi.utils.errors import InvalidConfiguration


class _StubSimulatorWrapper:
    def __init__(self, *args):
        self.simulation_save_callable = None


def _pipeline(tmp_path):
    return SingleEventPipeline(PipelineParameters("run", str(tmp_path), "job", False, 1))


def test_a_new_pipeline_declares_its_state(tmp_path):
    pipeline = _pipeline(tmp_path)

    for name in ("parameters", "data_cov_mat", "least_squares_solver", "mcmc_chain_for_mle"):
        assert getattr(pipeline, name) is None
    assert pipeline.default_receiver_time_shifts == {}


def test_the_pipeline_keeps_the_configured_receiver_time_shifts(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "GeneralSimulatorWrapper", _StubSimulatorWrapper)
    monkeypatch.setattr(pipeline_module.ParameterSampler, "from_configuration", classmethod(lambda *args: None))
    receivers = Receivers(receivers=[Receiver(37.0, -118.0, "XX", "AAA"), Receiver(38.0, -119.0, "XX", "BBB")])
    receivers.set_time_shifts({"AAA": 3})
    parameters = ModelParameters()
    parameters.theta_fiducial = {"moment_tensor": [1e15] * 6}
    simulation = SimulationParameters(receivers, "ZEN", 100.0, None, 1.0, {})
    pipeline = _pipeline(tmp_path)

    pipeline._load_base_pipeline_params(simulation, parameters, SimpleNamespace(sampling_method={}), None)
    receivers.set_time_shifts({"AAA": 0, "BBB": 0})

    assert pipeline.default_receiver_time_shifts == {"AAA": 3}


def test_multi_event_real_jobs_are_read_at_the_configured_trace_length():
    from seismo_sbi.sbi.pipeline_variants import MultiEventPipeline

    lengths_seen = []

    class _DataManager:
        data_length = 201

        def _create_job_data_from_real_events(self, real_event_jobs, test_noises, data_length=None):
            lengths_seen.append(data_length)
            return ["real job"]

    pipeline = MultiEventPipeline.__new__(MultiEventPipeline)
    pipeline.data_manager = _DataManager()
    pipeline.test_noises = {}

    job_data = pipeline.create_job_data([], {"event": "/events/event.h5"})

    assert job_data == ["real job"]
    assert lengths_seen == [201]


def test_the_real_trace_length_is_the_configured_duration_times_the_sampling_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "GeneralSimulatorWrapper", _StubSimulatorWrapper)
    monkeypatch.setattr(pipeline_module.ParameterSampler, "from_configuration", classmethod(lambda *args: None))
    receivers = Receivers(receivers=[Receiver(37.0, -118.0, "XX", "AAA")])
    parameters = ModelParameters()
    parameters.theta_fiducial = {"moment_tensor": [1e15] * 6}
    trace_lengths = {}
    for sampling_rate_hz in (1.0, 0.5):
        pipeline = _pipeline(tmp_path)
        simulation = SimulationParameters(receivers, "ZEN", 200.0, None, sampling_rate_hz, {})
        pipeline._load_base_pipeline_params(simulation, parameters, SimpleNamespace(sampling_method={}), None)
        trace_lengths[sampling_rate_hz] = pipeline.data_manager.data_length

    assert trace_lengths == {1.0: 201, 0.5: 101}


def test_load_configuration_takes_the_configured_seed_and_compression_methods(tmp_path, monkeypatch):
    pipeline = _pipeline(tmp_path)
    loaded = []
    monkeypatch.setattr(SingleEventPipeline, "load_seismo_parameters",
                        lambda self, *records: loaded.append(records))
    config = SimpleNamespace(compression_methods=[("optimal_score", {})], sbi_seed=17,
                             sim_parameters="sim", model_parameters="model", dataset_parameters="dataset")

    pipeline.load_configuration(config)

    assert pipeline.seed == 17
    assert pipeline.compression_methods == [("optimal_score", {})]
    assert loaded == [("sim", "model", "dataset")]


def test_training_sources_are_drawn_from_the_bounds_and_sampling_method_at_generation(tmp_path, monkeypatch):
    captured = []

    class _CapturingGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def run_and_save_simulations(self, simulation_inputs, output_paths):
            captured.extend(zip(simulation_inputs, output_paths))

    monkeypatch.setattr(pipeline_module, "DatasetGenerator", _CapturingGenerator)
    pipeline = _pipeline(tmp_path)
    pipeline.simulator_wrapper = _StubSimulatorWrapper()
    pipeline.parameters = ModelParameters()
    pipeline.parameters.names = {"moment_tensor": ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]}
    pipeline.parameters.theta_fiducial = {"moment_tensor": [1e15] * 6}
    pipeline.parameters.bounds = {"moment_tensor": [[-1e17] * 6, [1e17] * 6]}
    pipeline.parameter_sampler = pipeline_module.ParameterSampler.from_configuration(
        pipeline.parameters, {"moment_tensor": "constant"})
    pipeline.parameters.bounds["moment_tensor"] = [[0.0] * 6, [1e15] * 6]

    pipeline.generate_simulation_data(SimpleNamespace(num_simulations=4, sampling_method={"moment_tensor": "uniform"}))

    assert [path for _, path in captured] == [f"{pipeline.simulations_output_path}/train/sim_{i}.h5" for i in range(4)]
    assert all(0.0 <= value <= 1e15 for inputs, _ in captured for value in inputs["moment_tensor"])


def test_every_multi_event_test_job_carries_the_covariance_its_noise_was_rescaled_to(tmp_path):
    from seismo_sbi.sbi.noises.real_noise import RealNoiseSampler
    from seismo_sbi.sbi.pipeline_variants import MultiEventPipeline

    receivers = Receivers(receivers=[Receiver(0.0, 0.0, "XX", "AAA", ["Z"])])
    for window in range(4):
        with h5py.File(tmp_path / f"window_{window}.h5", "w") as noise_file:
            noise_file.create_dataset("outputs/AAA/Z", data=np.full(8, float(window + 1)))
            noise_file.create_dataset("misc/AAA/Z", data=np.array([float(window + 1) ** 2, 0.5]))
    sampler = RealNoiseSampler(SimulationParameters(receivers, "Z", 8.0, None, 1.0, {}), tmp_path, 8)

    class _DataManager:
        data_loader = SimpleNamespace(load_input_data=lambda path: None,
                                      load_simulation_data_array=lambda path: np.zeros(8))
        data_length = 8

        def _create_job_data_from_real_events(self, real_event_jobs, test_noises, data_length=None):
            return []

    pipeline = MultiEventPipeline.__new__(MultiEventPipeline)
    pipeline.data_manager = _DataManager()
    pipeline.test_noises = {"real_noise": sampler}
    np.random.seed(0)
    jobs = pipeline.create_job_data([tmp_path / f"sim_{k}.h5" for k in range(6)], {})

    first_covariance = jobs[0].covariance
    first_level = jobs[0].data_vector[0]
    assert all(job.covariance is first_covariance for job in jobs)
    assert all(np.allclose(job.data_vector, first_level) for job in jobs)
