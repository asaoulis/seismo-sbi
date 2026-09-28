"""The state a pipeline declares, and the receiver time shifts it keeps from its configuration."""
from types import SimpleNamespace

from seismo_sbi.sbi import pipeline as pipeline_module
from seismo_sbi.sbi.pipeline import SingleEventPipeline
from seismo_sbi.sbi.types.parameters import ModelParameters, PipelineParameters, SimulationParameters
from seismo_sbi.simulators.receivers import Receiver, Receivers


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
    monkeypatch.setattr(pipeline_module.DatasetGenerator, "create_samplers", staticmethod(lambda *args: {}))
    receivers = Receivers(receivers=[Receiver(37.0, -118.0, "XX", "AAA"), Receiver(38.0, -119.0, "XX", "BBB")])
    receivers.set_time_shifts({"AAA": 3})
    parameters = ModelParameters()
    parameters.theta_fiducial = {"moment_tensor": [1e15] * 6}
    simulation = SimulationParameters(receivers, "ZEN", 100.0, None, 1.0, {})
    pipeline = _pipeline(tmp_path)

    pipeline._load_base_pipeline_params(simulation, parameters, SimpleNamespace(sampling_method={}), None)
    receivers.set_time_shifts({"AAA": 0, "BBB": 0})

    assert pipeline.default_receiver_time_shifts == {"AAA": 3}
