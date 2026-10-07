"""Which training noise models are rescaled to a real event's pre-event variance, and the training data prepared from a small simulated set."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from seismo_sbi.sbi.datasets.training_data import rescale_training_noise_to_event
from seismo_sbi.sbi.noises.noise_model import NoiseModelConfiguration
from seismo_sbi.utils.errors import InvalidConfiguration


def configuration(noise_model, real_event_jobs=None):
    return SimpleNamespace(sbi_noise_model=NoiseModelConfiguration.from_yaml_block(noise_model),
                           real_event_jobs=real_event_jobs or {})


def test_white_gaussian_noise_trains_without_a_real_event():
    rescale_training_noise_to_event(None, configuration({"type": "gaussian", "noise_level": 1e-6}))


def test_frozen_real_noise_trains_without_a_real_event():
    rescale_training_noise_to_event(None, configuration({"type": "real_noise", "noise_catalogue_path": "/x", "rescale": False}))


def test_a_rescaled_noise_model_needs_a_real_event():
    with pytest.raises(InvalidConfiguration, match="real_events"):
        rescale_training_noise_to_event(None, configuration({"type": "gaussian_filtered"}))


@pytest.mark.requires_data
@pytest.mark.skipif(not os.path.isdir(os.environ.get("INSTASEIS_DB", "")), reason="needs INSTASEIS_DB")
def test_prepare_training_data_returns_the_pipeline_geometry(tmp_path, monkeypatch):
    from seismo_sbi.sbi.configuration import SBI_Configuration
    from seismo_sbi.sbi.datasets.training_data import build_pipeline, generate_training_dataset, prepare_training_data

    monkeypatch.chdir(Path(__file__).resolve().parents[2] / "examples")
    config = SBI_Configuration.from_file("configs/npe_example.yaml")
    config.sim_parameters = config.sim_parameters._replace(syngine_address=os.environ["INSTASEIS_DB"])
    config.pipeline_parameters = config.pipeline_parameters._replace(output_directory=str(tmp_path))
    config.test_job_simulations = config.test_job_simulations._replace(random_events=4)
    training = config.training.apply_overrides(epochs=1)
    pipeline = build_pipeline(config, "configs/npe_example.yaml")
    paths = generate_training_dataset(pipeline, config, training.skip_compression_stencil)

    data = prepare_training_data(pipeline, config, paths, training)

    assert len(data.simulation_paths) == 4
    assert data.trace_length == pipeline.trace_length
    assert data.station_locations.shape == (len(pipeline.simulation_parameters.receivers), 2)
