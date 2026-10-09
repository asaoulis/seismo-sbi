"""The typed training configuration: parsing, overrides and the checkpoint sidecar it produces.

The two fixtures under ``tests/fixtures/model_meta`` pin the ``model_meta.json`` a configuration
file must produce, so a change to the parsing that alters a checkpoint's metadata fails here.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from seismo_sbi.sbi.npe.training.train import CompressionTrainer, enable_mmd_loss
from seismo_sbi.sbi.training_configuration import TrainingConfiguration
from seismo_sbi.utils.errors import InvalidConfiguration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "model_meta"

COMPONENTS = "ZEN"
STATION_LOCATIONS = np.asarray([[36.4, 25.5], [37.0, 26.0], [36.0, 25.0]])
TRACE_LENGTH_SAMPLES = 200
THETA_SCALER_PROVENANCE = {"moment_tensor": "scale_shape",
                           "log10_m0_min": 15.5, "log10_m0_max": 19.5}


def _load_fixture_configuration(fixture_name):
    return yaml.safe_load((FIXTURES / f"{fixture_name}.yaml").read_text()) or {}


def _stub_mmd_sources(monkeypatch):
    """Stand in for the real events and the simulation suite; only the recorded block matters."""
    from seismo_sbi.sbi.npe.data import mmd_data
    monkeypatch.setattr(mmd_data, "build_real_context", lambda *a, **k: torch.zeros((2, 4)))
    monkeypatch.setattr(mmd_data, "build_psim_loader",
                        lambda *a, **k: SimpleNamespace(dataset=[0]))


@pytest.mark.parametrize("fixture_name", ["minimal", "full_architecture"])
def test_model_meta_matches_the_fixture(fixture_name, tmp_path, monkeypatch):
    """A configuration file produces exactly the checkpoint metadata recorded for it."""
    training = TrainingConfiguration.from_yaml_block(_load_fixture_configuration(fixture_name))
    trainer = CompressionTrainer.from_configuration(
        training, COMPONENTS, STATION_LOCATIONS, TRACE_LENGTH_SAMPLES, THETA_SCALER_PROVENANCE)

    _stub_mmd_sources(monkeypatch)
    enable_mmd_loss(trainer, training,
                    SimpleNamespace(data_manager=SimpleNamespace(data_loader=None),
                                    training_noise_sampler=None),
                    SimpleNamespace(augmentation_chain=None, augmentation_nuisance_params={}))

    written = trainer.write_model_meta(tmp_path).read_text()
    assert written == (FIXTURES / f"{fixture_name}.json").read_text()


def test_an_empty_configuration_gives_the_library_defaults():
    training = TrainingConfiguration.from_yaml_block({})
    assert training.encoder.station_encoder == "cnn"
    assert training.optimizer.lr == 1e-4 and training.optimizer.lr_schedule == "cosine"
    assert training.batch.train == 128 and training.batch.val_size == 256
    assert not training.logging.wandb and training.logging.csv
    assert training.to_model_config({}) == {"station_encoder": "cnn", "theta_scaler": {}}


def test_a_band_limit_takes_the_data_sampling_rate_and_reaches_the_model_config():
    training = TrainingConfiguration.from_yaml_block(
        {"ml_encoder": {"band_limit": {"cutoff_hz": 0.144}},
         "seismic_context": {"sampling_rate": 1.0, "processing": {"sampling_rate": 1.0}}})
    assert training.encoder.band_limit == {"cutoff_hz": 0.144, "sampling_rate_hz": 1.0}
    assert training.to_model_config({})["band_limit"] == {"cutoff_hz": 0.144, "sampling_rate_hz": 1.0}
    assert "band_limit" not in TrainingConfiguration.from_yaml_block({}).to_model_config({})


def test_skip_compression_data_sets_the_stencil_skip():
    assert not TrainingConfiguration.from_yaml_block({}).skip_compression_stencil
    assert TrainingConfiguration.from_yaml_block({"skip_compression_data": True}).skip_compression_stencil


def test_a_mistyped_training_block_raises():
    with pytest.raises(InvalidConfiguration, match="ml_encdoer"):
        TrainingConfiguration.from_yaml_block({"ml_encdoer": {"downsample": 4}})


def test_a_mistyped_key_of_a_named_block_raises():
    with pytest.raises(InvalidConfiguration, match="learning_rate"):
        TrainingConfiguration.from_yaml_block({"ml_optimizer": {"learning_rate": 1e-3}})


def test_conditioning_without_a_param_map_raises():
    with pytest.raises(InvalidConfiguration, match="param_map"):
        TrainingConfiguration.from_yaml_block({"ml_conditioning": {"d_cond": 16}})


def test_command_line_overrides_replace_only_what_they_set():
    training = TrainingConfiguration.from_yaml_block(
        {"ml_architecture": "tcn", "ml_batch": {"train": 64}})
    training.apply_overrides(station_encoder="pno", epochs=12, train_batch_size=8)
    assert training.encoder.station_encoder == "pno"
    assert training.epochs == 12
    assert training.batch.train == 8
    assert training.devices == 1


def test_the_validation_batch_follows_an_overridden_training_batch():
    """With no ``ml_batch.val`` the validation batch stays twice the batch actually used."""
    training = TrainingConfiguration.from_yaml_block({"ml_batch": {"train": 64}})
    training.apply_overrides(train_batch_size=16)
    assert training.batch.val_size == 32


def test_an_explicit_validation_batch_survives_an_overridden_training_batch():
    training = TrainingConfiguration.from_yaml_block({"ml_batch": {"train": 64, "val": 200}})
    training.apply_overrides(train_batch_size=16)
    assert training.batch.val_size == 200


def test_the_conditioning_noise_comes_from_a_training_augmentation_nuisance():
    config = {
        "ml_conditioning": {"param_map": {"source_location": ["latitude", "longitude"]}},
        "parameters": {"nuisance": {"source_location_error": {
            "stage": "training_augmentation", "coordinate_std": [0.05, 0.05]}}},
    }
    training = TrainingConfiguration.from_yaml_block(config)
    assert training.conditioning.coordinate_noise_std == [0.05, 0.05]
    assert training.conditioning.n_cond == 2


def test_a_simulation_staged_location_error_leaves_the_conditioning_unperturbed():
    config = {
        "ml_conditioning": {"param_map": {"source_location": ["latitude", "longitude"]}},
        "parameters": {"nuisance": {"source_location_error": {
            "stage": "simulation", "coordinate_std": [0.05, 0.05]}}},
    }
    training = TrainingConfiguration.from_yaml_block(config)
    assert training.conditioning.coordinate_noise_std is None


def test_dataloader_args_split_the_dataset_at_the_training_fraction():
    training = TrainingConfiguration.from_yaml_block({"ml_batch": {"train": 4}})
    pipeline = SimpleNamespace(
        data_manager=SimpleNamespace(data_loader="loader"),
        simulations_output_path="/sims",
        parameters=SimpleNamespace(names={"moment_tensor": []}),
        training_noise_sampler="sampler")
    data = SimpleNamespace(simulation_paths=list(range(200)), data_scaler="scaler",
                           augmentation_chain=None, augmentation_nuisance_params={},
                           post_noise_chain=None, post_noise_nuisance_params={})

    args = training.dataloader_args(pipeline, data)
    assert args["train_max_index"] == 180
    assert args["train_batch_size"] == 4 and args["val_batch_size"] == 8
    assert args["station_subsampler"] is None
    assert args["data_folder"] == "/sims"


def test_dataset_args_build_the_training_dataset_and_loader_args_complete_them():
    from seismo_sbi.sbi.npe.data.dataloading import TorchSimulationDataset

    training = TrainingConfiguration.from_yaml_block({"ml_batch": {"train": 4}})
    pipeline = SimpleNamespace(
        data_manager=SimpleNamespace(data_loader="loader"),
        simulations_output_path="/sims",
        parameters=SimpleNamespace(names={"moment_tensor": []}),
        training_noise_sampler="sampler")
    data = SimpleNamespace(simulation_paths=list(range(200)), data_scaler="scaler",
                           augmentation_chain=None, augmentation_nuisance_params={},
                           post_noise_chain=None, post_noise_nuisance_params={})

    dataset_args = training.dataset_args(pipeline, data)
    assert training.dataloader_args(pipeline, data) == {**dataset_args, **training.loader_args(200)}
    assert set(dataset_args) <= set(inspect.signature(TorchSimulationDataset).parameters)


def test_loader_args_split_any_number_of_samples_without_a_pipeline():
    training = TrainingConfiguration.from_yaml_block(
        {"ml_batch": {"train": 4, "train_fraction": 0.75}})
    args = training.loader_args(40)
    assert args["train_max_index"] == 30 and args["train_batch_size"] == 4
    assert "data_loader" not in args and "station_subsampler" not in args
