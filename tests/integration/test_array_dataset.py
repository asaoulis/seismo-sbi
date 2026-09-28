"""An ArraySimulationDataset built from a folder's arrays serves the same samples as the folder."""
import os

import h5py
import numpy as np
import pytest
import torch

from seismo_sbi.nuisance_effects.amplitude_effect import AmplitudeErrorEffect
from seismo_sbi.nuisance_effects.dropout_effects import ComponentDropoutEffect
from seismo_sbi.nuisance_effects.post_processing import PostProcessingChain
from seismo_sbi.nuisance_effects.time_shift_effect import TimeShiftErrorEffect
from seismo_sbi.simulators.receivers import Receiver, Receivers
from seismo_sbi.simulators.simulation_io import SimulationDataLoader
from seismo_sbi.sbi.compression.ML.array_dataset import ArraySimulationDataset
from seismo_sbi.sbi.compression.ML.dataloading import (
    StationSubsampler, TorchSimulationDataset, make_torch_dataloaders)

TRACE_LENGTH = 40
COMPONENTS = "ZEN"
N_SIMULATIONS = 6
PARAMETER_NAME_MAP = {"moment_tensor": ["m_rr", "m_tt"], "source_location": ["depth"]}
CONDITIONING_MAP = {"source_location": ["latitude", "longitude", "depth"]}


def _receivers():
    return Receivers(receivers=[
        Receiver(0.0, 0.0, "XX", "STA1", ["Z", "E", "N"]),
        Receiver(0.5, 1.0, "XX", "STA2", ["Z", "E", "N"]),
        Receiver(1.0, 0.2, "XX", "STA3", ["Z", "E"]),
    ])


def _write_simulations(folder):
    rng = np.random.default_rng(3)
    for simulation in range(N_SIMULATIONS):
        with h5py.File(os.path.join(folder, f"sim_{simulation}.h5"), "w") as f:
            inputs = f.create_group("inputs")
            moment_tensor = inputs.create_group("moment_tensor")
            moment_tensor.attrs["m_rr"], moment_tensor.attrs["m_tt"] = rng.normal(size=2) * 1e15
            location = inputs.create_group("source_location")
            for name, value in zip(["latitude", "longitude", "depth"], rng.normal(size=3) + 10.0):
                location.attrs[name] = value
            outputs = f.create_group("outputs")
            for receiver in _receivers().iterate():
                station = outputs.create_group(receiver.station_name)
                for component in receiver.components:
                    station.create_dataset(component, data=rng.normal(size=TRACE_LENGTH))


def _noise():
    return np.random.normal(0.0, 0.1, 8 * TRACE_LENGTH)


class _AffineScaler:
    def transform(self, theta):
        return 1e-15 * theta + 0.5


def _processing(station_subsampler=None):
    return dict(
        synthetic_noise_model_sampler=_noise, data_scaler=_AffineScaler(),
        augmentation_chain=PostProcessingChain([
            AmplitudeErrorEffect(scale_range=(0.5, 1.5)),
            TimeShiftErrorEffect(sampling_rate=1.0, uniform_offset=0.0, gaussian_sigma=2.0)]),
        augmentation_nuisance_params={"amplitude_error": 1.0, "time_shift_error": 1.0},
        post_noise_augmentation_chain=PostProcessingChain([ComponentDropoutEffect()]),
        post_noise_nuisance_params={"component_dropout": 0.5},
        station_subsampler=station_subsampler)


def _datasets(folder, preloaded, conditioned=False, station_subsampler=None):
    loader = SimulationDataLoader(COMPONENTS, _receivers())
    noise_std = [0.1, 0.1, 1.0] if conditioned else None
    from_files = TorchSimulationDataset(
        loader, str(folder), PARAMETER_NAME_MAP,
        conditioning_param_map=CONDITIONING_MAP if conditioned else None,
        conditioning_noise_std=noise_std, cache_in_memory=preloaded,
        **_processing(station_subsampler))
    loaded = [from_files._load_sim(path) for path in from_files.paths]
    theta = np.stack([theta for theta, _ in loaded])
    x = np.stack([data for _, data in loaded]).astype(np.float32 if preloaded else np.float64)
    conditioning = (np.stack([from_files._load_conditioning(path) for path in from_files.paths])
                    if conditioned else None)
    from_arrays = ArraySimulationDataset(
        theta, x, _receivers(), COMPONENTS, conditioning=conditioning,
        conditioning_noise_std=noise_std, **_processing(station_subsampler))
    return from_files, from_arrays


def _draw(dataset, index):
    np.random.seed(index)
    torch.manual_seed(index)
    theta, x = dataset[index]
    return [theta] + (list(x) if isinstance(x, tuple) else [x])


def _assert_same_samples(from_files, from_arrays):
    assert len(from_files) == len(from_arrays) == N_SIMULATIONS
    for index in range(N_SIMULATIONS):
        for file_part, array_part in zip(_draw(from_files, index), _draw(from_arrays, index)):
            assert file_part.dtype == array_part.dtype
            assert torch.equal(file_part, array_part)


@pytest.mark.parametrize("preloaded", [True, False])
def test_array_samples_equal_the_file_samples(tmp_path, preloaded):
    _write_simulations(str(tmp_path))
    _assert_same_samples(*_datasets(tmp_path, preloaded))


def test_array_samples_equal_the_file_samples_with_station_subsets_and_conditioning(tmp_path):
    _write_simulations(str(tmp_path))
    from_files, from_arrays = _datasets(tmp_path, True, conditioned=True,
                                        station_subsampler=StationSubsampler((0.3, 1.0)))
    _assert_same_samples(from_files, from_arrays)
    theta, (x, coords, source_vec) = from_arrays[0]
    assert x.shape[1:] == (3, TRACE_LENGTH) and coords.shape == (x.shape[0], 2)
    assert source_vec.shape == (3,)


def test_array_dataset_rejects_x_for_another_station_count():
    x = np.zeros((2, 4, 3, TRACE_LENGTH))
    with pytest.raises(ValueError, match="3 receivers"):
        ArraySimulationDataset(np.zeros((2, 2)), x, _receivers(), COMPONENTS, _noise)


def test_loaders_split_an_array_dataset_into_padded_station_batches():
    rng = np.random.default_rng(0)
    theta, x = rng.normal(size=(10, 2)), rng.normal(size=(10, 3, 3, TRACE_LENGTH))
    dataset = ArraySimulationDataset(theta, x, _receivers(), COMPONENTS, _noise,
                                     station_subsampler=StationSubsampler((0.3, 1.0)))
    train_loader, val_loader = make_torch_dataloaders(
        dataset=dataset, train_max_index=8, train_batch_size=4, val_batch_size=2)
    assert len(train_loader.dataset) == 8 and len(val_loader.dataset) == 2
    theta_batch, context = next(iter(train_loader))
    assert theta_batch.shape == (4, 2) and context.shape[0] == 4


def test_loaders_refuse_a_dataset_together_with_a_folder(tmp_path):
    dataset = ArraySimulationDataset(np.zeros((2, 2)), np.zeros((2, 3, 3, TRACE_LENGTH)),
                                     _receivers(), COMPONENTS, _noise)
    with pytest.raises(ValueError, match="not both"):
        make_torch_dataloaders(dataset=dataset, data_folder=str(tmp_path), train_max_index=1)
