"""NPE training samples from arrays held in memory instead of a folder of simulations.

:class:`ArraySimulationDataset` serves ``(theta, x)`` pairs the caller already holds through the
same clean-data augmentation, noise draw, parameter scaling, post-noise augmentation and station
selection as :class:`~seismo_sbi.sbi.npe.data.dataloading.TorchSimulationDataset`, so a
training run needs no HDF5 files on disk.
"""
import numpy as np
import torch

from seismo_sbi.simulators.simulation_io import SimulationDataLoader
from seismo_sbi.sbi.npe.data.dataloading import StationSubsampler, TorchSimulationDataset


class ArraySimulationDataset(TorchSimulationDataset):
    """Training samples drawn from ``theta`` and ``x`` arrays, each treated as one simulation.

    ``theta`` is ``(n_simulations, n_parameters)``, unscaled, in the units ``data_scaler``
    expects, and is held as float64 as the simulation files hold it. ``x`` is the clean data,
    ``(n_simulations, n_stations, n_components, trace_length)``: stations in the order of
    ``receivers``, components in the order of ``components`` (a string such as ``"ZEN"``), and
    zeros where a station does not record a component, which is the array
    :meth:`SimulationDataLoader.load_simulation_data_array` returns with ``stacked=True`` and
    ``fill_unused=True``. Its dtype is kept: float32 reproduces the preloaded file path, float64
    the per-file read. ``conditioning`` is ``(n_simulations, n_conditioning)``, the raw source
    vector a conditioned model receives, perturbed by ``conditioning_noise_std`` per draw.
    The other arguments are those of :class:`TorchSimulationDataset`.
    """

    def __init__(
        self,
        theta,
        x,
        receivers,
        components,
        synthetic_noise_model_sampler,
        conditioning=None,
        data_scaler=None,
        augmentation_chain=None,
        augmentation_nuisance_params=None,
        return_tensors: bool = True,
        torch_dtype=torch.float32,
        conditioning_noise_std=None,
        station_subsampler: StationSubsampler = None,
        post_noise_augmentation_chain=None,
        post_noise_nuisance_params=None,
    ):
        data_loader = SimulationDataLoader(components, receivers)
        self._set_sample_processing(
            data_loader, {}, synthetic_noise_model_sampler, data_scaler,
            augmentation_chain, augmentation_nuisance_params, return_tensors, torch_dtype,
            {}, conditioning_noise_std, station_subsampler,
            post_noise_augmentation_chain, post_noise_nuisance_params, None)
        self._hold_arrays(theta, x, conditioning, len(receivers.receivers))

    def _hold_arrays(self, theta, x, conditioning, n_stations):
        """Keep the arrays as the in-memory cache the per-sample steps read."""
        x = np.asarray(x)
        if x.ndim != 4 or x.shape[1] != n_stations:
            raise ValueError(f"x must be (n_simulations, {n_stations}, n_components, "
                             f"trace_length) for {n_stations} receivers, got {x.shape}")
        self.paths = [f"array_{index}" for index in range(x.shape[0])]
        self._cache_D = x
        self._cache_theta = None
        if theta is not None and np.size(theta):
            self._cache_theta = np.asarray(theta, dtype=np.float64).reshape(x.shape[0], -1)
        self._cache_cond = None
        if conditioning is not None:
            self._cache_cond = np.asarray(conditioning, dtype=np.float64).reshape(x.shape[0], -1)

    def _source_vector(self, idx, sim_path):
        """The perturbed conditioning row ``idx``, or ``None`` without conditioning."""
        if self._cache_cond is None:
            return None
        source_vec = torch.as_tensor(self._cache_cond[idx], dtype=self.torch_dtype)
        return self._perturb_conditioning(source_vec)
