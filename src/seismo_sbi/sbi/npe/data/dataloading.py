"""Torch datasets and loaders over a folder of HDF5 simulations for NPE training.

:class:`TorchSimulationDataset` loads each clean simulation, applies the training-time nuisance
augmentation, adds a noise draw, and returns ``(theta, x)``, optionally with a station subset and
a source-conditioning vector, and :func:`make_torch_dataloaders` builds the training and
validation loaders.
"""

import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from seismo_sbi.simulators.simulation_io import SimulationDataLoader
from seismo_sbi.nuisance_effects.post_processing import apply_chain_to_array
import numpy as np

from seismo_sbi.sbi.npe.data.simulation_cache import SimulationCache
from seismo_sbi.sbi.npe.data.station_selection import (
    StationSubsampler, select_stations, variable_station_collate)


class TorchSimulationDataset(Dataset):
    def __init__(
        self,
        data_loader: SimulationDataLoader,
        data_folder: str,
        parameter_name_map: dict,
        synthetic_noise_model_sampler,
        data_scaler=None,
        augmentation_chain=None,
        augmentation_nuisance_params=None,
        glob_pattern: str = "*.h5",
        return_tensors: bool = True,
        torch_dtype=torch.float32,
        conditioning_param_map: dict = None,
        conditioning_noise_std=None,
        station_subsampler: StationSubsampler = None,
        post_noise_augmentation_chain=None,
        post_noise_nuisance_params=None,
        cache_in_memory: bool = False,
        cache_preload_workers: int = 16,
        cache_dtype: str = "float32",
        fixed_item_masks=None,
    ):
        self._set_sample_processing(
            data_loader, parameter_name_map, synthetic_noise_model_sampler, data_scaler,
            augmentation_chain, augmentation_nuisance_params, return_tensors, torch_dtype,
            conditioning_param_map, conditioning_noise_std, station_subsampler,
            post_noise_augmentation_chain, post_noise_nuisance_params, fixed_item_masks)
        self._index_simulations(data_folder, glob_pattern, cache_in_memory, cache_preload_workers,
                                cache_dtype)

    def _set_sample_processing(self, data_loader, parameter_name_map, synthetic_noise_model_sampler,
                               data_scaler, augmentation_chain, augmentation_nuisance_params,
                               return_tensors, torch_dtype, conditioning_param_map,
                               conditioning_noise_std, station_subsampler,
                               post_noise_augmentation_chain, post_noise_nuisance_params,
                               fixed_item_masks):
        """What is done to each sample: augmentation, noise, scaling, station selection and conditioning."""
        self.data_loader = data_loader

        # Per-item ``(keep_indices, zero_channels)`` in the sorted order of ``paths``; they replace
        # the random subsampling and dropout, reproducing the parent event's station availability.
        self.fixed_item_masks = fixed_item_masks
        # Same alignment: a simulation stores its true source location, but the model must be
        # conditioned on the catalogue location, so their difference is the location error.
        self.fixed_conditioning = None

        # With a subsampler, __getitem__ draws a station subset per sample and returns
        # ``(theta, (x_sub, coords_sub, source_vec))`` for ``variable_station_collate`` to pack.
        self.station_subsampler = station_subsampler
        # Master station coordinates (N, 2) in canonical receiver-iteration order — matches
        # the station axis of the (N, C, T) data array, so subsampling indexes both alike.
        self.station_coords = data_loader.receivers.get_station_locations_array()

        # ``{input_type: [attribute names]}`` pulled from each simulation's stored inputs as a
        # raw, unscaled conditioning vector, returned packed as ``concat(flatten(x), source_vec)``.
        self.conditioning_param_map = conditioning_param_map or {}

        # Per-coordinate Gaussian widths, drawn fresh per sample, emulating the catalogue
        # location error. The order must match the conditioning vector's concatenation order.
        if conditioning_noise_std is None:
            self.conditioning_noise_std = None
        else:
            self.conditioning_noise_std = torch.as_tensor(
                np.asarray(conditioning_noise_std, dtype=float), dtype=torch_dtype)

        self.parameter_name_map = parameter_name_map or {}
        self.synthetic_noise_model_sampler = synthetic_noise_model_sampler
        self.data_scaler = data_scaler
        self.return_tensors = return_tensors
        self.torch_dtype = torch_dtype

        # A chain of post-processing effects folded into the clean data on the fly, before
        # noise is added.
        self.augmentation_chain = augmentation_chain
        self.augmentation_nuisance_params = augmentation_nuisance_params or {}

        # A chain applied to the noisy data, for effects like component dropout that must zero
        # a channel exactly and so cannot run before noise is added.
        self.post_noise_augmentation_chain = post_noise_augmentation_chain
        self.post_noise_nuisance_params = post_noise_nuisance_params or {}

    def _index_simulations(self, data_folder, glob_pattern, cache_in_memory, cache_preload_workers,
                           cache_dtype):
        """Where samples come from: the sorted HDF5 files under ``data_folder``, optionally held in RAM."""
        self.paths = sorted(glob.glob(os.path.join(data_folder, glob_pattern)))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No simulations matched {glob_pattern} under {data_folder}")
        else:
            print(f"Found {len(self.paths)} simulations matching {glob_pattern} under {data_folder}")

        self.simulation_cache = SimulationCache(self.paths, self.data_loader,
                                                self.parameter_name_map, self.conditioning_param_map)
        if cache_in_memory:
            self.simulation_cache.preload(int(cache_preload_workers),
                                          dtype=np.dtype(cache_dtype).type)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        theta, D = self.simulation_cache.load(idx)
        D = self._augment_clean(D)

        if self.data_scaler is not None and theta.size > 0:
            theta = self.data_scaler.transform(theta[np.newaxis, :]).flatten()
        theta = torch.as_tensor(theta, dtype=self.torch_dtype)
        D = torch.as_tensor(D, dtype=self.torch_dtype)

        x, noise_present = self._add_noise(D)
        x = self._augment_noisy(x)

        if self.return_tensors:
            x = torch.as_tensor(x, dtype=self.torch_dtype)
            theta = torch.as_tensor(theta, dtype=self.torch_dtype)

        source_vec = self._source_vector(idx)
        item_mask = None if self.fixed_item_masks is None else self.fixed_item_masks[idx]
        return select_stations(theta, x, source_vec, noise_present, item_mask=item_mask,
                               station_subsampler=self.station_subsampler,
                               station_coords=self.station_coords, torch_dtype=self.torch_dtype)

    def _augment_clean(self, D):
        """Apply the nuisance augmentation to the clean data ``(N, C, T)``, before noise is added."""
        if self.augmentation_chain is not None and self.augmentation_chain.effects:
            D = apply_chain_to_array(
                self.augmentation_chain,
                D,
                self.data_loader.receivers,
                self.data_loader.components,
                self.augmentation_nuisance_params,
            )
        return D

    def _add_noise(self, D):
        """``(x, noise_present)``: ``D`` plus one noise draw, and the stations the noise window
        carried (``None`` unless the sampler allows incomplete windows)."""
        draw = self.synthetic_noise_model_sampler.draw()
        noise_present = None if draw.present is None else np.asarray(draw.present, dtype=bool)
        noise = np.asarray(draw.noise)
        noise_rows = self.data_loader.zero_fill_unused_components(
            noise.reshape(-1, D.shape[-1]), D.shape[-1]
        )
        noise_arr = np.asarray(noise_rows)
        x = D + torch.as_tensor(noise_arr, dtype=self.torch_dtype).reshape(*D.shape)
        return x, noise_present

    def _augment_noisy(self, x):
        """Apply the post-noise chain (component dropout) on the full station set."""
        post_chain = getattr(self, "post_noise_augmentation_chain", None)
        if post_chain is not None and post_chain.effects:
            x_aug = apply_chain_to_array(
                post_chain,
                x.numpy(),
                self.data_loader.receivers,
                self.data_loader.components,
                self.post_noise_nuisance_params,
            )
            x = torch.as_tensor(x_aug, dtype=self.torch_dtype).reshape(*x.shape)
        return x

    def _source_vector(self, idx):
        """The perturbed raw source-conditioning vector, or ``None`` without conditioning."""
        fixed_cond = getattr(self, "fixed_conditioning", None)
        if fixed_cond is not None:
            raw_cond = np.asarray(fixed_cond[idx], dtype=float)
        else:
            raw_cond = self.simulation_cache.conditioning(idx)
        if raw_cond is None:
            return None
        source_vec = torch.as_tensor(raw_cond, dtype=self.torch_dtype)
        return self._perturb_conditioning(source_vec)

    def _perturb_conditioning(self, source_vec):
        """Source-location uncertainty augmentation: add per-coordinate Gaussian noise to the
        raw conditioning vector. Fresh draw per call (⇒ training augmentation). ``getattr`` keeps
        ``__new__`` test stubs working. No-op when ``conditioning_noise_std`` is unset. The model
        then sees a noisy source (and, in relative-coords mode, noisy source-relative station
        geometry) — matching the catalogue location error present at inference."""
        std = getattr(self, "conditioning_noise_std", None)
        if std is None:
            return source_vec
        return source_vec + torch.randn_like(source_vec) * std


def _seed_worker(worker_id):
    """Seed numpy in each DataLoader worker from torch's per-worker seed and the SLURM rank, so every worker and rank draws its own augmentation sequence."""
    rank = int(os.environ.get("SLURM_PROCID", 0))
    seed = (torch.initial_seed() + worker_id + rank * 100003) % (2 ** 32)
    np.random.seed(seed)


# Convenience factory to create a torch DataLoader for a dataset (no splitting)
def make_torch_dataloader(
    data_loader: SimulationDataLoader,
    data_folder: str,
    parameter_name_map: dict,
    synthetic_noise_model_sampler,
    augmentation_chain=None,
    augmentation_nuisance_params=None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = None,
    prefetch_factor: int = 4,
    glob_pattern: str = "*.h5",
    return_tensors: bool = True,
    torch_dtype=torch.float32,
    conditioning_param_map: dict = None,
    conditioning_noise_std=None,
    station_subsampler: StationSubsampler = None,
    post_noise_augmentation_chain=None,
    post_noise_nuisance_params=None,
) -> DataLoader:
    dataset = TorchSimulationDataset(
        data_loader=data_loader,
        data_folder=data_folder,
        parameter_name_map=parameter_name_map,
        synthetic_noise_model_sampler=synthetic_noise_model_sampler,
        augmentation_chain=augmentation_chain,
        augmentation_nuisance_params=augmentation_nuisance_params,
        glob_pattern=glob_pattern,
        return_tensors=return_tensors,
        torch_dtype=torch_dtype,
        conditioning_param_map=conditioning_param_map,
        conditioning_noise_std=conditioning_noise_std,
        station_subsampler=station_subsampler,
        post_noise_augmentation_chain=post_noise_augmentation_chain,
        post_noise_nuisance_params=post_noise_nuisance_params,
    )
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    # Variable-station samples are ragged ⇒ the default collate cannot stack them.
    collate_fn = variable_station_collate if station_subsampler is not None else None
    # prefetch_factor is only valid for multiprocessing loaders (num_workers > 0).
    extra = {"prefetch_factor": prefetch_factor} if num_workers > 0 else {}
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        collate_fn=collate_fn,
        **extra,
    )

def make_torch_dataloaders(
    *,
    data_loader: SimulationDataLoader = None,
    data_folder: str = None,
    parameter_name_map: dict = None,
    synthetic_noise_model_sampler=None,
    augmentation_chain=None,
    augmentation_nuisance_params=None,
    data_scaler=None,
    train_max_index: int,
    train_batch_size: int = 32,
    val_batch_size: int  = None,
    train_shuffle: bool = True,
    val_shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = None,
    prefetch_factor: int = 4,
    glob_pattern: str = "*.h5",
    return_tensors: bool = True,
    torch_dtype= torch.float32,
    conditioning_param_map: dict = None,
    conditioning_noise_std=None,
    station_subsampler: StationSubsampler = None,
    post_noise_augmentation_chain=None,
    post_noise_nuisance_params=None,
    cache_in_memory: bool = False,
    cache_preload_workers: int = 16,
    cache_dtype: str = "float32",
    dataset: Dataset = None,
):
    """``(train_loader, val_loader)`` over one dataset, split at ``train_max_index``.

    The dataset arguments go to :class:`TorchSimulationDataset`; a ``dataset`` already built, such
    as an :class:`~seismo_sbi.sbi.npe.data.array_dataset.ArraySimulationDataset`, is split
    as it is instead. ``val_batch_size`` defaults to ``train_batch_size``.
    """
    if val_batch_size is None:
        val_batch_size = train_batch_size

    # Train and val share one dataset, so nuisance augmentation is applied to BOTH
    # (val augmentation ON by design — val loss reflects the augmented distribution).
    if dataset is not None:
        if data_loader is not None or data_folder is not None:
            raise ValueError("Pass either a dataset or the folder to build one from, not both.")
        full_dataset = dataset
        station_subsampler = dataset.station_subsampler
    else:
        full_dataset = TorchSimulationDataset(
            data_loader=data_loader,
            data_folder=data_folder,
            parameter_name_map=parameter_name_map,
            synthetic_noise_model_sampler=synthetic_noise_model_sampler,
            augmentation_chain=augmentation_chain,
            augmentation_nuisance_params=augmentation_nuisance_params,
            data_scaler=data_scaler,
            glob_pattern=glob_pattern,
            return_tensors=return_tensors,
            torch_dtype=torch_dtype,
            conditioning_param_map=conditioning_param_map,
            conditioning_noise_std=conditioning_noise_std,
            station_subsampler=station_subsampler,
            post_noise_augmentation_chain=post_noise_augmentation_chain,
            post_noise_nuisance_params=post_noise_nuisance_params,
            cache_in_memory=cache_in_memory,
            cache_preload_workers=cache_preload_workers,
            cache_dtype=cache_dtype,
        )
    train_subset, val_subset = _split_at_index(full_dataset, train_max_index)
    loader_options = _loader_options(num_workers, pin_memory, persistent_workers, prefetch_factor,
                                     station_subsampler)

    # drop_last on TRAIN only: a trailing batch of one sample makes the flow's BatchNorm raise,
    # and sharding across ranks can produce one. Validation runs in eval mode, where it is safe.
    train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=train_shuffle,
                              drop_last=True, **loader_options)
    val_loader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=val_shuffle,
                            **loader_options)
    return train_loader, val_loader


def _split_at_index(dataset, train_max_index):
    """``(train, validation)`` subsets: the samples before ``train_max_index`` and the rest."""
    n = len(dataset)
    end = max(0, min(train_max_index, n))
    return Subset(dataset, range(0, end)), Subset(dataset, range(end, n))


def _loader_options(num_workers, pin_memory, persistent_workers, prefetch_factor, station_subsampler):
    """DataLoader keyword arguments shared by the training and validation loaders."""
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    options = {"num_workers": num_workers, "pin_memory": pin_memory,
               "persistent_workers": persistent_workers}
    # prefetch_factor is only valid with worker processes; a larger value keeps more augmented
    # batches queued ahead of the accelerator.
    if num_workers > 0:
        options["prefetch_factor"] = prefetch_factor
        options["worker_init_fn"] = _seed_worker
    # Variable-station samples are ragged ⇒ pad+pack via the custom collate.
    if station_subsampler is not None:
        options["collate_fn"] = variable_station_collate
    return options
