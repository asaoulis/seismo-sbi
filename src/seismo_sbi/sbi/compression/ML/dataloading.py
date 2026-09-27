"""Torch datasets and loaders over a folder of HDF5 simulations for NPE training.

:class:`TorchSimulationDataset` loads each clean simulation, applies the training-time nuisance
augmentation, adds a noise draw, and returns ``(theta, x)``, optionally with a station subset and
a source-conditioning vector. :class:`StationSubsampler` draws the subsets,
:func:`variable_station_collate` pads them into a batch, and :func:`make_torch_dataloaders`
builds the training and validation loaders.
"""

import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from seismo_sbi.simulators.simulation_io import SimulationDataLoader
from seismo_sbi.nuisance_effects.post_processing import apply_chain_to_array
import numpy as np

from .source_conditioning import pack_variable_context


class StationSubsampler:
    """Randomly select a subset of station indices to emulate variable station configs.

    Each draw samples a *keep fraction* from a configurable distribution and keeps that
    fraction of the master station set (with a ``min_stations`` floor), returning the
    sorted kept indices so the canonical station ordering is preserved.  Uses the global
    numpy RNG, which the DataLoader's ``_seed_worker`` re-seeds per worker for
    reproducible, per-worker-distinct augmentation.

    Parameters
    ----------
    keep_fraction:
        A fixed fraction (``float``) or a ``(low, high)`` range drawn uniformly per sample.
        Default ``(0.5, 1.0)``.
    min_stations:
        Lower bound on the number of kept stations (also clamped to the available count).
    """

    def __init__(self, keep_fraction=(0.5, 1.0), min_stations: int = 1):
        self.keep_fraction = keep_fraction if keep_fraction is not None else (0.5, 1.0)
        self.min_stations = int(min_stations)

    def _draw_fraction(self) -> float:
        kf = self.keep_fraction
        if isinstance(kf, (int, float)):
            return float(kf)
        return float(np.random.uniform(*kf))

    def __call__(self, num_stations: int, available: np.ndarray = None) -> np.ndarray:
        """Draw kept station indices.

        available:
            Optional boolean mask over the master station axis. When given, the draw is
            restricted to stations flagged available -- used with an incomplete real-noise
            window so a station without noise is never kept (it would otherwise enter the
            model as exactly-zero data). The keep FRACTION is still drawn from the
            configured distribution and applied to the available count, so the dropout
            distribution is unchanged in shape; only its support shrinks.
        """
        if available is None:
            pool = np.arange(num_stations)
        else:
            pool = np.flatnonzero(np.asarray(available, dtype=bool)[:num_stations])
            if pool.size == 0:
                raise ValueError("StationSubsampler: no stations available for this sample")
        n_keep = int(round(self._draw_fraction() * pool.size))
        n_keep = min(pool.size, max(min(self.min_stations, pool.size), n_keep))
        return np.sort(np.random.choice(pool, size=n_keep, replace=False))


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
        station_subsampler: "StationSubsampler" = None,
        post_noise_augmentation_chain=None,
        post_noise_nuisance_params=None,
        cache_in_memory: bool = False,
        cache_preload_workers: int = 16,
        cache_dtype: str = "float32",
        fixed_item_masks=None,
    ):
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

        self.paths = sorted(glob.glob(os.path.join(data_folder, glob_pattern)))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No simulations matched {glob_pattern} under {data_folder}")
        else:
            print(f"Found {len(self.paths)} simulations matching {glob_pattern} under {data_folder}")

        # Per-sample HDF5 reads are shared-file reads that do not scale with worker count, so
        # every clean array is preloaded into one buffer the workers share copy-on-write.
        self._cache_D = None
        self._cache_theta = None
        self._cache_cond = None
        if cache_in_memory:
            self._preload_cache(int(cache_preload_workers), dtype=np.dtype(cache_dtype).type)

    def __len__(self):
        return len(self.paths)

    def _preload_cache(self, max_workers: int = 16, dtype=np.float32):
        """Preload every sim's clean (theta, D[, conditioning]) into contiguous RAM buffers.

        Runs ONCE in the main process (before the DataLoader forks its workers) so the big
        ``_cache_D`` buffer is shared copy-on-write. Uses a thread pool because h5py releases the
        GIL on reads, so the per-file open/read I/O overlaps across threads (the on-disk path's
        scaling wall is exactly this serialised read). The cached arrays reproduce ``_load_sim`` /
        ``_load_conditioning`` exactly, so __getitem__ stays behaviour-identical.
        """
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        import time as _t
        n = len(self.paths)
        theta0, D0 = self._load_sim(self.paths[0])
        self._cache_D = np.empty((n,) + tuple(D0.shape), dtype=dtype)
        self._cache_theta = (
            np.empty((n, theta0.shape[0]), dtype=np.float64) if theta0.size else None
        )
        self._cache_cond = None
        if self.conditioning_param_map:
            cond0 = self._load_conditioning(self.paths[0])
            self._cache_cond = np.empty((n, cond0.shape[0]), dtype=np.float64)

        def _load_one(i):
            th, D = self._load_sim(self.paths[i])
            self._cache_D[i] = D
            if self._cache_theta is not None:
                self._cache_theta[i] = th
            if self._cache_cond is not None:
                self._cache_cond[i] = self._load_conditioning(self.paths[i])

        t0 = _t.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            # A progress bar over the in-order map, so a preload that runs for many minutes is
            # not silent, on a log file as well as a terminal.
            for _ in tqdm(ex.map(_load_one, range(n)), total=n,
                          desc="[sim-cache] preloading", unit="sim"):
                pass
        gb = self._cache_D.nbytes / 1e9
        print(f"[sim-cache] preloaded {n} sims into RAM ({gb:.2f} GB, {dtype.__name__}) "
              f"in {_t.perf_counter() - t0:.1f}s — per-sample HDF5 load removed.")

    def __getitem__(self, idx):
        sim_path = self.paths[idx]
        theta, D = self._load_clean(idx, sim_path)
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

        source_vec = self._source_vector(idx, sim_path)
        return self._select_stations(idx, theta, x, source_vec, noise_present)

    def _load_clean(self, idx, sim_path):
        """``(theta (D,), D (N, C, T))`` for one simulation, from the RAM cache when preloaded."""
        # getattr keeps datasets built via __new__ (test stubs) working without this attr.
        cache_D = getattr(self, "_cache_D", None)
        if cache_D is not None:
            # Copy out of the shared buffer so the aug/noise below never mutate the cache.
            D = np.array(cache_D[idx])
            theta = (np.array(self._cache_theta[idx]) if self._cache_theta is not None
                     else np.array([]))
        else:
            theta, D = self._load_sim(sim_path)
        return theta, D

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
        noise = self.synthetic_noise_model_sampler()
        noise_present = None
        if isinstance(noise, tuple):
            noise, second = noise
            if getattr(self.synthetic_noise_model_sampler, "allow_incomplete", False):
                noise_present = np.asarray(second, dtype=bool)
        noise = np.asarray(noise)
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

    def _source_vector(self, idx, sim_path):
        """The perturbed raw source-conditioning vector, or ``None`` without conditioning."""
        source_vec = None
        if self.conditioning_param_map:
            fixed_cond = getattr(self, "fixed_conditioning", None)
            if fixed_cond is not None:
                raw_cond = np.asarray(fixed_cond[idx], dtype=float)
            else:
                cache_cond = getattr(self, "_cache_cond", None)
                raw_cond = (cache_cond[idx] if cache_cond is not None
                            else self._load_conditioning(sim_path))
            source_vec = torch.as_tensor(raw_cond, dtype=self.torch_dtype)
            source_vec = self._perturb_conditioning(source_vec)
        return source_vec

    def _select_stations(self, idx, theta, x, source_vec, noise_present):
        """The returned item: ``(theta, (x_sub, coords_sub, source_vec))`` with a fixed mask or a
        station subsampler, else ``(theta, x)`` with the source vector packed into ``x``."""
        # getattr keeps datasets built via __new__ (test stubs) working without this attr.
        fixed_masks = getattr(self, "fixed_item_masks", None)
        if fixed_masks is not None:
            keep, zero_channels = fixed_masks[idx]
            for (si, ci) in (zero_channels or ()):
                x[si, ci, :] = 0.0
            keep = np.asarray(keep, dtype=int)
            x_sub = x[keep]
            coords_sub = torch.as_tensor(
                self.station_coords[keep], dtype=self.torch_dtype)
            return theta, (x_sub, coords_sub, source_vec)

        station_subsampler = getattr(self, "station_subsampler", None)
        if station_subsampler is not None:
            num_stations = x.shape[0]
            keep = station_subsampler(num_stations, available=noise_present)
            x_sub = x[keep]                                            # (N_sub, C, T)
            coords_sub = torch.as_tensor(
                self.station_coords[keep], dtype=self.torch_dtype
            )                                                          # (N_sub, 2)
            return theta, (x_sub, coords_sub, source_vec)

        if noise_present is not None and not noise_present.all():
            raise RuntimeError(
                "RealNoiseSampler(allow_incomplete=True) produced a window missing "
                f"{int((~noise_present).sum())} station(s), but this dataset has no "
                "station_subsampler to mask them out -- they would enter the model as "
                "exactly-zero data. Use incomplete noise windows only with "
                "variable-station training."
            )

        if source_vec is not None:
            from .source_conditioning import pack_context
            x = pack_context(x, source_vec)
        return theta, x

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

    def _load_conditioning(self, sim_path):
        """Extract the raw (unscaled) source-conditioning vector from a sim's stored inputs."""
        inputs_dict = self.data_loader.load_input_data(sim_path)
        return np.concatenate([
            [inputs_dict[input_type][attr] for attr in attrs]
            for input_type, attrs in self.conditioning_param_map.items()
        ]).astype(float)

    def _load_sim(self, sim_path):
        # The clean, unshifted array: time shifts and the other effects are applied as
        # augmentation in ``__getitem__``, not baked into the load.
        if len(self.parameter_name_map) > 0:
            # Single h5 open for BOTH theta (inputs) and the data array (outputs);
            # opening the file twice per sample is a measurable per-epoch cost.
            inputs_dict, D = self.data_loader.load_input_and_data_array(
                sim_path, stacked=True, fill_unused=True
            )
            fixed_keys = dict(
                (param_type, param_names)
                if param_names != ["earthquake_magnitude"]
                else ("moment_tensor", ["earthquake_magnitude"])
                for param_type, param_names in self.parameter_name_map.items()
            )
            theta = np.concatenate(
                [
                    [inputs_dict[param_type][param_name] for param_name in param_names]
                    for param_type, param_names in fixed_keys.items()
                ]
            )
        else:
            theta = np.array([])
            D = self.data_loader.load_simulation_data_array(sim_path, stacked=True, fill_unused=True)
        return theta, D


def variable_station_collate(batch):
    """Collate variable-station samples into a ragged-aware batch.

    Each item is ``(theta (D,), (x (N_i,C,T), coords (N_i,2), source_vec|None))``.  Pads
    every sample to the batch's ``max_N`` with zeros, builds a boolean validity mask
    ``(B, max_N)`` (True=real station), and packs each into the single flat context tensor
    the embedding net unpacks. Returns ``(theta (B,D), context (B,W))``.
    """
    thetas, samples = zip(*batch)
    xs, coords, source_vecs = zip(*samples)

    max_N = max(x.shape[0] for x in xs)
    C, T = xs[0].shape[1], xs[0].shape[2]
    dtype = xs[0].dtype
    has_source = source_vecs[0] is not None

    packed = []
    for x, crd, sv in zip(xs, coords, source_vecs):
        n = x.shape[0]
        x_pad = x.new_zeros((max_N, C, T)); x_pad[:n] = x
        crd_pad = crd.new_zeros((max_N, 2)); crd_pad[:n] = crd
        mask = torch.zeros(max_N, dtype=torch.bool); mask[:n] = True
        packed.append(pack_variable_context(x_pad, crd_pad, mask, sv if has_source else None))

    # packed vectors are already in xs[0].dtype (built from x.new_zeros / cast in
    # pack_variable_context), so no further dtype cast is needed here.
    context = torch.stack(packed, dim=0)
    theta = torch.stack([torch.as_tensor(t, dtype=dtype) for t in thetas], dim=0)
    return theta, context


def _seed_worker(worker_id):
    """Seed numpy per DataLoader worker so stochastic augmentation is reproducible.

    Each worker process inherits the same numpy global RNG state on fork; without
    re-seeding, all workers would draw the SAME augmentation sequence. Derive a
    distinct seed per worker from torch's per-worker initial seed.

    Under multi-GPU DDP (one srun rank per GPU) each rank draws its own noise and
    augmentation; folding in ``SLURM_PROCID`` guarantees the per-rank streams are
    provably independent (so two ranks never corrupt the same synthetic with the
    identical noise realisation) and robust to any future ``seed_everything``.
    ``SLURM_PROCID`` is unset off-cluster ⇒ rank 0 ⇒ byte-identical to before.
    """
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
    station_subsampler: "StationSubsampler" = None,
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
    data_loader: SimulationDataLoader,
    data_folder: str,
    parameter_name_map: dict,
    synthetic_noise_model_sampler,
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
    station_subsampler: "StationSubsampler" = None,
    post_noise_augmentation_chain=None,
    post_noise_nuisance_params=None,
    cache_in_memory: bool = False,
    cache_preload_workers: int = 16,
    cache_dtype: str = "float32",
):
    """``(train_loader, val_loader)`` over one dataset, split at ``train_max_index``.

    The dataset arguments go to :class:`TorchSimulationDataset`; ``val_batch_size`` defaults to
    ``train_batch_size``.
    """
    if val_batch_size is None:
        val_batch_size = train_batch_size

    # Train and val share one dataset, so nuisance augmentation is applied to BOTH
    # (val augmentation ON by design — val loss reflects the augmented distribution).
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
