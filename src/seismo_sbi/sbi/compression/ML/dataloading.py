# New imports for PyTorch dataset/dataloader and globbing
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader
from seismo_sbi.instaseis_simulator.post_processing import apply_chain_to_array
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

    def __call__(self, num_stations: int) -> np.ndarray:
        n_keep = int(round(self._draw_fraction() * num_stations))
        n_keep = min(num_stations, max(min(self.min_stations, num_stations), n_keep))
        return np.sort(np.random.choice(num_stations, size=n_keep, replace=False))

# New: Torch dataset that returns (theta, x) where x = D + noise
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
    ):
        self.data_loader = data_loader

        # Variable-station augmentation: when a subsampler is supplied, __getitem__ draws a
        # (possibly partial) subset of stations per sample and returns a tuple
        # (theta, (x_sub, coords_sub, source_vec|None)) that variable_station_collate packs
        # into a ragged-aware batch. Absent ⇒ the legacy fixed-N return path is unchanged.
        self.station_subsampler = station_subsampler
        # Master station coordinates (N, 2) in canonical receiver-iteration order — matches
        # the station axis of the (N, C, T) data array, so subsampling indexes both alike.
        self.station_coords = data_loader.receivers.get_station_locations_array()

        # Optional source-location conditioning: map of {input_type: [attr_names]} extracted
        # from each sim's stored inputs as a RAW (unscaled) conditioning vector. When set,
        # __getitem__ returns a packed context concat(flatten(x), source_vec); the embedding
        # net unpacks it. Absent ⇒ unchanged (N, C, T) data return.
        self.conditioning_param_map = conditioning_param_map or {}

        # Source-location UNCERTAINTY augmentation (v3): per-coordinate Gaussian std applied to
        # the raw conditioning vector on each __getitem__ (fresh draw ⇒ training augmentation).
        # Order MUST match the conditioning vector (param_map concatenation order). Emulates the
        # catalogue location error the conditioned model sees at inference. None ⇒ no perturbation.
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

        # Training-time nuisance augmentation: a PostProcessingChain of Category-2
        # effects (amplitude/dropout/time-shift/coda) folded into the CLEAN loaded
        # data on the fly, BEFORE noise is added. None / empty chain ⇒ no augmentation
        # (back-compat with the retired `random_shift_distribution` path at (0,0)).
        self.augmentation_chain = augmentation_chain
        self.augmentation_nuisance_params = augmentation_nuisance_params or {}

        # Post-noise augmentation: a PostProcessingChain applied to the NOISY data
        # ``x = D + noise`` (e.g. component_dropout, which zeros channels to mimic
        # genuinely-absent components and so must run after noise to be exactly zero).
        # None / empty chain ⇒ no post-noise augmentation.
        self.post_noise_augmentation_chain = post_noise_augmentation_chain
        self.post_noise_nuisance_params = post_noise_nuisance_params or {}

        self.paths = sorted(glob.glob(os.path.join(data_folder, glob_pattern)))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No simulations matched {glob_pattern} under {data_folder}")
        else:
            print(f"Found {len(self.paths)} simulations matching {glob_pattern} under {data_folder}")

        # --- Optional in-RAM sim cache (opt-in) ---
        # The per-sample HDF5 open/read (`_load_sim`, plus a SECOND open for `_load_conditioning`
        # on the conditioned path) is ~13 ms/sample and — being a shared-file read — does NOT scale
        # with more DataLoader workers (it plateaus the loader throughput). Once the model step is
        # fast (AMP/SDPA), this load dominates and starves the GPU. Preloading every clean array into
        # ONE contiguous numpy buffer in the MAIN process (before the workers fork) removes the load
        # entirely: __getitem__ indexes RAM, and the workers share the buffer copy-on-write (a single
        # large array's data pages aren't touched by Python refcounting, so no per-worker duplication).
        # The augmentation + noise + scaling still run per __getitem__ on a COPY, so behaviour is
        # byte-identical to the on-disk path (verified by checksum in scripts/bench_aug_dataloader.py).
        # cache_dtype trades RAM for storage precision. The model trains in float32, so
        # "float32" (default) is identical to the on-disk path at the model's input precision
        # (measured rel|Δ| 8e-8 < float32 ULP) at HALF the RAM of the native float64; use
        # "float64" for a byte-identical cache, or "float16" to halve RAM again on big (1M+)
        # datasets where the extra rounding is acceptable.
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
            list(ex.map(_load_one, range(n)))
        gb = self._cache_D.nbytes / 1e9
        print(f"[sim-cache] preloaded {n} sims into RAM ({gb:.2f} GB, {dtype.__name__}) "
              f"in {_t.perf_counter() - t0:.1f}s — per-sample HDF5 load removed.")

    def __getitem__(self, idx):
        # Load per-sample data — from the in-RAM cache when preloaded, else on demand from HDF5.
        sim_path = self.paths[idx]
        # getattr keeps datasets built via __new__ (test stubs) working without this attr.
        cache_D = getattr(self, "_cache_D", None)
        if cache_D is not None:
            # Copy out of the shared buffer so the aug/noise below never mutate the cache.
            D = np.array(cache_D[idx])
            theta = (np.array(self._cache_theta[idx]) if self._cache_theta is not None
                     else np.array([]))
        else:
            theta, D = self._load_sim(sim_path)

        # Fold in nuisance augmentation on the CLEAN data, before noise is added
        # (physical semantics: amplitude/dropout/shift act on signal, then noise).
        if self.augmentation_chain is not None and self.augmentation_chain.effects:
            D = apply_chain_to_array(
                self.augmentation_chain,
                D,
                self.data_loader.receivers,
                self.data_loader.components,
                self.augmentation_nuisance_params,
            )

        # Apply scaler to parameters if provided (consistent with previous behavior)
        if self.data_scaler is not None and theta.size > 0:
            theta = self.data_scaler.transform(theta[np.newaxis, :]).flatten()

        # Ensure types match prior behavior: use torch tensors for computations
        theta = torch.as_tensor(theta, dtype=self.torch_dtype)
        D = torch.as_tensor(D, dtype=self.torch_dtype)

        # Add synthetic noise on-the-fly. Keep the sampled noise as numpy through
        # zero-fill so the (N, C, T) block builds in one `np.asarray` (no per-row
        # conversion) and a single tensor cast — avoiding the old numpy->torch->numpy
        # ->torch round-trip. Same noise values and station/component placement.
        noise = self.synthetic_noise_model_sampler()
        if isinstance(noise, tuple):
            noise, _ = noise
        noise = np.asarray(noise)
        noise_rows = self.data_loader.zero_fill_unused_components(
            noise.reshape(-1, D.shape[-1]), D.shape[-1]
        )
        noise_arr = np.asarray(noise_rows)
        x = D + torch.as_tensor(noise_arr, dtype=self.torch_dtype).reshape(*D.shape)

        # Post-noise augmentation (e.g. component_dropout): applied to the NOISY x so a
        # dropped channel is EXACTLY zero (matching a genuinely-absent channel). Applied on the
        # full station set BEFORE any variable-station subsampling, keeping the array aligned
        # with the full `receivers` the adapter expects.
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

        if self.return_tensors:
            # x is already torch; ensure dtype
            x = torch.as_tensor(x, dtype=self.torch_dtype)
            theta = torch.as_tensor(theta, dtype=self.torch_dtype)

        # Optional raw source-conditioning vector (shared by both return paths).
        source_vec = None
        if self.conditioning_param_map:
            cache_cond = getattr(self, "_cache_cond", None)
            raw_cond = (cache_cond[idx] if cache_cond is not None
                        else self._load_conditioning(sim_path))
            source_vec = torch.as_tensor(raw_cond, dtype=self.torch_dtype)
            source_vec = self._perturb_conditioning(source_vec)

        # --- Variable-station path: subsample stations, carry per-sample coords ---
        # getattr keeps datasets built via __new__ (test stubs) working without this attr.
        station_subsampler = getattr(self, "station_subsampler", None)
        if station_subsampler is not None:
            num_stations = x.shape[0]
            keep = station_subsampler(num_stations)
            x_sub = x[keep]                                            # (N_sub, C, T)
            coords_sub = torch.as_tensor(
                self.station_coords[keep], dtype=self.torch_dtype
            )                                                          # (N_sub, 2)
            return theta, (x_sub, coords_sub, source_vec)

        # --- Legacy fixed-N path (unchanged) ---
        # Source-location conditioning: append the RAW conditioning vector → packed context.
        if source_vec is not None:
            from .source_conditioning import pack_context
            x = pack_context(x, source_vec)
        return theta, x

    def _perturb_conditioning(self, source_vec):
        """Source-location UNCERTAINTY augmentation (v3): add per-coordinate Gaussian noise to the
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
        # Load the CLEAN, un-shifted data array. Time shifts (and other nuisance
        # effects) are now applied as augmentation in __getitem__ via the
        # augmentation_chain, not baked into the load.
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
    """
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
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

# Build both train and val DataLoaders using a single split parameter train_max_index
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
    n = len(full_dataset)
    end = max(0, min(train_max_index, n))

    train_subset = Subset(full_dataset, range(0, end))
    val_subset = Subset(full_dataset, range(end, n))

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    # prefetch_factor is only valid for multiprocessing loaders (num_workers > 0);
    # passing it with num_workers=0 raises in torch >= 2.0. A larger prefetch keeps
    # more augmented batches queued ahead of the GPU so per-sample CPU augmentation
    # is hidden behind compute.
    extra = {"prefetch_factor": prefetch_factor} if num_workers > 0 else {}
    if num_workers > 0:
        extra["worker_init_fn"] = _seed_worker
    # Variable-station samples are ragged ⇒ pad+pack via the custom collate.
    if station_subsampler is not None:
        extra["collate_fn"] = variable_station_collate

    train_loader = DataLoader(
        train_subset,
        batch_size=train_batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        **extra,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=val_batch_size,
        shuffle=val_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        **extra,
    )
    return train_loader, val_loader