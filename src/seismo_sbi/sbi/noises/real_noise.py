from pathlib import Path
import numpy as np

from seismo_sbi.sbi.configuration import SimulationParameters
from seismo_sbi.sbi.noises.covariance_base import station_component_value
from seismo_sbi.simulators.simulation_io import SimulationDataLoader, component_alias


class RealNoiseSampler:

    # TODO: add components implementation

    def __init__(self, simulation_parameters : SimulationParameters, directory, data_length = None, adaptive_covariance= None,
                 freeze_scale: bool = False, allow_incomplete: bool = False):
        """
        allow_incomplete:
            When False (default, unchanged behaviour) a noise window missing ANY model
            station is skipped entirely, so the usable pool is
            ``n_windows * P(all stations present)`` -- which collapses as the station count
            grows (measured: 77.9% at 29 Iceland stations, far worse at 49).

            When True the window is used for the stations it DOES have: absent stations are
            zero-filled and ``__call__`` returns ``(noise_vector, present_mask)``. The
            consumer must mask those stations out -- in practice by intersecting the mask
            with the variable-station keep-set (see ``StationSubsampler(available=...)``),
            which is why this mode is only meaningful with variable-station training.

            Whole windows are still drawn intact, so inter-station noise coherence -- real
            at microseism periods -- is preserved. Assembling one sample's noise from
            several windows would destroy it and make the noise artificially
            uncorrelated across stations, biasing the posterior towards overconfidence.
        """
        self.allow_incomplete = bool(allow_incomplete)
        receivers = simulation_parameters.receivers
        self.num_stations = len(receivers.receivers)
        self.components = simulation_parameters.components
        self.components = component_alias(self.components)
        self.vector_length = round(simulation_parameters.seismogram_duration * simulation_parameters.sampling_rate)

        self.data_loader = SimulationDataLoader(self.components, simulation_parameters.receivers, data_length)

        # Flattened length of a complete noise window. A window sitting on a station data gap
        # comes back short, which would not broadcast against the data vector, so it is skipped.
        self._expected_length = (
            None if data_length is None
            else sum(len(rec.components) for rec in receivers.receivers) * data_length
        )

        self.noise_paths = self._find_noise_paths(directory)
        np.random.shuffle(self.noise_paths)

        # Without the cache every call opens an HDF5 file. Drawing uniformly with replacement
        # and never rescaling, a contiguous in-RAM pool is distributionally the same draw.
        self._noise_cache = None
        self._presence_cache = None
        self._presence_bits = None

        # When True the sampler draws windows verbatim and never rescales them to one event's
        # pre-event variance, which is what training amortised over events needs.
        self.freeze_scale = freeze_scale
        self.adaptive_covariance = adaptive_covariance
        if self.adaptive_covariance is not None:
            for receiver in self.adaptive_covariance.keys():
                for component in self.adaptive_covariance[receiver].keys():
                    self.adaptive_covariance[receiver][component] = self.adaptive_covariance[receiver][component][0]

        print(f"Found {len(self.noise_paths)} noise realisations.")
    
    def _build_presence_index(self):
        """Pack each cached window's station presence into one integer for O(n) subset queries.

        With <= 64 stations a window's presence is a single ``uint64``, so "which windows
        contain all of subset S" is one vectorised AND + compare over the whole pool
        (~40k windows => tens of microseconds), instead of a per-window Python loop or a
        rejection sampler that retries until it happens to hit a matching window.

        Only needed by :meth:`sample_containing`; the ordinary draw needs no index at all.
        """
        n = self._presence_cache.shape[1]
        if n > 64:
            # Fall back to the boolean matrix; the query below stays vectorised, just wider.
            self._presence_bits = None
            return
        weights = (np.uint64(1) << np.arange(n, dtype=np.uint64))
        self._presence_bits = (self._presence_cache.astype(np.uint64) * weights).sum(
            axis=1, dtype=np.uint64)

    def subset_window_count(self, station_indices):
        """How many cached windows contain every station in ``station_indices``."""
        return int(self._subset_matches(station_indices).size)

    def _subset_matches(self, station_indices):
        idx = np.asarray(station_indices, dtype=int)
        if getattr(self, "_presence_bits", None) is not None:
            want = np.uint64(0)
            for i in idx:
                want |= np.uint64(1) << np.uint64(int(i))
            return np.flatnonzero((self._presence_bits & want) == want)
        return np.flatnonzero(self._presence_cache[:, idx].all(axis=1))

    def sample_containing(self, station_indices, rng=None):
        """Draw a random cached window that contains ALL of ``station_indices``.

        For the case where the station subset is fixed in advance (inference on a real
        event, or a fixed evaluation config) and the window must be found to match it —
        the inverse of the ordinary draw. Uses the packed index, so it never rejection-
        samples. Raises if no window contains the subset.
        """
        matches = self._subset_matches(station_indices)
        if matches.size == 0:
            raise ValueError(
                f"No noise window contains all {len(station_indices)} requested stations.")
        r = (rng or np.random).randint(0, matches.size) if rng is None else rng.integers(matches.size)
        return self._noise_cache[matches[r]]

    def _find_noise_paths(self, directory):
        return np.array(list(Path(directory).glob('*.h5')))

    def preload_cache(self, max_workers: int = 16, dtype=np.float32):
        """Preload every VALID noise window into one contiguous RAM buffer (opt-in, generic mode).

        Loads each ``noise_paths`` file once (in parallel — h5py reads release the GIL), keeps the
        windows whose flattened length matches the data-vector length (skipping the
        missing-station / data-gap windows the on-disk ``__call__`` skips too), and stacks them into
        ``self._noise_cache`` ``(n_valid, L)``. Built in the MAIN process before the DataLoader forks
        workers, so the buffer is shared copy-on-write. ``__call__`` then returns a uniformly random
        row instead of opening a file. Only the generic (no-rescale, ``adaptive_covariance is None``)
        path uses the cache; the adaptive / ``no_rescale`` paths still read from disk.
        """
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        import time as _t
        paths = list(self.noise_paths)

        def _try_load(p):
            if self.allow_incomplete:
                # Absent stations are zero-filled, so every window has the canonical
                # length and NONE are discarded. The presence mask travels with the row.
                v, present = self.data_loader.load_flattened_simulation_vector_with_presence(p)
                v = np.asarray(v).reshape(-1)
                if not present.any():
                    return None
                if self._expected_length is not None and v.size != self._expected_length:
                    return None
                return v, present
            try:
                v = np.asarray(self._load_noise_file(p)).reshape(-1)
            except KeyError:
                return None
            if self._expected_length is not None and v.size != self._expected_length:
                return None
            return v, None

        t0 = _t.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            loaded = list(tqdm(ex.map(_try_load, paths), total=len(paths),
                               desc="[noise-cache] preloading", unit="win"))
        # Keep windows matching the modal length (the data-vector length); drop gaps/mismatches.
        kept = [vp for vp in loaded if vp is not None]
        if not kept:
            raise RuntimeError("RealNoiseSampler.preload_cache: no valid noise windows found.")
        ref_len = self._expected_length or kept[0][0].size
        kept = [vp for vp in kept if vp[0].size == ref_len]
        valid = [v for v, _ in kept]
        if self.allow_incomplete:
            self._presence_cache = np.ascontiguousarray(
                np.stack([m for _, m in kept], axis=0), dtype=bool)
            self._build_presence_index()
        self._noise_cache = np.ascontiguousarray(np.stack(valid, axis=0), dtype=dtype)
        gb = self._noise_cache.nbytes / 1e9
        print(f"[noise-cache] preloaded {len(valid)}/{len(paths)} noise windows into RAM "
              f"({gb:.2f} GB, {np.dtype(dtype).name}) in {_t.perf_counter() - t0:.1f}s — "
              f"per-sample HDF5 noise read removed.")

    def __call__(self, noise_path = None, no_rescale = False, noise_index = None, _attempts = 0):

        # Fast path: in-RAM pool for the generic ML draw (random window, no rescale). Returns a
        # uniformly random cached window — distributionally identical to the on-disk random draw.
        if (self._noise_cache is not None and not no_rescale
                and self.adaptive_covariance is None
                and noise_path is None and noise_index is None):
            row = np.random.randint(0, self._noise_cache.shape[0])
            if self.allow_incomplete:
                # O(1). No search and no rejection loop: the window is drawn first and the
                # station subset is derived from what it holds (see __init__ docstring).
                return self._noise_cache[row], self._presence_cache[row]
            return self._noise_cache[row]

        if _attempts > len(self.noise_paths):
            raise RuntimeError(
                f"RealNoiseSampler: no noise window matched the expected data-vector length "
                f"{self._expected_length} after scanning all {len(self.noise_paths)} windows.")
        if noise_index is not None:
            # Wrap: the retry below increments noise_index, which would otherwise run off the
            # end at the last window.
            noise_path = self.noise_paths[noise_index % len(self.noise_paths)]
        if noise_path is None:
            noise_index = np.random.randint(0, len(self.noise_paths))
            noise_path = self.noise_paths[noise_index]
        # A retry moves to the next window when walking sequentially, or draws afresh when the
        # window was requested by path.
        next_index = None if noise_index is None else noise_index + 1
        if self.allow_incomplete:
            noise_realisations, present = self.data_loader.load_flattened_simulation_vector_with_presence(
                noise_path)
            if not present.any():
                # Degenerate window (no model station at all): fall through to the next one.
                return self.__call__(noise_path=None, no_rescale=no_rescale,
                                     noise_index=next_index, _attempts=_attempts + 1)
            return noise_realisations, present
        try:
            noise_realisations = self._load_noise_file(noise_path)
        except KeyError:
            # this window lacks one of the event's stations; try the next one.
            return self.__call__(noise_path = None, no_rescale = no_rescale,
                                 noise_index=next_index, _attempts=_attempts + 1)
        # Skip windows on a station data gap: one trace is short, so the flattened vector
        # would not broadcast against the data vector.
        if self._expected_length is not None and noise_realisations.size != self._expected_length:
            return self.__call__(noise_path = None, no_rescale = no_rescale,
                                 noise_index=next_index, _attempts=_attempts + 1)
        # self.noise_index_counter += 1

        if no_rescale:
            misc_data = self.data_loader.load_misc_data(noise_path)
            noise_realisations = self._load_noise_file(noise_path)
            return noise_realisations, misc_data
        elif self.adaptive_covariance is  None:
            noise_realisations = self._load_noise_file(noise_path)
            return noise_realisations
        else:
            misc_data = self.data_loader.load_misc_data(noise_path)
            scales = self.calculate_scales(misc_data)
            noise_realisations = self._load_noise_file(noise_path, scale_dict = scales)
            return noise_realisations, misc_data


    
    def _load_noise_file(self, path : Path, *args, **kwargs):
        return self.data_loader.load_flattened_simulation_vector(path, *args, **kwargs)
    
    def calculate_scales(self, misc_data):
        scales = {}
        for receiver in misc_data.keys():
            scales[receiver] = {}
            for component in misc_data[receiver].keys():
                noise_instance_variance = misc_data[receiver][component] if misc_data[receiver][component].size == 1 else misc_data[receiver][component][0]
                data_noise_variance = station_component_value(self.adaptive_covariance, receiver, component)
  

                ratio = noise_instance_variance / data_noise_variance
                scales[receiver][component] = ratio

        return scales
    
    def set_adaptive_covariance_with_misc_data(self, misc_data):
        if self.freeze_scale:
            # Generic-event mode: ignore any attempt to rescale noise to one event's variance
            # (train_NPE.py and a few pipeline score-compression spots call this unconditionally).
            return
        adaptive_covariance = {}
        for receiver in misc_data.keys():
            adaptive_covariance[receiver] = {}
            for component in misc_data[receiver].keys():
                noise_instance_variance = misc_data[receiver][component] if misc_data[receiver][component].size == 1 else misc_data[receiver][component][0]
                adaptive_covariance[receiver][component] = [noise_instance_variance]
            
        self.adaptive_covariance = adaptive_covariance