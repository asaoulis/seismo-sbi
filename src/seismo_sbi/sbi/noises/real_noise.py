"""Recorded noise windows drawn as data-vector noise.

:class:`RealNoiseSampler` reads a directory of HDF5 noise windows in the receiver order of the
simulation, optionally rescaled to one event's pre-event variances, and can hold every valid
window in memory for training.
"""
from pathlib import Path
import numpy as np

from seismo_sbi.sbi.configuration import SimulationParameters
from seismo_sbi.sbi.noises.covariance_base import station_component_value
from seismo_sbi.simulators.simulation_io import SimulationDataLoader, component_alias


class RealNoiseSampler:
    """Draw recorded noise windows as data-vector noise.

    Each call picks one window from ``directory`` covering every model station (or, with
    ``allow_incomplete``, whichever it covers) and returns it flattened in receiver order,
    optionally rescaled to the variances set by :meth:`set_adaptive_covariance_with_misc_data`.
    """

    def __init__(self, simulation_parameters : SimulationParameters, directory, data_length = None, adaptive_covariance= None,
                 freeze_scale: bool = False, allow_incomplete: bool = False):
        """``simulation_parameters`` supplies the receivers and components; ``directory`` holds one
        HDF5 noise window per file, or is None when the windows are given in memory
        (:meth:`from_windows`).

        With ``allow_incomplete`` False a window missing any model station is skipped. With it True the
        window is used for the stations it has: absent stations are zero-filled and ``__call__`` returns
        ``(noise_vector, present_mask)`` for the caller to mask out, which is meaningful only with
        variable-station training. Whole windows are always drawn intact, preserving the inter-station
        noise coherence real at microseism periods.
        """
        self.allow_incomplete = bool(allow_incomplete)
        receivers = simulation_parameters.receivers
        self.num_stations = len(receivers.receivers)
        self.components = simulation_parameters.components
        self.components = component_alias(self.components)

        self.data_loader = SimulationDataLoader(self.components, simulation_parameters.receivers, data_length)

        # Flattened length of a complete noise window. A window sitting on a station data gap
        # comes back short, which would not broadcast against the data vector, so it is skipped.
        self._expected_length = (
            None if data_length is None
            else sum(len(rec.components) for rec in receivers.receivers) * data_length
        )

        self.noise_paths = self._find_noise_paths(directory)
        np.random.shuffle(self.noise_paths)

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

        if directory is not None:
            print(f"Found {len(self.noise_paths)} noise realisations.")

    @classmethod
    def from_receivers(cls, receivers, components, seismogram_duration_s, sampling_rate_hz, directory, **kwargs):
        """A sampler for ``receivers`` recording ``components`` (a string such as ``"ZEN"``) over
        ``seismogram_duration_s`` at ``sampling_rate_hz``; ``kwargs`` are those of the constructor.
        """
        simulation_parameters = SimulationParameters(
            receivers=receivers, components=components, seismogram_duration=seismogram_duration_s,
            syngine_address=None, sampling_rate=sampling_rate_hz, processing={},
        )
        return cls(simulation_parameters, directory, **kwargs)

    @classmethod
    def from_windows(cls, noise_windows, receivers, components, present=None):
        """A sampler drawing uniformly from ``noise_windows`` held in memory.

        ``noise_windows`` is ``(n_windows, data_vector_length)``: each row one window's traces for
        every component each receiver records, in receiver order, as :meth:`__call__` returns
        them. ``present`` ``(n_windows, n_stations)`` marks the stations each window holds; with
        it a call returns ``(noise_vector, present_mask)`` as with ``allow_incomplete``, and the
        rows are zero where a station is absent. Windows are drawn as recorded, never rescaled.
        """
        simulation_parameters = SimulationParameters(
            receivers=receivers, components=components, seismogram_duration=None,
            syngine_address=None, sampling_rate=None, processing={},
        )
        sampler = cls(simulation_parameters, None, freeze_scale=True,
                      allow_incomplete=present is not None)
        sampler._noise_cache = np.ascontiguousarray(noise_windows)
        if present is not None:
            sampler._presence_cache = np.ascontiguousarray(present, dtype=bool)
            sampler._build_presence_index()
        return sampler

    def _build_presence_index(self):
        """Pack each cached window's station presence into one integer, so :meth:`sample_containing` can find the windows holding a station subset in one comparison."""
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
        if directory is None:
            return np.array([])
        return np.array(list(Path(directory).glob('*.h5')))

    def preload_cache(self, max_workers: int = 16, dtype=np.float32):
        """Load every valid noise window into ``self._noise_cache`` ``(n_valid, L)``; ``__call__`` then
        returns a uniformly random row.

        Windows whose flattened length differs from the data-vector length (missing stations or data
        gaps) are skipped as the on-disk path skips them. Only the generic path (``adaptive_covariance``
        None, no rescaling) draws from the cache.
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

    def __call__(self, noise_path = None, no_rescale = False, noise_index = None):

        # Fast path: in-RAM pool for the generic ML draw (random window, no rescale). Returns a
        # uniformly random cached window — distributionally identical to the on-disk random draw.
        if (self._noise_cache is not None and not no_rescale
                and self.adaptive_covariance is None
                and noise_path is None and noise_index is None):
            row = np.random.randint(0, self._noise_cache.shape[0])
            if self.allow_incomplete:
                return self._noise_cache[row], self._presence_cache[row]
            return self._noise_cache[row]

        for _ in range(len(self.noise_paths) + 1):
            if noise_index is not None:
                noise_path = self.noise_paths[noise_index % len(self.noise_paths)]
            if noise_path is None:
                noise_index = np.random.randint(0, len(self.noise_paths))
                noise_path = self.noise_paths[noise_index]
            noise = self._read_window(noise_path, no_rescale)
            if noise is not None:
                return noise
            # An unusable window: walk on to the next one, or draw afresh when it was requested by path.
            noise_path = None
            noise_index = None if noise_index is None else noise_index + 1
        raise RuntimeError(
            f"RealNoiseSampler: no noise window matched the expected data-vector length "
            f"{self._expected_length} after scanning all {len(self.noise_paths)} windows.")

    def _read_window(self, noise_path, no_rescale):
        """The draw from the window at ``noise_path``, or None when the window is unusable: no model
        station at all, a model station missing, or a trace shorter than the data vector needs."""
        if self.allow_incomplete:
            noise_realisations, present = self.data_loader.load_flattened_simulation_vector_with_presence(
                noise_path)
            if not present.any():
                return None
            return noise_realisations, present
        try:
            noise_realisations = self._load_noise_file(noise_path)
        except KeyError:
            return None
        if self._expected_length is not None and noise_realisations.size != self._expected_length:
            return None

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
        """Rescale later draws to the per-trace variances in ``misc_data``; no-op if ``freeze_scale``."""
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