"""Recorded noise windows drawn as data-vector noise.

:class:`RealNoiseSampler` reads a directory of HDF5 noise windows in the receiver order of the
simulation, optionally rescaled to one event's pre-event variances, and can hold every valid
window in memory for training.
"""
from pathlib import Path
import h5py
import numpy as np

from seismo_sbi.sbi.types.parameters import SimulationParameters
from seismo_sbi.sbi.noises.covariance_base import pre_event_variances, station_component_value
from seismo_sbi.sbi.noises.noise_samplers import NoiseDraw, NoiseSampler
from seismo_sbi.simulators.simulation_io import SimulationDataLoader, component_alias


class RealNoiseSampler(NoiseSampler):
    """Draw recorded noise windows as data-vector noise.

    Each draw picks one window from ``directory`` covering every model station (or, with
    ``allow_incomplete``, whichever it covers) and returns it flattened in receiver order,
    optionally rescaled to the variances set by :meth:`rescale_to`.
    """

    def __init__(self, simulation_parameters : SimulationParameters, directory, data_length = None,
                 allow_incomplete: bool = False):
        """``simulation_parameters`` supplies the receivers and components; ``directory`` holds one
        HDF5 noise window per file, or is None when the windows are given in memory
        (:meth:`from_windows`).

        With ``allow_incomplete`` False a window missing any model station is skipped. With it True the
        window is used for the stations it has: absent stations are zero-filled and each draw carries
        the stations it holds in ``present`` for the caller to mask out, which is meaningful only with
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

        #: ``{station: {component: variance}}`` the draws are rescaled to, or None.
        self.target_variances = None

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
        every component each receiver records, in receiver order, as :meth:`draw` returns
        them. ``present`` ``(n_windows, n_stations)`` marks the stations each window holds; with
        it a draw carries its window's row of ``present`` as with ``allow_incomplete``, and the
        rows are zero where a station is absent. Windows are drawn as recorded.
        """
        simulation_parameters = SimulationParameters(
            receivers=receivers, components=components, seismogram_duration=None,
            syngine_address=None, sampling_rate=None, processing={},
        )
        sampler = cls(simulation_parameters, None, allow_incomplete=present is not None)
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
        """Load every valid noise window into ``self._noise_cache`` ``(n_valid, L)``; :meth:`draw` then
        returns a uniformly random row.

        Windows whose flattened length differs from the data-vector length (missing stations or data
        gaps) are skipped as the on-disk path skips them. Only a draw without a rescale target
        (``target_variances`` None) comes from the cache.
        """
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        import time as _t
        paths = list(self.noise_paths)

        def _try_load(p):
            if self.allow_incomplete:
                # Absent stations are zero-filled, so every window has the canonical
                # length and NONE are discarded. The presence mask travels with the row.
                v, present = self.data_loader.load_simulation_data_array_with_presence(p)
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

    def draw(self):
        """A noise window for a training sample.

        A random row of the in-memory pool when it is preloaded and no rescale target is set;
        otherwise a random window read from disk, rescaled to the target variances when they are
        set, its recorded covariance data then coming with it.
        """
        if self._noise_cache is not None and self.target_variances is None:
            row = np.random.randint(0, self._noise_cache.shape[0])
            present = self._presence_cache[row] if self.allow_incomplete else None
            return NoiseDraw(self._noise_cache[row], present)
        window_path, noise, present = self._usable_window(None)
        if self.allow_incomplete or self.target_variances is None:
            return NoiseDraw(noise, present)
        covariance_data = self.data_loader.load_misc_data(window_path)
        noise = self._load_noise_file(window_path, scale_dict=self.variance_ratios(covariance_data))
        return NoiseDraw(noise, None, covariance_data)

    def draw_with_covariance(self, window_index=None):
        """The window at ``window_index`` (a random one when None) as recorded, never rescaled or
        taken from the in-memory pool, with its recorded covariance data."""
        window_path, noise, present = self._usable_window(window_index)
        if self.allow_incomplete:
            return NoiseDraw(noise, present)
        return NoiseDraw(noise, None, self.data_loader.load_misc_data(window_path))

    def _usable_window(self, window_index):
        """``(path, noise, present)`` of the first usable window from ``window_index`` on, a random
        start when None, walking at most once round the pool."""
        if window_index is None:
            window_index = np.random.randint(0, len(self.noise_paths))
        for offset in range(len(self.noise_paths) + 1):
            window_path = self.noise_paths[(window_index + offset) % len(self.noise_paths)]
            noise, present = self._read_window(window_path)
            if noise is not None:
                return window_path, noise, present
        raise RuntimeError(
            f"RealNoiseSampler: no noise window matched the expected data-vector length "
            f"{self._expected_length} after scanning all {len(self.noise_paths)} windows.")

    def _read_window(self, window_path):
        """``(noise, present)`` read from the window at ``window_path``, or ``(None, None)`` when it
        is unusable: no model station at all, a model station missing, or a trace shorter than the
        data vector needs."""
        if self.allow_incomplete:
            noise, present = self.data_loader.load_simulation_data_array_with_presence(window_path)
            return (noise, present) if present.any() else (None, None)
        try:
            noise = self._load_noise_file(window_path)
        except KeyError:
            return None, None
        if self._expected_length is not None and noise.size != self._expected_length:
            return None, None
        return noise, None

    def _load_noise_file(self, path : Path, scale_dict=None):
        with h5py.File(path, 'r') as window:
            return self.data_loader.convert_sim_data_to_array(window, scale_dict=scale_dict)
    
    def variance_ratios(self, covariance_data):
        """``{station: {component: ratio}}``: each trace's pre-event variance in a window's
        ``covariance_data`` over the target variance it is rescaled to; the trace is divided by
        the square root of its ratio."""
        window_variances = pre_event_variances(covariance_data)
        return {station: {component: variance / station_component_value(self.target_variances, station, component)
                          for component, variance in components.items()}
                for station, components in window_variances.items()}

    def rescale_to(self, covariance_data):
        """Rescale later draws to the pre-event variances of ``covariance_data``
        ``{station: {component: autocovariance}}``."""
        self.target_variances = pre_event_variances(covariance_data)
