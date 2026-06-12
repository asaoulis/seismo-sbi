from pathlib import Path
import numpy as np
import obspy
import os

from seismo_sbi.sbi.configuration import SimulationParameters
from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader


class RealNoiseSampler:

    # TODO: add components implementation

    def __init__(self, simulation_parameters : SimulationParameters, directory, data_length = None, adaptive_covariance= None,
                 freeze_scale: bool = False):
        receivers = simulation_parameters.receivers
        self.num_stations = len(receivers.receivers)
        self.components = simulation_parameters.components
        self.components = self.components.replace('E', '1').replace('N', '2')
        self.vector_length = round(simulation_parameters.seismogram_duration * simulation_parameters.sampling_rate)

        self.data_loader = SimulationDataLoader(self.components, simulation_parameters.receivers, data_length)

        # Expected flattened length of a COMPLETE noise window for this receiver set.
        # A handful of windows (~0.1% in the v2 Santorini catalogue) sit on a station
        # data gap, so one trace is shorter than data_length; convert_sim_data_to_array
        # anchors the length to the first receiver and truncates, silently returning a
        # sub-length vector that will not broadcast against the full-length data vector D
        # (and would corrupt the empirical covariance). When data_length is known we skip
        # such windows in __call__ — the variable-length analogue of the missing-station
        # KeyError skip. data_length == trace_length == data_vector_length // num_traces,
        # so this expected length is exactly the data-vector length, self-consistent with
        # whatever the true per-trace sample count is.
        self._expected_length = (
            None if data_length is None
            else sum(len(rec.components) for rec in receivers.receivers) * data_length
        )

        self.noise_paths = self._find_noise_paths(directory)
        np.random.shuffle(self.noise_paths)

        # Optional in-RAM noise-window cache (opt-in via preload_cache()). Each __call__ otherwise
        # opens an HDF5 noise file (~12 ms) — a per-training-sample cost on top of the sim load. The
        # generic-event ML path draws noise windows uniformly at random WITH replacement and never
        # rescales (adaptive_covariance is None), so a contiguous in-RAM pool returning a random row
        # is distributionally identical to the on-disk draw, at no disk cost and with no
        # worker-scaling wall. Disabled (None) ⇒ unchanged on-disk behaviour.
        self._noise_cache = None

        # When True the sampler draws generic noise windows verbatim and never rescales them to
        # a single event's pre-event variance: set_adaptive_covariance_with_misc_data becomes a
        # no-op so adaptive_covariance stays None. Use for generic-event ("amortised over
        # events") training where rescaling to one event would defeat the purpose.
        self.freeze_scale = freeze_scale
        self.adaptive_covariance = adaptive_covariance
        if self.adaptive_covariance is not None:
            for receiver in self.adaptive_covariance.keys():
                for component in self.adaptive_covariance[receiver].keys():
                    self.adaptive_covariance[receiver][component] = self.adaptive_covariance[receiver][component][0]

        print(f"Found {len(self.noise_paths)} noise realisations.")
    
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
        import time as _t
        paths = list(self.noise_paths)

        def _try_load(p):
            try:
                v = np.asarray(self._load_noise_file(p)).reshape(-1)
            except KeyError:
                return None
            if self._expected_length is not None and v.size != self._expected_length:
                return None
            return v

        t0 = _t.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            loaded = list(ex.map(_try_load, paths))
        # Keep windows matching the modal length (the data-vector length); drop gaps/mismatches.
        valid = [v for v in loaded if v is not None]
        if not valid:
            raise RuntimeError("RealNoiseSampler.preload_cache: no valid noise windows found.")
        ref_len = self._expected_length or valid[0].size
        valid = [v for v in valid if v.size == ref_len]
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
            return self._noise_cache[np.random.randint(0, self._noise_cache.shape[0])]

        # if self.noise_index_counter == len(self.noise_paths):
        #     self.noise_index_counter = 0
        #     np.random.shuffle(self.noise_paths)
        if _attempts > len(self.noise_paths):
            raise RuntimeError(
                f"RealNoiseSampler: no noise window matched the expected data-vector length "
                f"{self._expected_length} after scanning all {len(self.noise_paths)} windows.")
        if noise_index is not None:
            # wrap: the KeyError retry below increments noise_index, which would
            # otherwise run off the end when the last window is hit (IndexError
            # 'index N out of bounds for axis 0 with size N').
            noise_path = self.noise_paths[noise_index % len(self.noise_paths)]
        if noise_path is None:
            noise_index = np.random.randint(0, len(self.noise_paths))
            noise_path = self.noise_paths[noise_index]
        # On a retry we move to the next window when walking sequentially (noise_index
        # set), or fall back to a fresh random draw when a window was requested by path
        # (noise_index is None).
        next_index = None if noise_index is None else noise_index + 1
        try:
            noise_realisations = self._load_noise_file(noise_path)
        except KeyError:
            # this window lacks one of the event's stations; try the next one.
            return self.__call__(noise_path = None, no_rescale = no_rescale,
                                 noise_index=next_index, _attempts=_attempts + 1)
        # Skip windows that sit on a station data gap: one trace is shorter than the rest,
        # so the flattened vector is < the data-vector length and would not broadcast
        # against D. Treated exactly like the missing-station case above.
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
                try:
                    data_noise_variance = self.adaptive_covariance[receiver][component]
                except KeyError:
                    new_comp = component.replace('E', '1').replace('N', '2')
                    data_noise_variance = self.adaptive_covariance[receiver][new_comp]
  

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