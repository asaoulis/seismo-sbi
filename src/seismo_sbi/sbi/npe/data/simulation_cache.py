"""The clean simulations an NPE training dataset draws its samples from.

:class:`SimulationCache` reads each simulation's parameters, clean data and source-conditioning
vector from its HDF5 file, per sample or once into RAM with :meth:`SimulationCache.preload`.
:class:`ArraySimulationCache` serves the same from arrays the caller already holds.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm


class SimulationCache:
    """The simulations at ``paths``, read with ``data_loader``.

    ``parameter_name_map`` is ``{parameter type: [names]}``, the order of ``theta``;
    ``conditioning_param_map`` is ``{input type: [attribute names]}``, the order of the raw,
    unscaled source-conditioning vector. Nothing is read until :meth:`load`, :meth:`conditioning`
    or :meth:`preload` is called.
    """

    def __init__(self, paths, data_loader, parameter_name_map, conditioning_param_map):
        self.paths = paths
        self.data_loader = data_loader
        self.parameter_name_map = parameter_name_map or {}
        self.conditioning_param_map = conditioning_param_map or {}
        self._cache_D = None
        self._cache_theta = None
        self._cache_cond = None

    def preload(self, max_workers: int = 16, dtype=np.float32):
        """Load every simulation's clean ``(theta, D[, conditioning])`` into contiguous buffers, reproducing ``_load_sim`` and ``_load_conditioning`` exactly."""
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

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            for _ in tqdm(ex.map(_load_one, range(n)), total=n,
                          desc="[sim-cache] preloading", unit="sim"):
                pass
        gb = self._cache_D.nbytes / 1e9
        print(f"[sim-cache] preloaded {n} sims into RAM ({gb:.2f} GB, {dtype.__name__}) "
              f"in {time.perf_counter() - t0:.1f}s — per-sample HDF5 load removed.")

    def load(self, idx):
        """``(theta (D,), D (N, C, T))`` for simulation ``idx``, from RAM when preloaded."""
        if self._cache_D is not None:
            # A copy, so the augmentation and noise applied to it never change the cache.
            D = np.array(self._cache_D[idx])
            theta = (np.array(self._cache_theta[idx]) if self._cache_theta is not None
                     else np.array([]))
        else:
            theta, D = self._load_sim(self.paths[idx])
        return theta, D

    def conditioning(self, idx):
        """The raw source-conditioning vector of simulation ``idx``, or None without conditioning."""
        if self._cache_cond is not None:
            return self._cache_cond[idx]
        if self.conditioning_param_map:
            return self._load_conditioning(self.paths[idx])
        return None

    def _load_conditioning(self, sim_path):
        """Extract the raw (unscaled) source-conditioning vector from a sim's stored inputs."""
        inputs_dict = self.data_loader.load_input_data(sim_path)
        return np.concatenate([
            [inputs_dict[input_type][attr] for attr in attrs]
            for input_type, attrs in self.conditioning_param_map.items()
        ]).astype(float)

    def _load_sim(self, sim_path):
        """``(theta, D)`` read from one file: the clean data, without time shifts or other effects."""
        if len(self.parameter_name_map) > 0:
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


class ArraySimulationCache(SimulationCache):
    """Simulations held as arrays: ``theta`` ``(n_simulations, n_parameters)`` or None, the clean
    ``x`` ``(n_simulations, n_stations, n_components, n_samples)`` and ``conditioning``
    ``(n_simulations, n_conditioning)`` or None. ``n_stations`` is the receiver count ``x`` must match.
    """

    def __init__(self, theta, x, conditioning, n_stations):
        x = np.asarray(x)
        if x.ndim != 4 or x.shape[1] != n_stations:
            raise ValueError(f"x must be (n_simulations, {n_stations}, n_components, "
                             f"trace_length) for {n_stations} receivers, got {x.shape}")
        super().__init__([f"array_{index}" for index in range(x.shape[0])], None, {}, {})
        self._cache_D = x
        if theta is not None and np.size(theta):
            self._cache_theta = np.asarray(theta, dtype=np.float64).reshape(x.shape[0], -1)
        if conditioning is not None:
            self._cache_cond = np.asarray(conditioning, dtype=np.float64).reshape(x.shape[0], -1)
