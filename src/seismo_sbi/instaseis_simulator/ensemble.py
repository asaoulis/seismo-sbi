import os
import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
import numpy as np

from .simulator import Simulator
from .wrapper import GenericPointSource, InstaseisDBQuerier


#: Per-station member resampling (see :class:`InstaseisEnsembleSimulator`) offsets the RNG seed by
#: this stride per station so that, under an EXPLICIT seed, each station draws a
#: distinct-but-reproducible member without colliding with the per-region offset
#: :class:`MultiModelSimulator` applies (``seed + region_index``). Production runs are unseeded, so
#: this only affects reproducibility tests.
PER_STATION_SEED_STRIDE = 10_000

#: Per-process LRU cache of open :class:`InstaseisDBQuerier` handles, keyed by
#: ``(pid, db_path, seismogram_length, processing_signature)``. ``instaseis.open_db`` (~168 ms) is
#: ~17x the cost of one ``get_seismograms`` (~10 ms), so reusing an open handle across stations AND
#: across simulations amortizes the open to ~once per member per worker process (per-event ~2x
#: faster; per-station kept ~1x instead of ~7x). This is a MODULE global — never an instance
#: attribute — precisely because :class:`InstaseisEnsembleSimulator` instances are deepcopied /
#: joblib-pickled out to dataset-generation workers and an open h5py handle is not picklable; keeping
#: the cache off the instance leaves the simulator picklable while still amortizing opens within each
#: worker. ``pid`` is in the key so a forked child never reuses a parent's handle. Outputs are
#: bit-identical to a fresh open (read-only DB), so cached runs reproduce existing datasets exactly.
_QUERIER_CACHE = OrderedDict()

#: LRU cap on :data:`_QUERIER_CACHE` (open DB handles held per process). Default comfortably covers a
#: Mode-A + Mode-B multi-model ensemble (~30 + ~30); each ``InstaseisEnsembleSimulator.__init__``
#: bumps it to at least its own member count so a single large ensemble never thrashes. Override with
#: env ``SEISMO_QUERIER_CACHE_MAXSIZE``.
_QUERIER_CACHE_MAXSIZE = max(1, int(os.environ.get("SEISMO_QUERIER_CACHE_MAXSIZE", "64")))


def _ensure_querier_cache_capacity(n_members: int) -> None:
    """Grow the global LRU cap to hold at least one full ensemble's worth of handles."""
    global _QUERIER_CACHE_MAXSIZE
    if n_members > _QUERIER_CACHE_MAXSIZE:
        _QUERIER_CACHE_MAXSIZE = n_members


class GFEnsembleSimulator(Simulator, ABC):
    """Simulator backed by an ensemble of precomputed 1D Earth-model GFs.

    One member is drawn per simulation; the fiducial member is used when
    use_fiducial=True.  Subclasses must expose `members` and `fiducial_member`
    and implement `_simulate_with_member` if they rely on the member-dispatch
    pattern (e.g. Instaseis).  CPS subclasses keep their own
    generic_point_source_simulation and call select_member() directly inside
    their GF-loading routine.
    """

    @property
    @abstractmethod
    def members(self) -> list:
        """Ordered list of ensemble members (e.g. folder paths or DB paths)."""
        ...

    @property
    @abstractmethod
    def fiducial_member(self):
        """The fiducial (reference) member of the ensemble."""
        ...

    @property
    def num_models(self) -> int:
        return len(self.members)

    def select_member(self, *, use_fiducial=False, seed=None):
        """Draw one member from the ensemble.

        Reproduces the legacy CPS call sequence exactly:
        np.random.seed(seed) then np.random.choice(members).
        """
        if use_fiducial:
            return self.fiducial_member
        if seed is not None:
            np.random.seed(seed)
        return np.random.choice(self.members)

    def _simulate_with_member(self, member, source, **kwargs) -> dict:
        """Simulate seismograms using a specific ensemble member.

        Override in subclasses that use the select_member →
        _simulate_with_member dispatch pattern (e.g. InstaseisEnsembleSimulator).
        CPS subclasses integrate member selection into their GF loading instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _simulate_with_member "
            "to use the member-dispatch pattern."
        )


class InstaseisEnsembleSimulator(GFEnsembleSimulator):
    """Instaseis simulator backed by an ensemble of Instaseis databases.

    By default one DB is drawn at random per simulation (via ``select_member``) and used for ALL
    stations, matching the CPS ensemble methodology.

    With ``resample_member_per_station=True`` (intra-ensemble / per-station sampling) an INDEPENDENT
    member is drawn for each station within this region for a given event. This is the physically
    faithful model when the ensemble members are *path-specific* 1-D models (e.g. the Santorini
    Mode-A caldera->station corridors): different stations have different paths, so their theory
    errors should be (partially) decorrelated rather than sharing one identical 1-D model.

    Both paths reuse open DB handles through the per-process :data:`_QUERIER_CACHE`, so per-station
    resampling stays ~as fast as the per-event default instead of paying one ``instaseis.open_db``
    per station (see the cache docstring).

    Parameters
    ----------
    instaseis_ensemble_dir : str or Path
        Directory whose immediate subdirectories are each a full Instaseis DB.
    instaseis_fiducial_loc : str or Path
        Path to the fiducial (reference) Instaseis DB.
    resample_member_per_station : bool, default False
        Draw an independent member per station per simulation (opt-in). ``use_fiducial=True`` always
        overrides this (every station uses the single fiducial member).
    """

    def __init__(self, instaseis_ensemble_dir, instaseis_fiducial_loc, *args,
                 resample_member_per_station=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.resample_member_per_station = resample_member_per_station
        # Hashable signature of the (small, fixed) processing config for the querier-cache key.
        self._processing_signature = json.dumps(
            self.synthetics_processing, sort_keys=True, default=str
        )
        ensemble_dir = Path(instaseis_ensemble_dir)
        self._members = sorted(
            [str(p) for p in ensemble_dir.iterdir() if p.is_dir()]
        )
        if not self._members:
            raise FileNotFoundError(
                f"No Instaseis DB directories found in {ensemble_dir}"
            )
        self._fiducial_member = str(instaseis_fiducial_loc)
        _ensure_querier_cache_capacity(len(self._members))

        # Derive sampling_rate from the fiducial DB. Open handles live in the MODULE-level
        # _QUERIER_CACHE (keyed by pid+path), never on the instance: GeneralSimulatorWrapper
        # deepcopies the simulation_callable and joblib pickles the simulator out to dataset workers,
        # and an open instaseis/h5py handle is not picklable ("h5py objects cannot be pickled").
        # Keeping the cache off the instance preserves picklability while still amortizing the
        # ~168 ms instaseis.open_db across stations and simulations within each worker process.
        self.sampling_rate = float(
            self._cached_querier(self._fiducial_member).sampling_rate
        )

    @property
    def members(self) -> list:
        return self._members

    @property
    def fiducial_member(self):
        return self._fiducial_member

    def _open_querier(self, db_path) -> InstaseisDBQuerier:
        # str() converts numpy.str_ (returned by np.random.choice on string lists) to
        # plain Python str to avoid "Can't mix strings and bytes" in os.walk inside
        # instaseis.open_db.
        return InstaseisDBQuerier(
            str(db_path), self.synthetics_processing, self.seismogram_length
        )

    def _cached_querier(self, db_path) -> InstaseisDBQuerier:
        """Return an open querier for ``db_path``, reusing the per-process LRU cache.

        See :data:`_QUERIER_CACHE`. On a miss, opens fresh via :meth:`_open_querier` and stores it;
        on a hit, marks it most-recently-used. Evicts least-recently-used handles once the cache
        exceeds :data:`_QUERIER_CACHE_MAXSIZE`.
        """
        key = (os.getpid(), str(db_path), self.seismogram_length, self._processing_signature)
        querier = _QUERIER_CACHE.get(key)
        if querier is None:
            querier = self._open_querier(db_path)
            _QUERIER_CACHE[key] = querier
            while len(_QUERIER_CACHE) > _QUERIER_CACHE_MAXSIZE:
                _QUERIER_CACHE.popitem(last=False)
        else:
            _QUERIER_CACHE.move_to_end(key)
        return querier

    def _simulate_with_member(self, member: str, source: GenericPointSource, *,
                              stf_duration=None, **kwargs) -> dict:
        querier = self._cached_querier(member)
        all_seismograms_map = {}
        for receiver in self.receivers.iterate():
            all_seismograms_map[receiver.station_name] = {}
            receiver_results = querier.get_seismograms(
                source, receiver, self.components, stf_duration=stf_duration
            )
            for component in self.components:
                all_seismograms_map[receiver.station_name][component] = (
                    receiver_results[component]
                )
        return all_seismograms_map

    def _simulate_per_station(self, source: GenericPointSource, *,
                              seed=None, stf_duration=None) -> dict:
        """Draw an INDEPENDENT member per station and serve each from the cached querier.

        Enabled by ``resample_member_per_station=True``. Unseeded (production) => N independent draws
        from the shared RNG; seeded => station ``i`` uses ``seed + i*PER_STATION_SEED_STRIDE`` so each
        station is distinct yet reproducible (and decorrelated from the per-region seed offset).
        """
        all_seismograms_map = {}
        for station_index, receiver in enumerate(self.receivers.iterate()):
            member_seed = None if seed is None else seed + station_index * PER_STATION_SEED_STRIDE
            member = self.select_member(use_fiducial=False, seed=member_seed)
            querier = self._cached_querier(member)
            receiver_results = querier.get_seismograms(
                source, receiver, self.components, stf_duration=stf_duration
            )
            all_seismograms_map[receiver.station_name] = {
                component: receiver_results[component] for component in self.components
            }
        return all_seismograms_map

    def generic_point_source_simulation(
        self, source: GenericPointSource, *, use_fiducial=False, seed=None,
        stf_duration=None, **kwargs
    ) -> dict:
        if self.resample_member_per_station and not use_fiducial:
            return self._simulate_per_station(source, seed=seed, stf_duration=stf_duration)
        member = self.select_member(use_fiducial=use_fiducial, seed=seed)
        return self._simulate_with_member(member, source, stf_duration=stf_duration)
