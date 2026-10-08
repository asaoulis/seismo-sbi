"""Instaseis forward model backed by an ensemble of Instaseis databases.

Members are the immediate subdirectories of the ensemble directory. A member is drawn per event,
per station, or per azimuthal sector, and the open database handles are reused through a
per-process LRU cache whose size is a memory budget (``SEISMO_QUERIER_CACHE_MAXSIZE``).
"""

import os
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np

from ..gf_ensemble import GFEnsembleSimulator
from ..sources import GenericPointSource
from ..simulation_io import SYNTHETICS_PRE_EVENT_PAD_S
from .querier import InstaseisDBQuerier


#: Stride the seed is offset by per station under per-station member resampling, so each station
#: draws a distinct but reproducible member without colliding with the per-region offset.
PER_STATION_SEED_STRIDE = 10_000

#: Open database handles, keyed by ``(pid, db_path, seismogram_length, processing_signature,
#: source_depth_offset_km, stf_alignment)``.
_QUERIER_CACHE = OrderedDict()

#: Cap on :data:`_QUERIER_CACHE`; ``SEISMO_QUERIER_CACHE_MAXSIZE`` overrides the default.
_QUERIER_CACHE_MAXSIZE = max(1, int(os.environ.get("SEISMO_QUERIER_CACHE_MAXSIZE", "64")))

#: True when the cap came from ``SEISMO_QUERIER_CACHE_MAXSIZE``, in which case it is
#: authoritative and nothing may grow it.
_QUERIER_CACHE_MAXSIZE_IS_EXPLICIT = "SEISMO_QUERIER_CACHE_MAXSIZE" in os.environ


def _ensure_querier_cache_capacity(n_members: int) -> None:
    """Let the open-database cache hold one full ensemble; a no-op against an explicit cap."""
    global _QUERIER_CACHE_MAXSIZE
    if _QUERIER_CACHE_MAXSIZE_IS_EXPLICIT:
        return
    if n_members > _QUERIER_CACHE_MAXSIZE:
        _QUERIER_CACHE_MAXSIZE = n_members
class InstaseisEnsembleSimulator(GFEnsembleSimulator):
    """Instaseis simulator backed by an ensemble of Instaseis databases.

    ``instaseis_ensemble_dir`` is a directory whose immediate subdirectories are each a full
    database; ``instaseis_fiducial_loc`` is the reference one. One member is drawn per
    simulation and used for every station, unless ``resample_member_per_station`` draws an
    independent member per station, which is the faithful model when the members are
    path-specific 1-D models. ``use_fiducial=True`` always overrides both.
    """

    pre_event_pad_s = SYNTHETICS_PRE_EVENT_PAD_S

    def __init__(self, instaseis_ensemble_dir, instaseis_fiducial_loc, *args,
                 resample_member_per_station=False, member_sampling=None,
                 sector_lambda=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.resample_member_per_station = resample_member_per_station
        # Sector sampling draws K ~ Poisson(sector_lambda) azimuthal boundaries per simulation
        # and gives every station inside a sector one member, so P(K = 0) is fully coherent.
        if member_sampling is None:
            member_sampling = 'per_station' if resample_member_per_station else 'per_event'
        member_sampling = str(member_sampling).lower()
        if member_sampling not in self.VALID_MEMBER_SAMPLING:
            raise ValueError(f"member_sampling must be one of {self.VALID_MEMBER_SAMPLING}; "
                             f"got {member_sampling!r}")
        if member_sampling == 'sector':
            if sector_lambda is None or float(sector_lambda) < 0.0:
                raise ValueError("member_sampling='sector' requires sector_lambda >= 0 "
                                 "(no default: calibrate it against the measured "
                                 "inter-station error correlation)")
            self.sector_lambda = float(sector_lambda)
        else:
            self.sector_lambda = None
        if member_sampling == 'per_station':
            self.resample_member_per_station = True
        self.member_sampling = member_sampling
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

        # The handle stays in the module-level cache, never on the instance: the simulator is
        # deepcopied and pickled out to workers, and an open database handle is not picklable.
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
        # str() converts the numpy.str_ that np.random.choice returns; instaseis.open_db walks
        # the path and cannot mix strings and bytes.
        return InstaseisDBQuerier(
            str(db_path), self.synthetics_processing, self.seismogram_length,
            self.source_depth_offset_km, self.stf_alignment
        )

    def _cached_querier(self, db_path) -> InstaseisDBQuerier:
        """An open querier for ``db_path``."""
        key = (os.getpid(), str(db_path), self.seismogram_length, self._processing_signature,
               self.source_depth_offset_km, self.stf_alignment)
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

    VALID_MEMBER_SAMPLING = ('per_event', 'per_station', 'sector')

    @staticmethod
    def sector_boundaries(lam: float, rng) -> np.ndarray:
        """K ~ Poisson(lam) boundaries, uniform on [0, 360), sorted (empty for K = 0)."""
        k = int(rng.poisson(lam)) if lam > 0.0 else 0
        return np.sort(rng.uniform(0.0, 360.0, size=k)) if k > 0 else np.zeros(0)

    @staticmethod
    def sector_index(azimuth_deg: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
        """Sector id per azimuth in deg. The arcs before the first and after the last boundary
        are the same sector, so K boundaries give K sectors, and K < 2 gives one."""
        az = np.mod(np.asarray(azimuth_deg, dtype=np.float64), 360.0)
        if len(boundaries) <= 1:
            return np.zeros(len(az), dtype=int)
        idx = np.searchsorted(boundaries, az, side='right')
        return np.where(idx == len(boundaries), 0, idx)

    def station_azimuths(self, source: GenericPointSource) -> np.ndarray:
        """Source -> station azimuths (deg, clockwise from north) in receiver iteration order."""
        loc = source.source_location
        la1, lo1 = np.deg2rad(loc.latitude), np.deg2rad(loc.longitude)
        out = []
        for r in self.receivers.iterate():
            la2, lo2 = np.deg2rad(r.latitude), np.deg2rad(r.longitude)
            dlon = lo2 - lo1
            az = np.arctan2(np.sin(dlon) * np.cos(la2),
                            np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlon))
            out.append(np.mod(np.rad2deg(az), 360.0))
        return np.asarray(out)

    def draw_sector_members(self, source: GenericPointSource, *, seed=None):
        """``(member per station, sector boundaries in deg)`` for one simulation.

        Seeded, sector ``j`` uses ``seed + j * PER_STATION_SEED_STRIDE``; unseeded, every draw
        comes from the shared global generator.
        """
        rng = np.random.default_rng(seed) if seed is not None else np.random
        bounds = self.sector_boundaries(self.sector_lambda, rng)
        sectors = self.sector_index(self.station_azimuths(source), bounds)
        members = {}
        for j in sorted(set(int(x) for x in sectors)):
            member_seed = None if seed is None else seed + j * PER_STATION_SEED_STRIDE
            members[j] = self.select_member(use_fiducial=False, seed=member_seed)
        return [members[int(j)] for j in sectors], bounds

    def _simulate_sector(self, source: GenericPointSource, *, seed=None, stf_duration=None) -> dict:
        per_station, _ = self.draw_sector_members(source, seed=seed)
        all_seismograms_map = {}
        for receiver, member in zip(self.receivers.iterate(), per_station):
            querier = self._cached_querier(member)
            receiver_results = querier.get_seismograms(
                source, receiver, self.components, stf_duration=stf_duration
            )
            all_seismograms_map[receiver.station_name] = {
                component: receiver_results[component] for component in self.components
            }
        return all_seismograms_map

    def _simulate_per_station(self, source: GenericPointSource, *,
                              seed=None, stf_duration=None) -> dict:
        """Seismograms with an independent member drawn per station.

        Seeded, station ``i`` uses ``seed + i * PER_STATION_SEED_STRIDE``; unseeded, every draw
        comes from the shared global generator.
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
        stf_duration=None, member=None, **kwargs
    ) -> dict:
        """Seismograms on one ensemble member: ``member`` if given, else the fiducial member or
        a random draw (per station or per azimuth sector when so configured)."""
        if member is None and not use_fiducial and getattr(self, 'member_sampling', None) == 'sector':
            return self._simulate_sector(source, seed=seed, stf_duration=stf_duration)
        if member is None and self.resample_member_per_station and not use_fiducial:
            return self._simulate_per_station(source, seed=seed, stf_duration=stf_duration)
        member = self.select_member(use_fiducial=use_fiducial, seed=seed, member=member)
        return self._simulate_with_member(member, source, stf_duration=stf_duration)
