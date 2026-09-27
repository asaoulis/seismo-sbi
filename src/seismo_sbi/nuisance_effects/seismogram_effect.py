"""The base class of every post-processing effect, and the behaviour effects share.

A ``SeismogramEffect`` owns exactly one nuisance key, is a strict no-op when that key is absent
from ``nuisance_params``, and returns a new dict rather than mutating its input. The base class
gates a transform per station, resolves the source location and measures source-station
distances; ``_bearing_and_distance_km`` is the spherical-Earth geometry under the last two.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SeismogramEffect(ABC):
    """One transformation of a ``{station: {component: waveform}}`` map.

    A subclass implements ``__call__``, takes its own key out of ``nuisance_params``, ignores
    every other key, and returns the input unchanged when its key is absent. An effect that
    needs the source keeps its configured ``(latitude, longitude)`` in ``_src``.
    """

    _src: tuple[float, float] | None = None

    @abstractmethod
    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        **nuisance_params,
    ) -> dict:
        ...

    @staticmethod
    def _apply_per_station_gated(seismograms_map: dict, probability, transform) -> dict:
        """Apply ``transform`` to each station independently with the given probability.

        ``probability`` is clipped to ``[0, 1]``; a station that does not fire is copied through
        as float64. The gate is drawn before whatever ``transform`` draws, which fixes the order
        the random number generator is called in.
        """
        p = float(np.clip(probability, 0.0, 1.0))
        result = {}
        for station, components in seismograms_map.items():
            if np.random.uniform() < p:
                result[station] = transform(components)
            else:
                result[station] = {
                    comp: trace.astype(np.float64) for comp, trace in components.items()
                }
        return result

    @staticmethod
    def _configured_source(source_latitude, source_longitude):
        """``(latitude, longitude)`` from an effect's configuration, or None if either is missing."""
        return (None if source_latitude is None or source_longitude is None
                else (float(source_latitude), float(source_longitude)))

    def _resolve_source(self, source_location):
        """``(latitude, longitude)`` from the nuisance dict's ``source_location``, else ``_src``."""
        src = (tuple(np.asarray(source_location, dtype=np.float64).ravel()[:2])
               if source_location is not None else self._src)
        if src is None:
            raise ValueError(
                f"{type(self).__name__} is active but no source location is available "
                "(pass source_location in nuisance_params or source_latitude/longitude "
                "in the effect config)")
        return src

    @staticmethod
    def _station_distances_km(receivers, source_latlon, distance_cap_km=None) -> dict:
        """``{station: source-station distance in km}``, capped at ``distance_cap_km`` if given."""
        out = {}
        for r in receivers.iterate():
            _, dist = _bearing_and_distance_km(source_latlon[0], source_latlon[1], r.latitude, r.longitude)
            out[r.station_name] = dist if distance_cap_km is None else min(dist, distance_cap_km)
        return out


def _bearing_and_distance_km(src_lat, src_lon, sta_lat, sta_lon):
    """``(azimuth in deg clockwise from north, distance in km)`` on a spherical Earth."""
    la1, lo1 = np.deg2rad(src_lat), np.deg2rad(src_lon)
    la2, lo2 = np.deg2rad(sta_lat), np.deg2rad(sta_lon)
    dlon = lo2 - lo1
    az = np.arctan2(
        np.sin(dlon) * np.cos(la2),
        np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlon))
    hav = (np.sin((la2 - la1) / 2.0) ** 2
           + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2.0) ** 2)
    dist = 2.0 * 6371.0 * np.arcsin(np.sqrt(hav))
    return np.rad2deg(az) % 360.0, dist
