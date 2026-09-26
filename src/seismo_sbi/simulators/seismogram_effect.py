"""The base class of every post-processing effect, and helpers several effects share.

A ``SeismogramEffect`` owns exactly one nuisance key, is a strict no-op when that key is absent
from ``nuisance_params``, and returns a new dict rather than mutating its input.
``_apply_per_station_gated`` fires a transform per station with a probability;
``_bearing_and_distance_km`` gives source-station azimuth and distance on a spherical Earth.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SeismogramEffect(ABC):
    """One transformation of a ``{station: {component: waveform}}`` map.

    A subclass implements ``__call__``, takes its own key out of ``nuisance_params``, ignores
    every other key, and returns the input unchanged when its key is absent.
    """

    @abstractmethod
    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        **nuisance_params,
    ) -> dict:
        ...


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
