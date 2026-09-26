"""Random per-station timing errors, flat or growing with path length.

``TimeShiftErrorEffect`` adds an array-wide common offset to independent per-station shifts
and applies them by Lanczos interpolation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from seismo_sbi.simulators.lanczos_shift import _shift_components
from seismo_sbi.simulators.seismogram_effect import SeismogramEffect


class TimeShiftErrorEffect(SeismogramEffect):
    """Shift a station's traces in time by a sub-sample amount, via Lanczos interpolation.

    The shift in s is an array-wide common offset plus an independent per-station draw from
    ``N(0, gaussian_sigma)``. The common offset models a constant velocity or source-time
    bias and is drawn once per call, from ``uniform(-uniform_offset, uniform_offset)`` or,
    under ``common_offset_dist='gaussian'``, from ``N(0, common_offset_sigma)``; measured
    array-wide offsets are peaked at zero rather than flat. Positive shifts delay.

    ``sigma_per_1000km`` and ``distance_cap_km`` grow the per-station width with path length
    and need the source location; zero keeps it flat. ``sampling_rate`` in samples per second
    converts the shift to samples and is injected by the caller rather than configured.

    Nuisance key ``time_shift_error``: a switch, not a probability. Absent or zero is the
    identity; any other value makes the effect active, with the magnitude set entirely by the
    configuration.
    """

    #: Default half-width of the array-wide common-offset uniform distribution (s).
    DEFAULT_UNIFORM_OFFSET: float = 0.0
    #: Default Gaussian standard deviation (seconds).
    DEFAULT_GAUSSIAN_SIGMA: float = 1.0
    #: Default Lanczos kernel order.
    DEFAULT_LANCZOS_ORDER: int = 5
    #: Default common-offset distribution ("uniform" for back-compat).
    DEFAULT_COMMON_OFFSET_DIST: str = "uniform"

    def __init__(
        self,
        sampling_rate: float,
        uniform_offset: Optional[float] = None,
        gaussian_sigma: Optional[float] = None,
        lanczos_order: Optional[int] = None,
        common_offset_dist: Optional[str] = None,
        common_offset_sigma: Optional[float] = None,
        sigma_per_1000km: float = 0.0,
        distance_cap_km: Optional[float] = None,
        source_latitude: Optional[float] = None,
        source_longitude: Optional[float] = None,
    ) -> None:
        self._sampling_rate = float(sampling_rate)
        # Timing error of one-dimensional synthetics grows with path length.
        self._sigma_per_1000km = float(sigma_per_1000km)
        if self._sigma_per_1000km < 0.0:
            raise ValueError("sigma_per_1000km must be >= 0")
        self._distance_cap = None if distance_cap_km is None else float(distance_cap_km)
        self._src = self._configured_source(source_latitude, source_longitude)
        self._uniform_offset = (
            float(uniform_offset)
            if uniform_offset is not None
            else self.DEFAULT_UNIFORM_OFFSET
        )
        self._sigma = (
            float(gaussian_sigma)
            if gaussian_sigma is not None
            else self.DEFAULT_GAUSSIAN_SIGMA
        )
        self._order = (
            int(lanczos_order)
            if lanczos_order is not None
            else self.DEFAULT_LANCZOS_ORDER
        )
        self._common_dist = (
            str(common_offset_dist).lower()
            if common_offset_dist is not None
            else self.DEFAULT_COMMON_OFFSET_DIST
        )
        if self._common_dist not in ("uniform", "gaussian"):
            raise ValueError(
                "common_offset_dist must be 'uniform' or 'gaussian'; got "
                f"{common_offset_dist!r}"
            )
        # Defaults to uniform_offset so an existing scale carries over when the distribution
        # is switched.
        self._common_sigma = (
            float(common_offset_sigma)
            if common_offset_sigma is not None
            else self._uniform_offset
        )

    def station_sigmas(self, receivers, source_location=None) -> dict:
        """``{station: width in s}`` of the per-station Gaussian, distance-scaled when
        ``sigma_per_1000km`` is positive and flat otherwise."""
        if self._sigma_per_1000km <= 0.0:
            return {}
        distances = self._station_distances_km(receivers, self._resolve_source(source_location),
                                               self._distance_cap)
        return {station: self._sigma + self._sigma_per_1000km * d / 1000.0
                for station, d in distances.items()}

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        *,
        time_shift_error: Optional[float] = None,
        source_location=None,
        **_ignored,
    ) -> dict:
        if time_shift_error is None or float(time_shift_error) == 0.0:
            return seismograms_map

        station_sigma = self.station_sigmas(receivers, source_location)

        if self._common_dist == "gaussian":
            common_offset_s = (
                np.random.normal(0.0, self._common_sigma)
                if self._common_sigma > 0.0
                else 0.0
            )
        else:
            common_offset_s = (
                np.random.uniform(-self._uniform_offset, self._uniform_offset)
                if self._uniform_offset > 0.0
                else 0.0
            )
        result = {}
        for station, components in seismograms_map.items():
            station_shift_s = common_offset_s + np.random.normal(
                0.0, station_sigma.get(station, self._sigma))
            shift_samples = station_shift_s * self._sampling_rate
            comps = list(components)
            if not comps:
                result[station] = {}
                continue
            result[station] = _shift_components(components, shift_samples, self._order)
        return result
