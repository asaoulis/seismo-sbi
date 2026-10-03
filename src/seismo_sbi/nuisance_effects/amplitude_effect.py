"""Random per-station or per-trace amplitude errors.

``AmplitudeErrorEffect`` multiplies traces by a factor drawn uniformly or log-normally, gated
per station or always on with a strength multiplier.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from seismo_sbi.nuisance_effects.seismogram_effect import SeismogramEffect, per_station, station_value


class AmplitudeErrorEffect(SeismogramEffect):
    """Multiply a station's traces by a random amplitude factor.

    Nuisance key ``amplitude_error``: a per-station activation probability, or, under
    ``always_on``, a strength multiplier on the width (0 is the identity, 1 the configured
    width, 2 double).

    By default one factor per station is drawn from ``uniform(*scale_range)``.
    ``distribution='lognormal'`` draws ``10 ** (log_sigma_dex * N(0, 1))`` instead;
    ``per_component`` draws once per trace rather than once per station. Measured per-trace
    amplitude errors of regional records against one-dimensional synthetics are log-normal at
    about 0.3 dex and independent between a station's components, which is what those two
    switches express. ``scale_range`` and ``log_sigma_dex`` take a ``{station: value}`` map as
    well as one value, with a ``"default"`` entry for the stations it does not name.
    """

    #: Default lower bound of the per-station scale factor distribution.
    DEFAULT_SCALE_LOW: float = 0.5
    #: Default upper bound of the per-station scale factor distribution.
    DEFAULT_SCALE_HIGH: float = 2.0
    #: Default log-normal width in dex; measured per-trace widths on regional broadband
    #: records run 0.31 to 0.38 dex.
    DEFAULT_LOG_SIGMA_DEX: float = 0.3
    VALID_DISTRIBUTIONS = ("uniform", "lognormal")

    def __init__(
        self,
        scale_range: tuple[float, float] | None = None,
        distribution: str = "uniform",
        log_sigma_dex: Optional[float] = None,
        per_component: bool = False,
        always_on: bool = False,
    ) -> None:
        if scale_range is not None:
            self._scale_low = per_station(scale_range, lambda pair: float(pair[0]))
            self._scale_high = per_station(scale_range, lambda pair: float(pair[1]))
        else:
            self._scale_low = self.DEFAULT_SCALE_LOW
            self._scale_high = self.DEFAULT_SCALE_HIGH
        self._distribution = str(distribution).lower()
        if self._distribution not in self.VALID_DISTRIBUTIONS:
            raise ValueError(
                f"distribution must be one of {self.VALID_DISTRIBUTIONS}; got {distribution!r}"
            )
        self._log_sigma = (
            per_station(log_sigma_dex) if log_sigma_dex is not None else self.DEFAULT_LOG_SIGMA_DEX
        )
        widths = self._log_sigma.values() if isinstance(self._log_sigma, dict) else [self._log_sigma]
        if min(widths) < 0.0:
            raise ValueError("log_sigma_dex must be >= 0")
        self._per_component = bool(per_component)
        self._always_on = bool(always_on)

    def _draw_scale(self, multiplier: float, station: str) -> float:
        """One amplitude scale factor at ``station``; the uniform distribution ignores ``multiplier``."""
        if self._distribution == "lognormal":
            log_sigma = station_value(self._log_sigma, station, self.DEFAULT_LOG_SIGMA_DEX)
            return float(10.0 ** (multiplier * log_sigma * np.random.normal()))
        return float(np.random.uniform(station_value(self._scale_low, station, self.DEFAULT_SCALE_LOW),
                                       station_value(self._scale_high, station, self.DEFAULT_SCALE_HIGH)))

    def _scale_components(self, components: dict, multiplier: float, station: str) -> dict:
        if self._per_component:
            return {
                comp: trace.astype(np.float64) * self._draw_scale(multiplier, station)
                for comp, trace in components.items()
            }
        scale = self._draw_scale(multiplier, station)
        return {comp: trace.astype(np.float64) * scale for comp, trace in components.items()}

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        *,
        amplitude_error: float | None = None,
        **_ignored,
    ) -> dict:
        if amplitude_error is None:
            return seismograms_map

        if self._always_on:
            multiplier = float(amplitude_error)
            if multiplier == 0.0:
                return seismograms_map
            return {
                station: self._scale_components(components, multiplier, station)
                for station, components in seismograms_map.items()
            }

        def _scale(components, station):
            return self._scale_components(components, 1.0, station)

        return self._apply_per_station_gated(seismograms_map, amplitude_error, _scale)
