"""Frequency-dependent travel-time errors, correlated across octaves and stations.

``DispersionSpreadEffect`` applies a per-station phase delay interpolated between octave
centres, its width growing with source-station distance.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from seismo_sbi.nuisance_effects.seismogram_effect import SeismogramEffect


class DispersionSpreadEffect(SeismogramEffect):
    """Delay each station by a frequency-dependent travel-time error: a pure phase delay
    ``u'(f) = u(f) exp(-2 pi i f tau(f))``.

    The delay is given at the octave centres ``octave_centres_s``, interpolated in log period
    between them and held constant outside. Per station and octave it is the product of the
    nuisance multiplier, a standard normal and a width that grows linearly with source-station
    distance from ``sigma_intercept_s`` at ``sigma_per_1000km_s`` per 1000 km.

    The correlation of those normals is the physics: a crust that is too fast delays every
    octave of the path the same way, so ``octave_correlation`` shares one draw across octaves
    by default, and ``common_fraction`` puts that share of the variance into a single
    array-wide draw, which is the coherent same-sign far-station delay independent per-station
    sampling can never produce. A systematic bias belongs in the reference model, not here.

    Nuisance key ``dispersion_spread``: a strength multiplier. Needs the source location, so
    it is a simulation-stage effect; ``sampling_rate`` is injected by the caller.
    """

    DEFAULT_OCTAVE_CENTRES_S = (12.5, 17.5, 25.0, 40.0)

    def __init__(
        self,
        sampling_rate: float,
        octave_centres_s=None,
        sigma_intercept_s=None,
        sigma_per_1000km_s=None,
        distance_cap_km: Optional[float] = None,
        octave_correlation: float = 1.0,
        common_fraction: float = 0.0,
        source_latitude: Optional[float] = None,
        source_longitude: Optional[float] = None,
    ) -> None:
        self._sampling_rate = float(sampling_rate)
        self._T = np.array(octave_centres_s if octave_centres_s is not None
                           else self.DEFAULT_OCTAVE_CENTRES_S, dtype=np.float64)
        n = len(self._T)
        if n < 1 or np.any(np.diff(self._T) <= 0):
            raise ValueError("octave_centres_s must be strictly increasing periods (s)")
        self._a = np.zeros(n) if sigma_intercept_s is None else np.asarray(sigma_intercept_s, float)
        self._b = np.zeros(n) if sigma_per_1000km_s is None else np.asarray(sigma_per_1000km_s, float)
        if self._a.shape != (n,) or self._b.shape != (n,):
            raise ValueError("sigma_intercept_s / sigma_per_1000km_s must have one entry per octave centre")
        if np.any(self._a < 0) or np.any(self._b < 0):
            raise ValueError("dispersion sigmas must be >= 0")
        self._cap = None if distance_cap_km is None else float(distance_cap_km)
        self._rho = float(octave_correlation)
        self._common = float(common_fraction)
        if not (0.0 <= self._rho <= 1.0) or not (0.0 <= self._common <= 1.0):
            raise ValueError("octave_correlation and common_fraction must lie in [0, 1]")
        self._src = self._configured_source(source_latitude, source_longitude)

    def station_sigmas(self, receivers, source_location=None) -> dict:
        """``{station: width per octave}`` in s."""
        distances = self._station_distances_km(receivers, self._resolve_source(source_location), self._cap)
        return {station: self._a + self._b * d / 1000.0 for station, d in distances.items()}

    def tau_of_freq(self, freqs: np.ndarray, tau_octaves: np.ndarray) -> np.ndarray:
        """Per-octave delays in s interpolated onto ``freqs`` in log period, held constant
        outside the octave centres."""
        with np.errstate(divide="ignore"):
            per = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-12), self._T[-1])
        per = np.clip(per, self._T[0], self._T[-1])
        if len(self._T) == 1:
            return np.full_like(freqs, tau_octaves[0], dtype=np.float64)
        return np.interp(np.log(per), np.log(self._T), tau_octaves)

    @staticmethod
    def _delay(trace: np.ndarray, tau_f: np.ndarray, dt: float) -> np.ndarray:
        n = len(trace)
        fr = np.fft.rfftfreq(n, d=dt)
        return np.fft.irfft(np.fft.rfft(trace) * np.exp(-2j * np.pi * fr * tau_f), n=n)

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        *,
        dispersion_spread: Optional[float] = None,
        source_location=None,
        **_ignored,
    ) -> dict:
        if dispersion_spread is None or float(dispersion_spread) == 0.0:
            return seismograms_map
        m = float(dispersion_spread)
        sig = self.station_sigmas(receivers, source_location)
        n_oct = len(self._T)
        z_common = np.random.normal(size=n_oct)
        z_common_shared = np.random.normal()
        dt = 1.0 / self._sampling_rate
        result = {}
        for station, components in seismograms_map.items():
            z_shared = np.random.normal()
            z_ind = np.random.normal(size=n_oct)
            z_sta = np.sqrt(self._rho) * z_shared + np.sqrt(1.0 - self._rho) * z_ind
            z_com = np.sqrt(self._rho) * z_common_shared + np.sqrt(1.0 - self._rho) * z_common
            z = np.sqrt(self._common) * z_com + np.sqrt(1.0 - self._common) * z_sta
            tau_oct = m * z * sig.get(station, np.zeros(n_oct))
            comps = list(components)
            if not comps or not np.any(tau_oct):
                result[station] = {c: components[c].astype(np.float64) for c in comps}
                continue
            n = len(components[comps[0]])
            fr = np.fft.rfftfreq(n, d=dt)
            tau_f = self.tau_of_freq(fr, tau_oct)
            result[station] = {c: self._delay(components[c].astype(np.float64), tau_f, dt)
                               for c in comps}
        return result
