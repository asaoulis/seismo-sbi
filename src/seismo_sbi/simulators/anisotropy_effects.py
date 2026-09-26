"""Deterministic anisotropy: azimuthal travel-time anomalies and shear-wave splitting.

``AzimuthalAnisotropyEffect`` delays each station by its path's azimuthal anomaly;
``ShearSplittingEffect`` splits the horizontals with the Silver and Chan operator.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from seismo_sbi.simulators.lanczos_shift import _apply_lanczos_shift_batch
from seismo_sbi.simulators.seismogram_effect import SeismogramEffect, _bearing_and_distance_km


class AzimuthalAnisotropyEffect(SeismogramEffect):
    """Delay each station by the travel-time anomaly weak azimuthal anisotropy gives it.

    With the speed varying as ``V(phi) = V0 (1 + A cos 2(phi - phi_fast))``, a path of length
    D at source-station azimuth phi accumulates ``-(D / V0) A cos(2 (phi - phi_fast))``, so
    the fast azimuth arrives early. The delay grows with path length and is coherent across
    the array, unlike the independent per-station shifts of :class:`TimeShiftErrorEffect`. All
    components of a station share it, applied by Lanczos interpolation; there is no randomness.

    Nuisance key ``azimuthal_anisotropy``: a strength multiplier on ``aniso_fraction``, absent
    or zero being the identity. The source location comes from the nuisance dict or the
    constructor, and an active effect raises without one rather than silently doing nothing.
    """

    DEFAULT_REF_VELOCITY_KMS: float = 3.5
    DEFAULT_LANCZOS_ORDER: int = 5

    def __init__(
        self,
        sampling_rate: float,
        fast_azimuth_deg: float = 45.0,
        aniso_fraction: float = 0.0125,
        ref_velocity_kms: Optional[float] = None,
        source_latitude: Optional[float] = None,
        source_longitude: Optional[float] = None,
        lanczos_order: Optional[int] = None,
    ) -> None:
        self._sampling_rate = float(sampling_rate)
        self._fast_az = float(fast_azimuth_deg)
        self._fraction = float(aniso_fraction)
        self._v0 = float(ref_velocity_kms if ref_velocity_kms is not None
                         else self.DEFAULT_REF_VELOCITY_KMS)
        self._src = (None if source_latitude is None or source_longitude is None
                     else (float(source_latitude), float(source_longitude)))
        self._order = int(lanczos_order if lanczos_order is not None
                          else self.DEFAULT_LANCZOS_ORDER)

    def station_delays(self, receivers, source_latlon) -> dict:
        """``{station: delay in s}`` at multiplier one."""
        src_lat, src_lon = source_latlon
        out = {}
        for r in receivers.iterate():
            az, dist = _bearing_and_distance_km(src_lat, src_lon,
                                                r.latitude, r.longitude)
            out[r.station_name] = float(
                -(dist / self._v0) * self._fraction
                * np.cos(2.0 * np.deg2rad(az - self._fast_az)))
        return out

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        *,
        azimuthal_anisotropy: Optional[float] = None,
        source_location=None,
        **_ignored,
    ) -> dict:
        if azimuthal_anisotropy is None or float(azimuthal_anisotropy) == 0.0:
            return seismograms_map
        mult = float(azimuthal_anisotropy)
        src = (tuple(np.asarray(source_location, float)[:2])
               if source_location is not None else self._src)
        if src is None:
            raise ValueError(
                "AzimuthalAnisotropyEffect is active but no source location is "
                "available (pass source_location in nuisance_params or "
                "source_latitude/longitude in the effect config)")
        delays = self.station_delays(receivers, src)
        result = {}
        for station, components in seismograms_map.items():
            comps = list(components)
            if station not in delays or not comps:
                result[station] = dict(components)
                continue
            shift_samples = mult * delays[station] * self._sampling_rate
            traces = np.stack([components[c] for c in comps])
            shifted = _apply_lanczos_shift_batch(traces, shift_samples, self._order)
            result[station] = {c: shifted[j] for j, c in enumerate(comps)}
        return result


class ShearSplittingEffect(SeismogramEffect):
    """Split the horizontals of each station, the Silver and Chan (1991) operator.

    Station-side and in the geographic frame: rotate north and east into the fast and slow
    frame given by ``fast_azimuth_deg``, delay the slow trace by ``delay_s``, rotate back. The
    vertical is untouched, a station missing either horizontal passes through, and there is no
    randomness.

    Nuisance key ``shear_wave_splitting``: a multiplier on ``delay_s``, absent or zero being
    the identity.
    """

    DEFAULT_LANCZOS_ORDER: int = 5

    def __init__(
        self,
        sampling_rate: float,
        fast_azimuth_deg: float = 45.0,
        delay_s: float = 0.149,
        lanczos_order: Optional[int] = None,
    ) -> None:
        self._sampling_rate = float(sampling_rate)
        self._fast_az = float(fast_azimuth_deg)
        self._delay = float(delay_s)
        self._order = int(lanczos_order if lanczos_order is not None
                          else self.DEFAULT_LANCZOS_ORDER)

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        *,
        shear_wave_splitting: Optional[float] = None,
        **_ignored,
    ) -> dict:
        if shear_wave_splitting is None or float(shear_wave_splitting) == 0.0:
            return seismograms_map
        delay_samples = (float(shear_wave_splitting) * self._delay
                         * self._sampling_rate)
        phi = np.deg2rad(self._fast_az)
        c, s = np.cos(phi), np.sin(phi)
        result = {}
        for station, components in seismograms_map.items():
            if "E" not in components or "N" not in components:
                result[station] = dict(components)
                continue
            north = np.asarray(components["N"], float)
            east = np.asarray(components["E"], float)
            fast = c * north + s * east
            slow = -s * north + c * east
            slow = _apply_lanczos_shift_batch(slow[None, :], delay_samples,
                                              self._order)[0]
            new = dict(components)
            new["N"] = c * fast - s * slow
            new["E"] = s * fast + c * slow
            result[station] = new
        return result
