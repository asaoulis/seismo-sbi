"""Local equirectangular (lat/lon) <-> local-Cartesian (km) conversions.

These are deliberately lightweight, dependency-free approximations valid over the
small spatial extent of a single seismic crisis (tens of km), used by the
catalogue source-location prior to apply a Gaussian perturbation specified in
kilometres.  A reference point fixes the local tangent plane; the longitude scale
uses ``cos(ref_lat)`` so the forward and inverse transforms round-trip exactly for
that reference latitude.
"""
from __future__ import annotations

import numpy as np

#: Mean great-circle kilometres per degree of latitude.
KM_PER_DEG: float = 111.195


def latlon_to_km_offsets(lat_deg, lon_deg, ref_lat_deg, ref_lon_deg):
    """Equirectangular (lat, lon) -> (east_km, north_km) offsets about a reference.

    Parameters
    ----------
    lat_deg, lon_deg:
        Point(s) to convert (degrees). Scalars or arrays.
    ref_lat_deg, ref_lon_deg:
        Reference / origin of the local tangent plane (degrees).

    Returns
    -------
    (dx_km, dy_km):
        East (x) and North (y) offsets in kilometres relative to the reference.
    """
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    dy_km = (lat - ref_lat_deg) * KM_PER_DEG
    dx_km = (lon - ref_lon_deg) * KM_PER_DEG * np.cos(np.radians(ref_lat_deg))
    return dx_km, dy_km


def km_offsets_to_latlon(dx_km, dy_km, ref_lat_deg, ref_lon_deg):
    """Inverse of :func:`latlon_to_km_offsets`.

    Parameters
    ----------
    dx_km, dy_km:
        East (x) and North (y) offsets in kilometres. Scalars or arrays.
    ref_lat_deg, ref_lon_deg:
        Reference / origin of the local tangent plane (degrees).

    Returns
    -------
    (lat_deg, lon_deg):
        Absolute latitude/longitude (degrees).
    """
    dx = np.asarray(dx_km, dtype=float)
    dy = np.asarray(dy_km, dtype=float)
    lat_deg = ref_lat_deg + dy / KM_PER_DEG
    lon_deg = ref_lon_deg + dx / (KM_PER_DEG * np.cos(np.radians(ref_lat_deg)))
    return lat_deg, lon_deg
