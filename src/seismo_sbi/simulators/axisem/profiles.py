"""Collapse a three-dimensional tomography model to one-dimensional profiles along corridors.

A corridor is the buffered polyline between two points, usually a source and a station. Its
profile is the slowness mean of the model cells inside it at each depth, taken over cells where
both wavespeeds are present so the ratio stays coherent, with the gaps interpolated in slowness
and the ends filled from a reference profile. The ensemble of corridors is then the spread an
inference carries as theory error, and each member can be spliced onto a deep reference and
written as a background model.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

#: Provenance recorded per depth: where that depth's wavespeeds came from.
SOURCE_CODES = {"corridor": 0, "interp": 1, "reference": 2, "empty": 3}
SOURCE_NAMES = {code: name for name, code in SOURCE_CODES.items()}

#: Coordinate reference system the model's latitudes and longitudes are in.
EPSG_LATLON = 4326


@dataclass
class Model3D:
    """The three-dimensional Vp/Vs model on its regular (depth, latitude, longitude) grid.

    ``vp`` and ``vs`` are ``(n_depths, n_latitudes, n_longitudes)`` in km/s, ``depth`` in km
    and ``lat``/``lon`` in degrees, all ascending. ``easting`` and ``northing`` are the
    file's own projected coordinates, kept only as a cross-check: path geometry uses the
    consistent frame this class builds by transforming the grid nodes into ``utm_epsg``.
    """

    vp: np.ndarray
    vs: np.ndarray
    depth: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    easting: np.ndarray
    northing: np.ndarray
    utm_epsg: int
    attrs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.to_utm = Transformer.from_crs(EPSG_LATLON, self.utm_epsg, always_xy=True)
        self.to_ll = Transformer.from_crs(self.utm_epsg, EPSG_LATLON, always_xy=True)
        lon2d, lat2d = np.meshgrid(self.lon, self.lat)
        gx, gy = self.to_utm.transform(lon2d, lat2d)
        self.grid_x = gx
        self.grid_y = gy
        self.cell_xy = np.column_stack([gx.ravel(), gy.ravel()])
        self.any_finite = np.isfinite(self.vp).any(axis=0) | np.isfinite(self.vs).any(axis=0)
        self.n_finite_depths = (np.isfinite(self.vp) | np.isfinite(self.vs)).sum(axis=0)
        self.lon_min, self.lon_max = float(self.lon.min()), float(self.lon.max())
        self.lat_min, self.lat_max = float(self.lat.min()), float(self.lat.max())

    @property
    def extent(self):
        """``[lon_min, lon_max, lat_min, lat_max]`` in degrees, for a map axis."""
        return [self.lon_min, self.lon_max, self.lat_min, self.lat_max]

    def lonlat_to_utm(self, lon, lat):
        return self.to_utm.transform(lon, lat)

    def utm_to_lonlat(self, x, y):
        return self.to_ll.transform(x, y)

    def domain_contains(self, lon, lat) -> bool:
        return (self.lon_min <= lon <= self.lon_max) and (self.lat_min <= lat <= self.lat_max)

    def clip_segment_to_domain(self, lon0, lat0, lon1, lat1):
        """Liang-Barsky clip of the segment to the model's bounding box.

        Returns ``(lon0, lat0, lon1, lat1, clipped)`` in degrees, or ``None`` when the whole
        segment lies outside.
        """
        dx, dy = lon1 - lon0, lat1 - lat0
        p = [-dx, dx, -dy, dy]
        q = [lon0 - self.lon_min, self.lon_max - lon0, lat0 - self.lat_min, self.lat_max - lat0]
        u0, u1 = 0.0, 1.0
        for pi, qi in zip(p, q):
            if pi == 0:
                if qi < 0:
                    return None
            else:
                t = qi / pi
                if pi < 0:
                    u0 = max(u0, t)
                else:
                    u1 = min(u1, t)
        if u0 > u1:
            return None
        clon0, clat0 = lon0 + u0 * dx, lat0 + u0 * dy
        clon1, clat1 = lon0 + u1 * dx, lat0 + u1 * dy
        clipped = (u0 > 1e-9) or (u1 < 1 - 1e-9)
        return clon0, clat0, clon1, clat1, clipped


@dataclass
class CorridorPath:
    """One corridor: its endpoints and the along-path columns the average is taken over.

    ``p0_lonlat`` and ``p1_lonlat`` are in degrees, ``sample_lonlat`` and ``sample_xy`` are
    ``(n_samples, 2)`` in degrees and in projected metres, ``corridor_halfwidth_m`` is the
    buffer half-width in m and ``length_km`` the path length in km.
    """

    path_id: str
    mode: str
    station: Optional[str]
    seed: int
    p0_lonlat: tuple
    p1_lonlat: tuple
    sample_lonlat: np.ndarray
    sample_xy: np.ndarray
    corridor_halfwidth_m: float
    clipped: bool = False
    length_km: float = 0.0


def sample_polyline(model: Model3D, p0_ll, p1_ll, spacing_m, halfwidth_m):
    """``(sample_lonlat, sample_xy, length_km)`` for the straight path from ``p0_ll`` to ``p1_ll``.

    The endpoints are in degrees; the columns are spaced every ``spacing_m`` in projected metres.
    """
    x0, y0 = model.lonlat_to_utm(*p0_ll)
    x1, y1 = model.lonlat_to_utm(*p1_ll)
    length = float(np.hypot(x1 - x0, y1 - y0))
    n = max(2, int(np.ceil(length / spacing_m)) + 1)
    t = np.linspace(0.0, 1.0, n)
    xs = x0 + t * (x1 - x0)
    ys = y0 + t * (y1 - y0)
    lons, lats = model.utm_to_lonlat(xs, ys)
    sample_xy = np.column_stack([xs, ys])
    sample_lonlat = np.column_stack([lons, lats])
    return sample_lonlat, sample_xy, length / 1000.0


def corridor_mask(model: Model3D, path: CorridorPath) -> np.ndarray:
    """``(n_latitudes, n_longitudes)`` mask of cells within the corridor half-width of the path.

    A point-cloud buffer: a cell is in when its nearest along-path column is close enough. With
    the column spacing well below the half-width this is a faithful buffered polyline.
    """
    tree = cKDTree(path.sample_xy)
    dist, _ = tree.query(model.cell_xy, k=1)
    return (dist <= path.corridor_halfwidth_m).reshape(model.vp.shape[1:])


def extract_profile(model: Model3D, path: CorridorPath, vp_ref, vs_ref) -> dict:
    """Collapse one corridor to Vp and Vs against depth, with coverage and provenance.

    Each depth is the slowness mean ``1 / mean(1 / v)`` over the cells where Vp and Vs are both
    present, which is the travel-time-preserving collapse and keeps the ratio internally
    coherent: a ratio of means over different cell populations can describe a pair of
    wavespeeds that never coexisted. Empty depths are filled from ``vp_ref`` and ``vs_ref``,
    both in km/s on ``model.depth``, or interpolated; see :func:`fill_ladder_joint`.
    """
    mask = corridor_mask(model, path)
    nz = model.vp.shape[0]
    vp = np.full(nz, np.nan)
    vs = np.full(nz, np.nan)
    cov_vp = np.zeros(nz)
    cov_vs = np.zeros(nz)
    cov_both = np.zeros(nz)
    n_cells = int(mask.sum())
    for k in range(nz):
        if n_cells == 0:
            break
        vvp = model.vp[k][mask]
        vvs = model.vs[k][mask]
        fvp = np.isfinite(vvp)
        fvs = np.isfinite(vvs)
        both = fvp & fvs
        cov_vp[k] = fvp.mean()
        cov_vs[k] = fvs.mean()
        cov_both[k] = both.mean()
        if both.any():
            vp[k] = 1.0 / np.mean(1.0 / vvp[both])
            vs[k] = 1.0 / np.mean(1.0 / vvs[both])

    src = fill_ladder_joint(vp, vs, model.depth, vp_ref, vs_ref)
    return {
        "vp": vp, "vs": vs, "coverage_vp": cov_vp, "coverage_vs": cov_vs, "coverage_both": cov_both,
        "source_vp": src, "source_vs": src, "n_corridor_cells": n_cells,
    }


def fill_ladder_joint(vp, vs, depth, vp_ref, vs_ref) -> np.ndarray:
    """Fill the missing depths of the pair ``(vp, vs)`` in place; returns the provenance codes.

    A depth counts as corridor-sourced only where both wavespeeds are present, so neither ever
    carries provenance the other does not. Gaps between covered depths are interpolated in
    slowness, which is travel-time consistent; the ends fall back to the reference profile.
    """
    nz = len(vp)
    src = np.full(nz, SOURCE_CODES["empty"], dtype=int)
    finite = np.isfinite(vp) & np.isfinite(vs)
    src[finite] = SOURCE_CODES["corridor"]
    if not finite.any():
        vp[:] = vp_ref
        vs[:] = vs_ref
        src[:] = SOURCE_CODES["reference"]
        return src
    kmin, kmax = np.argmax(finite), nz - 1 - np.argmax(finite[::-1])
    interior = np.zeros(nz, dtype=bool)
    interior[kmin:kmax + 1] = True
    gap = interior & ~finite
    if gap.any():
        vp[gap] = 1.0 / np.interp(depth[gap], depth[finite], 1.0 / vp[finite])
        vs[gap] = 1.0 / np.interp(depth[gap], depth[finite], 1.0 / vs[finite])
        src[gap] = SOURCE_CODES["interp"]
    ends = ~interior
    if ends.any():
        vp[ends] = vp_ref[ends]
        vs[ends] = vs_ref[ends]
        src[ends] = SOURCE_CODES["reference"]
    return src


def build_profile_ensemble(model: Model3D, paths, vp_ref, vs_ref):
    """Extract every corridor into stacked ``(n_paths, n_depths)`` arrays plus one record each.

    ``VP`` and ``VS`` are in km/s, ``COVP`` and ``COVS`` the covered fraction of corridor cells,
    and ``SRCP`` and ``SRCS`` the provenance codes of :data:`SOURCE_CODES`.
    """
    nz = model.vp.shape[0]
    n = len(paths)
    VP = np.full((n, nz), np.nan)
    VS = np.full((n, nz), np.nan)
    COVP = np.zeros((n, nz))
    COVS = np.zeros((n, nz))
    SRCP = np.zeros((n, nz), dtype=int)
    SRCS = np.zeros((n, nz), dtype=int)
    records = []
    for i, path in enumerate(paths):
        pr = extract_profile(model, path, vp_ref, vs_ref)
        VP[i], VS[i] = pr["vp"], pr["vs"]
        COVP[i], COVS[i] = pr["coverage_vp"], pr["coverage_vs"]
        SRCP[i], SRCS[i] = pr["source_vp"], pr["source_vs"]
        srchist = {name: int((pr["source_vp"] == code).sum()) for name, code in SOURCE_CODES.items()}
        records.append({
            "path_id": path.path_id, "mode": path.mode, "station": path.station,
            "seed": int(path.seed), "p0_lonlat": [round(c, 5) for c in path.p0_lonlat],
            "p1_lonlat": [round(c, 5) for c in path.p1_lonlat],
            "length_km": round(path.length_km, 2), "clipped": bool(path.clipped),
            "corridor_halfwidth_km": path.corridor_halfwidth_m / 1000.0,
            "n_corridor_cells": pr["n_corridor_cells"],
            "mean_coverage_vp": round(float(pr["coverage_vp"].mean()), 4),
            "frac_depth_reference_vp": round(srchist["reference"] / nz, 3),
            "source_hist_vp": srchist,
        })
    return {"VP": VP, "VS": VS, "COVP": COVP, "COVS": COVS, "SRCP": SRCP, "SRCS": SRCS,
            "records": records}
