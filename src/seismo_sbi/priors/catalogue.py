"""Unified earthquake-catalogue loader for catalogue-driven priors.

Loads either a Lomax-style CSV (as in
``scripts/santorini_pathbreaker/catalogue/Santorini_catalog.csv``) or any
obspy-readable catalogue (QuakeML, etc.) into a flat :class:`EventCatalogue` of
numpy arrays.  Using obspy for the non-CSV path lets us ingest standard catalogue
formats transparently (matching ``scripts/build_catalogue.py``).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class EventCatalogue:
    """Projection-free view of a seismic catalogue used by the priors.

    Attributes
    ----------
    latitude, longitude:
        Degrees, shape ``(N,)``.
    depth:
        Kilometres (positive down), shape ``(N,)``.
    magnitude:
        Event magnitudes, shape ``(N,)``.
    magnitude_type:
        Per-event magnitude type strings (e.g. ``"Ml"``), shape ``(N,)``.
    err_h, err_z:
        Optional horizontal / vertical location errors in km, shape ``(N,)`` or
        ``None`` when the source does not provide them.
    """

    latitude: np.ndarray
    longitude: np.ndarray
    depth: np.ndarray
    magnitude: np.ndarray
    magnitude_type: np.ndarray
    err_h: Optional[np.ndarray] = None
    err_z: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.latitude.shape[0])

    @property
    def lat_lon_depth(self) -> np.ndarray:
        """``(N, 3)`` array of ``[latitude, longitude, depth_km]``."""
        return np.column_stack([self.latitude, self.longitude, self.depth])


def load_catalogue(path, *, magnitude_type: Optional[str] = None) -> EventCatalogue:
    """Load a catalogue from a CSV or any obspy-readable file.

    Dispatch is by file extension: ``.csv`` is parsed directly (and reads the
    ``ErrH``/``Errz`` columns if present); anything else is read with
    :func:`obspy.read_events`.

    Parameters
    ----------
    path:
        Path to the catalogue file.
    magnitude_type:
        If given, keep only events whose magnitude type matches (e.g. ``"Ml"``).
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        catalogue = _load_csv(path)
    else:
        catalogue = _load_obspy(path)

    if magnitude_type is not None:
        mask = catalogue.magnitude_type == magnitude_type
        catalogue = _filter(catalogue, mask)

    if len(catalogue) == 0:
        raise ValueError(f"Catalogue {path} contained no usable events")
    return catalogue


def _filter(cat: EventCatalogue, mask: np.ndarray) -> EventCatalogue:
    return EventCatalogue(
        latitude=cat.latitude[mask],
        longitude=cat.longitude[mask],
        depth=cat.depth[mask],
        magnitude=cat.magnitude[mask],
        magnitude_type=cat.magnitude_type[mask],
        err_h=None if cat.err_h is None else cat.err_h[mask],
        err_z=None if cat.err_z is None else cat.err_z[mask],
    )


def _load_csv(path: Path) -> EventCatalogue:
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise ValueError(f"CSV catalogue {path} is empty")

    def col(name):
        return np.array([float(r[name]) for r in rows], dtype=float)

    fields = rows[0].keys()
    err_h = col("ErrH") if "ErrH" in fields else None
    err_z = col("Errz") if "Errz" in fields else None
    mag_type = np.array(
        [r.get("magnitude_type", "") for r in rows], dtype=object
    )
    return EventCatalogue(
        latitude=col("latitude"),
        longitude=col("longitude"),
        depth=col("depth"),  # CSV depth already in km
        magnitude=col("magnitude"),
        magnitude_type=mag_type,
        err_h=err_h,
        err_z=err_z,
    )


def _load_obspy(path: Path) -> EventCatalogue:
    import obspy  # local import: obspy is heavy and only needed for non-CSV catalogues

    events = obspy.read_events(str(path))
    lats, lons, depths, mags, mag_types = [], [], [], [], []
    for ev in events:
        origin = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
        magnitude = ev.preferred_magnitude() or (
            ev.magnitudes[0] if ev.magnitudes else None
        )
        if origin is None or magnitude is None:
            continue
        lats.append(origin.latitude)
        lons.append(origin.longitude)
        # obspy depth is metres; the prior works in km.
        depths.append((origin.depth or 0.0) / 1000.0)
        mags.append(magnitude.mag)
        mag_types.append(magnitude.magnitude_type or "")

    return EventCatalogue(
        latitude=np.array(lats, dtype=float),
        longitude=np.array(lons, dtype=float),
        depth=np.array(depths, dtype=float),
        magnitude=np.array(mags, dtype=float),
        magnitude_type=np.array(mag_types, dtype=object),
        err_h=None,
        err_z=None,
    )
