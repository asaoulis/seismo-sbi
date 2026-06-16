"""Unified earthquake-catalogue loader for catalogue-driven priors.

Loads either a Lomax-style CSV (as in
``scripts/santorini_pathbreaker/catalogue/Santorini_catalog.csv``) or any
obspy-readable catalogue (QuakeML, etc.) into a flat :class:`EventCatalogue` of
numpy arrays.  Using obspy for the non-CSV path lets us ingest standard catalogue
formats transparently (matching ``scripts/build_catalogue.py``).
"""
from __future__ import annotations

import csv
import datetime as _dt
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
    time:
        Optional per-event origin times as ``numpy.datetime64[us]``, shape
        ``(N,)`` or ``None`` when the source does not provide them. Populated by
        both CSV paths (built from the date columns) and the obspy path.
    """

    latitude: np.ndarray
    longitude: np.ndarray
    depth: np.ndarray
    magnitude: np.ndarray
    magnitude_type: np.ndarray
    err_h: Optional[np.ndarray] = None
    err_z: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None

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
        time=None if cat.time is None else cat.time[mask],
    )


def _load_csv(path: Path) -> EventCatalogue:
    """Parse a catalogue CSV, dispatching on the column layout.

    Two layouts are supported:

    * **Lomax / legacy** (``Santorini_catalog.csv``) — explicit
      ``latitude,longitude,depth,magnitude,magnitude_type`` columns, a split
      ``year,month,day,hour,minute,seconds`` time, and ``ErrH``/``Errz`` errors.
    * **NLL-SC** (``..._NLL-SC_se4.csv``) — a single ISO ``date-time`` column,
      magnitudes in ``Mamp`` (amplitude Ml, used here) / ``Mdur`` (duration Ml),
      ``errH``/``errZ`` errors, and leading whitespace on every header name.
    """
    with open(path) as fh:
        reader = csv.DictReader(fh)
        # Header names in the NLL-SC export carry leading spaces; normalise both
        # the keys and (lazily, at access time) the values.
        rows = [{(k.strip() if k else k): v for k, v in r.items()} for r in reader]
    if not rows:
        raise ValueError(f"CSV catalogue {path} is empty")

    def col(name):
        return np.array([float(r[name]) for r in rows], dtype=float)

    fields = set(rows[0].keys())
    if "date-time" in fields and "Mamp" in fields:
        return _load_csv_nllsc(rows, col)
    return _load_csv_lomax(rows, col, fields)


def _load_csv_lomax(rows, col, fields) -> EventCatalogue:
    err_h = col("ErrH") if "ErrH" in fields else None
    err_z = col("Errz") if "Errz" in fields else None
    mag_type = np.array(
        [r.get("magnitude_type", "") for r in rows], dtype=object
    )
    time = None
    if {"year", "month", "day", "hour", "minute", "seconds"} <= fields:
        time = np.array(
            [
                np.datetime64(
                    _dt.datetime(
                        int(r["year"]), int(r["month"]), int(r["day"]),
                        int(r["hour"]), int(r["minute"]),
                    )
                    + _dt.timedelta(seconds=float(r["seconds"]))
                )
                for r in rows
            ],
            dtype="datetime64[us]",
        )
    return EventCatalogue(
        latitude=col("latitude"),
        longitude=col("longitude"),
        depth=col("depth"),  # CSV depth already in km
        magnitude=col("magnitude"),
        magnitude_type=mag_type,
        err_h=err_h,
        err_z=err_z,
        time=time,
    )


def _load_csv_nllsc(rows, col) -> EventCatalogue:
    """Load the NLL-SC export: ISO ``date-time`` + ``Mamp`` magnitude."""
    time = np.array(
        [np.datetime64(r["date-time"].strip()) for r in rows],
        dtype="datetime64[us]",
    )
    return EventCatalogue(
        latitude=col("latitude"),
        longitude=col("longitude"),
        depth=col("depth"),  # km
        magnitude=col("Mamp"),  # amplitude Ml; closest analogue to legacy Ml
        magnitude_type=np.array(["Ml"] * len(rows), dtype=object),
        err_h=col("errH"),
        err_z=col("errZ"),
        time=time,
    )


def _load_obspy(path: Path) -> EventCatalogue:
    import obspy  # local import: obspy is heavy and only needed for non-CSV catalogues

    events = obspy.read_events(str(path))
    lats, lons, depths, mags, mag_types, times = [], [], [], [], [], []
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
        times.append(
            np.datetime64(origin.time.datetime) if origin.time is not None
            else np.datetime64("NaT")
        )

    return EventCatalogue(
        latitude=np.array(lats, dtype=float),
        longitude=np.array(lons, dtype=float),
        depth=np.array(depths, dtype=float),
        magnitude=np.array(mags, dtype=float),
        magnitude_type=np.array(mag_types, dtype=object),
        err_h=None,
        err_z=None,
        time=np.array(times, dtype="datetime64[us]"),
    )
