"""Unified earthquake-catalogue loader for catalogue-driven priors.

Reads a CSV in one of the named layouts below, or any obspy-readable catalogue (QuakeML and
friends), into a flat :class:`EventCatalogue` of numpy arrays. The CSV layout is named by the
caller through ``csv_format``, or detected from the header when none is given.
"""
from __future__ import annotations

import csv
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

#: CSV layouts this loader understands, named for how each carries the origin time.
CSV_FORMATS = ("split_time", "iso_time")


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


def load_catalogue(path, *, magnitude_type: Optional[str] = None,
                   csv_format: Optional[str] = None) -> EventCatalogue:
    """Load a catalogue from a CSV or any obspy-readable file.

    Dispatch is by file extension: ``.csv`` is parsed here, anything else with
    :func:`obspy.read_events`.

    Parameters
    ----------
    path:
        Path to the catalogue file.
    magnitude_type:
        If given, keep only events whose magnitude type matches (e.g. ``"Ml"``).
    csv_format:
        Which CSV layout to expect, one of :data:`CSV_FORMATS`. ``None`` detects it from the
        header, which is convenient but silent about a file that is neither.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        catalogue = _load_csv(path, csv_format)
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


def _load_csv(path: Path, csv_format: Optional[str]) -> EventCatalogue:
    """Parse a catalogue CSV in the named layout, or in the one its header implies."""
    with open(path) as fh:
        reader = csv.DictReader(fh)
        # Some exports pad every header name with a leading space.
        rows = [{(k.strip() if k else k): v for k, v in r.items()} for r in reader]
    if not rows:
        raise ValueError(f"CSV catalogue {path} is empty")

    def col(name):
        return np.array([float(r[name]) for r in rows], dtype=float)

    fields = set(rows[0].keys())
    if csv_format is None:
        csv_format = "iso_time" if {"date-time", "Mamp"} <= fields else "split_time"
    if csv_format not in CSV_FORMATS:
        raise ValueError(f"unknown csv_format {csv_format!r}; expected one of {CSV_FORMATS}")
    if csv_format == "iso_time":
        return _load_csv_iso_time(rows, col)
    return _load_csv_split_time(rows, col, fields)


def _load_csv_split_time(rows, col, fields) -> EventCatalogue:
    """``latitude,longitude,depth,magnitude,magnitude_type`` with the origin time split across
    ``year,month,day,hour,minute,seconds`` columns and ``ErrH``/``Errz`` location errors."""
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


def _load_csv_iso_time(rows, col) -> EventCatalogue:
    """A single ISO ``date-time`` column, amplitude magnitude ``Mamp`` (``Mdur`` is the
    duration magnitude and is not read), and ``errH``/``errZ`` location errors."""
    time = np.array(
        [np.datetime64(r["date-time"].strip()) for r in rows],
        dtype="datetime64[us]",
    )
    return EventCatalogue(
        latitude=col("latitude"),
        longitude=col("longitude"),
        depth=col("depth"),  # km
        magnitude=col("Mamp"),
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
