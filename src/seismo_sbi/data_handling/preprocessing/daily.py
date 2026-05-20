"""Process raw mseed data in daily chunks for efficient catalogue generation.

The daily-processing pattern:
  1. Process each (station, calendar-day) pair once: remove instrument response,
     bandpass filter, resample → save as a processed daily mseed file.
  2. Slice windows from the processed daily files (no further processing needed).

Benefits vs per-window processing:
  - Processing cost paid once per station-day instead of once per window.
  - Tapering artefacts (required before response removal) affect only the very
    edges of each day file; interior time windows are completely clean.
  - Processed daily files are resumable: existing files are skipped.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Dict, List, Optional

import joblib
from obspy import UTCDateTime

from .io import find_mseed_files, load_waveforms, load_inventory
from .processing import deconvolve_and_filter


def process_daily_files(
    data_dir: Path,
    station_networks: Dict[str, str],
    processed_dir: Path,
    t_start,
    t_end,
    stationxml_dir: Optional[Path] = None,
    prefilter_kwargs: Optional[dict] = None,
    filter_kwargs: Optional[dict] = None,
    sampling_rate: Optional[float] = None,
    channel_glob: str = "BH?",
    n_jobs: int = 1,
    overwrite: bool = False,
) -> List[Path]:
    """Process raw mseed into processed daily files, parallelised over (station, day).

    For each calendar day in [t_start, t_end] and each station in
    *station_networks*, this function:
      1. Finds the raw mseed files for that station-day.
      2. Removes instrument response (if stationxml_dir is given), applies
         bandpass filter, resamples to *sampling_rate*.
      3. Writes the result as a daily mseed file into *processed_dir* using
         the same ``{station}/{YYYY.DDD}/`` directory layout as raw data so
         that ``find_mseed_files`` works transparently on processed_dir.

    Existing files are silently skipped (resumable). Pass ``overwrite=True``
    to force reprocessing.

    Args:
        data_dir: Root raw mseed directory.
        station_networks: ``{station_code: network_code}`` mapping.
        processed_dir: Output directory for processed daily files.
        t_start: Start of the range to process (datetime or UTCDateTime).
        t_end: End of the range to process.
        stationxml_dir: Directory with StationXML response files (or None to
            skip response removal).
        prefilter_kwargs: Passed to ``deconvolve_and_filter``.
        filter_kwargs: Passed to ``deconvolve_and_filter``.
        sampling_rate: Target sampling rate (Hz); if None, keep original rate.
        channel_glob: Channel glob pattern (default ``'BH?'``).
        n_jobs: Number of parallel workers.
        overwrite: If True, reprocess even if the output file already exists.

    Returns:
        Flat list of Paths to successfully written processed mseed files.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    t0 = UTCDateTime(t_start)
    t1 = UTCDateTime(t_end)

    # Enumerate unique (station, network, date) tasks over the date range
    tasks: list = []
    seen: set = set()
    current = t0
    while current <= t1 + 86400:
        date = _dt.date(current.year, current.month, current.day)
        if date not in seen:
            seen.add(date)
            for station, network in station_networks.items():
                tasks.append((station, network, date))
        current += 86400

    results = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
        joblib.delayed(_process_one_station_day)(
            data_dir=data_dir,
            station=station,
            network=network,
            date=date,
            stationxml_dir=stationxml_dir,
            processed_dir=processed_dir,
            prefilter_kwargs=prefilter_kwargs,
            filter_kwargs=filter_kwargs,
            sampling_rate=sampling_rate,
            channel_glob=channel_glob,
            overwrite=overwrite,
        )
        for station, network, date in tasks
    )

    return [p for paths in results if paths for p in paths]


def _process_one_station_day(
    data_dir: Path,
    station: str,
    network: str,
    date: _dt.date,
    stationxml_dir: Optional[Path],
    processed_dir: Path,
    prefilter_kwargs: Optional[dict],
    filter_kwargs: Optional[dict],
    sampling_rate: Optional[float],
    channel_glob: str,
    overwrite: bool,
) -> List[Path]:
    """Worker: process and save all channels for one station on one calendar day.

    Args identical to ``process_daily_files``.  Returns list of Paths written
    (empty list if no raw data was found or processing failed).
    """
    year = date.year
    jday = date.timetuple().tm_yday

    out_dir = Path(processed_dir) / station / f"{year}.{jday:03d}"

    # Resumability: skip if processed files already exist
    if not overwrite and out_dir.is_dir():
        existing = list(out_dir.glob(f"*.{year}.{jday:03d}.mseed"))
        if existing:
            return existing

    # Locate raw files for this station-day
    t_day_start = UTCDateTime(date)
    t_day_end = t_day_start + 86400
    paths = find_mseed_files(
        data_dir, station, t_day_start, t_day_end,
        network=network, channel_glob=channel_glob,
    )
    if not paths:
        return []

    # Load inventory inside the worker (avoids serialisation issues across processes)
    inventory = None
    remove_response = False
    if stationxml_dir is not None:
        try:
            inventory = load_inventory(Path(stationxml_dir))
            remove_response = True
        except Exception:
            pass

    try:
        st = load_waveforms(paths)
        st = deconvolve_and_filter(
            st,
            inventory=inventory,
            remove_response=remove_response,
            prefilter_kwargs=prefilter_kwargs,
            filter_kwargs=filter_kwargs,
            target_sr=sampling_rate,
        )
    except Exception as exc:
        print(f"  daily: {station} {date} — processing error: {exc}")
        return []

    # Write each trace as a daily mseed file in the standard layout
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for tr in st:
        loc = tr.stats.location  # may be "" for empty location code
        fname = (
            f"{tr.stats.network}.{tr.stats.station}.{loc}."
            f"{tr.stats.channel}.{year}.{jday:03d}.mseed"
        )
        out_path = out_dir / fname
        try:
            tr.write(str(out_path), format="MSEED")
            written.append(out_path)
        except Exception as exc:
            print(f"  daily: {station} {date} {tr.stats.channel} — write error: {exc}")

    return written
