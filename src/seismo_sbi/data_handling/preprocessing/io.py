"""File I/O helpers: read waveforms, read inventory, write mseed windows."""

from pathlib import Path
from typing import Iterable, List, Optional
import datetime

import obspy
from obspy import Stream, Inventory, UTCDateTime


def find_mseed_files(
    data_dir: Path,
    station: str,
    t_start,
    t_end,
    network: str = "*",
    channel_glob: str = "BH?",
) -> List[Path]:
    """Find mseed files matching the custom_download.py directory layout.

    Layout: {data_dir}/{station}/{year}.{jday}/{net}.{sta}.{loc}.{cha}.{year}.{jday}.mseed

    Searches all julian-day subdirectories covered by [t_start, t_end] and
    returns every matching mseed file found.

    Args:
        data_dir: Root data directory.
        station: Station code (e.g. 'ANMO').
        t_start: Window start (datetime or UTCDateTime).
        t_end: Window end (datetime or UTCDateTime).
        network: Network code or glob pattern (default '*').
        channel_glob: Channel glob pattern (default 'BH?').

    Returns:
        Sorted list of matching Path objects.  May be empty if no data found.
    """
    t0 = UTCDateTime(t_start)
    t1 = UTCDateTime(t_end)
    paths = []

    # Collect all julian days in [t_start, t_end] (inclusive)
    current = t0
    seen_jdays = set()
    while current <= t1 + 86400:
        year = current.year
        jday = current.julday
        key = (year, jday)
        if key not in seen_jdays:
            seen_jdays.add(key)
            day_dir = Path(data_dir) / station / f"{year}.{jday:03d}"
            if day_dir.is_dir():
                for p in day_dir.glob(f"{network}.{station}.*.{channel_glob}.{year}.{jday:03d}.mseed"):
                    paths.append(p)
        current += 86400  # advance by one day

    return sorted(set(paths))


def load_waveforms(
    mseed_paths: Iterable[Path],
    starttime: UTCDateTime = None,
    endtime: UTCDateTime = None,
) -> Stream:
    """Load waveforms from one or more mseed files into a single Stream.

    Args:
        mseed_paths: Iterable of paths to .mseed files.
        starttime: Optional trim start (UTCDateTime or datetime).
        endtime: Optional trim end.

    Returns:
        Merged Stream containing all traces from the given files,
        trimmed to [starttime, endtime] if provided.
    """
    st = Stream()
    for path in mseed_paths:
        kw = {}
        if starttime is not None:
            kw["starttime"] = UTCDateTime(starttime)
        if endtime is not None:
            kw["endtime"] = UTCDateTime(endtime)
        st += obspy.read(str(path), format="MSEED", check_compression=False, **kw)
    return st


def load_inventory(resp_dir: Path) -> Inventory:
    """Read all StationXML files found under resp_dir into one Inventory.

    Args:
        resp_dir: Directory (searched recursively) containing *.xml files.

    Returns:
        Combined Inventory.
    """
    xml_files = list(Path(resp_dir).rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No StationXML files found under {resp_dir}")

    inv = obspy.read_inventory(str(xml_files[0]))
    for p in xml_files[1:]:
        inv += obspy.read_inventory(str(p))
    return inv


def write_window(stream: Stream, out_path: Path) -> None:
    """Write a Stream to a MiniSEED file, creating parent directories as needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stream.write(str(out_path), format="MSEED")
