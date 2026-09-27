"""File I/O for observed waveforms.

:func:`find_mseed_files` and :func:`load_waveforms` read miniSEED, :func:`load_inventory` reads
the station responses, and :func:`write_window` writes a cut window back to miniSEED.
"""

from pathlib import Path
from typing import Iterable, List

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
    """The mseed files of one station between two times, in the download layout
    ``{data_dir}/{station}/{year}.{jday}/{net}.{sta}.{loc}.{cha}.{year}.{jday}.mseed``.

    :param data_dir: root data directory.
    :param station: station code.
    :param t_start: window start, ``datetime`` or ``UTCDateTime``.
    :param t_end: window end.
    :param network: network code or glob (default ``'*'``).
    :param channel_glob: channel glob (default ``'BH?'``).
    :returns: sorted paths; empty when nothing matches.
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
    """One merged ``Stream`` from several mseed files.

    :param mseed_paths: paths of the files.
    :param starttime: optional trim start, ``UTCDateTime`` or ``datetime``.
    :param endtime: optional trim end.
    :returns: the merged ``Stream``, trimmed to ``[starttime, endtime]`` when given.
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
    """One ``Inventory`` from every ``*.xml`` StationXML file under ``resp_dir``, searched recursively."""
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
