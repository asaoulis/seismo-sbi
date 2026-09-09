"""A UTC day holding SEVERAL per-event windows must still read back the RIGHT one.

Cross-repo contract. The F-net fetcher (personal-page worker/fnet/fetch_fnet.py) writes short
per-event windows into a DAY-granular layout, tagging the filename's location field
(``BO.ABU.w143900.BHZ.2025.002.mseed``) so same-day events cannot overwrite each other. That
tag is only safe if THIS side -- ``find_mseed_files`` + ``load_waveforms`` -- still discovers
the files and selects the window belonging to the event being built.

Without these assertions a naming fix on the writer could look correct while
``build_catalogue.py`` silently stitched the wrong event's waveform into an event h5.
"""
import numpy as np
import obspy
import pytest

from seismo_sbi.data_handling.preprocessing.io import find_mseed_files, load_waveforms

NET, STA, CHA = "BO", "ABU", "BHZ"
DAY = obspy.UTCDateTime(2025, 1, 2)
# Two events on the SAME UTC day, hours apart.
W1 = obspy.UTCDateTime(2025, 1, 2, 3, 15, 0)
W2 = obspy.UTCDateTime(2025, 1, 2, 14, 39, 0)
DUR = 1080.0
SR = 2.0


def _write_window(day_dir, start, marker):
    """One tagged window file whose samples are a constant `marker` (identifiable)."""
    tr = obspy.Trace(np.full(int(DUR * SR), float(marker), dtype=np.float32))
    tr.stats.network, tr.stats.station, tr.stats.channel = NET, STA, CHA
    tr.stats.location = ""
    tr.stats.sampling_rate = SR
    tr.stats.starttime = start
    tag = f"w{start.datetime:%H%M%S}"
    p = day_dir / f"{NET}.{STA}.{tag}.{CHA}.{start.year}.{start.julday:03d}.mseed"
    tr.write(str(p), format="MSEED")
    return p


@pytest.fixture()
def archive(tmp_path):
    day_dir = tmp_path / STA / f"{DAY.year}.{DAY.julday:03d}"
    day_dir.mkdir(parents=True)
    _write_window(day_dir, W1, 11.0)
    _write_window(day_dir, W2, 22.0)
    return tmp_path


def test_find_mseed_files_discovers_every_tagged_window(archive):
    """The unchanged day glob must return BOTH windows (the tag rides the location field)."""
    paths = find_mseed_files(archive, STA, W1, W2 + DUR, network=NET, channel_glob="BH?")
    assert len(paths) == 2, f"expected both tagged windows, got {[p.name for p in paths]}"
    assert {p.name.split(".")[2] for p in paths} == {"w031500", "w143900"}


@pytest.mark.parametrize("start,marker", [(W1, 11.0), (W2, 22.0)])
def test_load_waveforms_returns_only_the_requested_window(archive, start, marker):
    """Given both files, trimming to one event's window must yield ONLY that event's data."""
    paths = find_mseed_files(archive, STA, W1, W2 + DUR, network=NET, channel_glob="BH?")
    st = load_waveforms(paths, starttime=start, endtime=start + DUR)
    assert len(st) >= 1, "requested window produced no data"
    data = np.concatenate([tr.data for tr in st])
    assert np.allclose(data, marker), (
        f"window at {start} returned foreign samples (expected all {marker}, "
        f"got {sorted(set(np.unique(data)))[:5]}) -> the WRONG event was selected")
    for tr in st:
        assert tr.stats.starttime >= start - 1.0
        assert tr.stats.endtime <= start + DUR + 1.0


def test_the_two_windows_are_distinguishable(archive):
    """Guard against the fixture accidentally writing the same payload twice."""
    paths = find_mseed_files(archive, STA, W1, W2 + DUR, network=NET, channel_glob="BH?")
    first = load_waveforms(paths, starttime=W1, endtime=W1 + DUR)
    second = load_waveforms(paths, starttime=W2, endtime=W2 + DUR)
    assert not np.allclose(first[0].data[0], second[0].data[0])
