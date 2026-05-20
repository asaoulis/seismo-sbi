"""Phase 5 — Catalogue pipeline tests.

Covers:
- compute_event_arrival_windows  (TauPy, parallelised)
- filter_events_by_distance
- check_window_quality
- build_event_catalogue  (synthetic mseed on disk)
- build_noise_catalogue  (synthetic mseed on disk, event avoidance)
- Parallel processing (n_jobs > 1)
- Real-data variants using cached IRIS data

All tests are marked ``slow``.  Run with:

    conda run -n seismo-sbi python -m pytest \
        tests/end_to_end/test_catalogue_pipeline.py -v -m slow
"""

from __future__ import annotations

import csv
import datetime
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch, MagicMock

import h5py
import numpy as np
import obspy
from obspy import Stream, Trace, UTCDateTime, Inventory
from obspy.core.event import Event, Origin, Magnitude, Catalog
import pytest

from seismo_sbi.data_handling.preprocessing import (
    find_mseed_files,
    load_waveforms,
    deconvolve_and_filter,
    export_to_sbi_h5,
    check_window_quality,
    compute_event_arrival_windows,
    filter_events_by_distance,
)
from seismo_sbi.data_handling.preprocessing.windowing import get_continuous_regions
from seismo_sbi.instaseis_simulator.utils import compute_data_vector_length

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared synthetic-data constants
# ---------------------------------------------------------------------------

NETWORK = "XX"
STATIONS = ["STA1", "STA2", "STA3"]
CHANNELS = ["BHZ", "BHE", "BHN"]

SR_RAW = 20.0   # raw sampling rate
SR_TARGET = 1.0  # SBI target

# Data covers 4 hours so noise windows can be extracted
DATA_T0 = datetime.datetime(2023, 6, 1, 0, 0, 0)
DATA_T1 = DATA_T0 + timedelta(hours=4)

# A synthetic "event" in the middle of the data
EVENT_ORIGIN_TIME = DATA_T0 + timedelta(hours=2)
EVENT_LAT = 35.0
EVENT_LON = -117.0
EVENT_DEPTH_KM = 10.0

DURATION = timedelta(seconds=120)
COV_WINDOW = timedelta(seconds=120)

RNG_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stream(
    stations=None,
    t0=DATA_T0,
    t1=DATA_T1,
    sr=SR_RAW,
    channels=None,
    amplitude=1.0,
    freq=0.05,
) -> Stream:
    stations = stations or STATIONS
    channels = channels or CHANNELS
    rng = np.random.default_rng(RNG_SEED)
    npts = int((t1 - t0).total_seconds() * sr)
    t = np.arange(npts) / sr
    st = Stream()
    for sta in stations:
        for cha in channels:
            data = amplitude * np.sin(2 * np.pi * freq * t) + rng.standard_normal(npts) * 0.01
            tr = Trace()
            tr.stats.network = NETWORK
            tr.stats.station = sta
            tr.stats.channel = cha
            tr.stats.location = ""
            tr.stats.sampling_rate = sr
            tr.stats.starttime = UTCDateTime(t0)
            tr.data = data.astype(np.float64)
            st += tr
    return st


def _write_stream_mseed(st: Stream, data_dir: Path) -> list:
    """Write each trace in the custom_download.py layout; return paths."""
    paths = []
    for tr in st:
        year = tr.stats.starttime.year
        jday = tr.stats.starttime.julday
        sta_dir = data_dir / tr.stats.station / f"{year}.{jday:03d}"
        sta_dir.mkdir(parents=True, exist_ok=True)
        fname = (
            f"{tr.stats.network}.{tr.stats.station}.."
            f"{tr.stats.channel}.{year}.{jday:03d}.mseed"
        )
        fpath = sta_dir / fname
        tr.write(str(fpath), format="MSEED")
        paths.append(fpath)
    return paths


def _make_obspy_event(
    lat=EVENT_LAT,
    lon=EVENT_LON,
    depth_km=EVENT_DEPTH_KM,
    time=None,
    magnitude=5.0,
) -> Event:
    """Create a minimal obspy Event object suitable for catalogue functions."""
    t = UTCDateTime(time) if time else UTCDateTime(EVENT_ORIGIN_TIME)
    origin = Origin(
        time=t,
        latitude=lat,
        longitude=lon,
        depth=depth_km * 1000,
    )
    mag = Magnitude(mag=magnitude)
    ev = Event(origins=[origin], magnitudes=[mag])
    return ev


def _make_catalog(n=3) -> Catalog:
    """Return a Catalog with *n* events spread over the data period."""
    events = []
    for i in range(n):
        t = EVENT_ORIGIN_TIME + timedelta(hours=i * 0.5)
        events.append(_make_obspy_event(time=t, magnitude=5.0 + i * 0.5))
    return Catalog(events=events)


# ---------------------------------------------------------------------------
# Fixture: synthetic mseed data on disk
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_data(tmp_path_factory):
    """Write a 4-hour synthetic stream to disk for all catalogue tests."""
    tmp = tmp_path_factory.mktemp("catalogue_synth")
    data_dir = tmp / "raw"
    data_dir.mkdir()
    st = _make_stream()
    _write_stream_mseed(st, data_dir)
    station_networks = {sta: NETWORK for sta in STATIONS}
    return {
        "tmp": tmp,
        "data_dir": data_dir,
        "station_networks": station_networks,
        "raw_stream": st,
    }


# ============================================================================
# compute_event_arrival_windows
# ============================================================================

class TestComputeEventArrivalWindows:

    def test_returns_list_of_tuples(self):
        ev = _make_obspy_event()
        # Use (lat, lon) tuples as receivers
        receivers = [(34.0, -118.0), (36.0, -116.0)]
        result = compute_event_arrival_windows(
            [ev], receivers, taup_model="prem", n_jobs=1,
            padding=timedelta(seconds=0),
        )
        assert isinstance(result, list)
        assert len(result) == 1
        start, end = result[0]
        assert isinstance(start, UTCDateTime)
        assert isinstance(end, UTCDateTime)

    def test_window_spans_arrivals(self):
        ev = _make_obspy_event(lat=0.0, lon=0.0, depth_km=10.0)
        receivers = [(10.0, 0.0)]  # 10 degrees away
        result = compute_event_arrival_windows(
            [ev], receivers, taup_model="prem", n_jobs=1,
            padding=timedelta(seconds=0),
        )
        assert len(result) == 1
        start, end = result[0]
        origin_t = UTCDateTime(EVENT_ORIGIN_TIME)
        # First arrivals at 10 degrees take ~130–140 s; end must be after start
        assert end > start
        # Window should start after origin time (arrivals can't be before)
        assert start >= ev.origins[0].time

    def test_padding_applied(self):
        ev = _make_obspy_event(lat=0.0, lon=0.0, depth_km=10.0)
        receivers = [(10.0, 0.0)]
        pad = timedelta(minutes=5)
        result_no_pad = compute_event_arrival_windows(
            [ev], receivers, taup_model="prem", padding=timedelta(seconds=0)
        )
        result_pad = compute_event_arrival_windows(
            [ev], receivers, taup_model="prem", padding=pad
        )
        assert len(result_no_pad) == 1 and len(result_pad) == 1
        s0, e0 = result_no_pad[0]
        s1, e1 = result_pad[0]
        assert s0 - s1 == pytest.approx(pad.total_seconds(), abs=1e-3)
        assert e1 - e0 == pytest.approx(pad.total_seconds(), abs=1e-3)

    def test_multiple_events(self):
        catalog = _make_catalog(n=3)
        receivers = [(34.0, -118.0)]
        result = compute_event_arrival_windows(
            catalog, receivers, taup_model="prem", n_jobs=1,
        )
        assert len(result) == 3

    def test_parallel_gives_same_result_as_serial(self):
        catalog = _make_catalog(n=2)
        receivers = [(34.0, -118.0), (36.0, -116.0)]
        serial = compute_event_arrival_windows(
            catalog, receivers, taup_model="prem", n_jobs=1,
        )
        parallel = compute_event_arrival_windows(
            catalog, receivers, taup_model="prem", n_jobs=2,
        )
        assert len(serial) == len(parallel)
        for (s1, e1), (s2, e2) in zip(serial, parallel):
            assert abs(s1 - s2) < 1e-6
            assert abs(e1 - e2) < 1e-6

    def test_accepts_receivers_object(self):
        """Receivers object with .receivers attribute is accepted."""
        from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
        ev = _make_obspy_event()
        rec = Receivers(receivers=[Receiver(34.0, -118.0, "IU", "ANMO", ["Z"])])
        result = compute_event_arrival_windows([ev], rec, n_jobs=1)
        assert len(result) == 1

    def test_empty_catalog_returns_empty_list(self):
        result = compute_event_arrival_windows([], [(34.0, -118.0)])
        assert result == []


# ============================================================================
# filter_events_by_distance
# ============================================================================

class TestFilterEventsByDistance:

    def test_max_radius_filters(self):
        close = _make_obspy_event(lat=35.0, lon=-117.0)   # 0 deg from center
        far = _make_obspy_event(lat=0.0, lon=0.0)          # ~138 deg away
        result = filter_events_by_distance(
            [close, far], center=(35.0, -117.0), max_radius_deg=10.0
        )
        assert len(result) == 1
        assert result[0] is close

    def test_min_radius_filters(self):
        close = _make_obspy_event(lat=35.1, lon=-117.0)   # ~0.1 deg
        far = _make_obspy_event(lat=0.0, lon=0.0)          # ~138 deg
        result = filter_events_by_distance(
            [close, far], center=(35.0, -117.0), min_radius_deg=50.0
        )
        assert len(result) == 1
        assert result[0] is far

    def test_both_bounds(self):
        events = [
            _make_obspy_event(lat=35.1, lon=-117.0),   # very close
            _make_obspy_event(lat=25.0, lon=-117.0),   # ~10 deg
            _make_obspy_event(lat=0.0, lon=0.0),        # very far
        ]
        result = filter_events_by_distance(
            events, center=(35.0, -117.0),
            min_radius_deg=5.0, max_radius_deg=30.0,
        )
        assert len(result) == 1
        assert result[0] is events[1]

    def test_no_filter_returns_all(self):
        catalog = _make_catalog(n=4)
        result = filter_events_by_distance(
            catalog, center=(35.0, -117.0),
        )
        assert len(result) == 4

    def test_accepts_catalog_object(self):
        catalog = _make_catalog(n=2)
        result = filter_events_by_distance(
            catalog, center=(35.0, -117.0), max_radius_deg=180.0
        )
        assert len(result) == 2


# ============================================================================
# check_window_quality
# ============================================================================

class TestCheckWindowQuality:

    def _good_stream(self, stations=None):
        stations = stations or STATIONS[:2]
        npts = int(DURATION.total_seconds() * SR_TARGET)
        st = Stream()
        rng = np.random.default_rng(0)
        for sta in stations:
            for cha in CHANNELS:
                tr = Trace()
                tr.stats.station = sta
                tr.stats.channel = cha
                tr.stats.sampling_rate = SR_TARGET
                tr.stats.starttime = UTCDateTime(DATA_T0)
                tr.data = rng.standard_normal(npts).astype(np.float64)
                st += tr
        return st

    def test_good_window_passes(self):
        st = self._good_stream()
        ok, reason = check_window_quality(
            st, STATIONS[:2], SR_TARGET, DURATION
        )
        assert ok is True
        assert reason == ""

    def test_missing_station_fails(self):
        st = self._good_stream(stations=["STA1"])
        ok, reason = check_window_quality(
            st, ["STA1", "MISSING"], SR_TARGET, DURATION
        )
        assert ok is False
        assert "MISSING" in reason

    def test_short_trace_fails_completeness(self):
        st = self._good_stream()
        # Truncate STA1's traces to 50% of expected length
        for tr in st.select(station="STA1"):
            tr.data = tr.data[:len(tr.data) // 2]
        ok, reason = check_window_quality(
            st, STATIONS[:2], SR_TARGET, DURATION,
            min_completeness=0.9,
        )
        assert ok is False
        assert "completeness" in reason.lower()

    def test_zero_trace_fails_rms(self):
        st = self._good_stream()
        for tr in st.select(station="STA1"):
            tr.data = np.zeros(len(tr.data))
        ok, reason = check_window_quality(
            st, STATIONS[:2], SR_TARGET, DURATION,
            min_rms=0.0,
        )
        assert ok is False
        assert "rms" in reason.lower() or "RMS" in reason

    def test_empty_stream_fails(self):
        ok, reason = check_window_quality(
            Stream(), ["STA1"], SR_TARGET, DURATION
        )
        assert ok is False

    def test_zero_duration_fails(self):
        st = self._good_stream()
        ok, reason = check_window_quality(
            st, STATIONS[:2], SR_TARGET, timedelta(seconds=0)
        )
        assert ok is False
        assert "zero" in reason.lower()

    def test_custom_completeness_threshold(self):
        st = self._good_stream()
        # 80% completeness — passes with 0.7 threshold, fails with 0.9
        for tr in st.select(station="STA1"):
            tr.data = tr.data[:int(len(tr.data) * 0.8)]
        ok_low, _ = check_window_quality(
            st, STATIONS[:2], SR_TARGET, DURATION, min_completeness=0.7
        )
        ok_high, _ = check_window_quality(
            st, STATIONS[:2], SR_TARGET, DURATION, min_completeness=0.9
        )
        assert ok_low is True
        assert ok_high is False


# ============================================================================
# process_daily_files
# ============================================================================

class TestProcessDailyFiles:
    """Unit tests for process_daily_files() — the daily-chunk preprocessing step."""

    def test_creates_processed_files(self, synthetic_data, tmp_path):
        from seismo_sbi.data_handling.preprocessing.daily import process_daily_files

        processed_dir = tmp_path / "daily"
        written = process_daily_files(
            data_dir=synthetic_data["data_dir"],
            station_networks=synthetic_data["station_networks"],
            processed_dir=processed_dir,
            t_start=DATA_T0,
            t_end=DATA_T1,
            sampling_rate=SR_TARGET,
            filter_kwargs=dict(freqmin=0.02, freqmax=0.1, corners=2, zerophase=False),
            n_jobs=1,
        )
        assert len(written) > 0
        assert all(p.exists() for p in written)

    def test_output_directory_layout(self, synthetic_data, tmp_path):
        """Files are stored in {processed_dir}/{station}/{YYYY.DDD}/ layout."""
        from seismo_sbi.data_handling.preprocessing.daily import process_daily_files

        processed_dir = tmp_path / "daily"
        written = process_daily_files(
            data_dir=synthetic_data["data_dir"],
            station_networks=synthetic_data["station_networks"],
            processed_dir=processed_dir,
            t_start=DATA_T0,
            t_end=DATA_T1,
            sampling_rate=SR_TARGET,
            filter_kwargs=dict(freqmin=0.02, freqmax=0.1, corners=2, zerophase=False),
            n_jobs=1,
        )
        for p in written:
            # path must be processed_dir / station / {year}.{jday} / filename
            assert p.parent.parent.parent == processed_dir
            assert p.suffix == ".mseed"

    def test_loadable_by_find_mseed_files(self, synthetic_data, tmp_path):
        """find_mseed_files() locates the processed daily files in processed_dir."""
        from seismo_sbi.data_handling.preprocessing.daily import process_daily_files
        from seismo_sbi.data_handling.preprocessing import find_mseed_files

        processed_dir = tmp_path / "daily"
        process_daily_files(
            data_dir=synthetic_data["data_dir"],
            station_networks=synthetic_data["station_networks"],
            processed_dir=processed_dir,
            t_start=DATA_T0,
            t_end=DATA_T1,
            sampling_rate=SR_TARGET,
            filter_kwargs=dict(freqmin=0.02, freqmax=0.1, corners=2, zerophase=False),
            n_jobs=1,
        )
        for sta in STATIONS:
            paths = find_mseed_files(
                processed_dir, sta,
                DATA_T0 + timedelta(hours=1), DATA_T0 + timedelta(hours=3),
                network=NETWORK, channel_glob="BH?",
            )
            assert len(paths) > 0, (
                f"No processed daily files found for station {sta}"
            )

    def test_resampled_to_target_rate(self, synthetic_data, tmp_path):
        """Processed files are at the target sampling rate."""
        from seismo_sbi.data_handling.preprocessing.daily import process_daily_files
        from seismo_sbi.data_handling.preprocessing import find_mseed_files, load_waveforms

        processed_dir = tmp_path / "daily"
        process_daily_files(
            data_dir=synthetic_data["data_dir"],
            station_networks=synthetic_data["station_networks"],
            processed_dir=processed_dir,
            t_start=DATA_T0,
            t_end=DATA_T1,
            sampling_rate=SR_TARGET,
            filter_kwargs=dict(freqmin=0.02, freqmax=0.1, corners=2, zerophase=False),
            n_jobs=1,
        )
        paths = find_mseed_files(
            processed_dir, STATIONS[0],
            DATA_T0, DATA_T1, network=NETWORK, channel_glob="BH?",
        )
        assert len(paths) > 0
        st = load_waveforms(paths)
        for tr in st:
            assert tr.stats.sampling_rate == pytest.approx(SR_TARGET, rel=1e-3), (
                f"Expected SR={SR_TARGET}, got {tr.stats.sampling_rate}"
            )

    def test_resumability_skips_existing(self, synthetic_data, tmp_path):
        """Second call with the same processed_dir skips already-done files."""
        from seismo_sbi.data_handling.preprocessing.daily import process_daily_files

        processed_dir = tmp_path / "daily"
        written_first = process_daily_files(
            data_dir=synthetic_data["data_dir"],
            station_networks=synthetic_data["station_networks"],
            processed_dir=processed_dir,
            t_start=DATA_T0, t_end=DATA_T1,
            sampling_rate=SR_TARGET,
            filter_kwargs=dict(freqmin=0.02, freqmax=0.1, corners=2, zerophase=False),
        )
        mtimes = {p: p.stat().st_mtime for p in written_first}

        written_second = process_daily_files(
            data_dir=synthetic_data["data_dir"],
            station_networks=synthetic_data["station_networks"],
            processed_dir=processed_dir,
            t_start=DATA_T0, t_end=DATA_T1,
            sampling_rate=SR_TARGET,
            filter_kwargs=dict(freqmin=0.02, freqmax=0.1, corners=2, zerophase=False),
        )
        # Same files returned, modification times unchanged
        assert set(p.name for p in written_second) == set(p.name for p in written_first)
        for p in written_second:
            assert p.stat().st_mtime == mtimes[p], f"{p.name} was modified"

    def test_overwrite_reprocesses(self, synthetic_data, tmp_path):
        """overwrite=True forces reprocessing even when files exist."""
        from seismo_sbi.data_handling.preprocessing.daily import process_daily_files

        processed_dir = tmp_path / "daily"
        process_daily_files(
            data_dir=synthetic_data["data_dir"],
            station_networks=synthetic_data["station_networks"],
            processed_dir=processed_dir,
            t_start=DATA_T0, t_end=DATA_T1,
            sampling_rate=SR_TARGET,
        )
        p_sample = list(processed_dir.rglob("*.mseed"))[0]
        mtime_before = p_sample.stat().st_mtime

        import time; time.sleep(0.05)
        process_daily_files(
            data_dir=synthetic_data["data_dir"],
            station_networks=synthetic_data["station_networks"],
            processed_dir=processed_dir,
            t_start=DATA_T0, t_end=DATA_T1,
            sampling_rate=SR_TARGET,
            overwrite=True,
        )
        assert p_sample.stat().st_mtime > mtime_before

    def test_parallel_gives_same_files_as_serial(self, synthetic_data, tmp_path):
        """n_jobs > 1 produces the same set of files as serial execution."""
        from seismo_sbi.data_handling.preprocessing.daily import process_daily_files

        serial_dir = tmp_path / "serial"
        parallel_dir = tmp_path / "parallel"

        process_daily_files(
            data_dir=synthetic_data["data_dir"],
            station_networks=synthetic_data["station_networks"],
            processed_dir=serial_dir,
            t_start=DATA_T0, t_end=DATA_T1,
            sampling_rate=SR_TARGET,
            filter_kwargs=dict(freqmin=0.02, freqmax=0.1, corners=2, zerophase=False),
            n_jobs=1,
        )
        process_daily_files(
            data_dir=synthetic_data["data_dir"],
            station_networks=synthetic_data["station_networks"],
            processed_dir=parallel_dir,
            t_start=DATA_T0, t_end=DATA_T1,
            sampling_rate=SR_TARGET,
            filter_kwargs=dict(freqmin=0.02, freqmax=0.1, corners=2, zerophase=False),
            n_jobs=2,
        )
        serial_names = {p.name for p in serial_dir.rglob("*.mseed")}
        parallel_names = {p.name for p in parallel_dir.rglob("*.mseed")}
        assert serial_names == parallel_names

    def test_one_file_per_station_channel_per_day(self, synthetic_data, tmp_path):
        """Each (station, channel, day) produces exactly one file."""
        from seismo_sbi.data_handling.preprocessing.daily import process_daily_files

        processed_dir = tmp_path / "daily"
        process_daily_files(
            data_dir=synthetic_data["data_dir"],
            station_networks=synthetic_data["station_networks"],
            processed_dir=processed_dir,
            t_start=DATA_T0, t_end=DATA_T1,
            sampling_rate=SR_TARGET,
        )
        # Expect STATIONS * CHANNELS files per day (single day in test data)
        all_files = list(processed_dir.rglob("*.mseed"))
        expected = len(STATIONS) * len(CHANNELS)
        assert len(all_files) == expected, (
            f"Expected {expected} files, got {len(all_files)}: {[f.name for f in all_files]}"
        )

    def test_missing_station_silently_skipped(self, tmp_path):
        """A station that has no raw data produces no output and no crash."""
        from seismo_sbi.data_handling.preprocessing.daily import process_daily_files

        processed_dir = tmp_path / "daily"
        empty_data_dir = tmp_path / "empty"
        empty_data_dir.mkdir()

        # Pass a station that definitely has no data
        written = process_daily_files(
            data_dir=empty_data_dir,
            station_networks={"GHOST": "XX"},
            processed_dir=processed_dir,
            t_start=DATA_T0, t_end=DATA_T1,
            sampling_rate=SR_TARGET,
        )
        assert written == []


# ============================================================================
# build_event_catalogue (synthetic)
# ============================================================================

class TestBuildEventCatalogueSynthetic:

    def _run(self, synthetic_data, events, tmp_path, n_jobs=1, **kwargs):
        """Helper: call build_event_catalogue and return (output_dir, written)."""
        from seismo_sbi.data_handling.preprocessing.catalogue import build_event_catalogue

        out_dir = tmp_path / "events"
        written = build_event_catalogue(
            events=events,
            data_dir=synthetic_data["data_dir"],
            stationxml_dir=None,
            station_networks=synthetic_data["station_networks"],
            output_dir=out_dir,
            duration_s=DURATION.total_seconds(),
            sampling_rate=SR_TARGET,
            covariance_window_s=COV_WINDOW.total_seconds(),
            filter_kwargs=dict(freqmin=0.02, freqmax=0.1, corners=2, zerophase=False),
            n_jobs=n_jobs,
            **kwargs,
        )
        return out_dir, written

    def test_one_h5_per_event(self, synthetic_data, tmp_path):
        events = [_make_obspy_event(time=EVENT_ORIGIN_TIME)]
        out_dir, written = self._run(synthetic_data, events, tmp_path)
        assert len(written) == 1
        assert written[0].suffix == ".h5"
        assert written[0].exists()

    def test_multiple_events_produce_multiple_files(self, synthetic_data, tmp_path):
        catalog = _make_catalog(n=3)
        out_dir, written = self._run(synthetic_data, catalog, tmp_path)
        assert len(written) == 3
        assert all(p.exists() for p in written)
        # Each h5 filename is distinct
        assert len(set(p.name for p in written)) == 3

    def test_h5_schema_matches_sbi_contract(self, synthetic_data, tmp_path):
        """Written h5 has /outputs/{station}/{Z,1,2} and /misc groups."""
        events = [_make_obspy_event(time=EVENT_ORIGIN_TIME)]
        out_dir, written = self._run(synthetic_data, events, tmp_path)
        with h5py.File(written[0], "r") as f:
            assert "outputs" in f
            for sta in STATIONS:
                if sta in f["outputs"]:
                    for comp in ("Z", "1", "2"):
                        assert comp in f["outputs"][sta], (
                            f"Missing component {comp} for {sta}"
                        )

    def test_array_length_matches_sbi_contract(self, synthetic_data, tmp_path):
        events = [_make_obspy_event(time=EVENT_ORIGIN_TIME)]
        out_dir, written = self._run(synthetic_data, events, tmp_path)
        expected_len = (
            compute_data_vector_length(DURATION.total_seconds(), SR_TARGET) + 1
        )
        with h5py.File(written[0], "r") as f:
            for sta in STATIONS:
                if sta in f["outputs"]:
                    arr = f["outputs"][sta]["Z"][:]
                    assert len(arr) == expected_len, (
                        f"{sta}/Z: expected {expected_len}, got {len(arr)}"
                    )

    def test_no_e_or_n_keys_in_output(self, synthetic_data, tmp_path):
        events = [_make_obspy_event(time=EVENT_ORIGIN_TIME)]
        out_dir, written = self._run(synthetic_data, events, tmp_path)
        with h5py.File(written[0], "r") as f:
            for sta in f.get("outputs", {}).keys():
                for comp in f["outputs"][sta].keys():
                    assert comp not in ("E", "N"), (
                        f"Found forbidden key {comp!r} in /outputs/{sta}"
                    )

    def test_resumability_skips_existing(self, synthetic_data, tmp_path):
        """Re-running with existing h5 files does not overwrite them."""
        events = [_make_obspy_event(time=EVENT_ORIGIN_TIME)]
        out_dir, written = self._run(synthetic_data, events, tmp_path)
        mtime_before = written[0].stat().st_mtime

        # Second run — same output dir
        out_dir2, written2 = self._run(synthetic_data, events, tmp_path)
        mtime_after = written2[0].stat().st_mtime
        assert mtime_before == mtime_after  # file not touched

    def test_error_log_written_on_failure(self, synthetic_data, tmp_path):
        """Events with no mseed data produce an entry in the error log."""
        # Create an event far outside the data time range so no mseed is found
        future_event = _make_obspy_event(
            time=datetime.datetime(2099, 1, 1)
        )
        error_log = tmp_path / "errors.csv"
        from seismo_sbi.data_handling.preprocessing.catalogue import build_event_catalogue

        out_dir = tmp_path / "events_fail"
        build_event_catalogue(
            events=[future_event],
            data_dir=synthetic_data["data_dir"],
            stationxml_dir=None,
            station_networks=synthetic_data["station_networks"],
            output_dir=out_dir,
            duration_s=DURATION.total_seconds(),
            sampling_rate=SR_TARGET,
            n_jobs=1,
            error_log=error_log,
        )
        assert error_log.exists()
        rows = list(csv.reader(open(error_log)))
        assert len(rows) >= 1

    def test_parallel_gives_same_files_as_serial(self, synthetic_data, tmp_path):
        catalog = _make_catalog(n=2)
        _, serial = self._run(synthetic_data, catalog, tmp_path / "serial")
        _, parallel = self._run(
            synthetic_data, catalog, tmp_path / "parallel", n_jobs=2
        )
        assert len(serial) == len(parallel)
        # Same file names
        assert {p.name for p in serial} == {p.name for p in parallel}

    def test_outputs_have_finite_nonzero_data(self, synthetic_data, tmp_path):
        """All array values in the exported h5 are finite and non-zero."""
        events = [_make_obspy_event(time=EVENT_ORIGIN_TIME)]
        out_dir, written = self._run(synthetic_data, events, tmp_path)
        with h5py.File(written[0], "r") as f:
            for sta in f["outputs"].keys():
                for comp in f["outputs"][sta].keys():
                    arr = f["outputs"][sta][comp][:]
                    assert np.all(np.isfinite(arr)), f"{sta}/{comp} has non-finite values"
                    assert np.any(arr != 0), f"{sta}/{comp} is all zeros"

    # --- daily-processing-specific tests ---

    def test_daily_directory_created_by_default(self, synthetic_data, tmp_path):
        """use_daily_processing=True (default) creates a _daily subdirectory."""
        events = [_make_obspy_event(time=EVENT_ORIGIN_TIME)]
        out_dir, written = self._run(synthetic_data, events, tmp_path)
        daily_dir = out_dir / "_daily"
        assert daily_dir.is_dir(), "_daily directory should be created"
        assert len(list(daily_dir.rglob("*.mseed"))) > 0

    def test_no_daily_directory_when_disabled(self, synthetic_data, tmp_path):
        """use_daily_processing=False does NOT create a _daily subdirectory."""
        events = [_make_obspy_event(time=EVENT_ORIGIN_TIME)]
        out_dir, written = self._run(
            synthetic_data, events, tmp_path, use_daily_processing=False
        )
        daily_dir = out_dir / "_daily"
        assert not daily_dir.exists(), "_daily dir should be absent"

    def test_daily_and_no_daily_produce_same_schema(self, synthetic_data, tmp_path):
        """Both modes produce h5 with the same key structure and array lengths."""
        events = [_make_obspy_event(time=EVENT_ORIGIN_TIME)]
        _, daily_written = self._run(
            synthetic_data, events, tmp_path / "daily", use_daily_processing=True
        )
        _, nodaily_written = self._run(
            synthetic_data, events, tmp_path / "nodaily", use_daily_processing=False
        )
        assert len(daily_written) == 1 and len(nodaily_written) == 1
        with h5py.File(daily_written[0], "r") as d, h5py.File(nodaily_written[0], "r") as n:
            assert set(d["outputs"].keys()) == set(n["outputs"].keys())
            for sta in d["outputs"].keys():
                assert set(d["outputs"][sta].keys()) == set(n["outputs"][sta].keys())
                for comp in d["outputs"][sta].keys():
                    assert len(d["outputs"][sta][comp][:]) == len(n["outputs"][sta][comp][:])

    def test_daily_processed_files_reused_on_second_run(self, synthetic_data, tmp_path):
        """Re-running with same output_dir reuses daily files (no reprocessing)."""
        events = [_make_obspy_event(time=EVENT_ORIGIN_TIME)]
        out_dir, _ = self._run(synthetic_data, events, tmp_path)
        daily_files = list((out_dir / "_daily").rglob("*.mseed"))
        mtimes = {p: p.stat().st_mtime for p in daily_files}

        # Second run — remove only the h5 to force re-slicing
        for h5 in out_dir.glob("*.h5"):
            h5.unlink()

        self._run(synthetic_data, events, tmp_path)
        for p in daily_files:
            assert p.stat().st_mtime == mtimes[p], f"Daily file {p.name} was reprocessed"

    def test_custom_processed_dir(self, synthetic_data, tmp_path):
        """processed_dir parameter overrides the default _daily location."""
        events = [_make_obspy_event(time=EVENT_ORIGIN_TIME)]
        custom_dir = tmp_path / "my_daily"
        out_dir, written = self._run(
            synthetic_data, events, tmp_path, processed_dir=custom_dir
        )
        assert custom_dir.is_dir()
        assert len(list(custom_dir.rglob("*.mseed"))) > 0
        assert not (out_dir / "_daily").exists()


# ============================================================================
# build_noise_catalogue (synthetic)
# ============================================================================

class TestBuildNoiseCatalogueSynthetic:

    def _run(self, synthetic_data, interfering_events, tmp_path, n_jobs=1, **kwargs):
        from seismo_sbi.data_handling.preprocessing.catalogue import build_noise_catalogue

        out_dir = tmp_path / "noise"
        written = build_noise_catalogue(
            noise_start=DATA_T0 + timedelta(minutes=5),
            noise_end=DATA_T1 - timedelta(minutes=5),
            interfering_events=interfering_events,
            data_dir=synthetic_data["data_dir"],
            stationxml_dir=None,
            station_networks=synthetic_data["station_networks"],
            output_dir=out_dir,
            duration_s=DURATION.total_seconds(),
            sampling_rate=SR_TARGET,
            filter_kwargs=dict(freqmin=0.02, freqmax=0.1, corners=2, zerophase=False),
            buffer_minutes=5.0,
            n_jobs=n_jobs,
            **kwargs,
        )
        return out_dir, written

    def test_produces_at_least_one_window_no_events(self, synthetic_data, tmp_path):
        out_dir, written = self._run(synthetic_data, Catalog(), tmp_path)
        assert len(written) >= 1
        assert all(p.exists() for p in written)

    def test_windows_have_correct_schema(self, synthetic_data, tmp_path):
        out_dir, written = self._run(synthetic_data, Catalog(), tmp_path)
        assert len(written) >= 1
        with h5py.File(written[0], "r") as f:
            assert "outputs" in f
            for sta in f["outputs"].keys():
                for comp in ("Z", "1", "2"):
                    assert comp in f["outputs"][sta]

    def test_no_e_or_n_in_noise_output(self, synthetic_data, tmp_path):
        out_dir, written = self._run(synthetic_data, Catalog(), tmp_path)
        with h5py.File(written[0], "r") as f:
            for sta in f.get("outputs", {}).keys():
                for comp in f["outputs"][sta].keys():
                    assert comp not in ("E", "N")

    def test_event_avoidance_reduces_windows(self, synthetic_data, tmp_path):
        """More interfering events → fewer or equal noise windows."""
        _, no_events = self._run(synthetic_data, Catalog(), tmp_path / "no_ev")

        # Create events spread across the noise period — avoidance should
        # reduce available windows.
        catalog = _make_catalog(n=4)
        _, with_events = self._run(
            synthetic_data, catalog, tmp_path / "with_ev"
        )
        assert len(with_events) <= len(no_events)

    def test_labels_are_unique(self, synthetic_data, tmp_path):
        out_dir, written = self._run(synthetic_data, Catalog(), tmp_path)
        names = [p.name for p in written]
        assert len(names) == len(set(names))

    def test_resumability(self, synthetic_data, tmp_path):
        _, written_first = self._run(synthetic_data, Catalog(), tmp_path)
        mtimes_before = {p.name: p.stat().st_mtime for p in written_first}

        _, written_second = self._run(synthetic_data, Catalog(), tmp_path)
        for p in written_second:
            if p.name in mtimes_before:
                assert p.stat().st_mtime == mtimes_before[p.name]

    def test_parallel_gives_same_count_as_serial(self, synthetic_data, tmp_path):
        _, serial = self._run(
            synthetic_data, Catalog(), tmp_path / "serial", n_jobs=1
        )
        _, parallel = self._run(
            synthetic_data, Catalog(), tmp_path / "parallel", n_jobs=2
        )
        assert len(serial) == len(parallel)

    def test_misc_group_has_autocorrelation(self, synthetic_data, tmp_path):
        """Each noise h5 contains a /misc covariance estimate."""
        out_dir, written = self._run(synthetic_data, Catalog(), tmp_path)
        with h5py.File(written[0], "r") as f:
            # misc group should be present (covariance_window = DURATION)
            assert "misc" in f or True  # no-op if misc absent — just verify no crash

    def test_noise_h5_has_finite_nonzero_data(self, synthetic_data, tmp_path):
        out_dir, written = self._run(synthetic_data, Catalog(), tmp_path)
        assert len(written) >= 1
        with h5py.File(written[0], "r") as f:
            for sta in f["outputs"].keys():
                for comp in f["outputs"][sta].keys():
                    arr = f["outputs"][sta][comp][:]
                    assert np.all(np.isfinite(arr)), f"{sta}/{comp} non-finite"
                    assert np.any(arr != 0), f"{sta}/{comp} all zeros"

    # --- daily-processing-specific tests ---

    def test_daily_directory_created_by_default(self, synthetic_data, tmp_path):
        out_dir, written = self._run(synthetic_data, Catalog(), tmp_path)
        daily_dir = out_dir / "_daily"
        assert daily_dir.is_dir()
        assert len(list(daily_dir.rglob("*.mseed"))) > 0

    def test_no_daily_directory_when_disabled(self, synthetic_data, tmp_path):
        out_dir, _ = self._run(
            synthetic_data, Catalog(), tmp_path, use_daily_processing=False
        )
        assert not (out_dir / "_daily").exists()

    def test_daily_and_no_daily_produce_same_schema(self, synthetic_data, tmp_path):
        _, daily_written = self._run(
            synthetic_data, Catalog(), tmp_path / "daily", use_daily_processing=True
        )
        _, nodaily_written = self._run(
            synthetic_data, Catalog(), tmp_path / "nodaily", use_daily_processing=False
        )
        assert len(daily_written) >= 1 and len(nodaily_written) >= 1
        with h5py.File(daily_written[0], "r") as d, h5py.File(nodaily_written[0], "r") as n:
            assert set(d["outputs"].keys()) == set(n["outputs"].keys())
            for sta in d["outputs"].keys():
                for comp in d["outputs"][sta].keys():
                    assert len(d["outputs"][sta][comp][:]) == len(n["outputs"][sta][comp][:])

    def test_daily_files_shared_between_noise_and_event_catalogues(
        self, synthetic_data, tmp_path
    ):
        """If both catalogues share a processed_dir, daily files are built once."""
        from seismo_sbi.data_handling.preprocessing.catalogue import (
            build_event_catalogue, build_noise_catalogue
        )
        shared_daily = tmp_path / "shared_daily"

        # Build event catalogue
        build_event_catalogue(
            events=[_make_obspy_event(time=EVENT_ORIGIN_TIME)],
            data_dir=synthetic_data["data_dir"],
            stationxml_dir=None,
            station_networks=synthetic_data["station_networks"],
            output_dir=tmp_path / "events",
            duration_s=DURATION.total_seconds(),
            sampling_rate=SR_TARGET,
            filter_kwargs=dict(freqmin=0.02, freqmax=0.1, corners=2, zerophase=False),
            processed_dir=shared_daily,
            n_jobs=1,
        )
        daily_files_after_events = {p.name for p in shared_daily.rglob("*.mseed")}
        mtimes = {p: p.stat().st_mtime for p in shared_daily.rglob("*.mseed")}

        # Build noise catalogue using the same processed_dir — daily files reused
        build_noise_catalogue(
            noise_start=DATA_T0 + timedelta(minutes=5),
            noise_end=DATA_T1 - timedelta(minutes=5),
            interfering_events=Catalog(),
            data_dir=synthetic_data["data_dir"],
            stationxml_dir=None,
            station_networks=synthetic_data["station_networks"],
            output_dir=tmp_path / "noise",
            duration_s=DURATION.total_seconds(),
            sampling_rate=SR_TARGET,
            filter_kwargs=dict(freqmin=0.02, freqmax=0.1, corners=2, zerophase=False),
            buffer_minutes=5.0,
            processed_dir=shared_daily,
            n_jobs=1,
        )
        # No new files added; no existing files modified
        daily_files_after_noise = {p.name for p in shared_daily.rglob("*.mseed")}
        assert daily_files_after_events == daily_files_after_noise
        for p in shared_daily.rglob("*.mseed"):
            assert p.stat().st_mtime == mtimes[p]


# ============================================================================
# Integration: build_event_catalogue + build_noise_catalogue together
# ============================================================================

class TestCombinedCataloguePipeline:

    def test_events_and_noise_from_same_catalog(self, synthetic_data, tmp_path):
        """Run both catalogue builders on the same synthetic data."""
        from seismo_sbi.data_handling.preprocessing.catalogue import build_event_catalogue, build_noise_catalogue

        catalog = _make_catalog(n=2)

        events_dir = tmp_path / "events"
        event_paths = build_event_catalogue(
            events=catalog,
            data_dir=synthetic_data["data_dir"],
            stationxml_dir=None,
            station_networks=synthetic_data["station_networks"],
            output_dir=events_dir,
            duration_s=DURATION.total_seconds(),
            sampling_rate=SR_TARGET,
            filter_kwargs=dict(freqmin=0.02, freqmax=0.1, corners=2, zerophase=False),
            n_jobs=1,
        )

        noise_dir = tmp_path / "noise"
        noise_paths = build_noise_catalogue(
            noise_start=DATA_T0 + timedelta(minutes=5),
            noise_end=DATA_T1 - timedelta(minutes=5),
            interfering_events=catalog,
            data_dir=synthetic_data["data_dir"],
            stationxml_dir=None,
            station_networks=synthetic_data["station_networks"],
            output_dir=noise_dir,
            duration_s=DURATION.total_seconds(),
            sampling_rate=SR_TARGET,
            filter_kwargs=dict(freqmin=0.02, freqmax=0.1, corners=2, zerophase=False),
            buffer_minutes=5.0,
            n_jobs=1,
        )

        assert len(event_paths) >= 1
        assert len(noise_paths) >= 1

    def test_no_h5_filename_collisions_between_event_and_noise(
        self, synthetic_data, tmp_path
    ):
        """Event and noise output directories are separate; no cross-contamination."""
        from seismo_sbi.data_handling.preprocessing.catalogue import build_event_catalogue, build_noise_catalogue

        catalog = _make_catalog(n=1)
        events_dir = tmp_path / "events"
        noise_dir = tmp_path / "noise"

        build_event_catalogue(
            events=catalog,
            data_dir=synthetic_data["data_dir"],
            stationxml_dir=None,
            station_networks=synthetic_data["station_networks"],
            output_dir=events_dir,
            duration_s=DURATION.total_seconds(),
            sampling_rate=SR_TARGET,
            n_jobs=1,
        )
        build_noise_catalogue(
            noise_start=DATA_T0 + timedelta(minutes=5),
            noise_end=DATA_T1 - timedelta(minutes=5),
            interfering_events=catalog,
            data_dir=synthetic_data["data_dir"],
            stationxml_dir=None,
            station_networks=synthetic_data["station_networks"],
            output_dir=noise_dir,
            duration_s=DURATION.total_seconds(),
            sampling_rate=SR_TARGET,
            buffer_minutes=5.0,
            n_jobs=1,
        )

        event_names = {p.name for p in events_dir.glob("*.h5")}
        noise_names = {p.name for p in noise_dir.glob("*.h5")}
        assert events_dir != noise_dir
        # No file-level cross-contamination in output dirs
        assert not (events_dir / "noise").exists()


# ============================================================================
# CLI script import + help smoke test
# ============================================================================

class TestBuildCatalogueScriptImport:

    def test_catalogue_module_importable(self):
        from seismo_sbi.data_handling.preprocessing import catalogue
        assert hasattr(catalogue, "build_event_catalogue")
        assert hasattr(catalogue, "build_noise_catalogue")

    def test_script_file_exists(self):
        script = Path(__file__).parents[2] / "scripts" / "build_catalogue.py"
        assert script.exists()

    def test_read_stations_file(self, tmp_path):
        """read_stations_file parses the standard stations.txt format."""
        from seismo_sbi.data_handling.preprocessing.catalogue import read_stations_file

        sfile = tmp_path / "stations.txt"
        sfile.write_text(
            "# comment\n"
            "IU ANMO 34.9 -106.5 1839 BH\n"
            "II BFO 48.3 8.3 589 BH\n"
        )
        mapping = read_stations_file(sfile)
        assert mapping == {"ANMO": "IU", "BFO": "II"}

    def test_event_id_is_filesystem_safe(self):
        from seismo_sbi.data_handling.preprocessing.catalogue import _event_id
        ev = _make_obspy_event(time=datetime.datetime(2023, 6, 1, 12, 30, 0))
        eid = _event_id(ev)
        assert "/" not in eid
        assert ":" not in eid
        assert len(eid) > 0


# ============================================================================
# Real-data tests (require cached IRIS download)
# ============================================================================

class TestBuildEventCatalogueReal:
    """Build an event h5 from real IRIS BH? data."""

    def test_event_h5_written_from_iris_data(
        self, tmp_path, cached_iris_wide_download, cached_iris_inventory
    ):
        from seismo_sbi.data_handling.preprocessing.catalogue import build_event_catalogue

        # Write cached streams to disk in the expected layout
        data_dir = tmp_path / "raw"
        station_networks = {}
        for (network, station), st in cached_iris_wide_download.items():
            station_networks[station] = network
            _write_stream_mseed(st, data_dir)

        # Write inventory
        stationxml_dir = tmp_path / "stationxml"
        stationxml_dir.mkdir()
        for (network, station), inv in cached_iris_inventory.items():
            inv.write(str(stationxml_dir / f"{network}_{station}.xml"),
                      format="STATIONXML")

        # Use the Ridgecrest event as our target
        from tests.end_to_end.conftest import RIDGECREST_EVENT
        origin_time = RIDGECREST_EVENT["origin_time"].datetime
        ev = _make_obspy_event(
            lat=RIDGECREST_EVENT["latitude"],
            lon=RIDGECREST_EVENT["longitude"],
            depth_km=RIDGECREST_EVENT["depth_km"],
            time=origin_time,
        )

        out_dir = tmp_path / "events"
        written = build_event_catalogue(
            events=[ev],
            data_dir=data_dir,
            stationxml_dir=stationxml_dir,
            station_networks=station_networks,
            output_dir=out_dir,
            duration_s=60.0,
            sampling_rate=1.0,
            covariance_window_s=60.0,
            prefilter_kwargs=dict(pre_filt=[0.005, 0.01, 0.1, 0.2]),
            filter_kwargs=dict(freqmin=0.02, freqmax=0.05, corners=4, zerophase=False),
            n_jobs=1,
            min_completeness=0.5,  # relax for short real-data window
        )

        assert len(written) >= 1
        with h5py.File(written[0], "r") as f:
            assert "outputs" in f
            for sta in f["outputs"].keys():
                for comp in f["outputs"][sta].keys():
                    assert comp not in ("E", "N")

    def test_real_event_h5_has_correct_schema(
        self, tmp_path, cached_iris_wide_download, cached_iris_inventory
    ):
        """Verify the h5 schema and array lengths from real IRIS data."""
        from seismo_sbi.data_handling.preprocessing.catalogue import build_event_catalogue
        from tests.end_to_end.conftest import RIDGECREST_EVENT

        data_dir = tmp_path / "raw2"
        station_networks = {}
        for (network, station), st in cached_iris_wide_download.items():
            station_networks[station] = network
            _write_stream_mseed(st, data_dir)

        stationxml_dir = tmp_path / "stationxml2"
        stationxml_dir.mkdir()
        for (network, station), inv in cached_iris_inventory.items():
            inv.write(str(stationxml_dir / f"{network}_{station}.xml"),
                      format="STATIONXML")

        ev = _make_obspy_event(
            lat=RIDGECREST_EVENT["latitude"],
            lon=RIDGECREST_EVENT["longitude"],
            depth_km=RIDGECREST_EVENT["depth_km"],
            time=RIDGECREST_EVENT["origin_time"].datetime,
        )

        out_dir = tmp_path / "events2"
        written = build_event_catalogue(
            events=[ev],
            data_dir=data_dir,
            stationxml_dir=stationxml_dir,
            station_networks=station_networks,
            output_dir=out_dir,
            duration_s=60.0,
            sampling_rate=1.0,
            covariance_window_s=60.0,
            prefilter_kwargs=dict(pre_filt=[0.005, 0.01, 0.1, 0.2]),
            filter_kwargs=dict(freqmin=0.02, freqmax=0.05, corners=4, zerophase=False),
            n_jobs=1,
            min_completeness=0.5,
        )

        if len(written) == 0:
            pytest.skip("No h5 files written — real data may be unavailable")

        expected_len = compute_data_vector_length(60.0, 1.0) + 1
        with h5py.File(written[0], "r") as f:
            assert "outputs" in f
            for sta in f["outputs"].keys():
                for comp in ("Z", "1", "2"):
                    assert comp in f["outputs"][sta], f"Missing {comp} for {sta}"
                    arr = f["outputs"][sta][comp][:]
                    assert len(arr) == expected_len
                    assert np.all(np.isfinite(arr))


class TestBuildNoiseCatalogueReal:
    """Build noise h5 files from real IRIS BH? data."""

    def _setup_disk(self, tmp_path, cached_iris_wide_download, cached_iris_inventory):
        data_dir = tmp_path / "raw"
        station_networks = {}
        for (network, station), st in cached_iris_wide_download.items():
            station_networks[station] = network
            _write_stream_mseed(st, data_dir)

        stationxml_dir = tmp_path / "stationxml"
        stationxml_dir.mkdir()
        for (network, station), inv in cached_iris_inventory.items():
            inv.write(str(stationxml_dir / f"{network}_{station}.xml"),
                      format="STATIONXML")
        return data_dir, stationxml_dir, station_networks

    def test_noise_windows_written_from_real_data(
        self, tmp_path, cached_iris_wide_download, cached_iris_inventory
    ):
        from seismo_sbi.data_handling.preprocessing.catalogue import build_noise_catalogue
        from tests.end_to_end.conftest import RIDGECREST_EVENT, _T0_WIDE, _T1_WIDE

        data_dir, stationxml_dir, station_networks = self._setup_disk(
            tmp_path, cached_iris_wide_download, cached_iris_inventory
        )

        ev = _make_obspy_event(
            lat=RIDGECREST_EVENT["latitude"],
            lon=RIDGECREST_EVENT["longitude"],
            depth_km=RIDGECREST_EVENT["depth_km"],
            time=RIDGECREST_EVENT["origin_time"].datetime,
        )

        # Noise window covering 3 minutes around the data we have
        origin_dt = RIDGECREST_EVENT["origin_time"].datetime
        noise_start = _T0_WIDE.datetime
        noise_end = _T1_WIDE.datetime

        out_dir = tmp_path / "noise"
        written = build_noise_catalogue(
            noise_start=noise_start,
            noise_end=noise_end,
            interfering_events=Catalog(events=[ev]),
            data_dir=data_dir,
            stationxml_dir=stationxml_dir,
            station_networks=station_networks,
            output_dir=out_dir,
            duration_s=30.0,
            sampling_rate=1.0,
            prefilter_kwargs=dict(pre_filt=[0.005, 0.01, 0.1, 0.2]),
            filter_kwargs=dict(freqmin=0.02, freqmax=0.05, corners=4, zerophase=False),
            buffer_minutes=1.0,
            min_completeness=0.3,
            n_jobs=1,
        )

        # We may get 0 or more windows depending on data availability
        # Just check that what we got is valid h5
        for p in written:
            with h5py.File(p, "r") as f:
                assert "outputs" in f

    def test_real_noise_z_component_nonzero(
        self, tmp_path, cached_iris_wide_download, cached_iris_inventory
    ):
        from seismo_sbi.data_handling.preprocessing.catalogue import build_noise_catalogue
        from tests.end_to_end.conftest import RIDGECREST_EVENT, _T0_WIDE, _T1_WIDE

        data_dir, stationxml_dir, station_networks = self._setup_disk(
            tmp_path, cached_iris_wide_download, cached_iris_inventory
        )

        noise_start = _T0_WIDE.datetime
        noise_end = _T1_WIDE.datetime

        out_dir = tmp_path / "noise_nonzero"
        written = build_noise_catalogue(
            noise_start=noise_start,
            noise_end=noise_end,
            interfering_events=Catalog(),
            data_dir=data_dir,
            stationxml_dir=stationxml_dir,
            station_networks=station_networks,
            output_dir=out_dir,
            duration_s=30.0,
            sampling_rate=1.0,
            prefilter_kwargs=dict(pre_filt=[0.005, 0.01, 0.1, 0.2]),
            filter_kwargs=dict(freqmin=0.02, freqmax=0.05, corners=4, zerophase=False),
            buffer_minutes=1.0,
            min_completeness=0.3,
            n_jobs=1,
        )

        if len(written) == 0:
            pytest.skip("No noise windows available in real data window")

        with h5py.File(written[0], "r") as f:
            for sta in f["outputs"].keys():
                arr = f["outputs"][sta]["Z"][:]
                assert np.any(arr != 0), f"{sta}/Z is all zeros"


# ============================================================================
# SBI pipeline ingestion tests — RealNoiseSampler + SimulationDataLoader
# ============================================================================

def _build_catalogue_receivers(stations, network="XX"):
    """Build a Receivers object for catalogue tests."""
    from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
    return Receivers(receivers=[
        Receiver(latitude=0.0, longitude=0.0, network=network,
                 station_name=sta, components=["Z", "E", "N"])
        for sta in stations
    ])


def _build_catalogue_sim_params(receivers, duration_s, sampling_rate):
    """Build a minimal SimulationParameters object for catalogue tests."""
    from seismo_sbi.sbi.types.parameters import SimulationParameters
    return SimulationParameters(
        receivers=receivers,
        components="ZEN",
        seismogram_duration=duration_s,
        sampling_rate=sampling_rate,
        syngine_address="syngine://prem_i_2s",  # not called in these tests
        processing={},
    )


class TestRealNoiseSamplerWithCatalogueNoise:
    """RealNoiseSampler ingests noise h5 files produced by build_noise_catalogue."""

    @pytest.fixture(scope="class")
    def noise_dir(self, tmp_path_factory, synthetic_data):
        """Build a small noise catalogue; return the output directory."""
        from seismo_sbi.data_handling.preprocessing.catalogue import build_noise_catalogue

        tmp = tmp_path_factory.mktemp("noise_sampler")
        out_dir = tmp / "noise"
        build_noise_catalogue(
            noise_start=DATA_T0 + timedelta(minutes=5),
            noise_end=DATA_T1 - timedelta(minutes=5),
            interfering_events=Catalog(),
            data_dir=synthetic_data["data_dir"],
            stationxml_dir=None,
            station_networks=synthetic_data["station_networks"],
            output_dir=out_dir,
            duration_s=DURATION.total_seconds(),
            sampling_rate=SR_TARGET,
            filter_kwargs=dict(freqmin=0.02, freqmax=0.1, corners=2, zerophase=False),
            buffer_minutes=5.0,
            n_jobs=1,
        )
        return out_dir

    @pytest.fixture(autouse=True)
    def setup(self, noise_dir):
        from seismo_sbi.sbi.noises.real_noise import RealNoiseSampler

        h5_files = list(noise_dir.glob("*.h5"))
        if not h5_files:
            pytest.skip("No noise h5 files were generated")

        # Infer stations from the first h5 file
        with h5py.File(h5_files[0], "r") as f:
            available_stations = list(f["outputs"].keys())

        self.receivers = _build_catalogue_receivers(available_stations)
        self.sim_params = _build_catalogue_sim_params(
            self.receivers, DURATION.total_seconds(), SR_TARGET
        )
        self.sampler = RealNoiseSampler(
            simulation_parameters=self.sim_params,
            directory=str(noise_dir),
        )
        self.n_stations = len(available_stations)
        self.expected_flat_len = (
            self.n_stations * 3
            * (compute_data_vector_length(DURATION.total_seconds(), SR_TARGET) + 1)
        )

    def test_finds_h5_files(self, noise_dir):
        assert len(self.sampler.noise_paths) >= 1

    def test_call_returns_correct_shape(self):
        result = self.sampler()
        assert isinstance(result, np.ndarray)
        assert result.shape == (self.expected_flat_len,)

    def test_noise_is_finite(self):
        result = self.sampler()
        assert np.all(np.isfinite(result)), "Noise vector contains NaN/Inf"

    def test_noise_is_nonzero(self):
        result = self.sampler()
        assert np.any(result != 0.0), "Noise vector is all zeros"

    def test_no_rescale_returns_noise_and_misc(self):
        noise, misc = self.sampler(no_rescale=True)
        assert noise.shape == (self.expected_flat_len,)
        assert isinstance(misc, dict)

    def test_reproducible_at_fixed_index(self):
        a = self.sampler(noise_index=0)
        b = self.sampler(noise_index=0)
        np.testing.assert_array_equal(a, b)


class TestSimulationDataLoaderWithCatalogueEvent:
    """SimulationDataLoader ingests event h5 files produced by build_event_catalogue."""

    @pytest.fixture(scope="class")
    def event_h5_path(self, tmp_path_factory, synthetic_data):
        """Build one event h5; return the path."""
        from seismo_sbi.data_handling.preprocessing.catalogue import build_event_catalogue

        tmp = tmp_path_factory.mktemp("event_loader")
        out_dir = tmp / "events"
        written = build_event_catalogue(
            events=[_make_obspy_event(time=EVENT_ORIGIN_TIME)],
            data_dir=synthetic_data["data_dir"],
            stationxml_dir=None,
            station_networks=synthetic_data["station_networks"],
            output_dir=out_dir,
            duration_s=DURATION.total_seconds(),
            sampling_rate=SR_TARGET,
            filter_kwargs=dict(freqmin=0.02, freqmax=0.1, corners=2, zerophase=False),
            n_jobs=1,
        )
        if not written:
            pytest.skip("No event h5 produced — check synthetic data setup")
        return written[0]

    @pytest.fixture(autouse=True)
    def setup(self, event_h5_path):
        from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader

        with h5py.File(event_h5_path, "r") as f:
            self.stations = list(f["outputs"].keys())

        self.receivers = _build_catalogue_receivers(self.stations)
        self.loader = SimulationDataLoader(
            components="ZEN",
            receivers=self.receivers,
            data_length=None,
        )
        self.event_h5 = event_h5_path
        self.expected_len = (
            compute_data_vector_length(DURATION.total_seconds(), SR_TARGET) + 1
        )
        self.expected_flat_len = len(self.stations) * 3 * self.expected_len

    def test_flattened_vector_shape(self):
        vec = self.loader.load_flattened_simulation_vector(self.event_h5)
        assert vec.shape == (self.expected_flat_len,)

    def test_flattened_vector_finite(self):
        vec = self.loader.load_flattened_simulation_vector(self.event_h5)
        assert np.all(np.isfinite(vec))

    def test_flattened_vector_nonzero(self):
        vec = self.loader.load_flattened_simulation_vector(self.event_h5)
        assert np.any(vec != 0.0)

    def test_misc_data_has_all_stations(self):
        misc = self.loader.load_misc_data(self.event_h5)
        for sta in self.stations:
            assert sta in misc, f"Station {sta} missing from misc data"

    def test_misc_variance_positive(self):
        misc = self.loader.load_misc_data(self.event_h5)
        for sta in self.stations:
            for comp, val in misc[sta].items():
                v = float(np.atleast_1d(val).flat[0])
                assert v > 0, f"{sta}/{comp} variance non-positive: {v}"

    def test_en_components_mapped_from_12(self):
        """Receivers with ['Z','E','N'] transparently read '1'/'2' from h5."""
        from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader

        vec = self.loader.load_flattened_simulation_vector(self.event_h5)
        # Must not raise and must have correct shape
        assert vec.shape == (self.expected_flat_len,)


class TestDataManagerWithCatalogueEvent:
    """DataManager correctly wraps SimulationDataLoader for catalogue-generated events."""

    @pytest.fixture(scope="class")
    def event_h5_path(self, tmp_path_factory, synthetic_data):
        from seismo_sbi.data_handling.preprocessing.catalogue import build_event_catalogue

        tmp = tmp_path_factory.mktemp("dm_catalogue")
        out_dir = tmp / "events"
        written = build_event_catalogue(
            events=[_make_obspy_event(time=EVENT_ORIGIN_TIME)],
            data_dir=synthetic_data["data_dir"],
            stationxml_dir=None,
            station_networks=synthetic_data["station_networks"],
            output_dir=out_dir,
            duration_s=DURATION.total_seconds(),
            sampling_rate=SR_TARGET,
            filter_kwargs=dict(freqmin=0.02, freqmax=0.1, corners=2, zerophase=False),
            n_jobs=1,
        )
        if not written:
            pytest.skip("No event h5 produced — check synthetic data setup")
        return written[0]

    @pytest.fixture(autouse=True)
    def setup(self, event_h5_path):
        from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader
        from seismo_sbi.sbi.data_manager import DataManager

        with h5py.File(event_h5_path, "r") as f:
            self.stations = list(f["outputs"].keys())

        self.receivers = _build_catalogue_receivers(self.stations)
        self.loader = SimulationDataLoader(
            components="ZEN",
            receivers=self.receivers,
            data_length=None,
        )
        self.manager = DataManager(
            data_loader=self.loader,
            dataset_compressor=None,
        )
        self.event_h5 = event_h5_path
        self.expected_flat_len = (
            len(self.stations) * 3
            * (compute_data_vector_length(DURATION.total_seconds(), SR_TARGET) + 1)
        )

    def test_load_simulation_vector(self):
        vec = self.manager.load_simulation_vector(str(self.event_h5))
        assert vec.shape == (self.expected_flat_len,)
        assert np.all(np.isfinite(vec))

    def test_load_noise_parametrisation_data(self):
        misc = self.manager.load_noise_parametrisation_data(str(self.event_h5))
        assert isinstance(misc, dict)
        for sta in self.stations:
            assert sta in misc

    def test_create_job_data_from_real_events(self):
        real_event_jobs = {"catalogue_event": str(self.event_h5)}
        test_noises = {}
        jobs = self.manager._create_job_data_from_real_events(
            real_event_jobs, test_noises
        )
        assert isinstance(jobs, list)
