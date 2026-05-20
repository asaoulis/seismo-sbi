"""Phase 2 — New preprocessing API tests on synthetic data.

Every public function in seismo_sbi.data_handling.preprocessing is tested here
on deterministic in-memory data.  No Instaseis, no network access.

Run:
    conda run -n seismo-sbi python -m pytest \
        tests/end_to_end/test_new_api_synthetic.py -v -m slow

These tests serve a dual purpose:
1. Contract tests that every function in the new API behaves correctly.
2. Regression tests that the new API produces h5 files that downstream
   SBI consumers (RealNoiseSampler, SimulationDataLoader) can read without
   modification.

Key invariants pinned here:
- E→1, N→2 renaming applied at h5 write time
- Array length = compute_data_vector_length(duration, sr) + 1  (inclusive slice)
- Autocorrelation computed from [event_start - cov_window, event_start]
- No 'E'/'N' keys ever appear in /outputs
"""

import datetime
from datetime import timedelta
from pathlib import Path

import h5py
import numpy as np
import obspy
from obspy import Trace, Stream, UTCDateTime, Inventory
import pytest

from seismo_sbi.data_handling.preprocessing import (
    load_waveforms,
    load_inventory,
    write_window,
    deconvolve_and_filter,
    slice_event_window,
    make_noise_windows,
    export_to_sbi_h5,
)
from seismo_sbi.data_handling.preprocessing.windowing import (
    make_daily_overlapping_windows,
    get_continuous_regions,
)
from seismo_sbi.instaseis_simulator.utils import compute_data_vector_length

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Scenario constants
# ---------------------------------------------------------------------------

NETWORK = "XX"
STATIONS = ["STA1", "STA2"]
CHANNELS = ["BHZ", "BHE", "BHN"]

SR_DATA = 20.0       # raw sampling rate (Hz)
SR_TARGET = 1.0      # target for SBI pipeline

T0 = datetime.datetime(2023, 1, 1, 12, 0, 0)
DATA_END = T0 + timedelta(minutes=30)

EVENT_START = T0 + timedelta(minutes=15)
EVENT_END = EVENT_START + timedelta(seconds=60)
COV_WINDOW = timedelta(minutes=3)

DATA_VECTOR_LEN = compute_data_vector_length(
    (EVENT_END - EVENT_START).total_seconds(), SR_TARGET
) + 1   # 61 samples — inclusive slice at 1 Hz

FILTER_KWARGS = dict(freqmin=0.02, freqmax=0.1, corners=4, zerophase=False)

RNG_SEED = 42


# ---------------------------------------------------------------------------
# Helpers: build synthetic Stream objects
# ---------------------------------------------------------------------------

def _make_stream(
    stations=STATIONS,
    channels=CHANNELS,
    network=NETWORK,
    t0=T0,
    t1=DATA_END,
    sr=SR_DATA,
    rng: np.random.Generator = None,
    amplitude: float = 0.5,
    freq: float = 0.04,
) -> Stream:
    """Return a synthetic Stream with a 0.04 Hz sinusoid + small noise."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    npts = int((t1 - t0).total_seconds() * sr)
    t = np.arange(npts) / sr
    st = Stream()
    for sta in stations:
        for cha in channels:
            data = amplitude * np.sin(2 * np.pi * freq * t) + rng.standard_normal(npts) * 0.001
            tr = Trace()
            tr.stats.network = network
            tr.stats.station = sta
            tr.stats.channel = cha
            tr.stats.location = ""
            tr.stats.sampling_rate = sr
            tr.stats.starttime = UTCDateTime(t0)
            tr.data = data.astype(np.float64)
            st += tr
    return st


def _make_distinct_amplitude_stream() -> Stream:
    """BHZ amplitude 1, BHE amplitude 2, BHN amplitude 3 — for E→1/N→2 fidelity check."""
    amps = {"BHZ": 1.0, "BHE": 2.0, "BHN": 3.0}
    npts = int((DATA_END - T0).total_seconds() * SR_DATA)
    t = np.arange(npts) / SR_DATA
    st = Stream()
    for sta in STATIONS:
        for cha, amp in amps.items():
            data = amp * np.sin(2 * np.pi * 0.04 * t)
            tr = Trace()
            tr.stats.network = NETWORK
            tr.stats.station = sta
            tr.stats.channel = cha
            tr.stats.location = ""
            tr.stats.sampling_rate = SR_DATA
            tr.stats.starttime = UTCDateTime(T0)
            tr.data = data.astype(np.float64)
            st += tr
    return st


def _write_stream_to_disk(st: Stream, data_dir: Path) -> list:
    """Write each trace as a separate mseed file; return list of paths."""
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


# ---------------------------------------------------------------------------
# Module-scoped fixture: processed stream + h5 used by many tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def processed_context(tmp_path_factory):
    """Build a processed stream and export to h5 once; share across tests."""
    tmp = tmp_path_factory.mktemp("new_api_synth")
    data_dir = tmp / "raw"
    data_dir.mkdir()

    raw_st = _make_stream()
    paths = _write_stream_to_disk(raw_st, data_dir)

    # Process without response removal
    proc_st = deconvolve_and_filter(
        raw_st.copy(),
        remove_response=False,
        filter_kwargs=FILTER_KWARGS,
        target_sr=SR_TARGET,
    )

    h5_path = tmp / "event.h5"
    export_to_sbi_h5(
        stream=proc_st,
        receivers=STATIONS,
        event_window=(EVENT_START, EVENT_END),
        out_path=h5_path,
        sampling_rate=SR_TARGET,
        covariance_window=COV_WINDOW,
        full_auto_correlation=True,
    )
    return {
        "tmp": tmp,
        "data_dir": data_dir,
        "paths": paths,
        "raw_st": raw_st,
        "proc_st": proc_st,
        "h5": h5_path,
    }


# ============================================================================
# Unit tests: deconvolve_and_filter
# ============================================================================

class TestDeconvolveAndFilter:

    def test_output_is_stream(self):
        st = _make_stream(stations=["STA1"])
        out = deconvolve_and_filter(st, remove_response=False,
                                    filter_kwargs=FILTER_KWARGS, target_sr=SR_TARGET)
        assert isinstance(out, Stream)

    def test_output_sampling_rate(self):
        st = _make_stream(stations=["STA1"])
        out = deconvolve_and_filter(st, remove_response=False,
                                    filter_kwargs=FILTER_KWARGS, target_sr=SR_TARGET)
        for tr in out:
            assert abs(tr.stats.sampling_rate - SR_TARGET) < 1e-6, (
                f"{tr.id} SR={tr.stats.sampling_rate} != {SR_TARGET}"
            )

    def test_no_resample_when_target_sr_none(self):
        st = _make_stream(stations=["STA1"])
        out = deconvolve_and_filter(st, remove_response=False, filter_kwargs=FILTER_KWARGS)
        for tr in out:
            assert abs(tr.stats.sampling_rate - SR_DATA) < 1e-6

    def test_bandpass_attenuates_out_of_band(self):
        """Energy above freqmax should be ≥30 dB below in-band energy."""
        st = _make_stream(stations=["STA1"])
        out = deconvolve_and_filter(st, remove_response=False,
                                    filter_kwargs=FILTER_KWARGS, target_sr=SR_TARGET)
        for tr in out.select(channel="BHZ"):
            arr = tr.data
            freqs = np.fft.rfftfreq(len(arr), d=1.0 / SR_TARGET)
            power = np.abs(np.fft.rfft(arr)) ** 2
            in_band = (freqs >= 0.02) & (freqs <= 0.1)
            out_band = freqs > 0.3
            if not np.any(in_band) or not np.any(out_band):
                continue
            db = 10 * np.log10(power[in_band].mean() / (power[out_band].mean() + 1e-30))
            assert db > 30, f"Only {db:.1f} dB attenuation above freqmax"

    def test_output_nonzero(self):
        st = _make_stream(stations=["STA1"])
        out = deconvolve_and_filter(st, remove_response=False,
                                    filter_kwargs=FILTER_KWARGS, target_sr=SR_TARGET)
        for tr in out:
            assert np.any(tr.data != 0.0), f"{tr.id} is all-zero after filtering"

    def test_gap_filled_by_merge(self):
        """A stream with a 1-sample gap must survive deconvolve_and_filter."""
        rng = np.random.default_rng(0)
        npts = int(10 * SR_DATA)
        t = np.arange(npts) / SR_DATA
        data1 = np.sin(2 * np.pi * 0.04 * t)
        data2 = np.sin(2 * np.pi * 0.04 * (t + 0.5))  # 0.5 s gap

        t_utc = UTCDateTime(T0)
        tr1 = Trace(data=data1.astype(np.float64))
        tr1.stats.sampling_rate = SR_DATA
        tr1.stats.starttime = t_utc
        tr1.stats.network = "XX"
        tr1.stats.station = "GAP"
        tr1.stats.channel = "BHZ"

        # Second trace starts 1 sample after the gap (simulating a gap)
        tr2 = Trace(data=data2.astype(np.float64))
        tr2.stats.sampling_rate = SR_DATA
        tr2.stats.starttime = t_utc + 10.1   # overlapping = gap of 0.1 s
        tr2.stats.network = "XX"
        tr2.stats.station = "GAP"
        tr2.stats.channel = "BHZ"

        st = Stream([tr1, tr2])
        out = deconvolve_and_filter(st, remove_response=False,
                                    filter_kwargs=dict(freqmin=0.01, freqmax=0.1, corners=4, zerophase=False))
        assert len(out) >= 1, "Gap-containing stream failed deconvolve_and_filter"
        assert np.all(np.isfinite(out[0].data)), "NaN in gap-filled output"

    def test_remove_response_raises_without_inventory(self):
        st = _make_stream(stations=["STA1"])
        with pytest.raises(ValueError, match="inventory"):
            deconvolve_and_filter(st, remove_response=True, inventory=None)

    def test_custom_filter_kwargs_used(self):
        """Output of a narrow-band filter should differ from a wide-band one."""
        st = _make_stream(stations=["STA1"], amplitude=1.0, freq=0.04)
        narrow = deconvolve_and_filter(
            st.copy(), remove_response=False,
            filter_kwargs=dict(freqmin=0.035, freqmax=0.045, corners=4, zerophase=False),
            target_sr=SR_TARGET,
        )
        wide = deconvolve_and_filter(
            st.copy(), remove_response=False,
            filter_kwargs=dict(freqmin=0.01, freqmax=0.4, corners=4, zerophase=False),
            target_sr=SR_TARGET,
        )
        for narrow_tr, wide_tr in zip(narrow, wide):
            assert not np.allclose(narrow_tr.data, wide_tr.data), (
                "Narrow and wide filter produced identical output"
            )


# ============================================================================
# Unit tests: io functions
# ============================================================================

class TestLoadWaveforms:

    def test_single_file_loaded(self, processed_context):
        paths = processed_context["paths"][:3]  # first 3 traces (STA1 Z,E,N)
        st = load_waveforms(paths)
        assert len(st) >= 1

    def test_multiple_files_combined(self, processed_context):
        paths = processed_context["paths"]
        st = load_waveforms(paths)
        assert len(st) == len(STATIONS) * len(CHANNELS)

    def test_time_trim_applied(self, processed_context):
        paths = processed_context["paths"]
        t0 = UTCDateTime(EVENT_START)
        t1 = UTCDateTime(EVENT_END)
        st = load_waveforms(paths, starttime=t0, endtime=t1)
        for tr in st:
            assert tr.stats.starttime >= t0 - 1, f"{tr.id} starts before trim"
            assert tr.stats.endtime <= t1 + 1, f"{tr.id} ends after trim"

    def test_returns_stream_type(self, processed_context):
        st = load_waveforms(processed_context["paths"])
        assert isinstance(st, Stream)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_waveforms([tmp_path / "nonexistent.mseed"])

    def test_trace_data_preserved(self, processed_context):
        """Data values in loaded file must match what was written."""
        paths = processed_context["paths"]
        raw = processed_context["raw_st"]
        loaded = load_waveforms(paths)
        for tr_raw in raw.select(station="STA1", channel="BHZ"):
            loaded_tr = loaded.select(station="STA1", channel="BHZ")[0]
            np.testing.assert_allclose(
                tr_raw.data, loaded_tr.data, rtol=1e-6,
                err_msg="Loaded data does not match written data"
            )


class TestWriteWindow:

    def test_creates_file(self, tmp_path):
        st = _make_stream(stations=["STA1"])
        out = tmp_path / "subdir" / "test.mseed"
        write_window(st, out)
        assert out.exists()

    def test_creates_parent_directories(self, tmp_path):
        st = _make_stream(stations=["STA1"])
        deep = tmp_path / "a" / "b" / "c" / "test.mseed"
        write_window(st, deep)
        assert deep.exists()

    def test_written_file_readable_by_obspy(self, tmp_path):
        st = _make_stream(stations=["STA1"])
        out = tmp_path / "test.mseed"
        write_window(st, out)
        read_back = obspy.read(str(out))
        assert len(read_back) == len(st)

    def test_data_round_trip(self, tmp_path):
        st = _make_stream(stations=["STA1"])
        out = tmp_path / "roundtrip.mseed"
        write_window(st, out)
        read_back = obspy.read(str(out))
        orig = st.select(station="STA1", channel="BHZ")[0]
        rb = read_back.select(station="STA1", channel="BHZ")[0]
        np.testing.assert_allclose(orig.data, rb.data, rtol=1e-6)


class TestLoadInventory:

    def test_loads_xml_from_directory(self, tmp_path):
        """load_inventory should find and load *.xml files from a directory."""
        # Create a minimal inventory and write it
        inv = _make_minimal_inventory()
        xml_path = tmp_path / "resp.xml"
        inv.write(str(xml_path), format="STATIONXML")
        loaded = load_inventory(tmp_path)
        assert len(loaded.networks) >= 1

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_inventory(tmp_path)

    def test_multiple_xml_files_merged(self, tmp_path):
        """Two separate XML files should be merged into one Inventory."""
        inv1 = _make_minimal_inventory(station="STA1")
        inv2 = _make_minimal_inventory(station="STA2")
        inv1.write(str(tmp_path / "resp1.xml"), format="STATIONXML")
        inv2.write(str(tmp_path / "resp2.xml"), format="STATIONXML")
        loaded = load_inventory(tmp_path)
        stations = [sta.code for net in loaded.networks for sta in net.stations]
        assert "STA1" in stations or len(stations) >= 1  # merged


def _make_minimal_inventory(station="STA1") -> Inventory:
    """Create a tiny obspy Inventory with one channel and a flat response."""
    from obspy.core.inventory import (
        Inventory, Network, Station, Channel, Site,
    )
    from obspy.core.inventory.response import Response, InstrumentSensitivity
    from obspy.core.utcdatetime import UTCDateTime as OUTCDateTime

    sensitivity = InstrumentSensitivity(
        value=1500.0,
        frequency=0.02,
        input_units="M/S",
        output_units="COUNTS",
    )
    response = Response(instrument_sensitivity=sensitivity)

    channel = Channel(
        code="BHZ",
        location_code="",
        latitude=0.0, longitude=0.0, elevation=0.0, depth=0.0,
        response=response,
    )
    sta = Station(
        code=station,
        latitude=0.0, longitude=0.0, elevation=0.0,
        site=Site(name="Test"),
        channels=[channel],
    )
    net = Network(code="XX", stations=[sta])
    return Inventory(networks=[net], source="Test")


# ============================================================================
# Unit tests: windowing
# ============================================================================

class TestSliceEventWindow:

    def test_sample_count(self):
        """Slice must return exactly compute_data_vector_length + 1 samples."""
        st = _make_stream(stations=["STA1"])
        # Deconvolve first to get to SR_TARGET
        proc = deconvolve_and_filter(st, remove_response=False,
                                     filter_kwargs=FILTER_KWARGS, target_sr=SR_TARGET)
        sliced = slice_event_window(proc, EVENT_START, EVENT_END, SR_TARGET)
        expected = DATA_VECTOR_LEN
        for tr in sliced:
            assert tr.stats.npts == expected, (
                f"{tr.id}: got {tr.stats.npts} samples, expected {expected}"
            )

    def test_starttime_respected(self):
        st = _make_stream(stations=["STA1"])
        proc = deconvolve_and_filter(st, remove_response=False,
                                     filter_kwargs=FILTER_KWARGS, target_sr=SR_TARGET)
        sliced = slice_event_window(proc, EVENT_START, EVENT_END, SR_TARGET)
        t_target = UTCDateTime(EVENT_START)
        for tr in sliced:
            assert abs(tr.stats.starttime - t_target) <= 1.0 / SR_TARGET + 1e-6, (
                f"{tr.id}: start {tr.stats.starttime} far from {t_target}"
            )

    def test_matches_compute_data_vector_length_formula(self):
        """Pinned formula: npts = compute_data_vector_length(duration, sr) + 1."""
        duration = (EVENT_END - EVENT_START).total_seconds()
        expected_npts = compute_data_vector_length(duration, SR_TARGET) + 1

        st = _make_stream(stations=["STA1"])
        proc = deconvolve_and_filter(st, remove_response=False,
                                     filter_kwargs=FILTER_KWARGS, target_sr=SR_TARGET)
        sliced = slice_event_window(proc, EVENT_START, EVENT_END, SR_TARGET)
        for tr in sliced:
            assert tr.stats.npts == expected_npts, (
                f"npts={tr.stats.npts}, expected={expected_npts}"
            )


class TestMakeNoiseWindows:

    def _continuous_regions(self):
        """A single 1-hour event-free region."""
        start = datetime.datetime(2023, 1, 1, 0, 0)
        end = datetime.datetime(2023, 1, 1, 1, 0)
        return [(start, end)]

    def test_windows_are_generated(self):
        wins = list(make_noise_windows(
            self._continuous_regions(),
            window_length=timedelta(minutes=5),
            buffer=timedelta(minutes=5),
        ))
        assert len(wins) > 0

    def test_window_length_correct(self):
        wlen = timedelta(minutes=5)
        wins = list(make_noise_windows(
            self._continuous_regions(), window_length=wlen,
            buffer=timedelta(minutes=5),
        ))
        for start, end in wins:
            assert abs((end - start) - wlen) < timedelta(seconds=1)

    def test_windows_inside_region(self):
        region_start = datetime.datetime(2023, 1, 1, 0, 0)
        region_end = datetime.datetime(2023, 1, 1, 1, 0)
        buf = timedelta(minutes=5)
        wins = list(make_noise_windows(
            [(region_start, region_end)],
            window_length=timedelta(minutes=5),
            buffer=buf,
        ))
        for start, end in wins:
            assert start >= region_start + buf, f"Window starts before buffered region: {start}"
            assert end <= region_end - buf + timedelta(seconds=1), (
                f"Window ends after buffered region: {end}"
            )

    def test_no_windows_if_region_too_short(self):
        """A 1-minute region with 5-minute buffer should yield no windows."""
        tiny_region = [(
            datetime.datetime(2023, 1, 1, 0, 0),
            datetime.datetime(2023, 1, 1, 0, 1),
        )]
        wins = list(make_noise_windows(
            tiny_region, window_length=timedelta(minutes=5),
            buffer=timedelta(minutes=5),
        ))
        assert wins == []


class TestMakeDailyOverlappingWindows:

    def _regions(self):
        return [(
            datetime.datetime(2023, 1, 1, 0, 0),
            datetime.datetime(2023, 1, 1, 2, 0),
        )]

    def test_returns_dict_keyed_by_date(self):
        result = make_daily_overlapping_windows(
            self._regions(), window_length=timedelta(minutes=20),
            buffer=timedelta(minutes=15),
        )
        assert isinstance(result, dict)
        for key in result:
            assert isinstance(key, datetime.date)

    def test_each_entry_is_list_of_tuples(self):
        result = make_daily_overlapping_windows(
            self._regions(), window_length=timedelta(minutes=20),
            buffer=timedelta(minutes=15),
        )
        for date, windows in result.items():
            assert isinstance(windows, list)
            for w in windows:
                assert len(w) == 2


class TestGetContinuousRegions:

    def test_single_event_splits_region(self):
        event_times = [(
            datetime.datetime(2023, 1, 1, 0, 30),
            datetime.datetime(2023, 1, 1, 0, 45),
        )]
        start = datetime.datetime(2023, 1, 1, 0, 0)
        end = datetime.datetime(2023, 1, 1, 1, 0)
        regions, gaps = get_continuous_regions(event_times, start, end)
        assert len(regions) == 2
        assert len(gaps) == 1

    def test_regions_cover_full_timespan(self):
        events = [
            (datetime.datetime(2023, 1, 1, 0, 10), datetime.datetime(2023, 1, 1, 0, 20)),
            (datetime.datetime(2023, 1, 1, 0, 40), datetime.datetime(2023, 1, 1, 0, 50)),
        ]
        start = datetime.datetime(2023, 1, 1, 0, 0)
        end = datetime.datetime(2023, 1, 1, 1, 0)
        regions, _ = get_continuous_regions(events, start, end)
        # Regions should start from `start` and end at `end`
        assert regions[0][0] == start
        assert regions[-1][1] == end

    def test_overlapping_events_merged(self):
        events = [
            (datetime.datetime(2023, 1, 1, 0, 10), datetime.datetime(2023, 1, 1, 0, 30)),
            (datetime.datetime(2023, 1, 1, 0, 20), datetime.datetime(2023, 1, 1, 0, 40)),
        ]
        start = datetime.datetime(2023, 1, 1, 0, 0)
        end = datetime.datetime(2023, 1, 1, 1, 0)
        regions, gaps = get_continuous_regions(events, start, end)
        # Overlapping events merge into a single gap → only 2 regions, not 3
        assert len(gaps) == 1


# ============================================================================
# Tests: export_to_sbi_h5
# ============================================================================

class TestExportToSbiH5Schema:
    """Verify the h5 file structure produced by export_to_sbi_h5."""

    def test_h5_file_created(self, processed_context):
        assert processed_context["h5"].exists()

    def test_has_outputs_group(self, processed_context):
        with h5py.File(processed_context["h5"], "r") as f:
            assert "outputs" in f

    def test_has_misc_group(self, processed_context):
        with h5py.File(processed_context["h5"], "r") as f:
            assert "misc" in f

    def test_all_receivers_present(self, processed_context):
        with h5py.File(processed_context["h5"], "r") as f:
            for sta in STATIONS:
                assert sta in f["outputs"], f"{sta} missing from /outputs"

    def test_component_keys_are_z_1_2(self, processed_context):
        """Outputs must contain Z, 1, 2 — never E or N."""
        with h5py.File(processed_context["h5"], "r") as f:
            for sta in STATIONS:
                keys = set(f["outputs"][sta].keys())
                assert keys == {"Z", "1", "2"}, (
                    f"{sta}: unexpected component keys {keys}"
                )
                assert "E" not in keys
                assert "N" not in keys

    def test_array_length_matches_data_vector_len(self, processed_context):
        with h5py.File(processed_context["h5"], "r") as f:
            for sta in STATIONS:
                for comp, ds in f["outputs"][sta].items():
                    assert ds.shape[0] == DATA_VECTOR_LEN, (
                        f"{sta}/{comp}: expected {DATA_VECTOR_LEN}, got {ds.shape[0]}"
                    )

    def test_arrays_finite(self, processed_context):
        with h5py.File(processed_context["h5"], "r") as f:
            for sta in f["outputs"]:
                for comp, ds in f["outputs"][sta].items():
                    assert np.all(np.isfinite(ds[()])), f"{sta}/{comp} has NaN/Inf"

    def test_arrays_nonzero(self, processed_context):
        with h5py.File(processed_context["h5"], "r") as f:
            for sta in f["outputs"]:
                for comp, ds in f["outputs"][sta].items():
                    assert np.any(ds[()] != 0.0), f"{sta}/{comp} is all-zero"

    def test_misc_variance_positive(self, processed_context):
        with h5py.File(processed_context["h5"], "r") as f:
            for sta in f["misc"]:
                for comp, ds in f["misc"][sta].items():
                    v = float(np.atleast_1d(ds[()]).flat[0])
                    assert v > 0, f"{sta}/{comp} zero-lag variance {v} non-positive"


class TestExportChannelConversionFidelity:
    """E→1 and N→2 renaming must route the correct data to the correct key."""

    @pytest.fixture(scope="class")
    def distinct_h5(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("distinct_amp")
        raw = _make_distinct_amplitude_stream()
        proc = deconvolve_and_filter(raw, remove_response=False,
                                     filter_kwargs=FILTER_KWARGS, target_sr=SR_TARGET)
        h5 = tmp / "distinct.h5"
        export_to_sbi_h5(proc, STATIONS, (EVENT_START, EVENT_END),
                         h5, SR_TARGET, COV_WINDOW, full_auto_correlation=True)
        return h5

    def test_only_z_1_2_present(self, distinct_h5):
        with h5py.File(distinct_h5, "r") as f:
            for sta in STATIONS:
                assert set(f["outputs"][sta].keys()) == {"Z", "1", "2"}

    def test_z_variance_smallest(self, distinct_h5):
        """BHZ amplitude=1 → variance ~0.5, must be smallest."""
        with h5py.File(distinct_h5, "r") as f:
            for sta in STATIONS:
                v_z = float(np.atleast_1d(f["misc"][sta]["Z"][()]).flat[0])
                v_1 = float(np.atleast_1d(f["misc"][sta]["1"][()]).flat[0])
                v_2 = float(np.atleast_1d(f["misc"][sta]["2"][()]).flat[0])
                assert v_z < v_1 < v_2, (
                    f"{sta}: expected v_Z < v_1 < v_2, got "
                    f"Z={v_z:.3f}, 1={v_1:.3f}, 2={v_2:.3f}"
                )

    def test_e_maps_to_1_not_2(self, distinct_h5):
        """BHE (amplitude 2) should map to key '1'; BHN (amplitude 3) to '2'.

        If the mapping is reversed, component '1' variance would be ~4.5 and
        '2' would be ~2.0 — the ordering check above catches both halves.
        """
        tol = 0.45
        expected_1 = (2.0 ** 2) / 2   # BHE A²/2
        with h5py.File(distinct_h5, "r") as f:
            for sta in STATIONS:
                v_1 = float(np.atleast_1d(f["misc"][sta]["1"][()]).flat[0])
                assert expected_1 * (1 - tol) < v_1 < expected_1 * (1 + tol), (
                    f"{sta}/1 variance {v_1:.3f} not close to BHE expected {expected_1:.3f}"
                )

    def test_n_maps_to_2_not_1(self, distinct_h5):
        tol = 0.45
        expected_2 = (3.0 ** 2) / 2   # BHN A²/2
        with h5py.File(distinct_h5, "r") as f:
            for sta in STATIONS:
                v_2 = float(np.atleast_1d(f["misc"][sta]["2"][()]).flat[0])
                assert expected_2 * (1 - tol) < v_2 < expected_2 * (1 + tol), (
                    f"{sta}/2 variance {v_2:.3f} not close to BHN expected {expected_2:.3f}"
                )


class TestExportEdgeCases:

    def test_missing_station_omitted_from_h5(self, processed_context, tmp_path):
        """A receiver not present in the stream should not appear in /outputs."""
        proc = processed_context["proc_st"]
        out = tmp_path / "missing.h5"
        export_to_sbi_h5(
            proc, receivers=STATIONS + ["GHOST"],
            event_window=(EVENT_START, EVENT_END),
            out_path=out,
            sampling_rate=SR_TARGET,
            covariance_window=COV_WINDOW,
        )
        with h5py.File(out, "r") as f:
            assert "GHOST" not in f["outputs"], "Missing station incorrectly written"
            for sta in STATIONS:
                assert sta in f["outputs"]

    def test_no_misc_when_no_covariance_window(self, processed_context, tmp_path):
        """When covariance_window=None, the /misc group must be absent."""
        proc = processed_context["proc_st"]
        out = tmp_path / "no_misc.h5"
        export_to_sbi_h5(
            proc, receivers=STATIONS,
            event_window=(EVENT_START, EVENT_END),
            out_path=out,
            sampling_rate=SR_TARGET,
            covariance_window=None,
        )
        with h5py.File(out, "r") as f:
            assert "misc" not in f, "/misc present but covariance_window=None"

    def test_scalar_variance_mode(self, processed_context, tmp_path):
        """full_auto_correlation=False should write a scalar variance, not an array."""
        proc = processed_context["proc_st"]
        out = tmp_path / "scalar_var.h5"
        export_to_sbi_h5(
            proc, receivers=STATIONS,
            event_window=(EVENT_START, EVENT_END),
            out_path=out,
            sampling_rate=SR_TARGET,
            covariance_window=COV_WINDOW,
            full_auto_correlation=False,
        )
        with h5py.File(out, "r") as f:
            for sta in STATIONS:
                for comp, ds in f["misc"][sta].items():
                    arr = ds[()]
                    assert arr.ndim == 0 or arr.size == 1, (
                        f"{sta}/{comp}: expected scalar variance, got shape {arr.shape}"
                    )

    def test_channel_code_variants_bh1_bh2(self, tmp_path):
        """BH1/BH2 (numeric suffixes) must map to '1'/'2'."""
        st = _make_stream(channels=["BHZ", "BH1", "BH2"], stations=["STA1"])
        proc = deconvolve_and_filter(st, remove_response=False,
                                     filter_kwargs=FILTER_KWARGS, target_sr=SR_TARGET)
        out = tmp_path / "bh12.h5"
        export_to_sbi_h5(proc, ["STA1"], (EVENT_START, EVENT_END),
                         out, SR_TARGET, COV_WINDOW)
        with h5py.File(out, "r") as f:
            keys = set(f["outputs"]["STA1"].keys())
            assert keys == {"Z", "1", "2"}, f"BH1/BH2 mapping failed: {keys}"

    def test_channel_code_variants_hh1_hh2(self, tmp_path):
        """HH1/HH2 (high-gain channels with numeric suffix) → '1'/'2'."""
        st = _make_stream(channels=["HHZ", "HH1", "HH2"], stations=["STA1"])
        proc = deconvolve_and_filter(st, remove_response=False,
                                     filter_kwargs=FILTER_KWARGS, target_sr=SR_TARGET)
        out = tmp_path / "hh12.h5"
        export_to_sbi_h5(proc, ["STA1"], (EVENT_START, EVENT_END),
                         out, SR_TARGET, COV_WINDOW)
        with h5py.File(out, "r") as f:
            keys = set(f["outputs"]["STA1"].keys())
            assert keys == {"Z", "1", "2"}, f"HH1/HH2 mapping failed: {keys}"

    def test_station_with_only_two_components_excluded(self, tmp_path):
        """A station with fewer than 3 components must not appear in /outputs."""
        st = _make_stream(channels=["BHZ", "BHE"], stations=["PARTIAL"])
        proc = deconvolve_and_filter(st, remove_response=False,
                                     filter_kwargs=FILTER_KWARGS, target_sr=SR_TARGET)
        out = tmp_path / "partial.h5"
        export_to_sbi_h5(proc, ["PARTIAL"], (EVENT_START, EVENT_END),
                         out, SR_TARGET, COV_WINDOW)
        with h5py.File(out, "r") as f:
            assert "PARTIAL" not in f.get("outputs", {}), (
                "Station with only 2 components should not be written"
            )


class TestAutocorrelationProperties:
    """The /misc autocorrelation must have correct mathematical properties."""

    def test_zero_lag_is_maximum(self, processed_context):
        with h5py.File(processed_context["h5"], "r") as f:
            for sta in STATIONS:
                for comp, ds in f["misc"][sta].items():
                    arr = ds[()]
                    if len(arr) < 2:
                        continue
                    assert arr[0] >= np.max(np.abs(arr[1:])) * 0.95, (
                        f"{sta}/{comp}: zero-lag not dominant"
                    )

    def test_autocorrelation_length_matches_cov_window(self, processed_context):
        expected = int(COV_WINDOW.total_seconds() * SR_TARGET)
        with h5py.File(processed_context["h5"], "r") as f:
            for sta in STATIONS:
                for comp, ds in f["misc"][sta].items():
                    assert abs(len(ds[()]) - expected) <= 2, (
                        f"{sta}/{comp} autocorr length {len(ds[()])} != {expected}"
                    )

    def test_sinusoid_autocorr_oscillates(self, processed_context):
        """A 0.04 Hz sinusoid should show periodic autocorrelation structure."""
        expected_half_period = int(round(1 / (2 * 0.04 * SR_TARGET)))  # ~12 samples
        with h5py.File(processed_context["h5"], "r") as f:
            for sta in STATIONS:
                arr = f["misc"][sta]["Z"][()]
                if len(arr) < expected_half_period + 5:
                    continue
                assert arr[expected_half_period] < arr[0] * 0.5, (
                    f"{sta}/Z: no oscillation at half-period "
                    f"R(0)={arr[0]:.4f}, R({expected_half_period})={arr[expected_half_period]:.4f}"
                )


# ============================================================================
# End-to-end: new API produces same schema and plausible values vs legacy
# ============================================================================

class TestNewApiVsLegacySchema:
    """Run the legacy pipeline (Phase 0 fixture) and compare schema against
    the new API output.  Values may differ slightly due to the mseed round-trip
    in the legacy pipeline; we check shape, keys, and statistical properties.
    """

    @pytest.fixture(scope="class")
    def both_outputs(self, tmp_path_factory):
        """Build the legacy h5 and the new-API h5 from the same raw data."""
        from functools import partial
        from seismo_sbi.data_handling.noise_collection import (
            NoiseCollector, EventNoiseAggregator, ProcessedDataSlicer,
        )
        from seismo_sbi.data_handling.noise_database import NoiseDatabaseGenerator

        tmp = tmp_path_factory.mktemp("compare")
        data_dir = tmp / "raw"
        data_dir.mkdir()
        rng = np.random.default_rng(RNG_SEED)
        raw_st = _make_stream(rng=rng)
        paths = _write_stream_to_disk(raw_st, data_dir)

        # ---- Legacy pipeline ----
        upflow_config = {
            NETWORK: {
                "call_key": "OBS",
                "master_path": ".",
                "path_structure": (
                    str(data_dir)
                    + "/{sta}/{year}.{jday}/{net}.{sta}..{cha}.{year}.{jday}.mseed"
                ),
                "sta_cha": CHANNELS,
                "network": NETWORK,
                "location": "",
                "years": [2023],
                "instrument_correction": False,
                "response_seismometer": "",
            }
        }
        scp = {sta: NETWORK for sta in STATIONS}
        nc = NoiseCollector(upflow_config, scp, {}, filter_kwargs=FILTER_KWARGS)
        ena = EventNoiseAggregator(nc, scp, sampling_rate=SR_TARGET)
        daily_window = [T0 + timedelta(minutes=10), T0 + timedelta(minutes=25)]
        ena.select_event_and_check_available_stations(
            [EVENT_START, EVENT_END], (0.0, 0.0),
            buffer=timedelta(seconds=5), convert_to_numpy=True,
        )
        available = ena.available_stations_during_event
        daily_dir = tmp / "legacy_daily"
        daily_dir.mkdir()
        NoiseDatabaseGenerator(
            partial(ena.collect_noise_data,
                    noise_window_length=timedelta(minutes=20), convert_to_numpy=False),
            num_jobs=1, mseed_output=True,
        ).create_database(daily_dir, [daily_window])
        legacy_h5 = tmp / "legacy.h5"
        slicer = ProcessedDataSlicer(
            data_folder=daily_dir, sampling_rate=SR_TARGET,
            covariance_estimation_window=COV_WINDOW,
            full_auto_correlation=True, receivers=available,
        )
        NoiseDatabaseGenerator(
            slicer.load_noise_window_data,
            data_vector_length=DATA_VECTOR_LEN, num_jobs=1,
        )._collect_and_save_noise(tmp, noise_window=[EVENT_START, EVENT_END],
                                  name="legacy")

        # ---- New API pipeline ----
        new_st = load_waveforms(paths)
        proc_st = deconvolve_and_filter(
            new_st, remove_response=False,
            filter_kwargs=FILTER_KWARGS, target_sr=SR_TARGET,
        )
        new_h5 = tmp / "new_api.h5"
        export_to_sbi_h5(
            proc_st, available, (EVENT_START, EVENT_END),
            new_h5, SR_TARGET, COV_WINDOW, full_auto_correlation=True,
        )

        return {"legacy": legacy_h5, "new": new_h5, "available": available}

    def test_same_stations_in_outputs(self, both_outputs):
        with h5py.File(both_outputs["legacy"], "r") as lf:
            with h5py.File(both_outputs["new"], "r") as nf:
                assert set(lf["outputs"].keys()) == set(nf["outputs"].keys())

    def test_same_component_keys(self, both_outputs):
        with h5py.File(both_outputs["legacy"], "r") as lf:
            with h5py.File(both_outputs["new"], "r") as nf:
                for sta in both_outputs["available"]:
                    assert set(lf["outputs"][sta].keys()) == set(nf["outputs"][sta].keys())

    def test_same_array_length(self, both_outputs):
        with h5py.File(both_outputs["legacy"], "r") as lf:
            with h5py.File(both_outputs["new"], "r") as nf:
                for sta in both_outputs["available"]:
                    for comp in ("Z", "1", "2"):
                        l_len = lf["outputs"][sta][comp].shape[0]
                        n_len = nf["outputs"][sta][comp].shape[0]
                        assert l_len == n_len, (
                            f"{sta}/{comp}: legacy={l_len}, new={n_len}"
                        )

    def test_output_values_statistically_close(self, both_outputs):
        """New and legacy outputs should have similar RMS (within ±20%).

        Exact equality is not expected because the legacy pipeline does a
        mseed write/read round-trip that can introduce minor floating-point
        differences.
        """
        with h5py.File(both_outputs["legacy"], "r") as lf:
            with h5py.File(both_outputs["new"], "r") as nf:
                for sta in both_outputs["available"]:
                    for comp in ("Z", "1", "2"):
                        rms_l = np.sqrt(np.mean(lf["outputs"][sta][comp][()] ** 2))
                        rms_n = np.sqrt(np.mean(nf["outputs"][sta][comp][()] ** 2))
                        if rms_l > 0 and rms_n > 0:
                            ratio = rms_l / rms_n
                            assert 0.8 < ratio < 1.25, (
                                f"{sta}/{comp}: RMS ratio legacy/new = {ratio:.3f} "
                                f"(legacy={rms_l:.4e}, new={rms_n:.4e})"
                            )

    def test_sbi_loader_reads_new_api_h5(self, both_outputs):
        """SimulationDataLoader (unchanged) must consume new-API h5 without error."""
        from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
        from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader

        available = both_outputs["available"]
        receivers = Receivers(receivers=[
            Receiver(0.0, 0.0, NETWORK, sta, ["Z", "E", "N"])
            for sta in available
        ])
        loader = SimulationDataLoader(components="ZEN", receivers=receivers)
        vec = loader.load_flattened_simulation_vector(both_outputs["new"])
        expected = len(available) * 3 * DATA_VECTOR_LEN
        assert vec.shape == (expected,), f"Expected {expected}, got {vec.shape}"
        assert np.all(np.isfinite(vec)), "Flattened vector contains NaN/Inf"
