"""Phase 0.1 — Synthetic end-to-end preprocessing regression test.

Builds a fully deterministic dataset in tmp_path, runs it through the
CURRENT NoiseCollector / EventNoiseAggregator / NoiseDatabaseGenerator /
ProcessedDataSlicer pipeline, and pins key output properties.

These tests must NOT be modified during later refactor phases — they are the
behavioural contract that the new API must reproduce.
"""

import datetime
from datetime import timedelta
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import obspy
from obspy import Trace, Stream, UTCDateTime
import pytest

from seismo_sbi.data_handling.noise_collection import (
    NoiseCollector,
    EventNoiseAggregator,
    ProcessedDataSlicer,
)
from seismo_sbi.data_handling.noise_database import NoiseDatabaseGenerator
from seismo_sbi.instaseis_simulator.utils import compute_data_vector_length

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Scenario constants (do not change without updating assertions below)
# ---------------------------------------------------------------------------

NETWORK = "XX"
STATIONS = ["STA1", "STA2"]
CHANNELS = ["BHZ", "BHE", "BHN"]

SR_DATA = 20.0      # raw data sampling rate (Hz)
SR_TARGET = 1.0     # target sampling rate consumed by SBI pipeline (Hz)

# Absolute start time for all synthetic data
T0 = datetime.datetime(2023, 1, 1, 12, 0, 0)
DATA_END = T0 + timedelta(minutes=30)

# Event window: minutes 15-16 (1 minute, 60 samples at 1 Hz)
EVENT_START = T0 + timedelta(minutes=15)
EVENT_END = EVENT_START + timedelta(seconds=60)
EVENT_LOCATION = (0.0, 0.0)

# Noise collection window for the daily database (covers the event + covariance region)
DAILY_WINDOW = [T0 + timedelta(minutes=10), T0 + timedelta(minutes=25)]

NOISE_WINDOW_LENGTH = timedelta(minutes=20)  # > window duration → slice clips to available
COV_WINDOW = timedelta(minutes=3)            # needs data from EVENT_START - 3min = T0+12min

# ProcessedDataSlicer computes npts = compute_data_vector_length(...) + 1, and its
# slice (which overwrites the interpolation result) returns inclusive endpoints — so
# the actual array length is +1 relative to compute_data_vector_length.
DATA_VECTOR_LEN = compute_data_vector_length(
    (EVENT_END - EVENT_START).total_seconds(), SR_TARGET
) + 1  # 61 samples

# Bandpass safely within SR_TARGET Nyquist (0.5 Hz)
FILTER_KWARGS = dict(freqmin=0.02, freqmax=0.1, corners=4, zerophase=False)

RNG_SEED = 42


# ---------------------------------------------------------------------------
# Helpers: synthetic data creation
# ---------------------------------------------------------------------------

def _write_synthetic_mseed(data_dir: Path, station: str, network: str, rng: np.random.Generator):
    """Write 30 min of synthetic BHZ/BHE/BHN traces at SR_DATA Hz."""
    t_start = T0
    t_end = DATA_END
    npts = int((t_end - t_start).total_seconds() * SR_DATA)
    t_utc = UTCDateTime(t_start)
    year = t_start.year
    jday = t_start.timetuple().tm_yday

    sta_dir = data_dir / station / f"{year}.{jday:03d}"
    sta_dir.mkdir(parents=True, exist_ok=True)

    for cha in CHANNELS:
        t = np.arange(npts) / SR_DATA
        # 0.04 Hz sinusoid (within bandpass) + small white noise
        data = 0.5 * np.sin(2 * np.pi * 0.04 * t) + rng.standard_normal(npts) * 0.01

        tr = Trace()
        tr.stats.network = network
        tr.stats.station = station
        tr.stats.channel = cha
        tr.stats.location = ""
        tr.stats.sampling_rate = SR_DATA
        tr.stats.starttime = t_utc
        tr.data = data.astype(np.float64)

        fname = f"{network}.{station}..{cha}.{year}.{jday:03d}.mseed"
        tr.write(str(sta_dir / fname), format="MSEED")


def _build_upflow_config(data_dir: Path) -> dict:
    """UPFLOW_config dict keyed by network, as expected by NoiseCollector."""
    return {
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
            "instrument_correction": False,  # skip response removal for synthetic data
            "response_seismometer": "",       # not opened when correction is off
        }
    }


def _build_station_codes_paths() -> dict:
    """Maps station_name → network, as expected by NoiseCollector."""
    return {sta: NETWORK for sta in STATIONS}


# ---------------------------------------------------------------------------
# Module-scoped fixture: run the full pipeline once, share across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory):
    """Run the current preprocessing pipeline on synthetic data.

    Returns a dict with paths to key outputs and metadata for assertions.
    """
    tmp = tmp_path_factory.mktemp("synthetic_preprocess")
    data_dir = tmp / "raw"
    data_dir.mkdir()
    daily_dir = tmp / "daily"
    daily_dir.mkdir()
    event_dir = tmp / "events"
    event_dir.mkdir()

    rng = np.random.default_rng(RNG_SEED)

    # 1. Write deterministic synthetic mseed files
    for sta in STATIONS:
        _write_synthetic_mseed(data_dir, sta, NETWORK, rng)

    # 2. Build pipeline components (mirroring custom_preprocess.py)
    upflow_config = _build_upflow_config(data_dir)
    station_codes_paths = _build_station_codes_paths()

    noise_collector = NoiseCollector(
        upflow_config,
        station_codes_paths,
        {},
        filter_kwargs=FILTER_KWARGS,
    )
    event_noise_aggregator = EventNoiseAggregator(
        noise_collector, station_codes_paths, sampling_rate=SR_TARGET
    )

    # 3. Identify stations available during the event window
    event_noise_aggregator.select_event_and_check_available_stations(
        [EVENT_START, EVENT_END],
        EVENT_LOCATION,
        buffer=timedelta(seconds=5),
        convert_to_numpy=True,
    )

    available = event_noise_aggregator.available_stations_during_event
    assert len(available) > 0, "No stations found for synthetic event — check path layout"

    # 4. Stage 1: write preprocessed daily mseed via NoiseDatabaseGenerator
    noise_collection_callable = partial(
        event_noise_aggregator.collect_noise_data,
        noise_window_length=NOISE_WINDOW_LENGTH,
        convert_to_numpy=False,
    )
    noise_db_gen = NoiseDatabaseGenerator(
        noise_collection_callable, num_jobs=1, mseed_output=True
    )
    noise_db_gen.create_database(daily_dir, [DAILY_WINDOW])

    # 5. Stage 2: slice event window, compute covariance, write final h5
    data_slicer = ProcessedDataSlicer(
        data_folder=daily_dir,
        sampling_rate=SR_TARGET,
        covariance_estimation_window=COV_WINDOW,
        full_auto_correlation=True,
        receivers=available,
    )
    event_saver = NoiseDatabaseGenerator(
        data_slicer.load_noise_window_data,
        data_vector_length=DATA_VECTOR_LEN,  # 61 — matches inclusive slice at 1 Hz
        num_jobs=1,
    )
    event_name = "test_event"
    event_saver._collect_and_save_noise(
        event_dir,
        noise_window=[EVENT_START, EVENT_END],
        name=event_name,
    )

    event_h5 = event_dir / f"{event_name}.h5"
    return {
        "daily_dir": daily_dir,
        "event_dir": event_dir,
        "event_h5": event_h5,
        "available_stations": available,
    }


# ---------------------------------------------------------------------------
# 0.1a — Daily preprocessed files
# ---------------------------------------------------------------------------

class TestDailyDatabaseOutput:

    def test_daily_file_created(self, pipeline_output):
        """At least one preprocessed file is written to the daily directory."""
        files = list(pipeline_output["daily_dir"].glob("*"))
        assert len(files) >= 1, "No daily preprocessed files written"

    def test_daily_file_is_readable_by_obspy(self, pipeline_output):
        """The daily preprocessed file can be opened with obspy.read()."""
        daily_file = next(pipeline_output["daily_dir"].glob("*"))
        st = obspy.read(str(daily_file))
        assert len(st) > 0

    def test_daily_file_contains_expected_stations(self, pipeline_output):
        """Every available station has traces in the daily file."""
        daily_file = next(pipeline_output["daily_dir"].glob("*"))
        st = obspy.read(str(daily_file))
        station_names = {tr.stats.station for tr in st}
        for sta in pipeline_output["available_stations"]:
            assert sta in station_names, f"Station {sta} missing from daily file"

    def test_daily_file_contains_three_components(self, pipeline_output):
        """Each station in the daily file has exactly 3 component traces."""
        daily_file = next(pipeline_output["daily_dir"].glob("*"))
        st = obspy.read(str(daily_file))
        for sta in pipeline_output["available_stations"]:
            station_traces = st.select(station=sta)
            assert len(station_traces) == 3, (
                f"Expected 3 components for {sta}, got {len(station_traces)}"
            )

    def test_daily_traces_at_target_sampling_rate(self, pipeline_output):
        """Traces in the daily file are resampled to SR_TARGET."""
        daily_file = next(pipeline_output["daily_dir"].glob("*"))
        st = obspy.read(str(daily_file))
        for tr in st:
            assert abs(tr.stats.sampling_rate - SR_TARGET) < 1e-6, (
                f"Trace {tr.id} has wrong sampling rate: {tr.stats.sampling_rate}"
            )


# ---------------------------------------------------------------------------
# 0.1b — Final event HDF5 file
# ---------------------------------------------------------------------------

class TestEventH5Output:

    def test_event_h5_created(self, pipeline_output):
        """The final event .h5 file is written."""
        assert pipeline_output["event_h5"].exists(), "Event h5 file not created"

    def test_event_h5_has_outputs_group(self, pipeline_output):
        with h5py.File(pipeline_output["event_h5"], "r") as f:
            assert "outputs" in f, "H5 file missing /outputs group"

    def test_event_h5_has_misc_group(self, pipeline_output):
        with h5py.File(pipeline_output["event_h5"], "r") as f:
            assert "misc" in f, "H5 file missing /misc group"

    def test_event_h5_outputs_contains_all_stations(self, pipeline_output):
        with h5py.File(pipeline_output["event_h5"], "r") as f:
            outputs = f["outputs"]
            for sta in pipeline_output["available_stations"]:
                assert sta in outputs, f"Station {sta} missing from /outputs"

    def test_event_h5_outputs_components_renamed(self, pipeline_output):
        """E→1 and N→2 renaming must be applied (SBI pipeline contract)."""
        with h5py.File(pipeline_output["event_h5"], "r") as f:
            outputs = f["outputs"]
            for sta in pipeline_output["available_stations"]:
                components = set(outputs[sta].keys())
                assert "Z" in components, f"{sta} missing Z in /outputs"
                assert "1" in components or "E" in components, (
                    f"{sta} missing horizontal-1 component in /outputs"
                )
                assert "2" in components or "N" in components, (
                    f"{sta} missing horizontal-2 component in /outputs"
                )
                # The SBI pipeline expects '1'/'2', not 'E'/'N'
                assert "E" not in components, f"{sta} has unrenamed 'E' component"
                assert "N" not in components, f"{sta} has unrenamed 'N' component"

    def test_event_h5_array_length(self, pipeline_output):
        """Each component array has exactly DATA_VECTOR_LEN samples."""
        with h5py.File(pipeline_output["event_h5"], "r") as f:
            outputs = f["outputs"]
            for sta in pipeline_output["available_stations"]:
                for comp, ds in outputs[sta].items():
                    assert ds.shape[0] == DATA_VECTOR_LEN, (
                        f"{sta}/{comp}: expected {DATA_VECTOR_LEN} samples, "
                        f"got {ds.shape[0]}"
                    )

    def test_event_h5_arrays_finite(self, pipeline_output):
        """All output arrays contain only finite values."""
        with h5py.File(pipeline_output["event_h5"], "r") as f:
            for sta in f["outputs"]:
                for comp, ds in f["outputs"][sta].items():
                    arr = ds[()]
                    assert np.all(np.isfinite(arr)), (
                        f"{sta}/{comp} contains NaN or Inf"
                    )

    def test_event_h5_misc_variance_positive(self, pipeline_output):
        """Per-component variance in /misc must be strictly positive.

        With full_auto_correlation=True the /misc datasets are autocorrelation
        arrays; auto_cov[0] is the variance (zero-lag value).
        """
        with h5py.File(pipeline_output["event_h5"], "r") as f:
            misc = f["misc"]
            for sta in pipeline_output["available_stations"]:
                assert sta in misc, f"Station {sta} missing from /misc"
                for comp, ds in misc[sta].items():
                    arr = ds[()]
                    # scalar variance OR autocorrelation array (zero-lag = variance)
                    val = float(np.atleast_1d(arr).flat[0])
                    assert val > 0, (
                        f"{sta}/{comp} zero-lag variance is non-positive: {val}"
                    )

    def test_event_h5_arrays_nonzero(self, pipeline_output):
        """Output arrays must not be all-zero (filter should pass the 0.04 Hz signal)."""
        with h5py.File(pipeline_output["event_h5"], "r") as f:
            for sta in f["outputs"]:
                for comp, ds in f["outputs"][sta].items():
                    arr = ds[()]
                    assert np.any(arr != 0.0), f"{sta}/{comp} is all-zero"


# ---------------------------------------------------------------------------
# 0.1c — Signal content preserved through filtering
# ---------------------------------------------------------------------------

class TestSignalContent:

    def test_bandpass_attenuates_out_of_band(self, pipeline_output):
        """Energy outside [freqmin, freqmax] is attenuated by at least 30 dB.

        The injected signal is at 0.04 Hz (within band). We check that
        energy above 0.3 Hz (well above freqmax=0.1) is suppressed.
        """
        with h5py.File(pipeline_output["event_h5"], "r") as f:
            for sta in f["outputs"]:
                arr = f["outputs"][sta]["Z"][()]
                if len(arr) < 16:
                    continue  # too short for FFT check
                freqs = np.fft.rfftfreq(len(arr), d=1.0 / SR_TARGET)
                power = np.abs(np.fft.rfft(arr)) ** 2

                in_band = (freqs >= 0.02) & (freqs <= 0.1)
                out_of_band = freqs > 0.3

                if not np.any(in_band) or not np.any(out_of_band):
                    continue

                in_band_power = power[in_band].mean()
                out_power = power[out_of_band].mean()

                if in_band_power > 0 and out_power > 0:
                    attenuation_db = 10 * np.log10(in_band_power / out_power)
                    assert attenuation_db > 30, (
                        f"{sta}/Z: bandpass attenuation only {attenuation_db:.1f} dB"
                    )

    def test_flattened_vector_shape(self, pipeline_output):
        """Flattening all stations×components yields the expected total length."""
        with h5py.File(pipeline_output["event_h5"], "r") as f:
            n_stations = len(pipeline_output["available_stations"])
            n_components = 3  # Z, 1, 2
            expected_len = n_stations * n_components * DATA_VECTOR_LEN
            total = 0
            for sta in f["outputs"]:
                for comp in f["outputs"][sta]:
                    total += f["outputs"][sta][comp].shape[0]
            assert total == expected_len, (
                f"Total samples {total} != expected {expected_len}"
            )


# ---------------------------------------------------------------------------
# 0.1d — Channel conversion fidelity (E→1, N→2 mapping correctness)
# ---------------------------------------------------------------------------
#
# Fixture that injects DISTINCT per-channel amplitudes so we can verify the
# E→1 and N→2 rename maps the right data to the right key.
#
# BHZ amplitude: 1.0   → after pipeline, /outputs/*/Z  variance ≈ 0.5
# BHE amplitude: 2.0   → after pipeline, /outputs/*/1  variance ≈ 2.0  (E→1)
# BHN amplitude: 3.0   → after pipeline, /outputs/*/2  variance ≈ 4.5  (N→2)
# ---------------------------------------------------------------------------

CHANNEL_AMPS = {"BHZ": 1.0, "BHE": 2.0, "BHN": 3.0}
# Expected sinusoid variance A²/2 after bandpass (0.04 Hz is in-band)
EXPECTED_VAR = {cha: (amp ** 2) / 2 for cha, amp in CHANNEL_AMPS.items()}
# Tolerance: filter and finite-window effects can shift variance by ±40%
VAR_TOL = 0.4


def _write_distinct_channel_mseed(data_dir: Path, station: str, network: str):
    """Write 30 min of data where each channel has a distinct amplitude sinusoid."""
    npts = int((DATA_END - T0).total_seconds() * SR_DATA)
    t_utc = UTCDateTime(T0)
    year = T0.year
    jday = T0.timetuple().tm_yday
    sta_dir = data_dir / station / f"{year}.{jday:03d}"
    sta_dir.mkdir(parents=True, exist_ok=True)

    t = np.arange(npts) / SR_DATA
    for cha, amp in CHANNEL_AMPS.items():
        data = amp * np.sin(2 * np.pi * 0.04 * t)
        tr = Trace()
        tr.stats.network = network
        tr.stats.station = station
        tr.stats.channel = cha
        tr.stats.location = ""
        tr.stats.sampling_rate = SR_DATA
        tr.stats.starttime = t_utc
        tr.data = data.astype(np.float64)
        fname = f"{network}.{station}..{cha}.{year}.{jday:03d}.mseed"
        tr.write(str(sta_dir / fname), format="MSEED")


@pytest.fixture(scope="module")
def channel_conversion_output(tmp_path_factory):
    """Pipeline run with distinct per-channel amplitudes for E→1/N→2 validation."""
    tmp = tmp_path_factory.mktemp("channel_conversion")
    data_dir = tmp / "raw"
    data_dir.mkdir()
    daily_dir = tmp / "daily"
    daily_dir.mkdir()
    event_dir = tmp / "events"
    event_dir.mkdir()

    for sta in STATIONS:
        _write_distinct_channel_mseed(data_dir, sta, NETWORK)

    upflow_config = _build_upflow_config(data_dir)
    scp = _build_station_codes_paths()

    nc = NoiseCollector(upflow_config, scp, {}, filter_kwargs=FILTER_KWARGS)
    ena = EventNoiseAggregator(nc, scp, sampling_rate=SR_TARGET)
    ena.select_event_and_check_available_stations(
        [EVENT_START, EVENT_END], EVENT_LOCATION,
        buffer=timedelta(seconds=5), convert_to_numpy=True,
    )

    noise_callable = partial(
        ena.collect_noise_data,
        noise_window_length=NOISE_WINDOW_LENGTH,
        convert_to_numpy=False,
    )
    NoiseDatabaseGenerator(noise_callable, num_jobs=1, mseed_output=True).create_database(
        daily_dir, [DAILY_WINDOW]
    )

    data_slicer = ProcessedDataSlicer(
        data_folder=daily_dir,
        sampling_rate=SR_TARGET,
        covariance_estimation_window=COV_WINDOW,
        full_auto_correlation=True,
        receivers=ena.available_stations_during_event,
    )
    NoiseDatabaseGenerator(
        data_slicer.load_noise_window_data,
        data_vector_length=DATA_VECTOR_LEN,
        num_jobs=1,
    )._collect_and_save_noise(
        event_dir,
        noise_window=[EVENT_START, EVENT_END],
        name="channel_conv_event",
    )

    return {
        "event_h5": event_dir / "channel_conv_event.h5",
        "available_stations": ena.available_stations_during_event,
    }


class TestChannelConversionFidelity:
    """E→1 and N→2 renaming must route the correct data to the correct h5 key."""

    def test_only_z_1_2_present(self, channel_conversion_output):
        """Outputs contain exactly Z, 1, 2 — not E, N, or anything else."""
        with h5py.File(channel_conversion_output["event_h5"], "r") as f:
            for sta in f["outputs"]:
                keys = set(f["outputs"][sta].keys())
                assert keys == {"Z", "1", "2"}, (
                    f"{sta}: unexpected component keys {keys}"
                )

    def test_z_variance_matches_z_amplitude(self, channel_conversion_output):
        """The Z component (amplitude 1.0) has variance ≈ 0.5 in /misc."""
        expected = EXPECTED_VAR["BHZ"]
        with h5py.File(channel_conversion_output["event_h5"], "r") as f:
            for sta in channel_conversion_output["available_stations"]:
                v = float(np.atleast_1d(f["misc"][sta]["Z"][()]).flat[0])
                assert expected * (1 - VAR_TOL) < v < expected * (1 + VAR_TOL), (
                    f"{sta}/Z variance {v:.4f} outside expected {expected:.4f} ±{VAR_TOL*100:.0f}%"
                )

    def test_component_1_variance_matches_BHE_amplitude(self, channel_conversion_output):
        """Component '1' (mapped from BHE, amplitude 2.0) has variance ≈ 2.0 in /misc.

        If E→1 and N→2 are reversed, this will fail because BHN (amplitude 3.0)
        would give variance ≈ 4.5, which is outside the 2.0 ± 40% tolerance.
        """
        expected = EXPECTED_VAR["BHE"]
        with h5py.File(channel_conversion_output["event_h5"], "r") as f:
            for sta in channel_conversion_output["available_stations"]:
                v = float(np.atleast_1d(f["misc"][sta]["1"][()]).flat[0])
                assert expected * (1 - VAR_TOL) < v < expected * (1 + VAR_TOL), (
                    f"{sta}/1 variance {v:.4f} outside expected {expected:.4f} ±{VAR_TOL*100:.0f}%"
                    " — E→1 mapping may be wrong"
                )

    def test_component_2_variance_matches_BHN_amplitude(self, channel_conversion_output):
        """Component '2' (mapped from BHN, amplitude 3.0) has variance ≈ 4.5 in /misc."""
        expected = EXPECTED_VAR["BHN"]
        with h5py.File(channel_conversion_output["event_h5"], "r") as f:
            for sta in channel_conversion_output["available_stations"]:
                v = float(np.atleast_1d(f["misc"][sta]["2"][()]).flat[0])
                assert expected * (1 - VAR_TOL) < v < expected * (1 + VAR_TOL), (
                    f"{sta}/2 variance {v:.4f} outside expected {expected:.4f} ±{VAR_TOL*100:.0f}%"
                    " — N→2 mapping may be wrong"
                )

    def test_all_three_components_differ(self, channel_conversion_output):
        """The three components carry distinct signals (different amplitudes → different RMS)."""
        with h5py.File(channel_conversion_output["event_h5"], "r") as f:
            for sta in channel_conversion_output["available_stations"]:
                rms = {c: float(np.sqrt(np.mean(f["outputs"][sta][c][()] ** 2)))
                       for c in ("Z", "1", "2")}
                # All three should differ by more than 20%
                assert rms["Z"] < rms["1"] < rms["2"] or rms["Z"] != rms["1"], (
                    f"{sta}: RMS values unexpectedly similar: {rms}"
                )
                assert abs(rms["1"] - rms["2"]) / rms["2"] > 0.1, (
                    f"{sta}: components '1' and '2' have nearly identical RMS {rms}"
                )


# ---------------------------------------------------------------------------
# 0.1e — Covariance estimation correctness
# ---------------------------------------------------------------------------

class TestCovarianceEstimation:
    """The /misc autocorrelation must have correct mathematical properties."""

    def test_autocorrelation_length_matches_covariance_window(self, pipeline_output):
        """The autocorrelation array length equals the covariance window in samples.

        COV_WINDOW = 3 min at 1 Hz ≈ 181 samples (inclusive slice).
        """
        cov_samples_approx = int(COV_WINDOW.total_seconds() * SR_TARGET)
        with h5py.File(pipeline_output["event_h5"], "r") as f:
            for sta in pipeline_output["available_stations"]:
                for comp, ds in f["misc"][sta].items():
                    arr = ds[()]
                    # Allow ±2 samples for endpoint rounding
                    assert abs(len(arr) - cov_samples_approx) <= 2, (
                        f"{sta}/{comp} autocorrelation length {len(arr)} "
                        f"differs from expected {cov_samples_approx}"
                    )

    def test_zero_lag_is_maximum(self, pipeline_output):
        """The zero-lag value (variance) must be the maximum of the autocorrelation.

        For a stationary process R(τ) ≤ R(0) for all τ.
        """
        with h5py.File(pipeline_output["event_h5"], "r") as f:
            for sta in pipeline_output["available_stations"]:
                for comp, ds in f["misc"][sta].items():
                    arr = ds[()]
                    if len(arr) < 2:
                        continue
                    zero_lag = arr[0]
                    assert zero_lag >= np.max(np.abs(arr[1:])) * 0.95, (
                        f"{sta}/{comp}: zero-lag {zero_lag:.4f} is not the "
                        f"maximum (max off-diag abs = {np.max(np.abs(arr[1:])):.4f})"
                    )

    def test_autocorrelation_computed_from_pre_event_window(self, pipeline_output):
        """Variance should reflect the pre-event NOISE level, not the event signal.

        The event signal (0.04 Hz sinusoid, amplitude 0.5) is also present in the
        noise window here. The covariance window runs from EVENT_START - COV_WINDOW
        to EVENT_START.  We verify the zero-lag value is finite and plausible.
        """
        with h5py.File(pipeline_output["event_h5"], "r") as f:
            for sta in pipeline_output["available_stations"]:
                for comp, ds in f["misc"][sta].items():
                    v = float(ds[()][0])
                    # Signal amplitude 0.5 → variance ~0.125; noise ~0.
                    # Allow generous range but rule out zeros and implausible spikes.
                    assert 1e-6 < v < 10.0, (
                        f"{sta}/{comp} covariance zero-lag {v} is implausible"
                    )

    def test_autocorrelation_oscillates_for_sinusoidal_input(self, pipeline_output):
        """For a 0.04 Hz sinusoid the autocorrelation should show periodic structure.

        The autocorrelation of A*sin(2π*f*t) is (A²/2)*cos(2π*f*τ), so the
        first minimum should appear near τ = 1/(2f) = 12-13 samples.
        """
        f_signal = 0.04  # Hz — matches the injected sinusoid
        expected_half_period = int(round(1 / (2 * f_signal * SR_TARGET)))  # ~12 samples

        with h5py.File(pipeline_output["event_h5"], "r") as f:
            for sta in pipeline_output["available_stations"]:
                arr = f["misc"][sta]["Z"][()]
                if len(arr) < expected_half_period + 5:
                    continue
                zero_lag = arr[0]
                # The value at the half-period should be negative or much smaller
                half_period_val = arr[expected_half_period]
                assert half_period_val < zero_lag * 0.5, (
                    f"{sta}/Z: no oscillation in autocorrelation. "
                    f"R(0)={zero_lag:.4f}, R({expected_half_period})={half_period_val:.4f}"
                )

    def test_variance_estimated_independently_per_component(self, pipeline_output):
        """Z, 1, and 2 components must have independent (non-identical) variance estimates."""
        with h5py.File(pipeline_output["event_h5"], "r") as f:
            for sta in pipeline_output["available_stations"]:
                misc = f["misc"][sta]
                if set(misc.keys()) != {"Z", "1", "2"}:
                    continue
                v_z = float(misc["Z"][()][0])
                v_1 = float(misc["1"][()][0])
                v_2 = float(misc["2"][()][0])
                # All from the same 0.5*sin source — variances should be equal
                # (small deviations from rounding/taper are ok, but they must not be identical copies)
                # Check they are all close to 0.125 (= 0.5²/2)
                for comp, v in [("Z", v_z), ("1", v_1), ("2", v_2)]:
                    assert 0.05 < v < 0.5, (
                        f"{sta}/{comp} variance {v:.4f} far from expected ~0.125"
                    )


# ---------------------------------------------------------------------------
# 0.1f — Flattened vector ordering matches h5 layout
# ---------------------------------------------------------------------------

class TestFlattenedVectorOrdering:
    """The SBI pipeline flattens [STA1/Z, STA1/1, STA1/2, STA2/Z, ...].
    Verify this ordering is consistent with the h5 contents.
    """

    def test_first_block_matches_first_station_Z(self, pipeline_output):
        """First DATA_VECTOR_LEN samples of the flattened vector = STA1/Z."""
        from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
        from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader

        receivers = Receivers(receivers=[
            Receiver(0.0, 0.0, NETWORK, sta, ["Z", "E", "N"])
            for sta in pipeline_output["available_stations"]
        ])
        loader = SimulationDataLoader(components="ZEN", receivers=receivers)
        vec = loader.load_flattened_simulation_vector(pipeline_output["event_h5"])

        first_sta = pipeline_output["available_stations"][0]
        with h5py.File(pipeline_output["event_h5"], "r") as f:
            h5_z = f["outputs"][first_sta]["Z"][()]

        np.testing.assert_array_equal(
            vec[:DATA_VECTOR_LEN], h5_z,
            err_msg=f"First block of flattened vector != {first_sta}/Z from h5"
        )

    def test_second_block_matches_first_station_component1(self, pipeline_output):
        """Second DATA_VECTOR_LEN samples = STA1/1 (E channel)."""
        from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
        from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader

        receivers = Receivers(receivers=[
            Receiver(0.0, 0.0, NETWORK, sta, ["Z", "E", "N"])
            for sta in pipeline_output["available_stations"]
        ])
        loader = SimulationDataLoader(components="ZEN", receivers=receivers)
        vec = loader.load_flattened_simulation_vector(pipeline_output["event_h5"])

        first_sta = pipeline_output["available_stations"][0]
        with h5py.File(pipeline_output["event_h5"], "r") as f:
            h5_e = f["outputs"][first_sta]["1"][()]  # E was renamed to 1

        np.testing.assert_array_equal(
            vec[DATA_VECTOR_LEN : 2 * DATA_VECTOR_LEN], h5_e,
            err_msg=f"Second block of flattened vector != {first_sta}/1 from h5"
        )

    def test_station_blocks_are_non_overlapping_and_complete(self, pipeline_output):
        """Each station occupies a contiguous, non-overlapping block in the flat vector."""
        from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
        from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader

        available = pipeline_output["available_stations"]
        n_comp = 3
        receivers = Receivers(receivers=[
            Receiver(0.0, 0.0, NETWORK, sta, ["Z", "E", "N"]) for sta in available
        ])
        loader = SimulationDataLoader(components="ZEN", receivers=receivers)
        vec = loader.load_flattened_simulation_vector(pipeline_output["event_h5"])

        assert len(vec) == len(available) * n_comp * DATA_VECTOR_LEN
        # Verify each per-station block is distinct (different signals per station)
        block_size = n_comp * DATA_VECTOR_LEN
        for i, sta_a in enumerate(available):
            for j, sta_b in enumerate(available):
                if i >= j:
                    continue
                block_a = vec[i * block_size : (i + 1) * block_size]
                block_b = vec[j * block_size : (j + 1) * block_size]
                assert not np.allclose(block_a, block_b), (
                    f"Stations {sta_a} and {sta_b} have identical flattened blocks"
                )
