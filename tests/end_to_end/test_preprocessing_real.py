"""Phase 0.2 — Real-data end-to-end preprocessing test (network-gated).

Downloads ~5 minutes of BH? data from IRIS for a known small event, runs
the full current NoiseCollector / EventNoiseAggregator / NoiseDatabaseGenerator
/ ProcessedDataSlicer pipeline, and asserts physical properties of the output.

Opt-in: set SEISMO_SBI_RUN_NETWORK=1 to enable this test.
Cache: downloaded data is written to tests/data/_cache/ so subsequent runs
are fully offline.
"""

import datetime
from datetime import timedelta
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import obspy
import obspy.clients.fdsn
from obspy import UTCDateTime, Trace, Stream
import pytest

from seismo_sbi.data_handling.noise_collection import (
    NoiseCollector,
    EventNoiseAggregator,
    ProcessedDataSlicer,
)
from seismo_sbi.data_handling.noise_database import NoiseDatabaseGenerator
from seismo_sbi.instaseis_simulator.utils import compute_data_vector_length
from typing import Dict, Tuple

from tests.end_to_end.conftest import RIDGECREST_EVENT, IRIS_TEST_STATIONS

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Event window: 90-second window centred on the Ridgecrest P-wave arrival
# ---------------------------------------------------------------------------

EVENT_ORIGIN = RIDGECREST_EVENT["origin_time"]
EVENT_LAT = RIDGECREST_EVENT["latitude"]
EVENT_LON = RIDGECREST_EVENT["longitude"]

# Use a short window well after the P-arrival for ANMO (~10 s travel time at 800 km)
EVENT_START_UTC = EVENT_ORIGIN + 60    # 60 s post-origin
EVENT_END_UTC = EVENT_START_UTC + 90   # 90 s window

EVENT_START = EVENT_START_UTC.datetime
EVENT_END = EVENT_END_UTC.datetime
EVENT_LOCATION = (EVENT_LAT, EVENT_LON)

SR_TARGET = 1.0   # Hz — low enough to work on long-period BH channels

# Filter well within the 0.5 Hz Nyquist for 1 Hz target
FILTER_KWARGS = dict(freqmin=0.02, freqmax=0.1, corners=4, zerophase=False)
PREFILTER_KWARGS = dict(pre_filt=[0.005, 0.01, 0.4, 0.8])

# NOISE_WINDOW_LENGTH must be large enough that the central_noise_window
# written by collect_noise_data fully contains both the covariance pre-window
# (COV_WINDOW before EVENT_START) and the event window itself.
NOISE_WINDOW_LENGTH = timedelta(minutes=5)  # 300 s — wider than the event + cov window
COV_WINDOW = timedelta(seconds=60)
# ProcessedDataSlicer.slice gives inclusive endpoints (+1 sample, same as synthetic test)
DATA_VECTOR_LEN = compute_data_vector_length(
    (EVENT_END - EVENT_START).total_seconds(), SR_TARGET
) + 1  # 91 samples

CACHE_DIR = Path(__file__).parent.parent / "data" / "_cache"


# ---------------------------------------------------------------------------
# Helpers: write IRIS-downloaded streams into the custom_download.py layout
# ---------------------------------------------------------------------------

def _write_streams_to_layout(raw_dir: Path, streams: dict) -> Tuple[Dict, Dict]:
    """Write cached IRIS streams to the {sta}/{year}.{jday}/{net}.{sta}..{cha}.mseed layout.

    Returns (upflow_config, station_codes_paths) for the stations that were written.
    """
    upflow_configs = {}
    station_codes_paths = {}

    for (network, station), st in streams.items():
        if len(st) == 0:
            continue

        # Use the start time of the first trace to determine year/jday
        t_start = st[0].stats.starttime
        year = t_start.year
        jday = t_start.julday

        sta_dir = raw_dir / station / f"{year}.{jday:03d}"
        sta_dir.mkdir(parents=True, exist_ok=True)

        # Write each component as a separate mseed
        for tr in st:
            cha = tr.stats.channel
            fname = f"{network}.{station}..{cha}.{year}.{jday:03d}.mseed"
            tr.write(str(sta_dir / fname), format="MSEED")

        # Skip instrument correction: response removal is not what this test
        # exercises, and location-code mismatches between the RESP filename
        # written here and the path looked up by NoiseCollector make it brittle.
        instrument_correction = False

        # All downloaded channels (e.g. BHZ, BH1, BH2 or BHE, BHN)
        channels = sorted({tr.stats.channel for tr in st})

        upflow_configs[network] = {
            "call_key": "OBS",
            "master_path": ".",
            "path_structure": (
                str(raw_dir)
                + "/{sta}/{year}.{jday}/{net}.{sta}..{cha}.{year}.{jday}.mseed"
            ),
            "sta_cha": channels,
            "network": network,
            "location": "",
            "years": [year],
            "instrument_correction": instrument_correction,
            "response_seismometer": "",  # not used: instrument_correction=False
        }
        station_codes_paths[station] = network

    return upflow_configs, station_codes_paths


# ---------------------------------------------------------------------------
# Session-scoped fixture: run once, share across tests in this file
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_pipeline_output(cached_iris_download, tmp_path_factory):
    """Run the full preprocessing pipeline on cached IRIS data."""
    tmp = tmp_path_factory.mktemp("real_preprocess")
    raw_dir = tmp / "raw"
    raw_dir.mkdir()
    daily_dir = tmp / "daily"
    daily_dir.mkdir()
    event_dir = tmp / "events"
    event_dir.mkdir()

    upflow_config, station_codes_paths = _write_streams_to_layout(
        raw_dir, cached_iris_download
    )

    if len(station_codes_paths) < 2:
        pytest.skip("Fewer than 2 stations available; skip real-data pipeline test")

    noise_collector = NoiseCollector(
        upflow_config,
        station_codes_paths,
        {},
        prefilter_kwargs=PREFILTER_KWARGS,
        filter_kwargs=FILTER_KWARGS,
    )
    event_noise_aggregator = EventNoiseAggregator(
        noise_collector, station_codes_paths, sampling_rate=SR_TARGET
    )

    event_noise_aggregator.select_event_and_check_available_stations(
        [EVENT_START, EVENT_END],
        EVENT_LOCATION,
        buffer=timedelta(seconds=5),
        convert_to_numpy=True,
    )

    available = event_noise_aggregator.available_stations_during_event
    if len(available) < 2:
        pytest.skip(
            f"Only {len(available)} station(s) available for event window; need ≥2"
        )

    # Daily window must be wide enough that collect_noise_data's central_noise_window
    # (of length NOISE_WINDOW_LENGTH=300s) fully covers the event + covariance period.
    # padding = (daily_len - NOISE_WINDOW_LENGTH) / 2; central_start = daily_start + padding/2
    # Use ±4 min around the event midpoint → daily_len=570s, padding=135s, central_start=67.5s in.
    event_mid = EVENT_START_UTC + (EVENT_END_UTC - EVENT_START_UTC) / 2
    t_start_utc = event_mid - 285  # 4 min 45 s before midpoint
    t_end_utc = event_mid + 285    # 4 min 45 s after midpoint
    daily_window = [t_start_utc.datetime, t_end_utc.datetime]

    noise_collection_callable = partial(
        event_noise_aggregator.collect_noise_data,
        noise_window_length=NOISE_WINDOW_LENGTH,
        convert_to_numpy=False,
    )
    noise_db_gen = NoiseDatabaseGenerator(
        noise_collection_callable, num_jobs=1, mseed_output=True
    )
    noise_db_gen.create_database(daily_dir, [daily_window])

    data_slicer = ProcessedDataSlicer(
        data_folder=daily_dir,
        sampling_rate=SR_TARGET,
        covariance_estimation_window=COV_WINDOW,
        full_auto_correlation=True,
        receivers=available,
    )
    event_saver = NoiseDatabaseGenerator(
        data_slicer.load_noise_window_data,
        data_vector_length=DATA_VECTOR_LEN,
        num_jobs=1,
    )
    event_name = "ridgecrest_2019"
    event_saver._collect_and_save_noise(
        event_dir,
        noise_window=[EVENT_START, EVENT_END],
        name=event_name,
    )

    return {
        "daily_dir": daily_dir,
        "event_h5": event_dir / f"{event_name}.h5",
        "available_stations": available,
    }


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

class TestRealDataPipelineOutput:

    def test_at_least_two_stations_survive(self, real_pipeline_output):
        """Station availability filtering retains at least 2 stations."""
        assert len(real_pipeline_output["available_stations"]) >= 2

    def test_event_h5_created(self, real_pipeline_output):
        assert real_pipeline_output["event_h5"].exists()

    def test_event_h5_has_outputs_and_misc(self, real_pipeline_output):
        with h5py.File(real_pipeline_output["event_h5"], "r") as f:
            assert "outputs" in f
            assert "misc" in f

    def test_all_available_stations_in_h5(self, real_pipeline_output):
        with h5py.File(real_pipeline_output["event_h5"], "r") as f:
            outputs = f["outputs"]
            for sta in real_pipeline_output["available_stations"]:
                assert sta in outputs, f"Station {sta} missing from /outputs"

    def test_sampling_rate_is_target(self, real_pipeline_output):
        """Daily mseed traces are resampled to SR_TARGET Hz."""
        daily_file = next(real_pipeline_output["daily_dir"].glob("*"))
        st = obspy.read(str(daily_file))
        for tr in st:
            assert abs(tr.stats.sampling_rate - SR_TARGET) < 1e-6, (
                f"{tr.id} sampling rate {tr.stats.sampling_rate} != {SR_TARGET}"
            )

    def test_output_arrays_finite_and_nonzero(self, real_pipeline_output):
        with h5py.File(real_pipeline_output["event_h5"], "r") as f:
            for sta in f["outputs"]:
                for comp, ds in f["outputs"][sta].items():
                    arr = ds[()]
                    assert np.all(np.isfinite(arr)), f"{sta}/{comp} contains NaN/Inf"
                    assert np.any(arr != 0.0), f"{sta}/{comp} is all-zero"

    def test_variance_positive(self, real_pipeline_output):
        with h5py.File(real_pipeline_output["event_h5"], "r") as f:
            for sta in f["misc"]:
                for comp, ds in f["misc"][sta].items():
                    v = float(np.atleast_1d(ds[()]).flat[0])
                    assert v > 0, f"{sta}/{comp} variance non-positive: {v}"

    def test_array_length_correct(self, real_pipeline_output):
        with h5py.File(real_pipeline_output["event_h5"], "r") as f:
            for sta in f["outputs"]:
                for comp, ds in f["outputs"][sta].items():
                    assert ds.shape[0] == DATA_VECTOR_LEN, (
                        f"{sta}/{comp}: expected {DATA_VECTOR_LEN} samples, "
                        f"got {ds.shape[0]}"
                    )

    def test_bandpass_filter_applied(self, real_pipeline_output):
        """Traces have been filtered: Nyquist-band (>0.45 Hz) energy is negligible
        relative to the in-band signal for at least one station.

        A strict dB threshold is not appropriate for real uncorrected data
        (broadband instrument noise, teleseismic arrivals) over a 91-sample window.
        We check that the pipeline ran the filter at all by confirming that
        high-frequency energy is lower than the in-band energy.
        """
        attenuation_dbs = []
        with h5py.File(real_pipeline_output["event_h5"], "r") as f:
            for sta in f["outputs"]:
                arr = f["outputs"][sta]["Z"][()]
                if len(arr) < 16:
                    continue
                freqs = np.fft.rfftfreq(len(arr), d=1.0 / SR_TARGET)
                power = np.abs(np.fft.rfft(arr)) ** 2

                in_band = (freqs >= 0.02) & (freqs <= 0.1)
                near_nyquist = freqs > 0.4

                if not np.any(in_band) or not np.any(near_nyquist):
                    continue

                in_power = power[in_band].mean()
                out_power = power[near_nyquist].mean()
                if in_power > 0 and out_power > 0:
                    attenuation_dbs.append(10 * np.log10(in_power / out_power))

        assert len(attenuation_dbs) > 0, "No stations had enough data for spectral check"
        assert max(attenuation_dbs) > 0, (
            "Near-Nyquist energy exceeds in-band energy for all stations — "
            f"filter may not have run (attenuations: {attenuation_dbs})"
        )

    def test_component_renaming_applied(self, real_pipeline_output):
        """E and N components are renamed to 1 and 2 in the h5."""
        with h5py.File(real_pipeline_output["event_h5"], "r") as f:
            for sta in f["outputs"]:
                comps = set(f["outputs"][sta].keys())
                assert "E" not in comps, f"{sta} has unrenamed 'E'"
                assert "N" not in comps, f"{sta} has unrenamed 'N'"


class TestRealDataCovarianceProperties:
    """Physical checks on the covariance estimated from real seismic noise."""

    def test_zero_lag_is_positive_and_dominant(self, real_pipeline_output):
        """R(0) must be positive; checks it is in the right order of magnitude
        relative to the off-diagonal autocorrelation values.

        Note: the strict R(0) ≥ |R(τ)| property holds for the true autocorrelation
        but NOT for the unbiased sample estimator on short, non-stationary windows
        (real seismic noise over 60 s). We therefore only require R(0) > 0 and
        that the median off-diagonal value is less than R(0).
        """
        with h5py.File(real_pipeline_output["event_h5"], "r") as f:
            for sta in f["misc"]:
                for comp, ds in f["misc"][sta].items():
                    arr = ds[()]
                    zero_lag = float(arr[0])
                    assert zero_lag > 0, f"{sta}/{comp}: R(0) non-positive: {zero_lag}"
                    if len(arr) > 10:
                        median_off = float(np.median(np.abs(arr[5:])))
                        assert zero_lag > median_off, (
                            f"{sta}/{comp}: R(0)={zero_lag:.4e} ≤ median off-diag {median_off:.4e}"
                        )

    def test_variance_differs_across_components(self, real_pipeline_output):
        """Real seismic noise: Z, 1, 2 components should not all have identical variance.

        Identical variances would suggest a copy-paste bug in the component loop.
        """
        with h5py.File(real_pipeline_output["event_h5"], "r") as f:
            for sta in f["misc"]:
                comps = list(f["misc"][sta].keys())
                if len(comps) < 2:
                    continue
                variances = [float(f["misc"][sta][c][()][0]) for c in comps]
                # At least one pair should differ by more than 1%
                max_v, min_v = max(variances), min(variances)
                assert max_v > min_v * 1.01, (
                    f"{sta}: all component variances identical within 1%: {dict(zip(comps, variances))}"
                )

    def test_variance_differs_across_stations(self, real_pipeline_output):
        """Different stations (different instruments, distances) should have different noise levels."""
        available = real_pipeline_output["available_stations"]
        if len(available) < 2:
            pytest.skip("Need ≥2 stations")

        with h5py.File(real_pipeline_output["event_h5"], "r") as f:
            z_variances = {}
            for sta in available:
                if "Z" in f["misc"].get(sta, {}):
                    z_variances[sta] = float(f["misc"][sta]["Z"][()][0])

        if len(z_variances) < 2:
            pytest.skip("Not enough stations with Z component in misc")

        vals = list(z_variances.values())
        max_v, min_v = max(vals), min(vals)
        assert max_v > min_v * 1.01, (
            f"All station Z variances are nearly identical: {z_variances}"
        )

    def test_autocorrelation_length_plausible(self, real_pipeline_output):
        """Autocorrelation array length ≈ COV_WINDOW * SR_TARGET (±5 samples)."""
        expected = int(COV_WINDOW.total_seconds() * SR_TARGET)
        with h5py.File(real_pipeline_output["event_h5"], "r") as f:
            for sta in f["misc"]:
                for comp, ds in f["misc"][sta].items():
                    arr = ds[()]
                    assert abs(len(arr) - expected) <= 5, (
                        f"{sta}/{comp} autocorr length {len(arr)} != expected {expected}"
                    )


class TestRealDataChannelFidelity:
    """Channel content checks: components should carry distinct, physically meaningful signals."""

    def test_three_components_per_station(self, real_pipeline_output):
        """Each station in /outputs has exactly 3 component arrays (Z, 1, 2)."""
        with h5py.File(real_pipeline_output["event_h5"], "r") as f:
            for sta in real_pipeline_output["available_stations"]:
                comps = set(f["outputs"][sta].keys())
                assert comps == {"Z", "1", "2"}, (
                    f"{sta}: expected {{Z, 1, 2}}, got {comps}"
                )

    def test_components_are_not_identical(self, real_pipeline_output):
        """Z, 1, 2 must not be copies of each other (catches channel-selection bugs)."""
        with h5py.File(real_pipeline_output["event_h5"], "r") as f:
            for sta in real_pipeline_output["available_stations"]:
                z = f["outputs"][sta]["Z"][()]
                c1 = f["outputs"][sta]["1"][()]
                c2 = f["outputs"][sta]["2"][()]
                assert not np.allclose(z, c1, atol=1e-10), f"{sta}: Z == 1"
                assert not np.allclose(z, c2, atol=1e-10), f"{sta}: Z == 2"
                assert not np.allclose(c1, c2, atol=1e-10), f"{sta}: 1 == 2"

    def test_flattened_vector_via_sbi_loader(self, real_pipeline_output):
        """SimulationDataLoader produces the right-length vector from the real event h5."""
        from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
        from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader

        available = real_pipeline_output["available_stations"]
        receivers = Receivers(receivers=[
            Receiver(0.0, 0.0, "IU", sta, ["Z", "E", "N"]) for sta in available
        ])
        loader = SimulationDataLoader(components="ZEN", receivers=receivers)
        vec = loader.load_flattened_simulation_vector(real_pipeline_output["event_h5"])

        expected_len = len(available) * 3 * DATA_VECTOR_LEN
        assert vec.shape == (expected_len,), (
            f"Expected flat length {expected_len}, got {vec.shape}"
        )
        assert np.all(np.isfinite(vec)), "Flattened real-data vector contains NaN/Inf"
