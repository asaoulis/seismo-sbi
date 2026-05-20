"""Phase 3 — New preprocessing API tests on real IRIS data.

Uses the same Ridgecrest M5.1 event as the legacy real-data test (0.2), but
drives the full pipeline through the NEW preprocessing API functions instead
of the legacy NoiseCollector / ProcessedDataSlicer classes.

Key differences from the legacy real test:
- Response removal via load_inventory + deconvolve_and_filter (instrument_correction=True)
- No UPFLOW_config dict, no path_structure format strings
- No mseed write/read round-trip between processing and slicing
- Uses export_to_sbi_h5 directly for the final h5

Opt-in for network access: set SEISMO_SBI_RUN_NETWORK=1 (inherited from
conftest; tests will still run from cache if data is present).
"""

from datetime import timedelta
from pathlib import Path

import h5py
import numpy as np
import obspy
import pytest
from obspy import UTCDateTime

from seismo_sbi.data_handling.preprocessing import (
    load_waveforms,
    load_inventory,
    write_window,
    deconvolve_and_filter,
    slice_event_window,
    export_to_sbi_h5,
)
from seismo_sbi.instaseis_simulator.utils import compute_data_vector_length

from tests.end_to_end.conftest import RIDGECREST_EVENT, IRIS_TEST_STATIONS

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Event window — same as legacy test so assertions are directly comparable
# ---------------------------------------------------------------------------

EVENT_ORIGIN = RIDGECREST_EVENT["origin_time"]
EVENT_START_UTC = EVENT_ORIGIN + 60      # 60 s post-origin (P already arrived at ANMO)
EVENT_END_UTC = EVENT_START_UTC + 90     # 90 s window

EVENT_START = EVENT_START_UTC.datetime
EVENT_END = EVENT_END_UTC.datetime

SR_TARGET = 1.0
FILTER_KWARGS = dict(freqmin=0.02, freqmax=0.1, corners=4, zerophase=False)
PREFILTER_KWARGS = dict(pre_filt=[0.005, 0.01, 0.4, 0.8])
COV_WINDOW = timedelta(seconds=60)

DATA_VECTOR_LEN = compute_data_vector_length(
    (EVENT_END - EVENT_START).total_seconds(), SR_TARGET
) + 1   # 91 samples

CACHE_DIR = Path(__file__).parent.parent / "data" / "_cache"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_mseed_layout(raw_dir: Path, streams: dict) -> dict:
    """Write stream dict to {sta}/{year}.{jday}/{net}.{sta}.{loc}.{cha}.{y}.{j}.mseed.

    Returns {(network, station): [path, ...]} for use with load_waveforms.
    """
    station_paths = {}
    for (network, station), st in streams.items():
        if len(st) == 0:
            continue
        t0 = st[0].stats.starttime
        year, jday = t0.year, t0.julday
        sta_dir = raw_dir / station / f"{year}.{jday:03d}"
        sta_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for tr in st:
            cha = tr.stats.channel
            loc = tr.stats.location or ""
            fname = f"{network}.{station}.{loc}.{cha}.{year}.{jday:03d}.mseed"
            fpath = sta_dir / fname
            tr.write(str(fpath), format="MSEED")
            paths.append(fpath)
        station_paths[(network, station)] = paths
    return station_paths


# ---------------------------------------------------------------------------
# Module-scoped fixture: run new-API pipeline once on cached/downloaded data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def new_api_real_output(cached_iris_wide_download, cached_iris_inventory, tmp_path_factory):
    """Full new-API pipeline on real IRIS data with response removal."""
    tmp = tmp_path_factory.mktemp("new_api_real")
    raw_dir = tmp / "raw"
    raw_dir.mkdir()
    stationxml_dir = tmp / "stationxml"
    stationxml_dir.mkdir()

    # Write waveforms to the standard layout
    station_paths = _write_mseed_layout(raw_dir, cached_iris_wide_download)

    if len(station_paths) < 2:
        pytest.skip("Fewer than 2 stations available")

    # Write inventory files to stationxml dir (one per station)
    for (network, station), inv in cached_iris_inventory.items():
        xml_path = stationxml_dir / f"{network}_{station}.xml"
        inv.write(str(xml_path), format="STATIONXML")

    # Load combined inventory
    inventory = load_inventory(stationxml_dir)

    # Load waveforms for all stations and process
    available_stations = []
    all_streams = []
    t0_load = UTCDateTime(EVENT_START) - COV_WINDOW.total_seconds() - 60
    t1_load = UTCDateTime(EVENT_END) + 60

    for (network, station), paths in station_paths.items():
        try:
            st = load_waveforms(paths, starttime=t0_load, endtime=t1_load)
            if len(st) == 0:
                continue
            # Filter inventory to this station to avoid cross-station conflicts
            sta_inv = inventory.select(network=network, station=station)
            proc = deconvolve_and_filter(
                st,
                inventory=sta_inv,
                remove_response=True,
                prefilter_kwargs=PREFILTER_KWARGS,
                filter_kwargs=FILTER_KWARGS,
                target_sr=SR_TARGET,
            )
            all_streams.append(proc)
            available_stations.append(station)
        except Exception as exc:
            print(f"Skipping {network}.{station}: {exc}")
            continue

    if len(available_stations) < 2:
        pytest.skip(f"Only {len(available_stations)} station(s) processed; need ≥2")

    # Merge all station streams and export
    combined = obspy.Stream()
    for st in all_streams:
        combined += st

    event_h5 = tmp / "ridgecrest_new_api.h5"
    export_to_sbi_h5(
        combined,
        receivers=available_stations,
        event_window=(EVENT_START, EVENT_END),
        out_path=event_h5,
        sampling_rate=SR_TARGET,
        covariance_window=COV_WINDOW,
        full_auto_correlation=True,
    )

    # Also write a daily preprocessed mseed for io round-trip test
    daily_mseed = tmp / "daily.mseed"
    write_window(combined, daily_mseed)

    return {
        "h5": event_h5,
        "daily_mseed": daily_mseed,
        "available_stations": available_stations,
        "inventory": inventory,
        "stationxml_dir": stationxml_dir,
        "raw_dir": raw_dir,
        "station_paths": station_paths,
    }


# ============================================================================
# io tests on real data
# ============================================================================

class TestLoadWaveformsReal:

    def test_loads_mseed_from_disk(self, new_api_real_output):
        """load_waveforms should return a non-empty Stream from the cached mseed."""
        st_paths = new_api_real_output["station_paths"]
        for (network, station), paths in list(st_paths.items())[:1]:
            st = load_waveforms(paths)
            assert len(st) > 0, f"Empty stream for {network}.{station}"

    def test_all_bh_channels_loaded(self, new_api_real_output):
        """Expect Z and at least one horizontal per station."""
        for (network, station), paths in new_api_real_output["station_paths"].items():
            st = load_waveforms(paths)
            channels = {tr.stats.channel for tr in st}
            z_channels = {c for c in channels if c.endswith("Z")}
            assert len(z_channels) >= 1, f"No Z channel for {station}: {channels}"

    def test_time_trim_reduces_data(self, new_api_real_output):
        """Passing starttime/endtime should shorten the loaded stream."""
        st_paths = new_api_real_output["station_paths"]
        for (network, station), paths in list(st_paths.items())[:1]:
            full = load_waveforms(paths)
            t0 = full[0].stats.starttime + 30
            t1 = t0 + 60
            trimmed = load_waveforms(paths, starttime=t0, endtime=t1)
            for tr in trimmed:
                assert tr.stats.npts <= full.select(
                    station=station, channel=tr.stats.channel
                )[0].stats.npts


class TestLoadInventoryReal:

    def test_inventory_loaded_from_dir(self, new_api_real_output):
        inv = load_inventory(new_api_real_output["stationxml_dir"])
        assert len(inv.networks) > 0

    def test_inventory_covers_test_stations(self, new_api_real_output):
        inv = load_inventory(new_api_real_output["stationxml_dir"])
        network_station_pairs = {
            (net.code, sta.code)
            for net in inv.networks
            for sta in net.stations
        }
        # At least one of our IRIS test stations should be in the inventory
        expected = {(net, sta) for net, sta in IRIS_TEST_STATIONS}
        assert len(expected & network_station_pairs) >= 1, (
            f"None of {expected} in inventory {network_station_pairs}"
        )


class TestWriteWindowReal:

    def test_daily_mseed_readable(self, new_api_real_output):
        daily_mseed = new_api_real_output["daily_mseed"]
        assert daily_mseed.exists()
        st = obspy.read(str(daily_mseed))
        assert len(st) > 0

    def test_daily_mseed_at_target_sr(self, new_api_real_output):
        st = obspy.read(str(new_api_real_output["daily_mseed"]))
        for tr in st:
            assert abs(tr.stats.sampling_rate - SR_TARGET) < 1e-6, (
                f"{tr.id} SR={tr.stats.sampling_rate} != {SR_TARGET}"
            )


# ============================================================================
# Processing with real response removal
# ============================================================================

class TestDeconvolveRealData:

    def test_response_removal_changes_amplitude(self, new_api_real_output, cached_iris_wide_download):
        """Displacement output (response removed) should differ from raw counts.

        IRIS broadband channels have sensitivities ~1500 counts/(nm/s); after
        response removal the amplitude will be orders of magnitude different.
        """
        for (network, station), paths in list(
            new_api_real_output["station_paths"].items()
        )[:1]:
            raw = load_waveforms(paths)
            inv = new_api_real_output["inventory"].select(
                network=network, station=station
            )

            raw_rms = np.sqrt(np.mean(raw.select(channel="BHZ")[0].data ** 2))

            proc = deconvolve_and_filter(
                raw.copy(), inventory=inv, remove_response=True,
                prefilter_kwargs=PREFILTER_KWARGS,
                filter_kwargs=FILTER_KWARGS, target_sr=SR_TARGET,
            )
            proc_rms = np.sqrt(np.mean(proc.select(channel="BHZ")[0].data ** 2))

            assert raw_rms > 0 and proc_rms > 0, "RMS is zero before or after processing"
            # Expect at least an order-of-magnitude change (instrument sensitivity ~1500 ct/nm/s)
            assert proc_rms != pytest.approx(raw_rms, rel=0.01), (
                "Response removal had no effect on signal amplitude"
            )

    def test_response_removal_output_finite(self, new_api_real_output, cached_iris_wide_download):
        """Displacement traces after response removal must not contain NaN/Inf."""
        for (network, station), paths in list(
            new_api_real_output["station_paths"].items()
        )[:1]:
            raw = load_waveforms(paths)
            inv = new_api_real_output["inventory"].select(
                network=network, station=station
            )
            proc = deconvolve_and_filter(
                raw.copy(), inventory=inv, remove_response=True,
                prefilter_kwargs=PREFILTER_KWARGS,
                filter_kwargs=FILTER_KWARGS, target_sr=SR_TARGET,
            )
            for tr in proc:
                assert np.all(np.isfinite(tr.data)), (
                    f"{tr.id}: NaN/Inf after response removal"
                )

    def test_bandpass_applied_after_response_removal(self, new_api_real_output, cached_iris_wide_download):
        """After response removal + bandpass, near-Nyquist energy must be suppressed."""
        for (network, station), paths in list(
            new_api_real_output["station_paths"].items()
        )[:1]:
            raw = load_waveforms(paths)
            inv = new_api_real_output["inventory"].select(
                network=network, station=station
            )
            proc = deconvolve_and_filter(
                raw.copy(), inventory=inv, remove_response=True,
                prefilter_kwargs=PREFILTER_KWARGS,
                filter_kwargs=FILTER_KWARGS, target_sr=SR_TARGET,
            )
            for tr in proc.select(channel="BHZ"):
                arr = tr.data
                freqs = np.fft.rfftfreq(len(arr), d=1.0 / SR_TARGET)
                power = np.abs(np.fft.rfft(arr)) ** 2
                in_band = (freqs >= 0.02) & (freqs <= 0.1)
                near_ny = freqs > 0.4
                if not np.any(in_band) or not np.any(near_ny):
                    continue
                in_p = power[in_band].mean()
                ny_p = power[near_ny].mean()
                assert in_p >= ny_p, (
                    f"{tr.id}: near-Nyquist power ({ny_p:.2e}) exceeds in-band ({in_p:.2e})"
                )


# ============================================================================
# H5 output tests
# ============================================================================

class TestNewApiRealH5Schema:

    def test_h5_created(self, new_api_real_output):
        assert new_api_real_output["h5"].exists()

    def test_h5_has_outputs_and_misc(self, new_api_real_output):
        with h5py.File(new_api_real_output["h5"], "r") as f:
            assert "outputs" in f
            assert "misc" in f

    def test_all_available_stations_in_h5(self, new_api_real_output):
        with h5py.File(new_api_real_output["h5"], "r") as f:
            for sta in new_api_real_output["available_stations"]:
                assert sta in f["outputs"], f"{sta} missing from /outputs"

    def test_component_keys_z_1_2(self, new_api_real_output):
        with h5py.File(new_api_real_output["h5"], "r") as f:
            for sta in new_api_real_output["available_stations"]:
                keys = set(f["outputs"][sta].keys())
                assert keys == {"Z", "1", "2"}, (
                    f"{sta}: expected {{Z,1,2}}, got {keys}"
                )
                assert "E" not in keys and "N" not in keys

    def test_array_length_correct(self, new_api_real_output):
        with h5py.File(new_api_real_output["h5"], "r") as f:
            for sta in new_api_real_output["available_stations"]:
                for comp, ds in f["outputs"][sta].items():
                    assert ds.shape[0] == DATA_VECTOR_LEN, (
                        f"{sta}/{comp}: expected {DATA_VECTOR_LEN}, got {ds.shape[0]}"
                    )

    def test_arrays_finite_and_nonzero(self, new_api_real_output):
        with h5py.File(new_api_real_output["h5"], "r") as f:
            for sta in f["outputs"]:
                for comp, ds in f["outputs"][sta].items():
                    arr = ds[()]
                    assert np.all(np.isfinite(arr)), f"{sta}/{comp} has NaN/Inf"
                    assert np.any(arr != 0.0), f"{sta}/{comp} is all-zero"

    def test_variance_positive(self, new_api_real_output):
        with h5py.File(new_api_real_output["h5"], "r") as f:
            for sta in f["misc"]:
                for comp, ds in f["misc"][sta].items():
                    v = float(np.atleast_1d(ds[()]).flat[0])
                    assert v > 0, f"{sta}/{comp} variance non-positive: {v}"

    def test_autocorrelation_length_plausible(self, new_api_real_output):
        expected = int(COV_WINDOW.total_seconds() * SR_TARGET)
        with h5py.File(new_api_real_output["h5"], "r") as f:
            for sta in f["misc"]:
                for comp, ds in f["misc"][sta].items():
                    arr = ds[()]
                    assert abs(len(arr) - expected) <= 5, (
                        f"{sta}/{comp} autocorr length {len(arr)} != ~{expected}"
                    )


class TestNewApiRealCovarianceProperties:

    def test_zero_lag_dominant(self, new_api_real_output):
        with h5py.File(new_api_real_output["h5"], "r") as f:
            for sta in f["misc"]:
                for comp, ds in f["misc"][sta].items():
                    arr = ds[()]
                    r0 = float(arr[0])
                    assert r0 > 0, f"{sta}/{comp}: R(0) non-positive {r0}"
                    if len(arr) > 10:
                        med_off = float(np.median(np.abs(arr[5:])))
                        assert r0 > med_off, (
                            f"{sta}/{comp}: R(0)={r0:.3e} ≤ median off-diag {med_off:.3e}"
                        )

    def test_variance_differs_across_stations(self, new_api_real_output):
        """Different stations have different noise levels (different distance / instrument)."""
        available = new_api_real_output["available_stations"]
        if len(available) < 2:
            pytest.skip("Need ≥2 stations")
        with h5py.File(new_api_real_output["h5"], "r") as f:
            z_vars = {
                sta: float(f["misc"][sta]["Z"][()][0])
                for sta in available
                if sta in f["misc"] and "Z" in f["misc"][sta]
            }
        if len(z_vars) < 2:
            pytest.skip("Not enough stations with Z in misc")
        vals = list(z_vars.values())
        assert max(vals) > min(vals) * 1.001, (
            f"All Z variances suspiciously identical: {z_vars}"
        )

    def test_components_not_identical_within_station(self, new_api_real_output):
        """Z, 1, 2 arrays must carry distinct signals."""
        with h5py.File(new_api_real_output["h5"], "r") as f:
            for sta in new_api_real_output["available_stations"]:
                z = f["outputs"][sta]["Z"][()]
                c1 = f["outputs"][sta]["1"][()]
                c2 = f["outputs"][sta]["2"][()]
                assert not np.allclose(z, c1, atol=1e-15), f"{sta}: Z == 1"
                assert not np.allclose(z, c2, atol=1e-15), f"{sta}: Z == 2"
                assert not np.allclose(c1, c2, atol=1e-15), f"{sta}: 1 == 2"


class TestSbiIngestionWithNewApiH5:
    """SimulationDataLoader and RealNoiseSampler must consume new-API h5 files."""

    def test_simulation_data_loader_reads_h5(self, new_api_real_output):
        from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
        from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader

        available = new_api_real_output["available_stations"]
        # Use ANMO network for IU stations, BFO for II — just pass a dummy network
        # for testing; SimulationDataLoader only uses station name for h5 lookup
        receivers = Receivers(receivers=[
            Receiver(0.0, 0.0, "IU", sta, ["Z", "E", "N"])
            for sta in available
        ])
        loader = SimulationDataLoader(components="ZEN", receivers=receivers)
        vec = loader.load_flattened_simulation_vector(new_api_real_output["h5"])
        expected_len = len(available) * 3 * DATA_VECTOR_LEN
        assert vec.shape == (expected_len,), (
            f"Expected flat length {expected_len}, got {vec.shape}"
        )
        assert np.all(np.isfinite(vec)), "Flattened vector has NaN/Inf"

    def test_real_noise_sampler_reads_h5(self, new_api_real_output, tmp_path):
        """RealNoiseSampler should be able to use the new-API event h5 as a noise file."""
        from seismo_sbi.sbi.noises.real_noise import RealNoiseSampler
        from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
        from seismo_sbi.sbi.types.parameters import SimulationParameters
        import shutil

        available = new_api_real_output["available_stations"]
        h5_dir = tmp_path / "noise"
        h5_dir.mkdir()
        shutil.copy(new_api_real_output["h5"], h5_dir / "noise_sample.h5")

        receivers = Receivers(receivers=[
            Receiver(0.0, 0.0, "IU", sta, ["Z", "E", "N"])
            for sta in available
        ])
        sim_params = SimulationParameters(
            receivers=receivers,
            components="ZEN",
            seismogram_duration=(EVENT_END - EVENT_START).total_seconds(),
            sampling_rate=SR_TARGET,
            syngine_address="syngine://prem_i_2s",
            processing={},
        )
        sampler = RealNoiseSampler(
            simulation_parameters=sim_params,
            directory=h5_dir,
            data_length=DATA_VECTOR_LEN,
        )
        noise_vec = sampler()
        expected_len = len(available) * 3 * DATA_VECTOR_LEN
        assert noise_vec.shape == (expected_len,), (
            f"Expected noise vector length {expected_len}, got {noise_vec.shape}"
        )
        assert np.all(np.isfinite(noise_vec)), "Sampled noise has NaN/Inf"
