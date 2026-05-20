"""End-to-end test fixtures for the data preprocessing pipeline.

Provides a cached IRIS download fixture for the real-data slow tests.
The download is attempted unconditionally when the slow marker is used;
if the network is unavailable the fixture skips gracefully.
"""

from pathlib import Path

import obspy
import obspy.clients.fdsn
import pytest
from obspy import UTCDateTime


CACHE_DIR = Path(__file__).parent.parent / "data" / "_cache"

# Known small event for real-data tests (M5.1, 2019-07-04, Ridgecrest CA)
RIDGECREST_EVENT = {
    "origin_time": UTCDateTime("2019-07-04T17:33:49"),
    "latitude": 35.705,
    "longitude": -117.504,
    "depth_km": 8.0,
}

# Stations with reliable BH? availability on IRIS
IRIS_TEST_STATIONS = [
    ("IU", "ANMO"),  # Albuquerque NM, ~800 km
    ("II", "BFO"),   # Black Forest Germany — long-period, always up
    ("IU", "COLA"),  # College AK
]

# Wider time window used by new-API tests so that the covariance pre-event
# window (COV_WINDOW = 60 s) is always available before EVENT_START.
# legacy tests used −120 s / +180 s; new-API tests need −300 s / +300 s.
_T0_WIDE = RIDGECREST_EVENT["origin_time"] - 300   # 5 min before
_T1_WIDE = RIDGECREST_EVENT["origin_time"] + 300   # 5 min after


@pytest.fixture(scope="session")
def cached_iris_download():
    """Download ~5 min of BH? from IRIS test stations, cache to disk.

    Returns a dict: {(network, station): Stream}.
    Skips if no cached data exists and the network is unavailable.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    t0 = RIDGECREST_EVENT["origin_time"] - 120  # 2 min before
    t1 = RIDGECREST_EVENT["origin_time"] + 180  # 3 min after

    streams = {}
    for network, station in IRIS_TEST_STATIONS:
        cache_key = f"{network}_{station}_{t0.strftime('%Y%m%dT%H%M%S')}.mseed"
        cache_path = CACHE_DIR / cache_key

        if cache_path.exists():
            streams[(network, station)] = obspy.read(str(cache_path))
        else:
            try:
                client = obspy.clients.fdsn.Client("IRIS")
                st = client.get_waveforms(
                    network=network, station=station,
                    location="*", channel="BH?",
                    starttime=t0, endtime=t1,
                )
                st.write(str(cache_path), format="MSEED")
                streams[(network, station)] = st
            except Exception as exc:
                print(f"Could not download {network}.{station}: {exc}")

    if len(streams) == 0:
        pytest.skip(
            "No cached IRIS data and network unavailable — "
            "run with internet access once to populate tests/data/_cache/"
        )

    return streams


@pytest.fixture(scope="session")
def cached_iris_wide_download():
    """Download a wider waveform window (±5 min around origin) for new-API tests.

    The wider window allows the covariance pre-event window to always be
    available before EVENT_START.  Returns dict: {(network, station): Stream}.
    Skips if nothing is available.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    streams = {}
    for network, station in IRIS_TEST_STATIONS:
        cache_key = (
            f"{network}_{station}_{_T0_WIDE.strftime('%Y%m%dT%H%M%S')}_wide.mseed"
        )
        cache_path = CACHE_DIR / cache_key

        if cache_path.exists():
            streams[(network, station)] = obspy.read(str(cache_path))
        else:
            try:
                client = obspy.clients.fdsn.Client("IRIS")
                st = client.get_waveforms(
                    network=network, station=station,
                    location="*", channel="BH?",
                    starttime=_T0_WIDE, endtime=_T1_WIDE,
                )
                st.write(str(cache_path), format="MSEED")
                streams[(network, station)] = st
            except Exception as exc:
                print(f"Could not download {network}.{station} (wide): {exc}")

    if len(streams) == 0:
        pytest.skip(
            "No cached IRIS wide data and network unavailable — "
            "run with internet access once to populate tests/data/_cache/"
        )
    return streams


@pytest.fixture(scope="session")
def cached_iris_inventory():
    """Download StationXML for IRIS test stations and cache to disk.

    Returns dict: {(network, station): Inventory}.
    Skips if nothing is available.  The inventory is required for response
    removal tests in test_new_api_real.py.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    inventories = {}
    for network, station in IRIS_TEST_STATIONS:
        cache_key = f"{network}_{station}_inventory.xml"
        cache_path = CACHE_DIR / cache_key

        if cache_path.exists():
            try:
                inventories[(network, station)] = obspy.read_inventory(
                    str(cache_path)
                )
                continue
            except Exception:
                pass

        try:
            client = obspy.clients.fdsn.Client("IRIS")
            inv = client.get_stations(
                network=network, station=station,
                location="*", channel="BH?",
                starttime=_T0_WIDE, endtime=_T1_WIDE,
                level="response",
            )
            inv.write(str(cache_path), format="STATIONXML")
            inventories[(network, station)] = inv
        except Exception as exc:
            print(f"Could not download inventory for {network}.{station}: {exc}")

    if len(inventories) == 0:
        pytest.skip(
            "No cached IRIS inventory and network unavailable — "
            "run with internet access once to populate tests/data/_cache/"
        )
    return inventories
