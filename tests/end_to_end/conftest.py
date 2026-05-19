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
