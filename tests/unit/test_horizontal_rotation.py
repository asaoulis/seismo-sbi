"""Horizontal channels coded 1/2 are rotated to north/east by their inventory azimuths."""
import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime
from obspy.core.inventory import Channel, Inventory, Network, Station

from seismo_sbi.data_handling.preprocessing.processing import rotate_horizontals_to_north_east

RNG = np.random.default_rng(0)
VERTICAL, NORTH, EAST = RNG.normal(size=(3, 100))


def inventory(channels):
    """One station ``XX.STA`` with ``(code, azimuth_deg, dip_deg)`` channels."""
    channel_list = [Channel(code=code, location_code="", latitude=0, longitude=0, elevation=0, depth=0,
                            azimuth=azimuth_deg, dip=dip_deg, sample_rate=1) for code, azimuth_deg, dip_deg in channels]
    return Inventory(networks=[Network("XX", stations=[Station("STA", 0, 0, 0, channels=channel_list)])], source="test")


def stream(traces):
    """A stream of ``(channel_code, data)`` traces at station ``XX.STA``."""
    header = {"network": "XX", "station": "STA", "location": "", "starttime": UTCDateTime(2020, 1, 1), "sampling_rate": 1}
    return Stream([Trace(data.copy(), header={**header, "channel": code}) for code, data in traces])


def horizontals_at(azimuth_1_deg):
    """The ground motion recorded by horizontals at ``azimuth_1_deg`` and ``azimuth_1_deg + 90``."""
    phi = np.radians(azimuth_1_deg)
    return (NORTH * np.cos(phi) + EAST * np.sin(phi),
            -NORTH * np.sin(phi) + EAST * np.cos(phi))


@pytest.mark.parametrize("azimuth_1_deg", [0.0, 30.0, 200.0])
def test_z12_channels_rotate_to_true_north_and_east(azimuth_1_deg):
    channel_1, channel_2 = horizontals_at(azimuth_1_deg)
    raw = stream([("BHZ", VERTICAL), ("BH1", channel_1), ("BH2", channel_2)])
    metadata = inventory([("BHZ", 0, -90), ("BH1", azimuth_1_deg, 0), ("BH2", azimuth_1_deg + 90, 0)])

    rotated = {trace.stats.channel: trace.data for trace in rotate_horizontals_to_north_east(raw, metadata)}

    assert sorted(rotated) == ["BHE", "BHN", "BHZ"]
    np.testing.assert_allclose(rotated["BHN"], NORTH, atol=1e-10)
    np.testing.assert_allclose(rotated["BHE"], EAST, atol=1e-10)
    np.testing.assert_allclose(rotated["BHZ"], VERTICAL, atol=1e-10)


def test_zne_channels_are_left_untouched_whatever_the_inventory_azimuths():
    raw = stream([("BHZ", VERTICAL), ("BHN", NORTH), ("BHE", EAST)])
    metadata = inventory([("BHZ", 0, -90), ("BHN", 3, 0), ("BHE", 93, 0)])

    rotated = rotate_horizontals_to_north_east(raw.copy(), metadata)

    assert [trace.stats.channel for trace in rotated] == ["BHZ", "BHN", "BHE"]
    for before, after in zip(raw, rotated):
        np.testing.assert_array_equal(after.data, before.data)


def test_a_1_2_channel_without_an_inventory_raises():
    raw = stream([("BHZ", VERTICAL), ("BH1", NORTH), ("BH2", EAST)])
    with pytest.raises(ValueError, match="inventory"):
        rotate_horizontals_to_north_east(raw, None)


def test_an_incomplete_z12_set_raises():
    raw = stream([("BHZ", VERTICAL), ("BH1", NORTH)])
    metadata = inventory([("BHZ", 0, -90), ("BH1", 0, 0), ("BH2", 90, 0)])
    with pytest.raises(ValueError, match="complete"):
        rotate_horizontals_to_north_east(raw, metadata)
