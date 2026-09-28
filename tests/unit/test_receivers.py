"""Receivers built from a station file, a components map, a time-shift map, arrays or an obspy Inventory, and written back."""
import json
from pathlib import Path

from seismo_sbi.simulators.receivers import Receiver, Receivers

STATIONS = """# name network latitude longitude
BKS BK 37.876221 -122.23558
CMB BK 38.03455 -120.386513
KCC BK 37.323631 -119.318703
"""
COMPONENTS = {"BKS": ["Z", "E", "N"], "CMB": ["Z"], "KCC": []}
SHIFTS = {"BKS": 4, "KCC": 1}


def _write_station_files(tmp_path):
    stations = tmp_path / "stations.txt"
    stations.write_text(STATIONS)
    components = tmp_path / "components.json"
    components.write_text(json.dumps(COMPONENTS))
    shifts = tmp_path / "shifts.json"
    shifts.write_text(json.dumps(SHIFTS))
    return stations, components, shifts


def test_receivers_from_station_file_round_trip(tmp_path):
    stations, components, shifts = _write_station_files(tmp_path)

    receivers = Receivers(str(stations), str(components), str(shifts))

    assert receivers.receivers == [
        Receiver(37.876221, -122.23558, "BK", "BKS", ["Z", "E", "N"], 4),
        Receiver(38.03455, -120.386513, "BK", "CMB", ["Z"], 0),
    ]
    assert receivers.receiver_time_shifts_map == SHIFTS
    written = tmp_path / "written.txt"
    receivers.write_to_file(str(written))
    assert [line.split()[0] for line in written.read_text().splitlines()] == ["BKS", "CMB"]


def test_receivers_without_a_components_map_record_three_components(tmp_path):
    stations, _, _ = _write_station_files(tmp_path)

    receivers = Receivers(str(stations))

    assert [receiver.components for receiver in receivers.iterate()] == [["Z", "E", "N"]] * 3
    assert receivers.receiver_time_shifts_map == {}


def test_from_station_file_matches_the_positional_constructor(tmp_path):
    stations, components, shifts = _write_station_files(tmp_path)

    built = Receivers.from_station_file(str(stations), str(components), str(shifts))

    assert built.receivers == Receivers(str(stations), str(components), str(shifts)).receivers
    assert built.receiver_time_shifts_map == SHIFTS


def test_from_arrays_builds_one_receiver_per_station():
    receivers = Receivers.from_arrays(["BKS", "CMB"], ["BK", "BK"], [37.9, 38.0], [-122.2, -120.4],
                                      components=("Z",))

    assert receivers.receivers == [Receiver(37.9, -122.2, "BK", "BKS", ["Z"]),
                                   Receiver(38.0, -120.4, "BK", "CMB", ["Z"])]


def test_from_inventory_reads_every_station_in_order():
    from obspy.core.inventory import Inventory, Network, Station

    stations = [Station("BKS", 37.9, -122.2, 244.0), Station("CMB", 38.0, -120.4, 719.0)]
    inventory = Inventory(networks=[Network("BK", stations=stations)], source="test")

    receivers = Receivers.from_inventory(inventory)

    assert [(rec.network, rec.station_name, rec.latitude, rec.components) for rec in receivers] == [
        ("BK", "BKS", 37.9, ["Z", "E", "N"]), ("BK", "CMB", 38.0, ["Z", "E", "N"])]


def test_receivers_built_from_a_list_have_no_time_shifts_map_entries():
    receivers = Receivers(receivers=[Receiver(0.0, 0.0)])

    assert receivers.receiver_time_shifts_map == {}


def test_receivers_report_their_length_iteration_and_names():
    receivers = Receivers(receivers=[Receiver(0.0, 0.0, "BK", "BKS"), Receiver(1.0, 1.0, "BK", "CMB")])

    assert len(receivers) == 2
    assert list(receivers) == list(receivers.iterate())
    assert repr(receivers) == "Receivers(2 stations: BKS, CMB)"


def _station_with_channels(code, channel_codes):
    from obspy.core.inventory import Channel, Station

    channels = [Channel(channel_code, "", 37.9, -122.2, 244.0, 0.0) for channel_code in channel_codes]
    return Station(code, 37.9, -122.2, 244.0, channels=channels)


def _inventory(*stations):
    from obspy.core.inventory import Inventory, Network

    return Inventory(networks=[Network("BK", stations=list(stations))], source="test")


def test_from_inventory_reads_each_station_s_channels_in_z_e_n_order():
    inventory = _inventory(_station_with_channels("BKS", ["BHN", "BHZ", "BHE"]),
                           _station_with_channels("CMB", ["BH2", "BHZ", "BH1"]))

    receivers = Receivers.from_inventory(inventory)

    assert [rec.components for rec in receivers] == [["Z", "E", "N"], ["Z", "E", "N"]]


def test_from_inventory_keeps_only_channels_matching_the_pattern():
    inventory = _inventory(_station_with_channels("BKS", ["BHZ", "BHE", "BHN"]))

    receivers = Receivers.from_inventory(inventory, channels="BHZ")

    assert [rec.components for rec in receivers] == [["Z"]]


def test_from_inventory_ignores_state_of_health_channels():
    inventory = _inventory(_station_with_channels("BKS", ["BHZ", "LCE", "ACE", "VEC", "LOG"]))

    receivers = Receivers.from_inventory(inventory)

    assert [rec.components for rec in receivers] == [["Z"]]


def test_from_inventory_drops_a_station_whose_channels_give_no_component():
    inventory = _inventory(_station_with_channels("BKS", ["BHZ"]),
                           _station_with_channels("CMB", ["LOG", "HHZ", "HHE"]))

    receivers = Receivers.from_inventory(inventory, channels="BH?")

    assert [rec.station_name for rec in receivers] == ["BKS"]


def test_from_inventory_given_components_overrides_the_channels():
    inventory = _inventory(_station_with_channels("BKS", ["BHZ"]), _station_with_channels("CMB", []))

    receivers = Receivers.from_inventory(inventory, components=("Z", "N"))

    assert [rec.components for rec in receivers] == [["Z", "N"], ["Z", "N"]]


def test_to_inventory_round_trips_through_from_inventory():
    receivers = Receivers(receivers=[Receiver(37.9, -122.2, "BK", "BKS", ["Z", "E", "N"], 4),
                                     Receiver(38.0, -120.4, "BK", "CMB", ["Z"]),
                                     Receiver(36.1, -117.8, "CI", "CLC", ["E", "N"])])

    inventory = receivers.to_inventory()
    rebuilt = Receivers.from_inventory(inventory)

    assert [channel.code for channel in inventory.select(station="BKS")[0][0]] == ["BHZ", "BHE", "BHN"]
    assert rebuilt.receivers == [rec._replace(time_shift=0) for rec in receivers]


def test_network_station_codes_follow_the_station_file_order():
    stations = Path(__file__).resolve().parents[2] / "examples" / "configs" / "stations.txt"

    codes = Receivers.from_station_file(str(stations)).network_station_codes()

    assert codes == [("BK", "BKS"), ("BK", "CMB"), ("BK", "KCC"), ("BK", "ORV"), ("BK", "PKD")]


def test_a_station_file_with_one_station_gives_one_receiver(tmp_path):
    stations = tmp_path / "stations.txt"
    stations.write_text("BKS BK 37.876221 -122.23558\n")

    receivers = Receivers.from_station_file(str(stations))

    assert receivers.receivers == [Receiver(37.876221, -122.23558, "BK", "BKS", ["Z", "E", "N"])]


def test_default_receivers_do_not_share_a_components_list():
    first, second = Receiver(0.0, 0.0), Receiver(1.0, 1.0)

    assert first.components == ("Z", "E", "N")
    assert not isinstance(first.components, list) and second.components == ("Z", "E", "N")
