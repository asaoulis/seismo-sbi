"""Receivers built from a station file, a components map and a time-shift map."""
import json

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
