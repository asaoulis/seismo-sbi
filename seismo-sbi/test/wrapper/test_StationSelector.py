import os
import sys
import unittest

from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

from src.axisem_simulator.wrapper.station_selector import StationSelector
from src.axisem_simulator.wrapper.input_settings.input_file_writer import ConfigurationError

class Testing_StationSelector(unittest.TestCase):

    def test_file_read(self):
        station_filepath = Path(__file__).parent / "../stubs/STATIONS"
        input_settings = StationSelector(str(station_filepath.resolve()))

        exp_stations_index_map = {"UP01":0, "UP02":1, "UP10": 2}

        self.assertEqual(input_settings.stations_index_map, exp_stations_index_map)
    
    def test_select_all_stations_gen_text(self):
        station_filepath = Path(__file__).parent / "../stubs/STATIONS"
        input_settings = StationSelector(str(station_filepath.resolve()))

        station_text = input_settings._select_station_and_gen_text('all')
        expected_station_text = ["UP01	UPFLOW	40.1	-20	0	0",
                                "UP02	UPFLOW	40.2	-21	0	0",
                                "UP10	UPFLOW	50.2	-22	0	0"]

        self.assertEqual(station_text, expected_station_text)

    def test_select_some_stations_gen_text_1(self):
        station_filepath = Path(__file__).parent / "../stubs/STATIONS"
        input_settings = StationSelector(str(station_filepath.resolve()))

        selected_stations = ["UP01", "UP02"]
        station_text = input_settings._select_station_and_gen_text(selected_stations)
        expected_station_text = ["UP01	UPFLOW	40.1	-20	0	0",
                                "UP02	UPFLOW	40.2	-21	0	0"]

        self.assertEqual(station_text, expected_station_text)

    def test_select_some_stations_gen_text_2(self):
        station_filepath = Path(__file__).parent / "../stubs/STATIONS"
        input_settings = StationSelector(str(station_filepath.resolve()))

        selected_stations = ["UP01", "UP10"]
        station_text = input_settings._select_station_and_gen_text(selected_stations)
        expected_station_text = ["UP01	UPFLOW	40.1	-20	0	0",
                                "UP10	UPFLOW	50.2	-22	0	0"]

        self.assertEqual(station_text, expected_station_text)

    def test_select_some_stations_gen_text_3(self):
        station_filepath = Path(__file__).parent / "../stubs/STATIONS"
        input_settings = StationSelector(str(station_filepath.resolve()))

        selected_stations = ["INVALID STATION"]
        self.assertRaises(ConfigurationError, input_settings._select_station_and_gen_text, selected_stations)


if __name__ == '__main__':
    unittest.main()