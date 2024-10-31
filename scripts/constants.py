import yaml
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname('.'), '..')))
from src.instaseis_simulator.receivers import Receivers

receivers = Receivers('data/UPFLOW_STATIONS')


with open("data/station_config.yaml", 'r') as stream:
    try:
        UPFLOW_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

UPFLOW_data = np.genfromtxt('data/UPFLOW_lookup', dtype=str)
UPFLOW_station_to_type = {station_name: station_type for station_name, station_type in UPFLOW_data[:, 6:8]}
STATION_LOCATIONS = {station_name: (float(station_lat), float(station_long)) \
                     for station_name, station_lat, station_long in np.hstack([UPFLOW_data[:, 6:7], UPFLOW_data[:, 1:3]])}


STATION_CODES_PATHS = \
    {
        **{station_code: 'short_period_OBS' for station_code in ['SJO01','SJO02','SJO03','SJO04', 'SJO05','SJO06']},
        **{station_code: 'civisa_land' for station_code in ['ADH', 'BART', 'HOR', 'PAGU', 'PDA', 'PGRA', 'PICO', 'PSET', 'PSMN', 'ROSA', 'SRBC']},
        # **{station_code: 'civisa_e_land' for station_code in ['PMAN', 'PSCM', 'PCED', 'PID', 'PCALD', 'PCAN', 'PGRON', 'PPNO']},
        **{station_code: 'temp_land' for station_code in ['FL01',  'PC01',  'PC02',  'PC03',  'SJ01',  'SJ02',  'SJ03',  'SJ04',  'SJ05',  'TC01']},
        **{receiver.station_name: UPFLOW_station_to_type[receiver.station_name] for receiver in receivers.receivers}
    }