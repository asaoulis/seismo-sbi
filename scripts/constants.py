import numpy as np

def create_ipma_data_format(data_path):
    civisa_land = {
        'call_key': 'OBS',
        'master_path': '.',
        'path_structure': '{}/{{sta}}/{{year}}.{{jday}}/{{net}}.{{sta}}..{{cha}}.{{year}}.{{jday}}.mseed'.format(data_path),
        'sta_cha': ['HHZ', 'HHE', 'HHN'],
        'network': 'PM',
        'location': '',
        'years': [2021, 2022],
        'instrument_correction': True,
        'response_seismometer': '{}/RESP/{{net}}.{{sta}}.{{loc}}.{{cha}}.RESP'.format(data_path),
    }
    return {'civisa_land': civisa_land}

STATION_CODES_PATHS = {station_code: 'civisa_land' for station_code in ['ADH', 'BART', 'HOR', 'PAGU', 'PDA', 'PGRA', 'PICO', 'PSET', 'PSMN', 'ROSA', 'SRBC']}