from typing import NamedTuple, List
import numpy as np
import json

from ..data_handling.noise_collection import NoiseCollector, convert_channel_type

class Receiver(NamedTuple):

    latitude : float
    longitude : float
    network : str = "XX"
    station_name : str = "XXXX"
    components : List[str] = ["Z", "E", "N"]


class Receivers:

    def __init__(self, path_to_stations =None, receiver_components_map = None, station_config=None, stations=None, station_codes_paths=None, receivers=None):
        if path_to_stations is not None:
            self.receivers = self._convert_to_instaseis_receivers(path_to_stations, receiver_components_map)
        elif station_config is not None:
            self.receivers = self._generate_receivers_from_config(station_config, stations, station_codes_paths)
        else:
            self.receivers = receivers

    def _convert_to_instaseis_receivers(self, path_to_stations, receiver_components_map_path) -> List[Receiver]:

        stations_details = np.genfromtxt(path_to_stations, comments="#", dtype='str')

        if receiver_components_map_path is None:
            components = ["Z", "E", "N"]
            receiver_components_map = None
        else:
            with open(receiver_components_map_path, 'r') as f:
                receiver_components_map = json.load(f)
        receivers = []
        for station_details in stations_details:
            lat = float(station_details[2])
            long = float(station_details[3])
            if receiver_components_map is not None:
                components = receiver_components_map[station_details[0]]
            
            rec = Receiver(latitude=lat,
                            longitude=long,
                            network=station_details[1], 
                            station_name=station_details[0],
                            components=components)

            if len(components) != 0:
                receivers.append(rec)

        return receivers
    
    def _generate_receivers_from_config(self, station_config, stations, station_codes_paths):
        receivers = []

        channel = 'Z'
        for station in stations:
            config = station_config[station_codes_paths[station]]
            formatted_channel =  convert_channel_type(channel, config['sta_cha'])
            instrument_response_path = NoiseCollector.evaluate_response_filepath(config['response_seismometer'], config['master_path'], "", station, config['network'], formatted_channel, config['location'])

            station_location = NoiseCollector.get_station_location(instrument_response_path)

            rec = Receiver(latitude=station_location[0],
                            longitude=station_location[1],
                            network=config['network'], 
                            station_name=station)
            
            receivers.append(rec)

        return receivers

    def iterate(self):
        for rec in self.receivers:
            yield rec
    
    def write_to_file(self, path_to_stations):
        with open(path_to_stations, 'w') as f:
            for rec in self.receivers:
                f.write("%s %s %s %s\n" % (rec.station_name, rec.network, rec.latitude, rec.longitude))
            

    def plot(self):
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        # ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        # Plot stations
        
        for rec in self.iterate():
            lat, lon = rec.latitude, rec.longitude
            net, sta = rec.network, rec.station_name
            ax.plot(lon, lat, marker='v', color='red', markersize=6, transform=ccrs.PlateCarree())
            ax.text(lon, lat + 0.2, f"{net}.{sta}", fontsize=10, transform=ccrs.PlateCarree())
        # Add title and labels
        ax.set_title("Receiver network")
        plt.show()