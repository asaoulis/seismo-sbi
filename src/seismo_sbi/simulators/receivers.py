"""Where the ground motion is recorded.

A :class:`Receiver` is one station: its position, network and station names, the components it
records, and a time shift in samples. :class:`Receivers` holds an ordered set of them, which is
the order every seismogram array in the pipeline is in, and builds one from a station file, a
components map and an optional per-station shift map.
"""

from typing import NamedTuple, List
import numpy as np
import json

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.patches import Rectangle
from pyproj import Geod
from ..data_handling.noise_collection import NoiseCollector, convert_channel_type

class Receiver(NamedTuple):

    latitude : float
    longitude : float
    network : str = "XX"
    station_name : str = "XXXX"
    components : List[str] = ["Z", "E", "N"]
    time_shift : int = 0


class Receivers:

    def __init__(self, path_to_stations =None, receiver_components_map = None, receiver_time_shifts_map = None, station_config=None, stations=None, station_codes_paths=None, receivers=None):
        if path_to_stations is not None:
            self.receivers = self._convert_to_instaseis_receivers(path_to_stations, receiver_components_map, receiver_time_shifts_map)
        elif station_config is not None:
            self.receivers = self._generate_receivers_from_config(station_config, stations, station_codes_paths)
        else:
            self.receivers = receivers
        print("At receiver init",[rec.time_shift for rec in self.receivers])
    def set_time_shifts(self, time_shifts_map):
        self.receiver_time_shifts_map = time_shifts_map
        new_receivers = []
        for rec in self.receivers:
            time_shift = self.receiver_time_shifts_map.get(rec.station_name, 0)
            new_rec = rec._replace(time_shift=time_shift)
            new_receivers.append(new_rec)
        self.receivers = new_receivers
        
    def _convert_to_instaseis_receivers(self, path_to_stations, receiver_components_map_path, receiver_time_shifts_map) -> List[Receiver]:

        stations_details = np.genfromtxt(path_to_stations, comments="#", dtype='str')

        if receiver_components_map_path is None:
            components = ["Z", "E", "N"]
            receiver_components_map = None
        else:
            with open(receiver_components_map_path, 'r') as f:
                receiver_components_map = json.load(f)
        if receiver_time_shifts_map is None:
            self.receiver_time_shifts_map = {}
        else:
            with open(receiver_time_shifts_map, 'r') as f:
                self.receiver_time_shifts_map = json.load(f)
        receivers = []
        for station_details in stations_details:
            lat = float(station_details[2])
            long = float(station_details[3])

            if receiver_components_map is not None:
                components = receiver_components_map[station_details[0]]

            time_shift = self.receiver_time_shifts_map.get(station_details[0], 0)

            rec = Receiver(
                latitude=lat,
                longitude=long,
                network=station_details[1],
                station_name=station_details[0],
                components=components,
                time_shift=time_shift
            )

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
            

    def plot(self, ax=None, projection=None, add_labels=True, add_scalebar=True, add_north=False, add_receiver_icons=True):
        """Plot the receiver network on a map, on ``ax`` or on a new figure."""


        if projection is None:
            projection = ccrs.PlateCarree()

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': projection})
            created_fig = True
        else:
            created_fig = False

        lats = [r.latitude for r in self.iterate()]
        lons = [r.longitude for r in self.iterate()]

        margin = 0.5
        min_lat, max_lat = min(lats) - margin, max(lats) + margin
        min_lon, max_lon = min(lons) - margin, max(lons) + margin
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, facecolor='0.95', zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':', zorder=1)
        ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='black', linewidth=0.3, zorder=1)
        ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.3, zorder=1)

        states = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='10m',
            facecolor='none')
        ax.add_feature(states, edgecolor='black', linewidth=0.4, zorder=1)

        if add_receiver_icons:
            self.add_receiver_icons(ax, add_labels=add_labels)

        gl = ax.gridlines(draw_labels=False, linewidth=0.4, color='gray', alpha=0.5, linestyle='--', zorder=0)

        ax.set_xticks(range(int(min_lon), int(max_lon) + 1, 1), crs=ccrs.PlateCarree())
        ax.set_yticks(range(int(min_lat), int(max_lat) + 1, 1), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
        ax.tick_params(labelsize=9, direction='in')

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.8)

        def add_zebra_border(ax, step=1, length=0.1):
            """Draw alternating black and white tick marks around the map extent."""
            from itertools import cycle
            colors = cycle(['black', 'white'])
            lon_min, lon_max, lat_min, lat_max = ax.get_extent(crs=ccrs.PlateCarree())

            for lon in range(int(lon_min), int(lon_max)):
                color = next(colors)
                ax.plot([lon, lon + step], [lat_min, lat_min], color=color, lw=2, transform=ccrs.PlateCarree(), zorder=10)
                ax.plot([lon, lon + step], [lat_max, lat_max], color=color, lw=2, transform=ccrs.PlateCarree(), zorder=10)
            colors = cycle(['black', 'white'])
            for lat in range(int(lat_min), int(lat_max)):
                color = next(colors)
                ax.plot([lon_min, lon_min], [lat, lat + step], color=color, lw=2, transform=ccrs.PlateCarree(), zorder=10)
                ax.plot([lon_max, lon_max], [lat, lat + step], color=color, lw=2, transform=ccrs.PlateCarree(), zorder=10)

        add_zebra_border(ax)
        if add_north:
            ax.text(0.05, 0.95, 'N', transform=ax.transAxes,
                    fontsize=12, fontweight='bold', ha='center', va='center')
            ax.arrow(0.05, 0.90, 0, 0.04, transform=ax.transAxes,
                    color='k', width=0.005, head_width=0.03, head_length=0.02)

        if add_scalebar:
            from matplotlib import patheffects

            def scale_bar(ax, length_km=50, linewidth=2):
                """Add a scale bar of ``length`` km, measured geodetically."""
                lon_min, lon_max, lat_min, lat_max = ax.get_extent(ccrs.PlateCarree())
                lon_c = (lon_min + lon_max) / 2
                lat_c = (lat_min + lat_max) / 2

                geod = Geod(ellps="WGS84")
                _, _, dist = geod.inv(lon_c, lat_c, lon_c + 1, lat_c)
                km_per_deg = dist / 1000.0
                deg_length = length_km / km_per_deg

                x0, y0 = lon_c - deg_length / 2, lat_min + 0.3
                ax.plot([x0, x0 + deg_length], [y0, y0],
                        color='k', linewidth=linewidth, transform=ccrs.PlateCarree(),
                        path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
                ax.text(x0 + deg_length / 2, y0 - 0.1, f'{length_km} km',
                        ha='center', va='top', 
                        transform=ccrs.PlateCarree())

            scale_bar(ax)

        if created_fig:
            plt.tight_layout()
            plt.show()

    def add_receiver_icons(self, ax, add_labels=True, color='darkred'):
        for rec in self.iterate():
            ax.plot(rec.longitude, rec.latitude, marker='v', color=color,
                    markersize=7, transform=ccrs.PlateCarree(), zorder=5)
            if add_labels:
                ax.text(
                    rec.longitude - 0.25, rec.latitude - 0.3,
                    f"{rec.network}.{rec.station_name}", color='black', transform=ccrs.PlateCarree(),
                    ha='left', va='bottom', zorder=6,
                    bbox=dict(
                        boxstyle='round,pad=0.2',
                        facecolor='white',
                        edgecolor='black',
                        alpha=0.8,
                        linewidth=0.5
                    )
                )

        
    def get_station_locations_array(self):
        """``(n_stations, 2)`` of latitude and longitude in degrees, in receiver order."""
        return np.array([[rec.latitude, rec.longitude] for rec in self.iterate()])