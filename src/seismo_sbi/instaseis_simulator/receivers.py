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
    time_shift : int = 0  # time shift indices


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

            # Components (existing logic)
            if receiver_components_map is not None:
                components = receiver_components_map[station_details[0]]

            # NEW: look up time shift
            time_shift = self.receiver_time_shifts_map.get(station_details[0], 0)

            rec = Receiver(
                latitude=lat,
                longitude=long,
                network=station_details[1],
                station_name=station_details[0],
                components=components,
                time_shift=time_shift  # NEW
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
            

    def plot(self, ax=None, projection=None, add_labels=True, add_scalebar=True, add_north=False):
        """
        Plot the receiver network on a map using Cartopy.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            Existing axis to plot on. If None, a new figure and axis will be created.
        projection : cartopy.crs, optional
            Projection for the map. Defaults to PlateCarree.
        add_labels : bool, optional
            Whether to add network.station text labels.
        add_scalebar : bool, optional
            Whether to add a map scale bar.
        add_north : bool, optional
            Whether to add a north arrow.
        """

        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        from matplotlib.patches import Rectangle
        from pyproj import Geod

        # Choose projection
        if projection is None:
            projection = ccrs.PlateCarree()

        # Create figure/axis if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': projection})
            created_fig = True
        else:
            created_fig = False

        # Extract coordinates
        lats = [r.latitude for r in self.iterate()]
        lons = [r.longitude for r in self.iterate()]

        # Auto extent (add margin)
        margin = 0.5
        min_lat, max_lat = min(lats) - margin, max(lats) + margin
        min_lon, max_lon = min(lons) - margin, max(lons) + margin
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        # Add base map features
        ax.add_feature(cfeature.LAND, facecolor='0.95', zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':', zorder=1)
        ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='black', linewidth=0.3, zorder=1)
        ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.3, zorder=1)

        # Add U.S. state boundaries (high-res)
        states = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='10m',
            facecolor='none')
        ax.add_feature(states, edgecolor='black', linewidth=0.4, zorder=1)

        # Plot stations with boxed labels
        for rec in self.iterate():
            ax.plot(rec.longitude, rec.latitude, marker='v', color='darkred',
                    markersize=7, transform=ccrs.PlateCarree(), zorder=5)
            if add_labels:
                ax.text(
                    rec.longitude - 0.25, rec.latitude - 0.3,
                    f"{rec.network}.{rec.station_name}",
                    fontsize=8, color='black', transform=ccrs.PlateCarree(),
                    ha='left', va='bottom', zorder=6,
                    bbox=dict(
                        boxstyle='round,pad=0.2',
                        facecolor='white',
                        edgecolor='black',
                        alpha=0.8,
                        linewidth=0.5
                    )
                )

        # Add clean gridlines (no labels inside plot)
        gl = ax.gridlines(draw_labels=False, linewidth=0.4, color='gray', alpha=0.5, linestyle='--', zorder=0)

        # Add tick labels on outer edges
        ax.set_xticks(range(int(min_lon), int(max_lon) + 1, 1), crs=ccrs.PlateCarree())
        ax.set_yticks(range(int(min_lat), int(max_lat) + 1, 1), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
        ax.tick_params(labelsize=9, direction='in')

        # Add black frame
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.8)

        # Add zebra-style lat/lon border (alternating black/white segments)
        def add_zebra_border(ax, step=1, length=0.1):
            """
            Draws black/white alternating tick marks (zebra border) around map extent.
            """
            from itertools import cycle
            colors = cycle(['black', 'white'])
            lon_min, lon_max, lat_min, lat_max = ax.get_extent(crs=ccrs.PlateCarree())

            # Bottom and top borders
            for lon in range(int(lon_min), int(lon_max)):
                color = next(colors)
                ax.plot([lon, lon + step], [lat_min, lat_min], color=color, lw=2, transform=ccrs.PlateCarree(), zorder=10)
                ax.plot([lon, lon + step], [lat_max, lat_max], color=color, lw=2, transform=ccrs.PlateCarree(), zorder=10)
            # Reset and draw left/right
            colors = cycle(['black', 'white'])
            for lat in range(int(lat_min), int(lat_max)):
                color = next(colors)
                ax.plot([lon_min, lon_min], [lat, lat + step], color=color, lw=2, transform=ccrs.PlateCarree(), zorder=10)
                ax.plot([lon_max, lon_max], [lat, lat + step], color=color, lw=2, transform=ccrs.PlateCarree(), zorder=10)

        add_zebra_border(ax)
        # Optional: north arrow
        if add_north:
            ax.text(0.05, 0.95, 'N', transform=ax.transAxes,
                    fontsize=12, fontweight='bold', ha='center', va='center')
            ax.arrow(0.05, 0.90, 0, 0.04, transform=ax.transAxes,
                    color='k', width=0.005, head_width=0.03, head_length=0.02)

        # Optional: scale bar
        if add_scalebar:
            from matplotlib import patheffects

            def scale_bar(ax, length_km=50, linewidth=2):
                """
                Add a simple scale bar using pyproj.Geod for accurate length.
                """
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
                        ha='center', va='top', fontsize=8,
                        transform=ccrs.PlateCarree())

            scale_bar(ax)

        if created_fig:
            plt.tight_layout()
            plt.show()

        
    def get_station_locations_array(self):
        """
        Returns an array of station locations in the format [latitude, longitude].
        """
        return np.array([[rec.latitude, rec.longitude] for rec in self.iterate()])