import numpy as np

from datetime import timedelta, datetime

from obspy.geodetics import locations2degrees
from obspy.taup import tau
import obspy
import obspy.clients.fdsn
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.core.event import Catalog

import joblib

def get_epicentral_distances_function(event_lat, event_long, station):
    lat, long = station
    return locations2degrees(event_lat, event_long, lat, long)

class EventWindowSelector:

    def __init__(self, obspy_client='iris', earth_model='prem', num_jobs = 1) -> None:

        self.taup = tau.TauPyModel(model=earth_model)

        self.client = obspy.clients.fdsn.Client(obspy_client)

        self.parallel_execution = lambda input_list, function: joblib.Parallel(n_jobs=num_jobs)(joblib.delayed(function)(*args) for args in input_list)

    def select_events(self, event_lat, event_long, start_time, end_time, distant_min_radius = 15, distant_min_magnitude = 6.0,
                        close_max_radius = 15, close_min_magnitude=4):
        try:
            distant_events = self.client.get_events(latitude=event_lat, longitude=event_long,
                                        minradius = distant_min_radius,
                                        minmagnitude=distant_min_magnitude,
                                    starttime=obspy.UTCDateTime(start_time),
                                    endtime=obspy.UTCDateTime(end_time))
        except FDSNNoDataException:
            print('No distant event data found for event at {}, {}'.format(event_lat, event_long))
            distant_events = Catalog()
        try:
            near_events = self.client.get_events(latitude=event_lat, longitude=event_long,
                                        maxradius = close_max_radius,
                                        minmagnitude=close_min_magnitude,
                                    starttime=obspy.UTCDateTime(start_time),
                                    endtime=obspy.UTCDateTime(end_time))
        except FDSNNoDataException:
            print('No close event data found for event at {}, {}'.format(event_lat, event_long))
            near_events = Catalog()

        return distant_events, near_events



    def get_single_event_range(self, receivers, event, only_first_arrival=False):
        earliest_arrival = np.inf
        latest_arrival = -np.inf
        for station_details in receivers.receivers:
            station = station_details.latitude, station_details.longitude
            if not only_first_arrival:
                arrivals = self.taup.get_travel_times(source_depth_in_km=event.origins[0].depth/1000,
                                                    distance_in_degree=get_epicentral_distances_function(event.origins[0].latitude, event.origins[0].longitude, station)
                                                    )
            else:
                arrivals = self.taup.get_travel_times(source_depth_in_km=event.origins[0].depth/1000,
                                                    distance_in_degree=get_epicentral_distances_function(event.origins[0].latitude, event.origins[0].longitude, station),
                                                    phase_list=['P']
                                                    )          
            
            if len(arrivals) > 0:
                earliest_arrival = min(earliest_arrival, arrivals[0].time)
                latest_arrival = max(latest_arrival, arrivals[-1].time)
        return (event.origins[0].time + earliest_arrival, event.origins[0].time + latest_arrival)
    
    def get_unavailability_time_pairs(self, receivers, near_events, distant_events):
    
        event_start_end_times = self.parallel_execution([(receivers, event) for event in near_events + distant_events], self.get_single_event_range)

        return event_start_end_times
    
    @staticmethod
    def get_continuous_regions(event_start_end_times, start_time, end_time):
        continuous_regions = []
        gaps = []

        # Sort the list of start and end datetime pairs
        sorted_ranges = sorted(event_start_end_times, key=lambda x: x[0])

        # Initialize the start and end times with the first event
        current_start, current_end = sorted_ranges[0]
        continuous_regions.append((start_time, current_start))
        # Iterate through the sorted_ranges
        for start, end in sorted_ranges[1:]:
            if start > current_end:
                continuous_regions.append((current_end, start))
                gaps.append((current_start, current_end))
                current_start, current_end = start, end
            else:
                current_end = max(current_end, end)
        gaps.append((current_start, current_end))
        continuous_regions.append((current_end, end_time))

        # Append the last continuous region and any remaining gap
        return continuous_regions, gaps
    
    @staticmethod
    def create_daily_overlapping_windows_from_regions(contiguous_regions, window_length, buffer=timedelta(minutes=15), overlap_offset=None):
        daily_windows = {}
        if overlap_offset is None:
            overlap_offset = buffer
        for start, end in contiguous_regions:
            if isinstance(start, obspy.UTCDateTime):
                start = start.datetime
            if isinstance(end, obspy.UTCDateTime):
                end = end.datetime
            buffered_start = start + buffer
            buffered_end = end - buffer
            while buffered_start + window_length < buffered_end:
                date = buffered_start.date()
                window_end = buffered_start + window_length
                if date not in daily_windows:
                    daily_windows[date] = []
                if date == window_end.date():
                    daily_windows[date].append((buffered_start, window_end))

                buffered_start += overlap_offset
        return daily_windows

    @staticmethod
    def create_windows_from_regions(continuous_regions, window_length, buffer=timedelta(minutes=15)):

        for start, end in continuous_regions:
            if isinstance(start, obspy.UTCDateTime):
                start = start.datetime
            if isinstance(end, obspy.UTCDateTime):
                end = end.datetime
            buffered_start = start + buffer
            buffered_end = end - buffer
            while buffered_start + window_length < buffered_end:
                yield buffered_start, buffered_start + window_length
                buffered_start += buffer