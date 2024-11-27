import obspy
from obspy.signal.rotate import rotate_ne_rt
from scipy import signal

import numpy as np
import datetime
from datetime import timedelta
import math as m
import traceback
from seismo_sbi.instaseis_simulator.utils import compute_data_vector_length


def convert_channel_type(channel_type, sta_cha):
    if any('N' in channel_type for channel_type in sta_cha):
        if channel_type == '1':
            channel_type =  'E'
        elif channel_type == '2':
            channel_type = 'N'
        
    for sta_cha_type in sta_cha:
        if channel_type == sta_cha_type[-1]:
            return sta_cha_type

    return sta_cha + ' chan conversion'

class NoiseCollector:
    def __init__(self, UPFLOW_config, station_codes_paths, station_locations, channel_basis = None, prefilter_kwargs = {}, filter_kwargs = {}):
        self.UPFLOW_config = UPFLOW_config
        self.station_codes_paths = station_codes_paths
        self.station_locations = station_locations
        self.channel_basis = channel_basis

        self.prefilter_kwargs = dict(pre_filt=[0.005, 0.01, 2.5, 5], taper=True, taper_fraction=0.05)
        self.prefilter_kwargs = {**self.prefilter_kwargs, **prefilter_kwargs}

        self.filter_kwargs = dict(freqmin=0.02, freqmax=0.04, corners=4, zerophase=False)
        self.filter_kwargs = {**self.filter_kwargs, **filter_kwargs}

    @staticmethod
    def evaluate_data_filepath(fstring, master_path, current_datetime, station_code, network, channel, loc):
        sta = station_code
        net = network
        cha = channel
        loc = loc
        year, jday, month, day = current_datetime.strftime("%Y %j %m %d").split()
        return fstring.format(master_path, sta=sta, net=net, cha=cha, loc=loc, year=year, jday=jday, month=month, day=day)

    @staticmethod
    def evaluate_response_filepath(fstring, master_path, current_datetime, station_code, network, channel, loc):
        sta = station_code
        net = network
        cha = channel
        loc = loc
        
        return fstring.format(master_path, sta=sta, net=net, cha=cha, loc=loc)
    
    @staticmethod
    def get_station_location(instrument_response_path):
        inventory_response = obspy.read_inventory(instrument_response_path)
        station_location = inventory_response[0][0][0].latitude, inventory_response[0][0][0].longitude
        return station_location

    def load_noise_data(self, station_code, datetime_window, event_location, max_frequency):

        config = self.UPFLOW_config[self.station_codes_paths[station_code]]

        event_datetime = datetime_window[0]

        st_data_map = {}

        for channel in ['Z', '1', '2']:
            try:
                formatted_channel =  convert_channel_type(channel, config['sta_cha'])
                instrument_response_path = self.evaluate_response_filepath(config['response_seismometer'], config['master_path'], event_datetime, station_code, config['network'], formatted_channel, config["location"])
                
                filepath = self.evaluate_data_filepath(config['path_structure'], config['master_path'], event_datetime, station_code, config['network'], formatted_channel, config["location"])
                new_st = obspy.read(filepath, starttime=obspy.UTCDateTime(datetime_window[0]), endtime=obspy.UTCDateTime(datetime_window[1]), format='MSEED', check_compression=False)
                new_st = self.process_seismograms(new_st, max_frequency, instrument_response_path)

                st_data_map[channel]  = new_st[0]
                
            except Exception as e:
                error_message = f"Data couldn't be collected for {station_code}: {e}"
                print(error_message)
                raise Exception(error_message) from e
        # print(f"Data collected for {station_code}: ", np.mean(np.abs(new_st[0].data)))
        if self.channel_basis is not None:
            try:
                station_location = self.station_locations[station_code]
            except KeyError:
                station_location = self.get_station_location(instrument_response_path)
            _, _, baz = obspy.geodetics.base.gps2dist_azimuth(event_location[0], event_location[1], station_location[0], station_location[1])
            traces = rotate_ne_rt(st_data_map['1'].data, st_data_map['2'].data, ba=baz)
            st_R = obspy.Trace(data=traces[0], header=st_data_map['1'].stats)
            st_T = obspy.Trace(data=traces[1], header=st_data_map['1'].stats)
            st_data_map['R'] = st_R
            st_data_map['T'] = st_T

        return st_data_map

    def process_seismograms(self, new_st, max_frequency, instrument_response_path):
        # print(f"Sampling rate: {new_st[0].stats.sampling_rate}")
        if new_st[0].stats.sampling_rate > 5/max_frequency:
            new_st[0] = resample_trace(new_st[0], 1/20, "decimate")
        inventory_response = obspy.read_inventory(instrument_response_path)
        new_st = new_st.detrend('demean')
        new_st = new_st.detrend('linear')
        # new_st = new_st.remove_response(inventory=inventory_response, output="DISP", pre_filt=[0.01, 0.02, 2.5, 5], taper=True, taper_fraction=0.1)
        new_st = new_st.remove_response(inventory=inventory_response, output="DISP", **self.prefilter_kwargs)
        # taper
        new_st = new_st.taper(max_percentage=0.05, type='cosine')
        # new_st = new_st.filter('bandpass', freqmin=0.04, freqmax=0.07, corners=4, zerophase=False)
        new_st = new_st.filter('bandpass', **self.filter_kwargs)


        # # print(f"Decimation factor: {decimation_factor}")
        # new_st = new_st.resample(max_frequency, window="hann")
        new_st[0] = resample_trace(new_st[0], 1/max_frequency, "lanczos")


        return new_st

def resample_trace(tr, dt, method, lanczos_a=20):
    """
    resample ObsPy Trace (tr) with dt as delta (1/sampling_rate).
    This code is from LASIF repository (Lion Krischer) with some
    minor modifications.
    :param tr:
    :param dt:
    :param method:
    :param lanczos_a:
    :return:
    """
    while True:
        if method == "decimate":
            decimation_factor = int(dt / tr.stats.delta)
        elif method == "lanczos":
            decimation_factor = float(dt) / tr.stats.delta
        # decimate in steps for large sample rate reductions.
        if decimation_factor > 10:
            decimation_factor = 10
        if decimation_factor > 1:
            new_nyquist = tr.stats.sampling_rate / 2.0 / decimation_factor
            zerophase_chebychev_lowpass_filter(tr, new_nyquist)
            if method == "decimate":
                tr.decimate(factor=decimation_factor, no_filter=True)
            elif method == "lanczos":
                tr.taper(max_percentage=0.01)
                current_sr = float(tr.stats.sampling_rate)
                tr.interpolate(
                    method="lanczos",
                    sampling_rate=current_sr / decimation_factor,
                    a=lanczos_a,
                )
        else:
            return tr

def zerophase_chebychev_lowpass_filter(trace, freqmax):
    """
    Custom Chebychev type two zerophase lowpass filter useful for
    decimation filtering.
    This filter is stable up to a reduction in frequency with a factor of
    10. If more reduction is desired, simply decimate in steps.
    Partly based on a filter in ObsPy.
    :param trace: The trace to be filtered.
    :param freqmax: The desired lowpass frequency.
    Will be replaced once ObsPy has a proper decimation filter.
    This code is from LASIF repository (Lion Krischer).
    """
    # rp - maximum ripple of passband, rs - attenuation of stopband
    rp, rs, order = 1, 96, 1e99
    # stop band frequency
    ws = freqmax / (trace.stats.sampling_rate * 0.5)
    # pass band frequency
    wp = ws

    while True:
        if order <= 12:
            break
        wp *= 0.99
        order, wn = signal.cheb2ord(wp, ws, rp, rs, analog=0)

    b, a = signal.cheby2(order, rs, wn, btype="low", analog=0, output="ba")

    # Apply twice to get rid of the phase distortion.
    trace.data = signal.filtfilt(b, a, trace.data)
        
class EventNoiseAggregator:
        
        def __init__(self, noise_collector, station_codes_paths, sampling_rate):

            self.sampling_rate = sampling_rate
            self.noise_collector = noise_collector

            self.available_stations_during_event = []
            self.station_codes_paths = station_codes_paths

            self.event_data = None
            
        def select_event_and_check_available_stations(self, event_window, event_location, buffer = timedelta(0), convert_to_numpy = False):

            self.event_data = {}
            self.event_location = event_location
            fixed_num_seconds = m.ceil(((event_window[1] - event_window[0]).seconds/self.sampling_rate)) * self.sampling_rate
            exact_end_time = event_window[0] + timedelta(seconds=fixed_num_seconds)
            for station in self.station_codes_paths.keys():
                try:
                    st = self.noise_collector.load_noise_data(station, [event_window[0] - buffer, event_window[1] + buffer], event_location, max_frequency=self.sampling_rate)
                    self.available_stations_during_event.append(station)
                    for channel in st.keys():

                        st[channel] = st[channel].slice(obspy.UTCDateTime(event_window[0]),obspy.UTCDateTime( exact_end_time))

                        if convert_to_numpy:
                            st[channel] = st[channel].data

                    self.event_data[station] = st
                except Exception as e:
                    print(f"Error with station {station}: {e}")
                    continue
        
        def collect_noise_data(self, padded_noise_window, noise_window_length, convert_to_numpy = False):
            noise_stream = {}

            padding = ((padded_noise_window[1] - padded_noise_window[0]) - noise_window_length)/2
            start = padded_noise_window[0] + padding/2
            central_noise_window = [start, start + noise_window_length ]
            for station in self.available_stations_during_event:
                st = self.noise_collector.load_noise_data(station, padded_noise_window, self.event_location, max_frequency=self.sampling_rate)
                for channel in st.keys():
                    
                    st[channel] = st[channel].slice(obspy.UTCDateTime(central_noise_window[0]), obspy.UTCDateTime(central_noise_window[1]))
                    if convert_to_numpy:
                        st[channel] = st[channel].data
                noise_stream[station] = st


            return noise_stream
    
        def flatten_all_streams(self, stream, components, desired_length):
            numpy_data = []
            for station in self.available_stations_during_event:
                for component in components:
                    numpy_data.append(stream[station][component].data[:desired_length])
            return np.array(numpy_data)
        
import obspy
class ProcessedDataSlicer:

        def __init__(self, data_folder, sampling_rate, covariance_estimation_window=None, full_auto_correlation=False,receivers=None):

            self.data_folder = data_folder
            self.sampling_rate = sampling_rate
            self.covariance_estimation_window = covariance_estimation_window
            self.full_auto_correlation = full_auto_correlation
            self.receivers = receivers  

        def rename_component(self, component):
            if 'Z' in component:
                return 'Z'
            elif '1' in component or 'E'  == component[-1]:
                return '1'
            elif '2' in component or 'N'  == component[-1]:
                return '2'
            
        
        def load_noise_window_data(self, datetime_window):
            current_date = datetime_window[0].date()
            starting_datetime = datetime.datetime.combine(current_date, datetime.time(0, 0))
            filename = self.data_folder  / (starting_datetime.strftime("%Y.%m.%d.%H.%M") + '.h5')
            filename = str(filename.resolve())
            data = obspy.read(filename)
            data_copy = data.copy()
            station_component_map = {}
            for trace in data:
                station = trace.stats.station
                if station not in self.receivers:
                    continue
                component = trace.stats.channel
                if station not in station_component_map:
                    station_component_map[station] = []
                station_component_map[station].append(component)
            fixed_num_seconds = m.ceil(((datetime_window[1] - datetime_window[0]).seconds/self.sampling_rate)) * self.sampling_rate

            exact_end_time = datetime_window[0] + timedelta(seconds=fixed_num_seconds)
            # select only the stations that are in the receivers list
            data = obspy.Stream()
            for receiver in self.receivers:
                try:
                    data += data_copy.select(station=receiver).copy()
                except Exception as e:
                    continue
            data = data.slice(obspy.UTCDateTime(datetime_window[0] - timedelta(seconds=2)), obspy.UTCDateTime(exact_end_time + timedelta(2)), nearest_sample=False)
            npts = compute_data_vector_length(fixed_num_seconds, self.sampling_rate) + 1
            for i, trace in enumerate(data):

                try:
                    interped_trace = trace.interpolate(
                        sampling_rate=self.sampling_rate,
                        starttime=obspy.UTCDateTime(datetime_window[0]),
                        npts=npts,
                        method='lanczos',
                        a=20
                    )
                except Exception as e:
                    # print("trace staart time", trace.stats.starttime, trace.stats.endtime)
                    # traceback.print_exc()
                    # print(f"Error in Trace {i}: {trace.id}")
                    # print(f"Exception: {e}")
                    pass

            try:
                interped_data = data.interpolate(self.sampling_rate, starttime=obspy.UTCDateTime(datetime_window[0]), npts=npts, method='lanczos', a=20)
            except Exception as e:
                print(datetime_window[0])
                # traceback.print_exc()
                return station  + ' interp falied'
            # get all stations and the associated components
            data_map = {station:{} for station in station_component_map.keys()}
            for station in station_component_map.keys():
                station_data =  interped_data.select(station=station)
                
                if len(station_component_map[station]) != 3:
                    return station + ' comp'
                if len(station_data) == 0:
                    return station + ' empty'
                for component in station_component_map[station]:
                    renamed_component = self.rename_component(component)
                    try:
                        data_map[station][renamed_component] = station_data.select(channel=component)[0].data
                    except Exception as e:
                        return station + ' no data'
                if '2' not in data_map[station].keys():
                    print(station, list(data_map[station].keys()), station_component_map[station])
            if self.covariance_estimation_window is not None:
                try:
                    cov_start_time = datetime_window[0] - self.covariance_estimation_window
                    cov_estimation_data = data_copy.slice(obspy.UTCDateTime(cov_start_time), obspy.UTCDateTime(datetime_window[0]))
                    variance_dict = {}
                    for trace in cov_estimation_data:
                        station = trace.stats.station
                        component = trace.stats.channel
                        renamed_component = self.rename_component(component)
                        data_array = trace.data
                        if self.full_auto_correlation:
                            data_length = data_array.shape[0]

                            # compute autocorrelation
                            auto_correlate = np.correlate(data_array, data_array, mode='full')
                            averaged_auto_correlations = auto_correlate[:data_length][::-1]/np.arange(data_length, 0, -1)

                            auto_cov = averaged_auto_correlations
                            variance = auto_cov[0]

                        else:
                            auto_cov = np.var(data_array)
                            variance = auto_cov
                        # Add the variance to the dictionary
                        if station not in variance_dict:
                            variance_dict[station] = {}
                        # print(auto_cov)
                        variance_dict[station][renamed_component] = auto_cov
                        if variance <= 1e-30:
                            print(station, renamed_component, variance)
                            return station + ' variance'

                except Exception as e:
                    print(e)
                    return station + ' exception'
                
            else:
                variance_dict = None
                    
            return (data_map, variance_dict )