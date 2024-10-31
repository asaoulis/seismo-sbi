import numpy as np
from abc import ABC, abstractmethod
import obspy

from typing import NamedTuple
from datetime import timedelta

from .receivers import Receiver

import instaseis

class SyntheticsPreprocessing:

    def __init__(self, processing_config):
        self.processing_config = processing_config

    def __call__(self, seismograms):
        start, end = seismograms[0].stats.starttime, seismograms[0].stats.endtime
        length = end - start
        seismograms = seismograms.trim(starttime=start - length * 0.3, endtime=end + length * 0.3, pad = True, fill_value=0)
        seismograms = seismograms.taper(max_percentage=0.05, type='cosine')
        # seismograms = seismograms.filter('bandpass', freqmin=0.04, freqmax=0.07, corners=4, zerophase=False)
        seismograms = seismograms.filter(**self.processing_config['filter'])

        seismograms = seismograms.trim(starttime=start-60, endtime=end - length * 0.1)
        seismograms = seismograms.trim(starttime=start-60, endtime=end-60, pad=True, fill_value=0)

        return seismograms

class MomentTensor(ABC):
    
    @abstractmethod
    def _asdict(self):
        pass

class SimpleMomentTensor(MomentTensor):

    def __init__(self, source_magnitude):

        self.source_magnitude = source_magnitude
        moment_tensor_elements = np.sqrt(source_magnitude**2/3)
        self.components = np.concatenate([np.ones(3)*moment_tensor_elements, np.zeros(3)])

    def _asdict(self):
        return {"components": self.components, "earthquake_magnitude": self.source_magnitude}

class GeneralMomentTensor(MomentTensor):
    component_strings = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]

    def __init__(self, moment_tensor_components):
        self.components = moment_tensor_components

    def _asdict(self):
        return {component_string: component for component_string, component in zip(self.component_strings, self.components)}

class SourceLocation(NamedTuple):

    latitude : float
    longitude : float
    depth : float
    time_shift : float

class GenericPointSource(NamedTuple):

    source_location : SourceLocation
    moment_tensor : MomentTensor

class InstaseisDBQuerier:

    def __init__(self, instaseis_model_loc, processing_config, seismogram_duration_in_s = None) -> None:

        self.instaseis_database = instaseis.open_db(instaseis_model_loc)
        self.preprocessing = SyntheticsPreprocessing(processing_config)
        self._seismogram_duration_in_s = seismogram_duration_in_s
        
        self.sampling_rate = self._get_db_attribute('sampling_rate')
        self._raw_seismogram_duration_in_s = self._get_db_attribute('length')
        self._raw_seismogram_length = self._get_db_attribute('npts')
        self._dt = self._get_db_attribute('dt')

    def _get_db_attribute(self, key):
        return self.instaseis_database.info[key]

    def get_seismograms(self, source : GenericPointSource, receiver : Receiver, components):

        instaseis_source = self._create_source_object(source)
        instaseis_receiver = self._create_receiver_object(receiver)

        seismograms =  self.instaseis_database.get_seismograms(
                                                    instaseis_source,
                                                    instaseis_receiver,
                                                    components,
                                                    kind='displacement',
                                                    return_obspy_stream=True,
                                                    remove_source_shift=False,
                                                    reconvolve_stf = True)
                                                    # dt=1) ## hard coding to hack iasp92 to work with prem sampling rate
                                                    #dt=1/1.9521351683137325) # TODO: fix this hardcoding
        starttime = seismograms[0].stats.starttime
        if self._seismogram_duration_in_s is not None:
            seismograms  = seismograms.slice(starttime=starttime,
                                             endtime= starttime + timedelta(seconds=self._seismogram_duration_in_s + 10))

        # seismograms = seismograms.decimate(factor=2, no_filter=True)
        seismograms = self.preprocessing(seismograms)
        starttime = seismograms[0].stats.starttime

        seismograms = seismograms.interpolate(1, starttime=starttime, npts=901, method='lanczos', a=20)
        seismograms = {component: seismograms.select(component=component)[0].data for component in components}

        # seismograms = {component: seismograms[component] for component in components}

        return seismograms



    def _create_source_object(self, source: GenericPointSource):

        location = source.source_location
        m_tensor = source.moment_tensor.components


        if location.depth < 0:
            print('Warning: depth is negative. Setting depth to 0')
            location = location._replace(depth = 0)
        custom_scale = 1
        source = instaseis.Source(
            latitude=location.latitude,
            longitude=location.longitude,
            depth_in_m = location.depth * 1e3,
            time_shift = location.time_shift,
            dt = self._dt,
            m_rr = custom_scale*m_tensor[0],
            m_tt = custom_scale*m_tensor[1],
            m_pp = custom_scale*m_tensor[2],
            m_rt = custom_scale*m_tensor[3],
            m_rp = custom_scale*m_tensor[4],
            m_tp = custom_scale*m_tensor[5]
        )

        # Dirac source time function
        sliprate = np.zeros(1000)

        # sliprate[0] = 1.0/self._dt
        sliprate[0] = 1.0

        source.set_sliprate(sliprate, self._dt, time_shift=location.time_shift, normalize=True)

        return source
    
    def _create_receiver_object(self, receiver : Receiver):

        instaseis_receiver = instaseis.Receiver(
            latitude=receiver.latitude,
            longitude=receiver.longitude,
            network=receiver.network,
            station=receiver.station_name
        )

        return instaseis_receiver
    
    def _apply_time_shift(self, seismograms, time_shift):
        if time_shift !=0:
            time_shift_direction_is_positive = (time_shift >=0)
            num_elements_to_shift = round(self.sampling_rate * time_shift)
            for component, seismogram_array in seismograms.items():
                rolled_seismogram_component = np.roll(seismogram_array, num_elements_to_shift)
                if time_shift_direction_is_positive:
                    rolled_seismogram_component[:num_elements_to_shift] = 0
                else:
                    rolled_seismogram_component[num_elements_to_shift:] = 0

                seismograms[component] = rolled_seismogram_component
        return seismograms
    
    def _slice_seismograms(self, seismograms : dict, seismogram_duration):

        new_length = (self._raw_seismogram_length * seismogram_duration) \
                                            // self._raw_seismogram_duration_in_s
        new_length = int(new_length)
        for component, seismogram_array in seismograms.items():
            seismograms[component] = seismogram_array[:new_length]

        return seismograms