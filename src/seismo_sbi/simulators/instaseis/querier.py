"""Read seismograms out of one Instaseis database.

:class:`InstaseisDBQuerier` opens the database, converts a
:class:`~seismo_sbi.simulators.sources.GenericPointSource` into an Instaseis source with its
sliprate, and returns ``{component: waveform}`` per receiver. :class:`SyntheticsPreprocessing`
is the taper-filter-trim chain applied to every raw synthetic.
"""

import math

import numpy as np

from datetime import timedelta

from ..receivers import Receiver
from ..sources import GenericPointSource, _gcmt_half_duration, build_stf_sliprate
from seismo_sbi.utils.seismograms import compute_data_vector_length

import instaseis


#: Seconds of pre-origin pad every simulated seismogram carries, so the source origin sits at
#: t = +this in the exported window. Observed windows must use the same lead or they are
#: misaligned with the synthetics by this much.
SYNTHETICS_PRE_EVENT_PAD_S = 60.0

class SyntheticsPreprocessing:
    """Taper, filter and trim raw synthetics that start at the origin.

    The filter runs at ``processing['filter_sampling_rate']`` (Hz), the rate the observed data are
    filtered at, so the two paths see the same filter response. The result starts
    ``SYNTHETICS_PRE_EVENT_PAD_S`` before the origin exactly, whatever the database's sample
    interval, so the 1 Hz grid the querier interpolates to lines up with the observed windows.
    """

    def __init__(self, processing_config):
        self.processing_config = processing_config
        self.sampling_rate = processing_config['sampling_rate']
        self.filter_sampling_rate = float(processing_config['filter_sampling_rate'])

    def __call__(self, seismograms):

        start, end = seismograms[0].stats.starttime, seismograms[0].stats.endtime
        length = (end - start)*self.sampling_rate
        seismograms = seismograms.trim(starttime=start - length * 0.3, endtime=end + length * 0.3, pad = True, fill_value=0)
        seismograms = seismograms.taper(max_percentage=0.05, type='cosine')
        pad = SYNTHETICS_PRE_EVENT_PAD_S
        seismograms = self._resample_onto_anchored_grid(seismograms, start - pad)
        seismograms = seismograms.filter(**self.processing_config['filter'])

        seismograms = seismograms.trim(starttime=start - pad, endtime=end - length * 0.1)
        seismograms = seismograms.trim(starttime=start - pad, endtime=end - pad, pad=True, fill_value=0)

        return seismograms

    def _resample_onto_anchored_grid(self, seismograms, anchor):
        """The traces Lanczos-interpolated to the filter rate, on the sample grid through ``anchor``."""
        dt = 1.0 / self.filter_sampling_rate
        first_sample = anchor + math.ceil((seismograms[0].stats.starttime - anchor) / dt - 1e-9) * dt
        return seismograms.interpolate(self.filter_sampling_rate, method='lanczos', a=20,
                                       starttime=first_sample)


class InstaseisDBQuerier:

    def __init__(self, instaseis_model_loc, processing_config, seismogram_duration_in_s = None,
                 source_depth_offset_km: float = 0.0) -> None:
        """``source_depth_offset_km`` is the distance, positive downwards in km, from the
        model's free surface to the datum the catalogue measures depth from.

        Instaseis measures source depth from the free surface, which need not be sea level. The
        offset is applied only at the handoff to Instaseis, so everything upstream stays in the
        catalogue's own datum. It defaults to zero, the two being the same.
        """
        self.instaseis_database = instaseis.open_db(instaseis_model_loc)
        self.preprocessing = SyntheticsPreprocessing(processing_config)
        self._seismogram_duration_in_s = seismogram_duration_in_s
        self.source_depth_offset_km = float(source_depth_offset_km)
        
        self.sampling_rate = self._get_db_attribute('sampling_rate')
        self._raw_seismogram_duration_in_s = self._get_db_attribute('length')
        self._raw_seismogram_length = self._get_db_attribute('npts')
        self._dt = self._get_db_attribute('dt')

    def _get_db_attribute(self, key):
        return self.instaseis_database.info[key]

    def get_seismograms(self, source: GenericPointSource, receiver: Receiver, components, stf_duration=None):

        instaseis_source = self._create_source_object(source, stf_duration=stf_duration)
        instaseis_receiver = self._create_receiver_object(receiver)

        seismograms =  self.instaseis_database.get_seismograms(
                                                    instaseis_source,
                                                    instaseis_receiver,
                                                    components,
                                                    kind='displacement',
                                                    return_obspy_stream=True,
                                                    remove_source_shift=False,
                                                    reconvolve_stf = True)
        starttime = seismograms[0].stats.starttime
        if self._seismogram_duration_in_s is not None:
            seismograms  = seismograms.slice(starttime=starttime,
                                             endtime= starttime + timedelta(seconds=self._seismogram_duration_in_s + 10))

        seismograms = self.preprocessing(seismograms)
        starttime = seismograms[0].stats.starttime
        npts = compute_data_vector_length(self._seismogram_duration_in_s, self.preprocessing.sampling_rate)
        seismograms = seismograms.interpolate(self.preprocessing.sampling_rate, starttime=starttime, npts=npts +1, method='lanczos', a=20)
        seismograms = {component: seismograms.select(component=component)[0].data for component in components}

        return seismograms



    def _create_source_object(self, source: GenericPointSource, stf_duration=None):
        """An Instaseis source with its sliprate set, from a ``GenericPointSource``.

        ``stf_duration`` is ``None`` for a Dirac, or a multiplicative factor on the GCMT
        half-duration this source's scalar moment predicts, giving a triangular sliprate.
        """
        location = source.source_location
        m_tensor = source.moment_tensor.components

        # The guard tests the post-offset depth, which is what Instaseis rejects.
        depth_below_free_surface_km = location.depth + self.source_depth_offset_km

        if depth_below_free_surface_km < 0:
            print('Warning: depth is negative. Setting depth to 0')
            location = location._replace(depth=0)
            raise ValueError('Depth cannot be negative')
        custom_scale = 1
        instaseis_source = instaseis.Source(
            latitude=location.latitude,
            longitude=location.longitude,
            depth_in_m=depth_below_free_surface_km * 1e3,
            time_shift=location.time_shift,
            dt=self._dt,
            m_rr=custom_scale * m_tensor[0],
            m_tt=custom_scale * m_tensor[1],
            m_pp=custom_scale * m_tensor[2],
            m_rt=custom_scale * m_tensor[3],
            m_rp=custom_scale * m_tensor[4],
            m_tp=custom_scale * m_tensor[5],
        )

        # The only place the moment tensor feeds back into the source time function.
        gcmt_t_half = _gcmt_half_duration(m_tensor) if stf_duration is not None else 0.0
        sliprate = build_stf_sliprate(stf_duration, self._dt, gcmt_half_duration=gcmt_t_half)
        # build_stf_sliprate already carries unit moment under the sum*dt convention; letting
        # Instaseis normalise with np.trapz would double a boundary-spike Dirac.
        instaseis_source.set_sliprate(sliprate, self._dt, time_shift=location.time_shift, normalize=False)

        return instaseis_source
    
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
