"""Read seismograms out of one Instaseis database.

:class:`InstaseisDBQuerier` opens the database, converts a
:class:`~seismo_sbi.simulators.sources.GenericPointSource` into an Instaseis source with its
sliprate, and returns ``{component: waveform}`` per receiver. :class:`SyntheticsPreprocessing`
is the taper-filter-trim chain applied to every raw synthetic.
"""

import numpy as np

from datetime import timedelta

from ..receivers import Receiver
from ..sources import GenericPointSource, _gcmt_half_duration, build_stf_sliprate
from seismo_sbi.utils.seismograms import compute_data_vector_length

import instaseis


# Seconds of pre-origin pad every simulated seismogram carries: the final trims below place the
# source origin at t = +this in the exported window. The OBSERVED event catalogue MUST be windowed
# with the same lead (build_event_catalogue ``pre_event_window_s`` defaults to this), or obs and
# synthetics are misaligned by this many seconds -> out-of-distribution inference. Single source of
# truth for that convention.
SYNTHETICS_PRE_EVENT_PAD_S = 60.0

class SyntheticsPreprocessing:

    def __init__(self, processing_config):
        self.processing_config = processing_config
        self.sampling_rate = processing_config['sampling_rate']

    def __call__(self, seismograms):

        start, end = seismograms[0].stats.starttime, seismograms[0].stats.endtime
        length = (end - start)*self.sampling_rate
        seismograms = seismograms.trim(starttime=start - length * 0.3, endtime=end + length * 0.3, pad = True, fill_value=0)
        seismograms = seismograms.taper(max_percentage=0.05, type='cosine')
        # seismograms = seismograms.filter('bandpass', freqmin=0.04, freqmax=0.07, corners=4, zerophase=False)
        seismograms = seismograms.filter(**self.processing_config['filter'])

        pad = SYNTHETICS_PRE_EVENT_PAD_S
        seismograms = seismograms.trim(starttime=start - pad, endtime=end - length * 0.1)
        seismograms = seismograms.trim(starttime=start - pad, endtime=end - pad, pad=True, fill_value=0)

        return seismograms
class InstaseisDBQuerier:

    def __init__(self, instaseis_model_loc, processing_config, seismogram_duration_in_s = None,
                 source_depth_offset_km: float = 0.0) -> None:
        """
        source_depth_offset_km:
            Datum offset added to a source's depth at the Instaseis boundary, in km.

            Instaseis measures source depth from the *model's free surface*, which is
            not always sea level: an AxiSEM model built with its surface at mean ground
            elevation sits above the sea-level datum that catalogues use.  This offset
            is the (positive downward) distance from the model free surface to the
            catalogue datum, so that catalogues, prior boxes, conditioning vectors and
            posteriors can all stay in the catalogue's own datum (normally km b.s.l.)
            while only the handoff to Instaseis is corrected.

            Defaults to ``0.0`` (model surface == catalogue datum), which is the
            behaviour of every configuration that does not set it.
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
                                                    # dt=1) ## hard coding to hack iasp92 to work with prem sampling rate
                                                    #dt=1/1.9521351683137325) # TODO: fix this hardcoding
        starttime = seismograms[0].stats.starttime
        if self._seismogram_duration_in_s is not None:
            seismograms  = seismograms.slice(starttime=starttime,
                                             endtime= starttime + timedelta(seconds=self._seismogram_duration_in_s + 10))

        # seismograms = seismograms.decimate(factor=2, no_filter=True)
        seismograms = self.preprocessing(seismograms)
        starttime = seismograms[0].stats.starttime
        npts = compute_data_vector_length(self._seismogram_duration_in_s, self.preprocessing.sampling_rate)
        seismograms = seismograms.interpolate(self.preprocessing.sampling_rate, starttime=starttime, npts=npts +1, method='lanczos', a=20)
        seismograms = {component: seismograms.select(component=component)[0].data for component in components}

        # seismograms = {component: seismograms[component] for component in components}

        return seismograms



    def _create_source_object(self, source: GenericPointSource, stf_duration=None):
        """Build an Instaseis Source object.

        Parameters
        ----------
        source:
            Point source with location and moment tensor.
        stf_duration:
            If ``None`` (default), use a Dirac delta source time function
            (original behaviour — backward compatible).

            Otherwise, a multiplicative scatter factor applied to the GCMT
            empirical half-duration derived from this source's scalar moment:

            .. math::

                T_{\\text{eff}} = \\text{stf\\_duration} \\times 2.4\\times10^{-6}
                \\cdot M_0^{1/3}

            A value of 1.0 reproduces the scaling-law prediction; 0.5 halves
            it; 2.0 doubles it.  The STF is an isosceles triangle of
            half-duration T_eff (see :func:`_gcmt_half_duration` and
            :func:`build_stf_sliprate`).
        """
        location = source.source_location
        m_tensor = source.moment_tensor.components

        # Convert from the catalogue datum to depth below the model's free surface.
        # The guard below tests the *post-offset* depth because that is the quantity
        # Instaseis rejects; with the default offset of 0.0 this is unchanged.
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

        # Derive the GCMT baseline half-duration from M₀ when the STF nuisance
        # is active.  This is the only place where the moment tensor feeds back
        # into the STF — no upstream plumbing changes required.
        gcmt_t_half = _gcmt_half_duration(m_tensor) if stf_duration is not None else 0.0
        sliprate = build_stf_sliprate(stf_duration, self._dt, gcmt_half_duration=gcmt_t_half)
        # normalize=False: build_stf_sliprate already normalises to unit moment using the DC
        # convention (sum*dt). Instaseis's normalize=True uses np.trapz, which half-weights
        # endpoints and so doubles a boundary-spike Dirac -> synthetics 2x too loud (dMw -0.2007).
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
