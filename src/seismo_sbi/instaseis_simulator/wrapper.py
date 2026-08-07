import numpy as np
from abc import ABC, abstractmethod
import obspy

from typing import NamedTuple
from datetime import timedelta

from .receivers import Receiver
from .utils import compute_data_vector_length

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

# ---------------------------------------------------------------------------
# Source time function helpers
# ---------------------------------------------------------------------------

#: Minimum sliprate array length expected by Instaseis.
_MIN_STF_SAMPLES: int = 1000

#: GCMT empirical constant: T_half (s) = GCMT_SCALE_FACTOR · M₀^(1/3), M₀ in N·m.
#: Calibrated by Ekström & Dziewonski to fit the Global CMT catalogue.
GCMT_SCALE_FACTOR: float = 2.262e-6


def _scalar_moment(mt_components: np.ndarray) -> float:
    """Scalar seismic moment M₀ (N·m) from a 6-component moment tensor.

    Uses the Frobenius-norm convention
    ``M₀ = (1/√2) · ‖m_ij‖_F = sqrt(0.5 · Σ mᵢⱼ²)``,
    consistent with the Hanks-Kanamori moment-magnitude relation.

    Parameters
    ----------
    mt_components:
        ``[m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]`` in N·m.

    Returns
    -------
    float
        Scalar moment in N·m.
    """
    return float(np.sqrt(0.5 * np.dot(mt_components, mt_components)))


def _gcmt_half_duration(mt_components: np.ndarray) -> float:
    """GCMT empirical half-duration (s) for an isosceles-triangle STF.

    Applies the Ekström / Dziewonski relation used by the Global CMT routine:

    .. math::

        T_{\\text{half}} = 2.262 \\times 10^{-6} \\cdot M_0^{1/3}

    where M₀ is in N·m and T_half is in seconds.  The relation encodes the
    *mean* duration for a given magnitude; individual earthquakes scatter by
    roughly a factor of 2 around this.

    Parameters
    ----------
    mt_components:
        ``[m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]`` in N·m.

    Returns
    -------
    float
        Predicted STF half-duration in seconds.
    """
    m0 = _scalar_moment(mt_components)
    return GCMT_SCALE_FACTOR * m0 ** (1.0 / 3.0)


def _build_triangular_stf(half_duration: float, dt: float) -> np.ndarray:
    """Build an isosceles-triangle sliprate with half-duration *half_duration*.

    The triangle rises linearly from 0 at t = 0 to its peak at t = T_half,
    then falls linearly back to 0 at t = 2·T_half:

    .. code-block::

        sliprate(t) = t / T_half²             for 0 ≤ t ≤ T_half
        sliprate(t) = (2·T_half − t) / T_half²  for T_half < t ≤ 2·T_half

    The analytical area equals 1 (unit moment), but the *discrete* area generally does not, so
    :func:`build_stf_sliprate` renormalises by the DC convention ``sum * dt`` before returning.
    Note this is emphatically **not** a no-op for an under-resolved triangle, and the caller
    must pass ``normalize=False`` to ``set_sliprate`` (see :func:`_unit_moment_dirac`).

    Parameters
    ----------
    half_duration:
        Half-duration T_half in seconds (rise-time = fall-time).
    dt:
        Sample interval in seconds matching the Instaseis database.

    Returns
    -------
    np.ndarray
        1-D float64 array covering [0, 2·T_half] (inclusive), non-negative.
    """
    half_duration = float(half_duration)
    # Cover [0, 2·T_half] inclusive; the +0.5*dt tolerance absorbs floating-point
    # rounding so the falling edge at t = 2·T_half is always included.
    t = np.arange(0.0, 2.0 * half_duration + 0.5 * dt, dt)
    sliprate = np.where(
        t <= half_duration,
        t / half_duration ** 2,
        np.maximum(0.0, (2.0 * half_duration - t) / half_duration ** 2),
    )
    return sliprate.astype(np.float64)


def _unit_moment_dirac(dt: float) -> np.ndarray:
    """Discrete Dirac sliprate carrying exactly unit moment.

    A sliprate is a *normalised moment-rate* function: ``∫ sliprate dt = 1``, so that the
    released moment is exactly M₀.  For a discrete FFT convolution the relevant area is the
    DC component ``sliprate.sum() * dt``, so a unit-moment impulse has height ``1/dt`` — the
    same convention as Instaseis's own :meth:`instaseis.source.SourceTimeFunction.set_sliprate_dirac`.

    .. warning::

       Do **not** build this as a unit-height spike and delegate normalisation to
       ``set_sliprate(..., normalize=True)``.  Instaseis normalises by ``np.trapz``, whose
       trapezoidal rule half-weights the *endpoints*; a spike on the first sample integrates
       to ``dt/2`` rather than ``dt``, so the impulse comes back as ``2/dt`` — carrying
       **twice** the intended moment and making every synthetic exactly 2x too loud
       (``dMw = -(2/3)·log10 2 = -0.2007``).  That was a real bug here; see
       ``.claude/runs/santorini-paper-prep/mw-bias-investigation/artifacts/ROOT_CAUSE.md``.
    """
    sliprate = np.zeros(_MIN_STF_SAMPLES)
    sliprate[0] = 1.0 / dt
    return sliprate


def build_stf_sliprate(
    stf_duration,
    dt: float,
    gcmt_half_duration: float = 0.0,
) -> np.ndarray:
    """Return a sliprate array for use with Instaseis ``set_sliprate``.

    Two modes
    ---------
    **Dirac delta** (``stf_duration is None``)
        Returns a 1000-sample spike at index 0 — the original backward-compatible
        behaviour that delegates all moment-rate shaping to the database Green's
        function.

    **Triangular STF** (``stf_duration`` is a scalar)
        ``stf_duration`` is treated as a *multiplicative scatter factor* around
        the GCMT-predicted half-duration:

        .. math::

            T_{\\text{eff}} = \\text{stf\\_duration} \\times \\text{gcmt\\_half\\_duration}

        A value of 1.0 reproduces the scaling-law prediction exactly; 0.5 halves
        it; 2.0 doubles it.  The sliprate is an isosceles triangle of half-duration
        T_eff (see :func:`_build_triangular_stf`), zero-padded to at least
        :data:`_MIN_STF_SAMPLES` samples.

    Parameters
    ----------
    stf_duration:
        ``None`` for a Dirac delta, or a multiplicative scale factor (float or
        1-element array) applied to ``gcmt_half_duration``.
    dt:
        Sample interval in seconds (must match the Instaseis database).
    gcmt_half_duration:
        Baseline half-duration in seconds, typically computed via
        :func:`_gcmt_half_duration` from the moment tensor.  Must be positive
        when ``stf_duration`` is not ``None``.

    Returns
    -------
    np.ndarray
        1-D float64 array of length ≥ :data:`_MIN_STF_SAMPLES`.  Non-negative, and **already
        normalised to unit moment** (``sliprate.sum() * dt == 1``).  Pass it to Instaseis with
        ``set_sliprate(..., normalize=False)`` — letting Instaseis normalise via ``np.trapz``
        doubles a Dirac impulse (see :func:`_unit_moment_dirac`).

    Raises
    ------
    ValueError
        If ``stf_duration`` is not ``None`` and ``gcmt_half_duration <= 0``.
    """
    if stf_duration is None:
        return _unit_moment_dirac(dt)

    scale = float(np.squeeze(stf_duration))
    if gcmt_half_duration <= 0.0:
        raise ValueError(
            f"gcmt_half_duration must be positive when stf_duration is not None "
            f"(got {gcmt_half_duration!r}).  Pass the M₀-derived half-duration from "
            "_gcmt_half_duration()."
        )
    effective_half = scale * float(gcmt_half_duration)
    sliprate = _build_triangular_stf(effective_half, dt)
    if len(sliprate) < _MIN_STF_SAMPLES:
        sliprate = np.concatenate([sliprate, np.zeros(_MIN_STF_SAMPLES - len(sliprate))])
    # A half-duration at or below ~dt/2 discretises to a (near-)zero-area triangle, which would
    # divide by ~0 -> NaN seismograms. Such an STF is unresolvable at this sample interval
    # (physically a delta), so fall back to the Dirac impulse. This keeps stf_duration sampling
    # numerically safe at coarse dt / long periods, where a sub-sample STF has no effect on the
    # band-limited waveform anyway.
    #
    # Normalise to unit moment HERE, using the DC convention (sum * dt) rather than np.trapz --
    # the discrete FFT convolution's long-period gain is exactly sum*dt, and trapz half-weights
    # endpoints. The caller must therefore pass normalize=False to set_sliprate. For a
    # well-resolved triangle (which starts and ends at zero) the two conventions agree; they
    # differ by exactly 2x for a boundary spike, which is what made every synthetic 2x too loud.
    area = float(sliprate.sum()) * dt
    if not area > 0.0:
        return _unit_moment_dirac(dt)
    return sliprate / area


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

        if location.depth < 0:
            print('Warning: depth is negative. Setting depth to 0')
            location = location._replace(depth=0)
            raise ValueError('Depth cannot be negative')
        custom_scale = 1
        instaseis_source = instaseis.Source(
            latitude=location.latitude,
            longitude=location.longitude,
            depth_in_m=location.depth * 1e3,
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

        return seismograms