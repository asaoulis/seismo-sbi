"""Point sources and their source time functions.

A :class:`GenericPointSource` pairs a :class:`SourceLocation` with a moment tensor, either the
six components (``m_rr, m_tt, m_pp, m_rt, m_rp, m_tp``, in N.m) or an isotropic magnitude.
:func:`build_stf_sliprate` builds the unit-moment sliprate handed to the forward model: a Dirac
by default, or a triangle whose half-duration scales the GCMT law.
"""

import numpy as np
from abc import ABC, abstractmethod

from typing import NamedTuple

from seismo_sbi.utils.mt_conventions import scalar_moment


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

#: Shortest sliprate array Instaseis accepts, in samples.
_MIN_STF_SAMPLES: int = 1000

#: Ekstrom and Dziewonski's constant in T_half (s) = GCMT_SCALE_FACTOR * M0^(1/3), M0 in N.m.
GCMT_SCALE_FACTOR: float = 2.262e-6


def _gcmt_half_duration(mt_components: np.ndarray) -> float:
    """Half-duration in s of the triangular source time function the GCMT law predicts.

    ``mt_components`` is ``[m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]`` in N.m. The law gives the
    mean duration at a magnitude; individual earthquakes scatter by about a factor of two.
    """
    m0 = scalar_moment(mt_components)
    return GCMT_SCALE_FACTOR * m0 ** (1.0 / 3.0)


def _build_triangular_stf(half_duration: float, dt: float) -> np.ndarray:
    """Isosceles-triangle sliprate over ``[0, 2 * half_duration]``, sampled every ``dt`` s.

    ``half_duration`` is the rise time, equal to the fall time, in s. The analytical area is
    one but the discrete area is not, so :func:`build_stf_sliprate` renormalises.
    """
    half_duration = float(half_duration)
    # The half-sample tolerance keeps the falling edge in despite floating-point rounding.
    t = np.arange(0.0, 2.0 * half_duration + 0.5 * dt, dt)
    sliprate = np.where(
        t <= half_duration,
        t / half_duration ** 2,
        np.maximum(0.0, (2.0 * half_duration - t) / half_duration ** 2),
    )
    return sliprate.astype(np.float64)


def _unit_moment_dirac(dt: float) -> np.ndarray:
    """Discrete Dirac sliprate of unit moment: a spike of height ``1/dt``.

    A sliprate integrates to one, and the discrete FFT convolution\'s area is ``sum * dt``,
    which is why the height is ``1/dt``. Pass it to Instaseis with ``normalize=False``:
    Instaseis normalises with ``np.trapz``, which half-weights a boundary spike and so
    doubles the moment.
    """
    sliprate = np.zeros(_MIN_STF_SAMPLES)
    sliprate[0] = 1.0 / dt
    return sliprate


def build_stf_sliprate(
    stf_duration,
    dt: float,
    gcmt_half_duration: float = 0.0,
) -> np.ndarray:
    """Unit-moment sliprate for Instaseis ``set_sliprate``, at least ``_MIN_STF_SAMPLES`` long.

    ``stf_duration`` is ``None`` for a Dirac, leaving all moment-rate shaping to the Green\'s
    function, or a multiplicative factor on ``gcmt_half_duration`` (in s) giving an isosceles
    triangle of that half-duration: 1.0 is the scaling law, 0.5 halves it. ``dt`` is the
    database sample interval in s. The array satisfies ``sliprate.sum() * dt == 1``, so the
    caller must pass ``normalize=False``. Raises ValueError for a non-positive
    ``gcmt_half_duration`` when ``stf_duration`` is not ``None``.
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
    # A half-duration below about dt/2 discretises to a zero-area triangle; it is unresolvable
    # at this sample interval, so fall back to the Dirac rather than divide by zero.
    area = float(sliprate.sum()) * dt
    if not area > 0.0:
        return _unit_moment_dirac(dt)
    return sliprate / area
