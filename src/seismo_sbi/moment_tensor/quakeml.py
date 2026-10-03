"""Moment tensors and hypocentres to and from ObsPy's QuakeML event classes.

QuakeML's ``Tensor`` holds ``m_rr, m_tt, m_pp, m_rt, m_rp, m_tp`` in N.m in r, theta, phi = up,
south, east: the library's ``m6`` order and convention, so the components carry over unchanged.
An ``Origin`` gives depth in m and an absolute time; a :class:`SourceLocation` gives depth in km
and the time in s relative to a stated origin time.
"""
import numpy as np
from obspy.core.event import Tensor

from seismo_sbi.simulators.sources import SourceLocation

#: QuakeML ``Tensor`` attribute names in ``m6`` order.
M6_COMPONENTS = ("m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp")


def moment_tensor_from_tensor(tensor: Tensor) -> np.ndarray:
    """``m6`` of a QuakeML ``Tensor``, shape ``(6,)``, N.m, up-south-east."""
    return np.array([getattr(tensor, component) for component in M6_COMPONENTS], dtype=float)


def tensor_from_moment_tensor(moment_tensor) -> Tensor:
    """The QuakeML ``Tensor`` of ``moment_tensor``, ``m6`` of shape ``(6,)`` in N.m, up-south-east."""
    return Tensor(**{component: float(value) for component, value in zip(M6_COMPONENTS, moment_tensor)})


def source_location_from_origin(origin, origin_time_utc=None) -> SourceLocation:
    """The :class:`SourceLocation` of an ObsPy ``Origin``: depth in km, and the origin's time in s
    after ``origin_time_utc`` (a ``UTCDateTime``; the origin's own time when None, giving 0).

    An origin without a depth or a time raises ``ValueError``.
    """
    if origin.depth is None or origin.time is None:
        raise ValueError(f"Origin {origin.resource_id} has no depth or no time")
    reference_time = origin.time if origin_time_utc is None else origin_time_utc
    return SourceLocation(latitude=float(origin.latitude), longitude=float(origin.longitude),
                          depth=float(origin.depth) / 1000.0, time_shift=float(origin.time - reference_time))
