"""Moment tensors and hypocentres to and from ObsPy's QuakeML event classes.

QuakeML's ``Tensor`` holds ``m_rr, m_tt, m_pp, m_rt, m_rp, m_tp`` in N.m in r, theta, phi = up,
south, east: the library's ``m6`` order and convention, so the components carry over unchanged.
An ``Origin`` gives depth in m and an absolute time; a :class:`SourceLocation` gives depth in km
and the time in s relative to a stated origin time. :func:`posterior_event` reports a posterior as
an ``Event`` with its moment tensor, focal mechanism and moment magnitude.
"""
import numpy as np
from obspy import UTCDateTime
from obspy.core.event import (Axis, CreationInfo, Event, FocalMechanism, Magnitude, MomentTensor, NodalPlane,
                              NodalPlanes, PrincipalAxes, QuantityError, ResourceIdentifier, Tensor)

from seismo_sbi.moment_tensor.comparison import mt_axes, pyrocko_mt
from seismo_sbi.moment_tensor.conventions import create_matrix, moment_magnitude, scalar_moment
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


def posterior_event(posterior_samples, origin, point_estimate=None, method_id="smi:local/seismo-sbi") -> Event:
    """An ObsPy ``Event`` reporting ``posterior_samples``, ``(n_samples, 6)`` moment tensors in N.m
    (up-south-east), for the source at ``origin`` (an ObsPy ``Origin``).

    The tensor is ``point_estimate`` (``m6``; the posterior mean when None), each component's
    uncertainty its posterior standard deviation. The ``MomentTensor`` also carries the scalar moment
    (:func:`~seismo_sbi.moment_tensor.conventions.scalar_moment`) and the isotropic, double-couple
    and CLVD fractions of Jost and Herrmann (1989); the ``FocalMechanism`` the nodal planes and principal axes;
    the ``Magnitude`` Mw with its posterior standard deviation. These describe the point estimate
    and the marginal spreads only: correlations between components, the posterior mass on the lune
    and any second mode are not carried, nor are the samples.
    """
    posterior_samples = np.asarray(posterior_samples, dtype=float)
    m6 = posterior_samples.mean(axis=0) if point_estimate is None else np.asarray(point_estimate, dtype=float)
    method = ResourceIdentifier(method_id)
    creation_info = CreationInfo(author="seismo-sbi", creation_time=UTCDateTime())

    magnitude = Magnitude(mag=float(moment_magnitude(m6)), magnitude_type="Mw", origin_id=origin.resource_id,
                          mag_errors=QuantityError(uncertainty=float(np.std(moment_magnitude(posterior_samples)))),
                          method_id=method, creation_info=creation_info)
    moment_tensor = _moment_tensor(m6, posterior_samples, origin, magnitude, method, creation_info)
    focal_mechanism = FocalMechanism(nodal_planes=_nodal_planes(m6), principal_axes=_principal_axes(m6),
                                     moment_tensor=moment_tensor, triggering_origin_id=origin.resource_id,
                                     method_id=method, creation_info=creation_info)
    event = Event(origins=[origin], magnitudes=[magnitude], focal_mechanisms=[focal_mechanism],
                  creation_info=creation_info)
    event.preferred_origin_id = origin.resource_id
    event.preferred_magnitude_id = magnitude.resource_id
    event.preferred_focal_mechanism_id = focal_mechanism.resource_id
    return event


def _moment_tensor(m6, posterior_samples, origin, magnitude, method, creation_info) -> MomentTensor:
    """The QuakeML ``MomentTensor`` of ``m6`` with the posterior standard deviations as errors."""
    tensor = tensor_from_moment_tensor(m6)
    for component, spread in zip(M6_COMPONENTS, posterior_samples.std(axis=0)):
        setattr(tensor, f"{component}_errors", QuantityError(uncertainty=float(spread)))
    iso, double_couple, clvd = (ratio for _, ratio, _ in pyrocko_mt(m6).standard_decomposition()[:3])
    return MomentTensor(derived_origin_id=origin.resource_id, moment_magnitude_id=magnitude.resource_id,
                        scalar_moment=scalar_moment(m6),
                        scalar_moment_errors=QuantityError(uncertainty=float(np.std(scalar_moment(posterior_samples)))),
                        tensor=tensor, iso=float(iso), double_couple=float(double_couple), clvd=float(clvd),
                        method_id=method, creation_info=creation_info)


def _nodal_planes(m6) -> NodalPlanes:
    """The two nodal planes of ``m6`` (strike, dip, rake in degrees) from pyrocko."""
    planes = [NodalPlane(strike=float(strike), dip=float(dip), rake=float(rake))
              for strike, dip, rake in pyrocko_mt(m6).both_strike_dip_rake()]
    return NodalPlanes(nodal_plane_1=planes[0], nodal_plane_2=planes[1])


def _principal_axes(m6) -> PrincipalAxes:
    """T, P and null axes of ``m6``: azimuth and plunge in degrees, length the eigenvalue in N.m."""
    axes = mt_axes(m6)
    p_length, n_length, t_length = np.linalg.eigvalsh(create_matrix(m6))
    t_axis = Axis(azimuth=float(axes["t_az"][0]), plunge=float(axes["t_plunge"][0]), length=float(t_length))
    p_axis = Axis(azimuth=float(axes["p_az"][0]), plunge=float(axes["p_plunge"][0]), length=float(p_length))
    n_axis = Axis(azimuth=float(axes["n_az"][0]), plunge=float(axes["n_plunge"][0]), length=float(n_length))
    return PrincipalAxes(t_axis=t_axis, p_axis=p_axis, n_axis=n_axis)
