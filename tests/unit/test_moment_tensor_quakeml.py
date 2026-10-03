"""Moment tensors and hypocentres read from and written to ObsPy's QuakeML classes."""
from pathlib import Path

import numpy as np
import pytest
from obspy import read_events
from obspy.core.event import Origin

from seismo_sbi.moment_tensor.comparison import pyrocko_mt
from seismo_sbi.moment_tensor.conventions import moment_magnitude
from seismo_sbi.moment_tensor.quakeml import (
    moment_tensor_from_tensor, posterior_event, source_location_from_origin, tensor_from_moment_tensor)

GCMT_EVENT = Path(__file__).parent.parent / "fixtures" / "gcmt_C200604092050A.ndk"
#: The NDK file's tensor row (exponent 24, dyne-cm) in N.m, order rr, tt, pp, rt, rp, tp.
GCMT_M6_NM = np.array([4.180, -1.700, -2.480, -1.050, -2.410, -2.280]) * 1e17


def gcmt_event():
    return read_events(str(GCMT_EVENT), format="NDK")[0]


def test_gcmt_tensor_reads_as_m6_in_newton_metres():
    tensor = gcmt_event().focal_mechanisms[0].moment_tensor.tensor
    assert np.allclose(moment_tensor_from_tensor(tensor), GCMT_M6_NM, rtol=1e-12)


def test_tensor_round_trips_through_a_quakeml_file(tmp_path):
    event = gcmt_event()
    event.focal_mechanisms[0].moment_tensor.tensor = tensor_from_moment_tensor(GCMT_M6_NM)
    path = tmp_path / "event.xml"
    event.write(str(path), format="QUAKEML")

    tensor = read_events(str(path))[0].focal_mechanisms[0].moment_tensor.tensor
    assert np.array_equal(moment_tensor_from_tensor(tensor), GCMT_M6_NM)


def test_centroid_location_is_in_km_and_seconds_after_the_hypocentre():
    origins = {origin.origin_type: origin for origin in gcmt_event().origins}
    hypocentre, centroid = origins["hypocenter"], origins["centroid"]

    location = source_location_from_origin(centroid, origin_time_utc=hypocentre.time)

    assert (location.latitude, location.longitude) == (-20.46, -70.73)
    assert location.depth == pytest.approx(39.0)
    assert location.time_shift == pytest.approx(5.3)
    assert source_location_from_origin(centroid).time_shift == 0.0


def test_origin_without_depth_is_rejected():
    with pytest.raises(ValueError):
        source_location_from_origin(Origin(latitude=0.0, longitude=0.0, time=gcmt_event().origins[0].time))


def gcmt_posterior(n_samples=2000):
    spread = np.array([0.05, 0.04, 0.06, 0.03, 0.02, 0.04]) * 1e17
    return np.random.default_rng(0).normal(GCMT_M6_NM, spread, size=(n_samples, 6))


def hypocentre_origin():
    return next(origin for origin in gcmt_event().origins if origin.origin_type == "hypocenter")


def test_posterior_event_round_trips_its_tensor_and_spreads_through_quakeml(tmp_path):
    samples = gcmt_posterior()
    path = tmp_path / "posterior.xml"
    posterior_event(samples, hypocentre_origin()).write(str(path), format="QUAKEML")

    event = read_events(str(path))[0]
    moment_tensor = event.preferred_focal_mechanism().moment_tensor
    assert np.allclose(moment_tensor_from_tensor(moment_tensor.tensor), samples.mean(axis=0), rtol=1e-12)
    assert moment_tensor.tensor.m_rp_errors.uncertainty == pytest.approx(samples[:, 4].std())
    assert event.preferred_magnitude().mag == pytest.approx(moment_magnitude(samples.mean(axis=0)))
    assert event.preferred_origin().resource_id == hypocentre_origin().resource_id


def test_posterior_event_mechanism_matches_the_published_gcmt_solution():
    event = posterior_event(gcmt_posterior(), hypocentre_origin(), point_estimate=GCMT_M6_NM)
    mechanism = event.preferred_focal_mechanism()

    planes = [(plane.strike, plane.dip, plane.rake) for plane in
              (mechanism.nodal_planes.nodal_plane_1, mechanism.nodal_planes.nodal_plane_2)]
    assert np.allclose(planes, pyrocko_mt(GCMT_M6_NM).both_strike_dip_rake())
    assert np.allclose(sorted(planes), [[49, 30, 106], [211, 61, 81]], atol=1)
    axes = mechanism.principal_axes
    assert (round(axes.t_axis.azimuth), round(axes.t_axis.plunge)) == (100, 73)
    assert (round(axes.p_axis.azimuth), round(axes.p_axis.plunge)) == (308, 15)
    assert axes.t_axis.length == pytest.approx(4.975e17, rel=1e-3)
    assert mechanism.moment_tensor.scalar_moment == pytest.approx(5.035e17, rel=1e-3)
    assert mechanism.moment_tensor.double_couple + mechanism.moment_tensor.clvd == pytest.approx(1.0)
