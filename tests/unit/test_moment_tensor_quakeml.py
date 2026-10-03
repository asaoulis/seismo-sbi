"""Moment tensors and hypocentres read from and written to ObsPy's QuakeML classes."""
from pathlib import Path

import numpy as np
import pytest
from obspy import read_events
from obspy.core.event import Origin

from seismo_sbi.moment_tensor.quakeml import (
    moment_tensor_from_tensor, source_location_from_origin, tensor_from_moment_tensor)

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
