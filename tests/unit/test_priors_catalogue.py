"""Unit tests for the catalogue loader (CSV + obspy paths)."""

from pathlib import Path

import numpy as np
import pytest

from seismo_sbi.priors.catalogue import EventCatalogue, load_catalogue

REPO_ROOT = Path(__file__).resolve().parents[2]
SANTORINI_CSV = (
    REPO_ROOT / "scripts" / "santorini_pathbreaker" / "catalogue" / "Santorini_catalog.csv"
)


@pytest.mark.skipif(not SANTORINI_CSV.exists(), reason="Santorini catalogue CSV not present")
def test_load_real_santorini_csv():
    cat = load_catalogue(SANTORINI_CSV)
    assert isinstance(cat, EventCatalogue)
    assert len(cat) == 4088
    assert cat.latitude.dtype == float and cat.depth.dtype == float
    # plausible Aegean ranges
    assert 35.0 < cat.latitude.mean() < 38.0
    assert 24.0 < cat.longitude.mean() < 27.0
    # depth is in km, not metres
    assert cat.depth.max() < 100.0
    # error columns present
    assert cat.err_h is not None and cat.err_z is not None
    assert cat.lat_lon_depth.shape == (4088, 3)


@pytest.mark.skipif(not SANTORINI_CSV.exists(), reason="Santorini catalogue CSV not present")
def test_csv_magnitude_type_filter():
    cat = load_catalogue(SANTORINI_CSV, magnitude_type="Ml")
    assert len(cat) > 0
    assert set(np.unique(cat.magnitude_type)) == {"Ml"}


def test_load_obspy_quakeml_depth_in_km(tmp_path):
    obspy = pytest.importorskip("obspy")
    from obspy.core.event import Catalog, Event, Magnitude, Origin
    from obspy import UTCDateTime

    def make(lat, lon, depth_km, mag):
        o = Origin(time=UTCDateTime(2024, 1, 1), latitude=lat, longitude=lon,
                   depth=depth_km * 1000.0)  # obspy depth is metres
        m = Magnitude(mag=mag, magnitude_type="Mw")
        ev = Event(origins=[o], magnitudes=[m])
        ev.preferred_origin_id = o.resource_id
        ev.preferred_magnitude_id = m.resource_id
        return ev

    cat = Catalog(events=[make(36.5, 25.5, 10.0, 4.2), make(36.6, 25.4, 5.0, 3.1)])
    path = tmp_path / "mini.xml"
    cat.write(path, format="QUAKEML")

    loaded = load_catalogue(path)
    assert len(loaded) == 2
    # depth converted metres -> km
    assert np.allclose(np.sort(loaded.depth), [5.0, 10.0])
    assert np.allclose(np.sort(loaded.magnitude), [3.1, 4.2])
