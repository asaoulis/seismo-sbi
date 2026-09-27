"""Unit tests for the catalogue loader (CSV + obspy paths)."""

import numpy as np
import pytest

from seismo_sbi.priors.catalogue import EventCatalogue, load_catalogue

SPLIT_TIME_CSV = (
    "event_id,latitude,longitude,depth,magnitude,magnitude_type,year,month,day,hour,minute,"
    "seconds,RMS,Nphases,AzGap,dist_sta,ErrH,Errz\n"
    "1,36.51,25.46,3.0,1.06,Ml,2024,1,1,2,36,19.488,0.11,14,171,6.9,1.49,4.24\n"
    "2,36.32,25.59,12.9,1.33,Ml,2024,1,1,3,18,27.135,0.16,16,286,11.2,3.34,3.71\n"
    "3,36.40,25.50,7.5,2.10,Mw,2024,1,2,10,5,1.500,0.09,22,120,4.1,0.80,1.20\n"
)
ISO_TIME_CSV = (
    "date-time, latitude, longitude, depth, RMS, Nphs, Gap, Dist, errH, errZ, Mamp, Mdur\n"
    "2025-01-01T14:42:24.000907, 36.4319, 25.4245, 5.8936, 0.066, 15, 134.0, 1.93, 0.25, 0.32, 1.78, -9.9\n"
    "2025-01-01T18:51:54.460904, 36.4188, 25.4074, 5.9277, 0.066, 14, 131.0, 3.47, 0.46, 1.06, 1.44, -9.9\n"
)


def test_split_time_columns_give_km_depths_errors_and_origin_times(tmp_path):
    path = tmp_path / "split_time.csv"
    path.write_text(SPLIT_TIME_CSV)
    cat = load_catalogue(path)
    assert isinstance(cat, EventCatalogue) and len(cat) == 3
    assert cat.latitude.dtype == float and cat.depth.dtype == float
    assert np.allclose(cat.depth, [3.0, 12.9, 7.5])
    assert cat.err_h is not None and cat.err_z is not None
    assert cat.lat_lon_depth.shape == (3, 3)
    assert cat.time.dtype == np.dtype("datetime64[us]")
    assert cat.time[0] == np.datetime64("2024-01-01T02:36:19.488")


def test_iso_time_layout_reads_padded_headers_and_mamp_as_ml(tmp_path):
    path = tmp_path / "iso_time.csv"
    path.write_text(ISO_TIME_CSV)
    cat = load_catalogue(path)
    assert len(cat) == 2
    assert set(np.unique(cat.magnitude_type)) == {"Ml"}
    assert np.allclose(cat.magnitude, [1.78, 1.44])
    assert cat.err_h is not None and cat.err_z is not None
    assert cat.time.min() == np.datetime64("2025-01-01T14:42:24.000907")


def test_magnitude_type_filter_keeps_only_that_type(tmp_path):
    path = tmp_path / "split_time.csv"
    path.write_text(SPLIT_TIME_CSV)
    cat = load_catalogue(path, magnitude_type="Ml")
    assert len(cat) == 2
    assert set(np.unique(cat.magnitude_type)) == {"Ml"}


def test_load_obspy_quakeml_depth_in_km(tmp_path):
    pytest.importorskip("obspy")
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


def test_csv_format_selects_the_layout_instead_of_sniffing(tmp_path):
    """A named layout is read as named; an unknown name raises rather than guessing."""
    path = tmp_path / "events.csv"
    path.write_text(
        "date-time,latitude,longitude,depth,Mamp,errH,errZ\n"
        "2025-02-12T01:14:55,36.5,25.6,8.0,3.2,0.4,0.9\n"
    )
    catalogue = load_catalogue(path, csv_format="iso_time")
    assert len(catalogue) == 1 and catalogue.magnitude[0] == 3.2

    with pytest.raises(ValueError, match="unknown csv_format"):
        load_catalogue(path, csv_format="lomax")
