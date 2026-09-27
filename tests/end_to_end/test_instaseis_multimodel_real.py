"""InstaseisMultiModelSimulator against a real Instaseis database.

Two regions with disjoint station sets are each backed by an ensemble; every station's merged
seismogram must equal a single-ensemble simulation of that station. Both ensembles link to the
database at ``INSTASEIS_DB`` (one member and the fiducial), and ``use_fiducial=True`` keeps the
draw deterministic. Skipped when the database is absent.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from seismo_sbi.simulators.instaseis.ensemble import InstaseisEnsembleSimulator
from seismo_sbi.simulators.instaseis.multi_model import InstaseisMultiModelSimulator
from seismo_sbi.simulators.receivers import Receiver, Receivers
from seismo_sbi.simulators.sources import GenericPointSource, GeneralMomentTensor, SourceLocation

pytestmark = pytest.mark.slow

INSTASEIS_DB = Path(os.environ.get("INSTASEIS_DB", "/data/shared/ROSA_PREM_10s_disc"))


@pytest.fixture(scope="module")
def ensemble(tmp_path_factory):
    """``(ensemble_dir, fiducial_dir)``: a one-member ensemble linking to ``INSTASEIS_DB``."""
    if not INSTASEIS_DB.is_dir():
        pytest.skip(f"no Instaseis database at {INSTASEIS_DB}")
    root = tmp_path_factory.mktemp("ensemble")
    (root / "members").mkdir()
    (root / "members" / "member_0").symlink_to(INSTASEIS_DB)
    (root / "fiducial").symlink_to(INSTASEIS_DB)
    return str(root / "members"), str(root / "fiducial")


_PROC = {
    "filter": {"type": "bandpass", "freqmin": 0.03, "freqmax": 0.08,
               "corners": 4, "zerophase": False},
    "sampling_rate": 1.0,
}
_DURATION = 200.0
_COMPONENTS = ["Z"]

# One station per region.
_STA_A = Receiver(latitude=36.47090, longitude=25.40560, network="HT",
                  station_name="CMBO", components=["Z"])   # region A
_STA_B = Receiver(latitude=36.63001, longitude=25.67795, network="HT",
                  station_name="ANYD", components=["Z"])   # region B

_SOURCE = GenericPointSource(
    SourceLocation(36.45, 25.55, 12.0, 0.0),
    GeneralMomentTensor([1e16, 1e16, 1e16, 1e16, 1e16, 1e16]),
)


@pytest.fixture(scope="module")
def multimodel(ensemble):
    ensemble_dir, fiducial_dir = ensemble
    region_a = Receivers(receivers=[_STA_A])
    region_b = Receivers(receivers=[_STA_B])
    union = Receivers(receivers=[_STA_A, _STA_B])
    return InstaseisMultiModelSimulator(
        models=[
            {"receivers": region_a, "ensemble_dir": ensemble_dir, "fiducial_dir": fiducial_dir},
            {"receivers": region_b, "ensemble_dir": ensemble_dir, "fiducial_dir": fiducial_dir},
        ],
        components=_COMPONENTS, receivers=union,
        seismogram_duration_in_s=_DURATION, synthetics_processing=_PROC,
    )


@pytest.fixture(scope="module")
def single_region_a(ensemble):
    ensemble_dir, fiducial_dir = ensemble
    return InstaseisEnsembleSimulator(
        instaseis_ensemble_dir=ensemble_dir, instaseis_fiducial_loc=fiducial_dir,
        components=_COMPONENTS, receivers=Receivers(receivers=[_STA_A]),
        seismogram_duration_in_s=_DURATION, synthetics_processing=_PROC,
    )


@pytest.fixture(scope="module")
def single_region_b(ensemble):
    ensemble_dir, fiducial_dir = ensemble
    return InstaseisEnsembleSimulator(
        instaseis_ensemble_dir=ensemble_dir, instaseis_fiducial_loc=fiducial_dir,
        components=_COMPONENTS, receivers=Receivers(receivers=[_STA_B]),
        seismogram_duration_in_s=_DURATION, synthetics_processing=_PROC,
    )


class TestInstaseisMultiModelReal:

    def test_sampling_rate_exposed(self, multimodel):
        assert multimodel.sampling_rate > 0

    def test_merged_output_covers_both_stations_and_is_finite(self, multimodel):
        result = multimodel.generic_point_source_simulation(_SOURCE, use_fiducial=True)
        assert set(result.keys()) == {"CMBO", "ANYD"}
        for sta in ("CMBO", "ANYD"):
            assert np.all(np.isfinite(result[sta]["Z"]))
            assert len(result[sta]["Z"]) > 0

    def test_each_station_matches_its_single_ensemble(
        self, multimodel, single_region_a, single_region_b
    ):
        """The merged trace for a station must be bit-equal to a single-ensemble
        simulation of that same station (proves correct dispatch + merge)."""
        merged = multimodel.generic_point_source_simulation(_SOURCE, use_fiducial=True)
        a = single_region_a.generic_point_source_simulation(_SOURCE, use_fiducial=True)
        b = single_region_b.generic_point_source_simulation(_SOURCE, use_fiducial=True)
        assert np.allclose(merged["CMBO"]["Z"], a["CMBO"]["Z"])
        assert np.allclose(merged["ANYD"]["Z"], b["ANYD"]["Z"])

    def test_two_stations_are_distinct(self, multimodel):
        result = multimodel.generic_point_source_simulation(_SOURCE, use_fiducial=True)
        assert not np.allclose(result["CMBO"]["Z"], result["ANYD"]["Z"]), (
            "Different stations must yield distinct seismograms"
        )

    def test_run_simulation_roundtrip(self, multimodel):
        source_params = {
            "source_location": [36.45, 25.55, 12.0, 0.0],
            "moment_tensor": [1e16] * 6,
            "use_fiducial": True,
        }
        _, seismograms = multimodel.run_simulation(dict(source_params))
        assert set(seismograms.keys()) == {"CMBO", "ANYD"}
        assert all(np.all(np.isfinite(seismograms[s]["Z"])) for s in seismograms)
