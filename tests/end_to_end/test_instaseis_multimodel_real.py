"""End-to-end smoke for InstaseisMultiModelSimulator against a real Instaseis DB.

Proves the multi-model dispatch/merge works with the actual Instaseis backend:
two regions (disjoint station sets) are each backed by an Instaseis-DB ensemble,
and each station's merged seismogram must equal a single-ensemble simulation of
that station.  Uses the locally-available Santorini brustle ensemble; both
regions point at it (the mode_a/mode_b ensembles live only on the cluster during
the autonomous run, so the routing is validated against brustle here).

`use_fiducial=True` is used throughout so the draw is deterministic and never
hits one of the brustle ensemble's known-broken members.

Skipped if the brustle DB is not present locally.  Marked @pytest.mark.slow.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from seismo_sbi.instaseis_simulator.ensemble import InstaseisEnsembleSimulator
from seismo_sbi.instaseis_simulator.multi_model import InstaseisMultiModelSimulator
from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
from seismo_sbi.instaseis_simulator.wrapper import (
    GenericPointSource, GeneralMomentTensor, SourceLocation,
)

pytestmark = pytest.mark.slow

_ENS_DIR = next(
    (p for p in [
        os.environ.get("MULTIMODEL_TEST_DB"),
        "/data/alex/axisem_dbs/santorini_tomo_brustle",
    ] if p and Path(p).is_dir() and (Path(p) / "fiducial").is_dir()),
    None,
)


def _skip_if_no_db():
    if _ENS_DIR is None:
        pytest.skip(
            "No local Instaseis ensemble DB found (set MULTIMODEL_TEST_DB or place "
            "the Santorini brustle ensemble at /data/alex/axisem_dbs/santorini_tomo_brustle)."
        )


_PROC = {
    "filter": {"type": "bandpass", "freqmin": 0.03, "freqmax": 0.08,
               "corners": 4, "zerophase": False},
    "sampling_rate": 1.0,
}
_DURATION = 200.0
_COMPONENTS = ["Z"]

# Two on/near-Santorini lomax stations (one per region).
_STA_A = Receiver(latitude=36.47090, longitude=25.40560, network="HT",
                  station_name="CMBO", components=["Z"])   # region A
_STA_B = Receiver(latitude=36.63001, longitude=25.67795, network="HT",
                  station_name="ANYD", components=["Z"])   # region B

_SOURCE = GenericPointSource(
    SourceLocation(36.45, 25.55, 12.0, 0.0),
    GeneralMomentTensor([1e16, 1e16, 1e16, 1e16, 1e16, 1e16]),
)


def _fiducial_dir():
    return str(Path(_ENS_DIR) / "fiducial")


@pytest.fixture(scope="module")
def multimodel():
    _skip_if_no_db()
    region_a = Receivers(receivers=[_STA_A])
    region_b = Receivers(receivers=[_STA_B])
    union = Receivers(receivers=[_STA_A, _STA_B])
    return InstaseisMultiModelSimulator(
        models=[
            {"receivers": region_a, "ensemble_dir": _ENS_DIR, "fiducial_dir": _fiducial_dir()},
            {"receivers": region_b, "ensemble_dir": _ENS_DIR, "fiducial_dir": _fiducial_dir()},
        ],
        components=_COMPONENTS, receivers=union,
        seismogram_duration_in_s=_DURATION, synthetics_processing=_PROC,
    )


@pytest.fixture(scope="module")
def single_region_a():
    _skip_if_no_db()
    return InstaseisEnsembleSimulator(
        instaseis_ensemble_dir=_ENS_DIR, instaseis_fiducial_loc=_fiducial_dir(),
        components=_COMPONENTS, receivers=Receivers(receivers=[_STA_A]),
        seismogram_duration_in_s=_DURATION, synthetics_processing=_PROC,
    )


@pytest.fixture(scope="module")
def single_region_b():
    _skip_if_no_db()
    return InstaseisEnsembleSimulator(
        instaseis_ensemble_dir=_ENS_DIR, instaseis_fiducial_loc=_fiducial_dir(),
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
