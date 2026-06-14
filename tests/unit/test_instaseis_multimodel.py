"""Unit tests for the multi-model (multi-region) simulators.

Dependency-free: no Instaseis DB or CPS binaries required.  Each region's
sub-simulator is a tiny tagged mock so the per-station MERGE and routing can be
asserted by provenance (each station's trace carries its owning region's tag).

Covers:
  * station -> region routing over the union receivers,
  * a station absent from every region raises KeyError,
  * use_fiducial=True routes each region to ITS OWN fiducial (per-mode fiducial),
  * INDEPENDENT-per-region member draws (seed -> seed+i; unseeded -> no seed key),
  * the refactor: MultiModelCPSSimulator stays a CPSSimulator AND a
    MultiModelSimulator and merges identically through the shared base.
"""

import numpy as np
import pytest

from seismo_sbi.instaseis_simulator.simulator import Simulator
from seismo_sbi.instaseis_simulator.multi_model import (
    MultiModelSimulator,
    InstaseisMultiModelSimulator,
)
from seismo_sbi.instaseis_simulator.wrapper import (
    GenericPointSource, GeneralMomentTensor, SourceLocation,
)
from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
from seismo_sbi.cps_simulator.simulator import MultiModelCPSSimulator, CPSSimulator

TRACE_LEN = 16
_PROC = {
    "sampling_rate": 1.0,
    "filter": {"type": "bandpass", "freqmin": 0.01, "freqmax": 0.1},
}


# ---------------------------------------------------------------------------
# Tagged mock sub-simulator
# ---------------------------------------------------------------------------

class _TaggedSimulator(Simulator):
    """Returns a constant trace = tag (+100 under use_fiducial), so the merge's
    per-station routing is checkable.  Records the kwargs it was called with so
    the per-region seed semantics can be asserted."""

    def __init__(self, receivers, tag, num_models=3):
        super().__init__(
            components=["Z"],
            receivers=receivers,
            seismogram_duration_in_s=TRACE_LEN,
            synthetics_processing=_PROC,
        )
        self.tag = float(tag)
        self.num_models = num_models
        self.sampling_rate = 1.0
        self.last_kwargs = None
        self.last_seed = "UNCALLED"

    def generic_point_source_simulation(self, source, *, use_fiducial=False, **kwargs):
        self.last_kwargs = dict(kwargs)
        self.last_seed = kwargs.get("seed", "ABSENT")
        value = self.tag + (100.0 if use_fiducial else 0.0)
        return {
            rec.station_name: {comp: np.full(TRACE_LEN, value) for comp in rec.components}
            for rec in self.receivers.iterate()
        }


def _rec(name, lat=0.0, lon=0.0):
    return Receiver(latitude=lat, longitude=lon, network="XX",
                    station_name=name, components=["Z"])


def _dummy_source():
    return GenericPointSource(SourceLocation(0.0, 0.0, 10.0, 0.0),
                              GeneralMomentTensor([1e14] * 6))


# ---------------------------------------------------------------------------
# Fixtures: a 2-region geometry (region A = 2 stations, region B = 1 station)
# ---------------------------------------------------------------------------

@pytest.fixture
def geometry():
    a1, a2, b1 = _rec("STA_A1"), _rec("STA_A2"), _rec("STA_B1")
    region_a = Receivers(receivers=[a1, a2])
    region_b = Receivers(receivers=[b1])
    union = Receivers(receivers=[a1, a2, b1])
    return region_a, region_b, union


@pytest.fixture
def multimodel(geometry):
    region_a, region_b, union = geometry
    sim_a = _TaggedSimulator(region_a, tag=1.0)
    sim_b = _TaggedSimulator(region_b, tag=2.0)
    models = [
        {"receivers": region_a, "simulator": sim_a},
        {"receivers": region_b, "simulator": sim_b},
    ]
    sim = InstaseisMultiModelSimulator(
        models=models, components=["Z"], receivers=union,
        seismogram_duration_in_s=TRACE_LEN, synthetics_processing=_PROC,
    )
    return sim, sim_a, sim_b


# ---------------------------------------------------------------------------
# Construction / type
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_is_simulator_and_multimodel(self, multimodel):
        sim, _, _ = multimodel
        assert isinstance(sim, MultiModelSimulator)
        assert isinstance(sim, Simulator)

    def test_num_models_from_first_region(self, multimodel):
        sim, sim_a, _ = multimodel
        assert sim.num_models == sim_a.num_models

    def test_sampling_rate_exposed(self, multimodel):
        sim, _, _ = multimodel
        assert sim.sampling_rate == 1.0

    def test_empty_models_raises(self):
        with pytest.raises(ValueError):
            InstaseisMultiModelSimulator(
                models=[], components=["Z"], receivers=Receivers(receivers=[_rec("S")]),
                seismogram_duration_in_s=TRACE_LEN, synthetics_processing=_PROC,
            )

    def test_missing_receivers_key_raises(self):
        with pytest.raises(KeyError):
            InstaseisMultiModelSimulator(
                models=[{"ensemble_dir": "x", "fiducial_dir": "y"}],
                components=["Z"], receivers=Receivers(receivers=[_rec("S")]),
                seismogram_duration_in_s=TRACE_LEN, synthetics_processing=_PROC,
            )

    def test_non_dict_entry_raises(self):
        with pytest.raises(TypeError):
            InstaseisMultiModelSimulator(
                models=["not-a-dict"], components=["Z"],
                receivers=Receivers(receivers=[_rec("S")]),
                seismogram_duration_in_s=TRACE_LEN, synthetics_processing=_PROC,
            )


# ---------------------------------------------------------------------------
# Station -> region routing (the core merge)
# ---------------------------------------------------------------------------

class TestRouting:

    def test_each_station_routed_to_its_region(self, multimodel):
        sim, _, _ = multimodel
        result = sim.generic_point_source_simulation(_dummy_source())
        assert set(result.keys()) == {"STA_A1", "STA_A2", "STA_B1"}
        assert np.all(result["STA_A1"]["Z"] == 1.0)   # region A tag
        assert np.all(result["STA_A2"]["Z"] == 1.0)
        assert np.all(result["STA_B1"]["Z"] == 2.0)   # region B tag

    def test_trace_length_preserved(self, multimodel):
        sim, _, _ = multimodel
        result = sim.generic_point_source_simulation(_dummy_source())
        assert len(result["STA_A1"]["Z"]) == TRACE_LEN

    def test_station_not_in_any_region_raises_keyerror(self, geometry):
        region_a, region_b, _ = geometry
        # Union includes STA_C which belongs to no region -> must raise.
        union = Receivers(receivers=[_rec("STA_A1"), _rec("STA_C")])
        sim = InstaseisMultiModelSimulator(
            models=[
                {"receivers": region_a, "simulator": _TaggedSimulator(region_a, 1.0)},
                {"receivers": region_b, "simulator": _TaggedSimulator(region_b, 2.0)},
            ],
            components=["Z"], receivers=union,
            seismogram_duration_in_s=TRACE_LEN, synthetics_processing=_PROC,
        )
        with pytest.raises(KeyError):
            sim.generic_point_source_simulation(_dummy_source())

    def test_run_simulation_roundtrip_routes_and_is_finite(self, multimodel):
        sim, _, _ = multimodel
        source_params = {
            "source_location": [0.0, 0.0, 10.0, 0.0],
            "moment_tensor": [1e14] * 6,
        }
        _, seismograms = sim.run_simulation(dict(source_params))
        assert set(seismograms.keys()) == {"STA_A1", "STA_A2", "STA_B1"}
        assert np.all(seismograms["STA_A1"]["Z"] == 1.0)
        assert np.all(seismograms["STA_B1"]["Z"] == 2.0)
        assert all(np.all(np.isfinite(seismograms[s]["Z"])) for s in seismograms)


# ---------------------------------------------------------------------------
# Per-mode fiducial + independent-per-region member draws
# ---------------------------------------------------------------------------

class TestFiducialAndSeed:

    def test_use_fiducial_routes_each_region_to_its_own_fiducial(self, multimodel):
        sim, _, _ = multimodel
        result = sim.generic_point_source_simulation(_dummy_source(), use_fiducial=True)
        assert np.all(result["STA_A1"]["Z"] == 101.0)   # tag 1 + fiducial 100
        assert np.all(result["STA_B1"]["Z"] == 102.0)   # tag 2 + fiducial 100

    def test_unseeded_call_injects_no_seed(self, multimodel):
        """Production path: seed=None must NOT be forwarded (byte-identical to the
        historical single-class behaviour; sub-sims never see a `seed` key)."""
        sim, sim_a, sim_b = multimodel
        sim.generic_point_source_simulation(_dummy_source())
        assert sim_a.last_seed == "ABSENT"
        assert sim_b.last_seed == "ABSENT"

    def test_seed_is_offset_per_region(self, multimodel):
        """Independent-per-region: region i is given seed+i (decorrelated draws)."""
        sim, sim_a, sim_b = multimodel
        sim.generic_point_source_simulation(_dummy_source(), seed=42)
        assert sim_a.last_seed == 42
        assert sim_b.last_seed == 43

    def test_static_seed_helper(self):
        assert MultiModelSimulator._sub_model_seed(None, 0) is None
        assert MultiModelSimulator._sub_model_seed(None, 3) is None
        assert MultiModelSimulator._sub_model_seed(7, 0) == 7
        assert MultiModelSimulator._sub_model_seed(7, 2) == 9


# ---------------------------------------------------------------------------
# CPS refactor: MultiModelCPSSimulator now shares the generic base
# ---------------------------------------------------------------------------

class TestCPSMultiModelRefactor:

    def _build(self, geometry):
        region_a, region_b, union = geometry
        sim_a = _TaggedSimulator(region_a, tag=1.0)
        sim_b = _TaggedSimulator(region_b, tag=2.0)
        return MultiModelCPSSimulator(
            models=[
                {"receivers": region_a, "simulator": sim_a},
                {"receivers": region_b, "simulator": sim_b},
            ],
            components=["Z"], receivers=union,
            seismogram_duration_in_s=TRACE_LEN, synthetics_processing=_PROC,
            cps_path=None,
        ), sim_a, sim_b

    def test_is_both_cps_and_multimodel(self, geometry):
        sim, _, _ = self._build(geometry)
        assert isinstance(sim, CPSSimulator)        # still a CPSSimulator instance
        assert isinstance(sim, MultiModelSimulator)  # shares the generic merge

    def test_num_traces_retained_from_cps_base(self, geometry):
        sim, _, _ = self._build(geometry)
        # CPSSimulator.__init__ sets num_traces from the union (3 stations x 1 comp)
        assert sim.num_traces == 3

    def test_cps_merge_routes_by_station(self, geometry):
        sim, _, _ = self._build(geometry)
        result = sim.generic_point_source_simulation(_dummy_source())
        assert np.all(result["STA_A1"]["Z"] == 1.0)
        assert np.all(result["STA_B1"]["Z"] == 2.0)

    def test_compute_or_load_greens_functions_not_implemented(self, geometry):
        sim, _, _ = self._build(geometry)
        with pytest.raises(NotImplementedError):
            sim.compute_or_load_greens_functions(None, None)


# ---------------------------------------------------------------------------
# resample_member_per_station forwarding (multi-model -> each region's ensemble)
# ---------------------------------------------------------------------------

class _RecordingEnsemble:
    """Fake InstaseisEnsembleSimulator that records the flag it was built with (no DB)."""

    def __init__(self, **kwargs):
        self.resample_member_per_station = kwargs.get("resample_member_per_station")
        self.receivers = kwargs["receivers"]
        self.num_models = 3
        self.sampling_rate = 1.0

    def generic_point_source_simulation(self, source, **kwargs):
        return {
            rec.station_name: {comp: np.zeros(TRACE_LEN) for comp in rec.components}
            for rec in self.receivers.iterate()
        }


def _ensemble_models():
    a, b = _rec("STA_A1"), _rec("STA_B1")
    return [
        {"receivers": Receivers(receivers=[a]), "ensemble_dir": "/x/a", "fiducial_dir": "/x/a/f"},
        {"receivers": Receivers(receivers=[b]), "ensemble_dir": "/x/b", "fiducial_dir": "/x/b/f"},
    ], Receivers(receivers=[a, b])


class TestResampleMemberPerStationForwarding:

    def _build(self, monkeypatch, **flag):
        import seismo_sbi.instaseis_simulator.multi_model as mm
        monkeypatch.setattr(mm, "InstaseisEnsembleSimulator", _RecordingEnsemble)
        models, union = _ensemble_models()
        sim = InstaseisMultiModelSimulator(
            models=models, components=["Z"], receivers=union,
            seismogram_duration_in_s=TRACE_LEN, synthetics_processing=_PROC, **flag,
        )
        return sim

    def test_flag_forwarded_to_every_region(self, monkeypatch):
        sim = self._build(monkeypatch, resample_member_per_station=True)
        assert sim.resample_member_per_station is True
        assert [s.resample_member_per_station for s in sim.sub_sims] == [True, True]

    def test_default_flag_false_for_every_region(self, monkeypatch):
        sim = self._build(monkeypatch)
        assert sim.resample_member_per_station is False
        assert [s.resample_member_per_station for s in sim.sub_sims] == [False, False]
