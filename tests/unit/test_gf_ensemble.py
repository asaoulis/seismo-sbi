"""Unit tests for GFEnsembleSimulator and EnsembleTheoryCovarianceEstimationSimulator.

Dependency-free: no Instaseis DB or CPS binaries required.
"""

import numpy as np
import pytest
from copy import deepcopy

from seismo_sbi.instaseis_simulator.ensemble import GFEnsembleSimulator
from seismo_sbi.instaseis_simulator.wrapper import GenericPointSource
from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
from seismo_sbi.sbi.compression.theory_covariance import (
    EnsembleTheoryCovarianceEstimationSimulator,
    CPSTheoryCovarianceEstimationSimulator,
)

TRACE_LEN = 20
N_MEMBERS = 5


# ---------------------------------------------------------------------------
# Mock ensemble simulator
# ---------------------------------------------------------------------------

class MockEnsembleSimulator(GFEnsembleSimulator):
    """Concrete GFEnsembleSimulator returning distinct constant traces per member."""

    def __init__(self, receivers, n_members=N_MEMBERS, trace_len=TRACE_LEN):
        super().__init__(
            components=["Z"],
            receivers=receivers,
            seismogram_duration_in_s=trace_len,
            synthetics_processing={
                "sampling_rate": 1.0,
                "filter": {"type": "bandpass", "freqmin": 0.01, "freqmax": 0.1},
            },
        )
        self._trace_len = trace_len
        # Members are labelled 0..n_members-1
        self._members = list(range(n_members))
        self._fiducial = -1  # distinct sentinel

    @property
    def members(self) -> list:
        return self._members

    @property
    def fiducial_member(self):
        return self._fiducial

    def _simulate_with_member(self, member, source, **kwargs) -> dict:
        return {
            rec.station_name: {
                comp: np.full(self._trace_len, float(member))
                for comp in rec.components
            }
            for rec in self.receivers.iterate()
        }

    def generic_point_source_simulation(self, source: GenericPointSource,
                                        *, use_fiducial=False, seed=None,
                                        **kwargs) -> dict:
        member = self.select_member(use_fiducial=use_fiducial, seed=seed)
        return self._simulate_with_member(member, source)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def receivers():
    r = Receiver(0.0, 0.0, "XX", "STA1", ["Z"])
    return Receivers(receivers=[r])


@pytest.fixture
def mock_sim(receivers):
    return MockEnsembleSimulator(receivers)


# ---------------------------------------------------------------------------
# num_models
# ---------------------------------------------------------------------------

class TestNumModels:

    def test_num_models_equals_len_members(self, mock_sim):
        assert mock_sim.num_models == N_MEMBERS

    def test_num_models_matches_members_length(self, mock_sim):
        assert mock_sim.num_models == len(mock_sim.members)


# ---------------------------------------------------------------------------
# select_member equivalence with legacy CPS pattern
# ---------------------------------------------------------------------------

class TestSelectMember:

    def test_use_fiducial_returns_fiducial_member(self, mock_sim):
        result = mock_sim.select_member(use_fiducial=True)
        assert result == mock_sim.fiducial_member

    def test_use_fiducial_ignores_seed(self, mock_sim):
        for seed in [0, 42, 999]:
            assert mock_sim.select_member(use_fiducial=True, seed=seed) == mock_sim.fiducial_member

    def test_fixed_seed_matches_legacy_np_random(self, mock_sim):
        """select_member(seed=s) must produce the same member as the legacy pattern."""
        members = mock_sim.members
        for seed in [0, 7, 42, 100]:
            # Legacy pattern
            np.random.seed(seed)
            legacy_choice = np.random.choice(members)
            # New API
            chosen = mock_sim.select_member(seed=seed)
            assert chosen == legacy_choice, (
                f"seed={seed}: select_member returned {chosen}, "
                f"legacy np.random.choice returned {legacy_choice}"
            )

    def test_repeated_draws_cover_all_members(self, mock_sim):
        """With enough draws and no seeding, all members should appear."""
        seen = set()
        for _ in range(500):
            seen.add(mock_sim.select_member())
        assert seen == set(mock_sim.members)

    def test_no_seed_is_random(self, mock_sim):
        """Two consecutive unseeded draws should not always agree (probabilistic)."""
        draws = {mock_sim.select_member() for _ in range(20)}
        assert len(draws) > 1, "Unseeded draws produced only one unique member"


# ---------------------------------------------------------------------------
# generic_point_source_simulation via member dispatch
# ---------------------------------------------------------------------------

class TestGenericPointSourceSimulation:

    def _dummy_source(self):
        from seismo_sbi.instaseis_simulator.wrapper import (
            GeneralMomentTensor, SourceLocation
        )
        loc = SourceLocation(0.0, 0.0, 10.0, 0.0)
        mt = GeneralMomentTensor([1e14] * 6)
        return GenericPointSource(loc, mt)

    def test_fiducial_simulation_uses_fiducial_member(self, mock_sim):
        source = self._dummy_source()
        result = mock_sim.generic_point_source_simulation(source, use_fiducial=True)
        trace = result["STA1"]["Z"]
        assert np.all(trace == float(mock_sim.fiducial_member))

    def test_seeded_simulation_is_reproducible(self, mock_sim):
        source = self._dummy_source()
        r1 = mock_sim.generic_point_source_simulation(source, seed=42)
        r2 = mock_sim.generic_point_source_simulation(source, seed=42)
        assert np.array_equal(r1["STA1"]["Z"], r2["STA1"]["Z"])

    def test_output_has_correct_keys(self, mock_sim, receivers):
        source = self._dummy_source()
        result = mock_sim.generic_point_source_simulation(source, seed=0)
        assert set(result.keys()) == {"STA1"}
        assert "Z" in result["STA1"]

    def test_output_trace_length(self, mock_sim):
        source = self._dummy_source()
        result = mock_sim.generic_point_source_simulation(source, seed=0)
        assert len(result["STA1"]["Z"]) == TRACE_LEN


# ---------------------------------------------------------------------------
# EnsembleTheoryCovarianceEstimationSimulator
# ---------------------------------------------------------------------------

class TestEnsembleTheoryCovarianceEstimationSimulator:

    def _make_cov_sim(self, receivers, mock_sim):
        def data_flattening(d):
            outputs = d["outputs"]
            arrays = [outputs[rec.station_name][comp]
                      for rec in receivers.iterate()
                      for comp in rec.components]
            return np.concatenate(arrays)

        return EnsembleTheoryCovarianceEstimationSimulator(
            simulator=mock_sim,
            data_flattening=data_flattening,
            components=["Z"],
            receivers=deepcopy(receivers),
            seismogram_duration_in_s=TRACE_LEN,
            synthetics_processing={
                "sampling_rate": 1.0,
                "filter": {"type": "bandpass", "freqmin": 0.01, "freqmax": 0.1},
            },
        )

    def _dummy_source(self):
        from seismo_sbi.instaseis_simulator.wrapper import (
            GeneralMomentTensor, SourceLocation
        )
        loc = SourceLocation(0.0, 0.0, 10.0, 0.0)
        mt = GeneralMomentTensor([1e14] * 6)
        return GenericPointSource(loc, mt)

    def test_output_has_correct_keys(self, receivers, mock_sim):
        cov_sim = self._make_cov_sim(receivers, mock_sim)
        source = self._dummy_source()
        result = cov_sim.generic_point_source_simulation(source)
        assert "STA1" in result
        assert "Z" in result["STA1"]

    def test_output_covariance_shape(self, receivers, mock_sim):
        cov_sim = self._make_cov_sim(receivers, mock_sim)
        source = self._dummy_source()
        result = cov_sim.generic_point_source_simulation(source)
        trace = result["STA1"]["Z"]
        # covariance block flattened: trace_len * trace_len
        assert trace.shape == (TRACE_LEN * TRACE_LEN,)

    def test_alias_is_same_class(self):
        assert CPSTheoryCovarianceEstimationSimulator is EnsembleTheoryCovarianceEstimationSimulator

    def test_num_realisations_equals_num_models(self, receivers, mock_sim):
        cov_sim = self._make_cov_sim(receivers, mock_sim)
        assert cov_sim.num_realisations == mock_sim.num_models


# ---------------------------------------------------------------------------
# CPSPrecomputedSimulator regression — verify select_member is used and
# produces bit-identical results to the legacy np.random pattern.
# No CPS binaries required (update_with_Gtensor is monkeypatched).
# ---------------------------------------------------------------------------

class TestCPSPrecomputedSelectMemberRegression:

    @pytest.fixture
    def cps_gf_dirs(self, tmp_path):
        """Create a fake CPS GF directory tree with 3 sub-folders + a separate fiducial."""
        ensemble_dir = tmp_path / "ensemble"
        ensemble_dir.mkdir()
        for name in ["model_0", "model_1", "model_2"]:
            sub = ensemble_dir / name
            sub.mkdir()
            (sub / "GF.mseed").touch()
        fiducial = tmp_path / "fiducial"
        fiducial.mkdir()
        (fiducial / "GF.mseed").touch()
        return ensemble_dir, fiducial

    def test_cps_precomputed_is_gf_ensemble_simulator(self, cps_gf_dirs, receivers, monkeypatch):
        tmp_path, fiducial = cps_gf_dirs
        # Patch update_with_Gtensor so no CPS binary is needed
        import seismo_sbi.cps_simulator.simulator as cps_mod
        chosen_folders = []
        def fake_update(objstats, velocity_model, **kwargs):
            chosen_folders.append(str(kwargs.get("gf_directory", "")))
            return np.zeros((6, 1))
        monkeypatch.setattr(cps_mod, "update_with_Gtensor", fake_update)

        from seismo_sbi.cps_simulator.simulator import CPSPrecomputedSimulator
        sim = CPSPrecomputedSimulator(
            fiducial_model_path=str(fiducial),
            components=["Z"],
            receivers=receivers,
            seismogram_duration_in_s=TRACE_LEN,
            synthetics_processing={
                "sampling_rate": 1.0,
                "filter": {"type": "bandpass", "freqmin": 0.01, "freqmax": 0.1},
            },
            gf_storage_root=str(tmp_path),
        )
        assert isinstance(sim, GFEnsembleSimulator)
        assert sim.num_models == 3

    def test_cps_select_member_matches_legacy_for_fixed_seed(self, cps_gf_dirs, receivers):
        tmp_path, fiducial = cps_gf_dirs
        from seismo_sbi.cps_simulator.simulator import CPSPrecomputedSimulator

        class FakeCPS(CPSPrecomputedSimulator):
            def compute_or_load_greens_functions(self, *a, **kw):
                pass

        sim = FakeCPS(
            fiducial_model_path=str(fiducial),
            components=["Z"],
            receivers=receivers,
            seismogram_duration_in_s=TRACE_LEN,
            synthetics_processing={
                "sampling_rate": 1.0,
                "filter": {"type": "bandpass", "freqmin": 0.01, "freqmax": 0.1},
            },
            gf_storage_root=str(tmp_path),
        )
        members = sim.members
        for seed in [0, 1, 7, 42]:
            # Legacy pattern
            np.random.seed(seed)
            legacy = np.random.choice(members)
            # New select_member
            chosen = sim.select_member(seed=seed)
            assert str(chosen) == str(legacy), (
                f"seed={seed}: select_member={chosen}, legacy={legacy}"
            )
