"""End-to-end tests for InstaseisEnsembleSimulator.

Tests run against two real local Instaseis databases:
  - ROSA_PREM_10s_disc  (also used by test_pipeline_simulators)
  - prem_i_10s          (different period parameterisation of PREM)

They share the same Earth model family but differ in dt/discretization,
so after filtering and resampling they produce distinct waveforms — giving
a realistic heterogeneous ensemble to exercise the theory-error path.

Skipped if INSTASEIS_DB (primary DB) is not available at the usual location.
The secondary DB is expected at /data/shared/prem_i_10s; the test is skipped
if it is also absent.

Marked @pytest.mark.slow — opens two Instaseis DBs and runs a stencil.
"""

import os
import numpy as np
import pytest
from copy import deepcopy
from pathlib import Path

from seismo_sbi.instaseis_simulator.ensemble import (
    InstaseisEnsembleSimulator,
    GFEnsembleSimulator,
)
from seismo_sbi.instaseis_simulator.simulator import InstaseisSourceSimulator
from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
from seismo_sbi.instaseis_simulator.wrapper import (
    GenericPointSource,
    GeneralMomentTensor,
    SourceLocation,
)
from seismo_sbi.sbi.compression.theory_covariance import (
    EnsembleTheoryCovarianceEstimationSimulator,
)

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# DB availability
# ---------------------------------------------------------------------------

_DB_PRIMARY = next(
    (p for p in [
        os.environ.get("INSTASEIS_DB"),
        "/data/shared/ROSA_PREM_10s_disc",
    ] if p and Path(p).is_dir()),
    None,
)
_DB_SECONDARY = next(
    (p for p in [
        os.environ.get("INSTASEIS_DB_2"),
        "/data/shared/prem_i_10s",
    ] if p and Path(p).is_dir()),
    None,
)

_BOTH_DBS_AVAILABLE = _DB_PRIMARY is not None and _DB_SECONDARY is not None


def _skip_if_no_dbs():
    if not _BOTH_DBS_AVAILABLE:
        missing = []
        if _DB_PRIMARY is None:
            missing.append("primary DB (ROSA_PREM_10s_disc)")
        if _DB_SECONDARY is None:
            missing.append("secondary DB (prem_i_10s)")
        pytest.skip(
            "Instaseis DBs not found: " + ", ".join(missing) +
            ". Set INSTASEIS_DB / INSTASEIS_DB_2 or place DBs under /data/shared/."
        )


# ---------------------------------------------------------------------------
# Shared seismic setup
# ---------------------------------------------------------------------------

_SYNTHETICS_PROCESSING = {
    "filter": {
        "type": "bandpass",
        "freqmin": 0.02,
        "freqmax": 0.05,
        "corners": 4,
        "zerophase": False,
    },
    "sampling_rate": 1.0,
}
_SEISMOGRAM_DURATION = 200.0  # seconds
_COMPONENTS = ["Z"]

_SOURCE_LOC = SourceLocation(
    latitude=37.636,
    longitude=-118.936,
    depth=5.0,
    time_shift=0.0,
)
_MOMENT = GeneralMomentTensor([1e14, 1e14, 1e14, 1e14, 1e14, 1e14])
_SOURCE = GenericPointSource(_SOURCE_LOC, _MOMENT)

_SOURCE_PARAMS = {
    "source_location": [37.636, -118.936, 5.0, 0.0],
    "moment_tensor": [1e14] * 6,
    "velocity_model": None,
}


def _build_receivers():
    return Receivers(
        receivers=[
            Receiver(
                latitude=35.945,
                longitude=-120.541,
                network="BK",
                station_name="PKD",
                components=["Z"],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ensemble_dir(tmp_path_factory):
    """Create a temp dir with two symlinks pointing at the real Instaseis DBs.

    InstaseisEnsembleSimulator.members is discovered by iterdir(), so we need
    physical (or symlinked) subdirectories inside a common parent.
    """
    _skip_if_no_dbs()
    tmp = tmp_path_factory.mktemp("instaseis_ensemble")
    (tmp / "db_0").symlink_to(_DB_PRIMARY, target_is_directory=True)
    (tmp / "db_1").symlink_to(_DB_SECONDARY, target_is_directory=True)
    return tmp


@pytest.fixture(scope="module")
def ensemble_sim(ensemble_dir):
    """InstaseisEnsembleSimulator backed by the two-member ensemble."""
    _skip_if_no_dbs()
    return InstaseisEnsembleSimulator(
        instaseis_ensemble_dir=str(ensemble_dir),
        instaseis_fiducial_loc=_DB_PRIMARY,
        components=_COMPONENTS,
        receivers=_build_receivers(),
        seismogram_duration_in_s=_SEISMOGRAM_DURATION,
        synthetics_processing=_SYNTHETICS_PROCESSING,
    )


@pytest.fixture(scope="module")
def single_db_sim():
    """InstaseisSourceSimulator using the primary DB, for equivalence checks."""
    _skip_if_no_dbs()
    return InstaseisSourceSimulator(
        instaseis_model_loc=_DB_PRIMARY,
        components=_COMPONENTS,
        receivers=_build_receivers(),
        seismogram_duration_in_s=_SEISMOGRAM_DURATION,
        synthetics_processing=_SYNTHETICS_PROCESSING,
    )


# ---------------------------------------------------------------------------
# Basic interface tests
# ---------------------------------------------------------------------------

class TestInstaseisEnsembleInterface:

    def test_is_gf_ensemble_simulator(self, ensemble_sim):
        assert isinstance(ensemble_sim, GFEnsembleSimulator)

    def test_num_models_is_two(self, ensemble_sim):
        assert ensemble_sim.num_models == 2

    def test_members_are_two_paths(self, ensemble_sim):
        assert len(ensemble_sim.members) == 2

    def test_fiducial_member_is_primary_db(self, ensemble_sim):
        assert ensemble_sim.fiducial_member == str(_DB_PRIMARY)

    def test_sampling_rate_is_set(self, ensemble_sim):
        # After resampling the DB should have sampling_rate matching the processing config
        assert ensemble_sim.sampling_rate > 0


# ---------------------------------------------------------------------------
# Simulation correctness tests
# ---------------------------------------------------------------------------

class TestInstaseisEnsembleSimulation:

    def test_fiducial_simulation_returns_seismograms(self, ensemble_sim):
        result = ensemble_sim.generic_point_source_simulation(
            _SOURCE, use_fiducial=True
        )
        assert "PKD" in result
        assert "Z" in result["PKD"]
        assert len(result["PKD"]["Z"]) > 0

    def test_fiducial_simulation_is_reproducible(self, ensemble_sim):
        r1 = ensemble_sim.generic_point_source_simulation(_SOURCE, use_fiducial=True)
        r2 = ensemble_sim.generic_point_source_simulation(_SOURCE, use_fiducial=True)
        assert np.allclose(r1["PKD"]["Z"], r2["PKD"]["Z"]), (
            "Fiducial simulation must be deterministic"
        )

    def test_fiducial_matches_single_db_simulator(self, ensemble_sim, single_db_sim):
        """The fiducial ensemble output must be bit-identical to InstaseisSourceSimulator
        using the same DB, since both call the same InstaseisDBQuerier path."""
        fid_result = ensemble_sim.generic_point_source_simulation(
            _SOURCE, use_fiducial=True
        )
        single_result = single_db_sim.generic_point_source_simulation(
            _SOURCE, stf_duration=None
        )
        assert np.allclose(fid_result["PKD"]["Z"], single_result["PKD"]["Z"]), (
            "Fiducial ensemble output must match InstaseisSourceSimulator with the same DB"
        )

    def test_seeded_draw_is_reproducible(self, ensemble_sim):
        r1 = ensemble_sim.generic_point_source_simulation(_SOURCE, seed=42)
        r2 = ensemble_sim.generic_point_source_simulation(_SOURCE, seed=42)
        assert np.allclose(r1["PKD"]["Z"], r2["PKD"]["Z"]), (
            "Fixed-seed draw must give identical seismograms on repeated calls"
        )

    def test_different_seeds_can_give_different_traces(self, ensemble_sim):
        """With only 2 members, sufficiently many seeds will hit both.

        The two PREM DBs are near-identical at default tolerance but differ at
        strict (double-precision) tolerance due to different dt interpolation.
        We use strict tolerance to distinguish them.
        """
        traces = []
        for seed in range(20):
            r = ensemble_sim.generic_point_source_simulation(_SOURCE, seed=seed)
            traces.append(r["PKD"]["Z"].copy())
        # Use strict tolerance: the two DBs differ at the 1e-10 relative level
        unique_traces = []
        for t in traces:
            if not any(np.allclose(t, u, rtol=1e-10, atol=1e-25) for u in unique_traces):
                unique_traces.append(t)
        assert len(unique_traces) == 2, (
            "Expected exactly 2 distinct trace outputs (one per DB member); "
            f"got {len(unique_traces)}"
        )

    def test_two_members_give_distinct_seismograms(self, ensemble_sim):
        """The two DBs have different dt, so their interpolated traces are not bit-identical.

        Both are PREM-family at 10s period, so the filtered waveforms are very similar
        but not numerically identical at strict (double-precision) tolerance.
        """
        members = ensemble_sim.members
        t0 = ensemble_sim._simulate_with_member(members[0], _SOURCE)
        t1 = ensemble_sim._simulate_with_member(members[1], _SOURCE)
        z0 = t0["PKD"]["Z"]
        z1 = t1["PKD"]["Z"]
        # Not bit-identical at strict tolerance (different dt → different interpolation)
        assert not np.allclose(z0, z1, rtol=1e-10, atol=1e-25), (
            "Two PREM DBs with different dt must produce numerically distinct traces "
            "at strict tolerance, even if physically similar"
        )
        # Highly correlated (both are PREM-based at 10s)
        norm0 = z0 / (np.linalg.norm(z0) + 1e-30)
        norm1 = z1 / (np.linalg.norm(z1) + 1e-30)
        corr = float(np.abs(np.dot(norm0, norm1)))
        assert corr > 0.5, (
            f"Members are uncorrelated (corr={corr:.3f}); expected PREM-family similarity"
        )

    def test_ensemble_draws_cover_both_members(self, ensemble_sim):
        """Over many random draws (no seed), both members should appear."""
        seen = set()
        for _ in range(100):
            member = ensemble_sim.select_member()
            seen.add(member)
        assert len(seen) == 2, (
            f"Only {len(seen)} of 2 members observed in 100 draws"
        )

    def test_run_simulation_roundtrip(self, ensemble_sim):
        """Simulator.run_simulation() with source_parameters dict works end-to-end."""
        source, seismograms = ensemble_sim.run_simulation(
            dict(_SOURCE_PARAMS), use_fiducial=True
        )
        assert "PKD" in seismograms
        assert "Z" in seismograms["PKD"]
        assert np.all(np.isfinite(seismograms["PKD"]["Z"])), "Seismogram contains NaN/Inf"


# ---------------------------------------------------------------------------
# Theory covariance estimation
# ---------------------------------------------------------------------------

class TestInstaseisEnsembleTheoryCovariance:

    @pytest.fixture(scope="class")
    def cov_sim(self, ensemble_sim):
        receivers = _build_receivers()

        def data_flattening(d):
            outputs = d["outputs"]
            arrays = [
                outputs[rec.station_name][comp]
                for rec in receivers.iterate()
                for comp in rec.components
            ]
            return np.concatenate(arrays)

        return EnsembleTheoryCovarianceEstimationSimulator(
            simulator=ensemble_sim,
            data_flattening=data_flattening,
            components=_COMPONENTS,
            receivers=deepcopy(receivers),
            seismogram_duration_in_s=_SEISMOGRAM_DURATION,
            synthetics_processing=_SYNTHETICS_PROCESSING,
            # InstaseisDBQuerier holds open HDF5 handles that can't be pickled,
            # so we must use sequential execution (no joblib workers).
            internal_jobs=1,
        )

    def test_num_realisations_equals_num_models(self, cov_sim, ensemble_sim):
        assert cov_sim.num_realisations == ensemble_sim.num_models

    def _run_cov_sim(self, cov_sim):
        """Run the covariance simulation without use_fiducial (that makes all draws fiducial).

        The ensemble draws are unseeded — with 2 members and 2 draws (with replacement),
        there is a ~50% chance both draws pick the same member (zero covariance).
        We seed numpy before the call to get a deterministic, non-degenerate draw.
        """
        np.random.seed(1)  # empirically gives [member0, member1] for a 2-member ensemble
        return cov_sim.generic_point_source_simulation(_SOURCE)

    def test_covariance_output_has_correct_keys(self, cov_sim):
        result = self._run_cov_sim(cov_sim)
        assert "PKD" in result
        assert "Z" in result["PKD"]

    def test_covariance_block_is_finite(self, cov_sim):
        result = self._run_cov_sim(cov_sim)
        block = result["PKD"]["Z"]
        assert np.all(np.isfinite(block)), "Covariance block contains NaN/Inf"

    def test_covariance_block_is_positive_semidefinite(self, cov_sim):
        """The covariance block must reshape into a PSD matrix."""
        result = self._run_cov_sim(cov_sim)
        block = result["PKD"]["Z"]
        n = int(np.sqrt(len(block)))
        assert n * n == len(block), "Covariance block length is not a perfect square"
        C = block.reshape(n, n)
        assert np.allclose(C, C.T, atol=1e-10), "Covariance matrix is not symmetric"
        # PSD: all eigenvalues >= 0 (allow small numerical negatives)
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals >= -1e-10 * np.abs(eigvals).max()), (
            f"Covariance matrix has significantly negative eigenvalues: {eigvals.min():.3e}"
        )

    def test_covariance_is_nonzero(self, cov_sim):
        """When the two draws pick different members the covariance must be non-zero.

        The two 10s PREM DBs differ at the double-precision level (different dt
        interpolation), so the ensemble covariance is non-zero at that level.
        np.random.seed(1) is chosen to produce a draw that picks member 0 then
        member 1, giving a non-degenerate 2-member estimate.
        """
        result = self._run_cov_sim(cov_sim)
        block = result["PKD"]["Z"]
        assert np.linalg.norm(block) > 0.0, (
            "Covariance block is exactly zero — both ensemble draws picked the same member. "
            "Check that np.random.seed(1) picks two distinct members."
        )
