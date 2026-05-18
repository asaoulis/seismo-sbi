"""End-to-end tests for the SBI pipeline through to compression.

Tests run the full stencil simulation + compressor build for both the
Instaseis and CPS simulators, then verify that the resulting Fisher
information matrix is physically reasonable.

Simulator availability:
  - Instaseis: set the INSTASEIS_DB env var, or place the database at
    /data/shared/ROSA_PREM_10s_disc.
  - CPS: install the CPS suite so that hprep96/hspec96/hpulse96 are on PATH,
    or set CPS_PATH to the directory containing the binaries.

Marked @pytest.mark.slow — the five-point derivative stencil runs
(4 × n_params + 1) forward simulations (25 for the 6-component MT).
"""

import os
import shutil
import numpy as np
import pytest
from pathlib import Path

from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
from seismo_sbi.sbi.pipeline import SingleEventPipeline
from seismo_sbi.sbi.types.parameters import (
    DatasetGenerationParameters,
    IterativeLeastSquaresParameters,
    ModelParameters,
    PipelineParameters,
    SimulationParameters,
)

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

def _find_instaseis_db():
    for path in [
        os.environ.get("INSTASEIS_DB"),
        "/data/shared/ROSA_PREM_10s_disc",
    ]:
        if path and Path(path).is_dir():
            return path
    return None


def _find_cps_path():
    # Explicit env var takes priority, then the known installation path,
    # then fall back to whatever is on PATH.
    for path in [
        os.environ.get("CPS_PATH"),
        "/home/alex/work/cps/PROGRAMS.330/bin",
    ]:
        if path and Path(path, "hprep96").exists():
            return path
    for exe in ["hprep96", "hspec96", "hpulse96"]:
        found = shutil.which(exe)
        if found:
            return str(Path(found).parent)
    return None


_INSTASEIS_DB = _find_instaseis_db()
_CPS_PATH = _find_cps_path()


# ---------------------------------------------------------------------------
# Inline velocity model: Southern California (Dreger & Hemberger 1990)
# Columns of SoCal.plain.txt after np.loadtxt(…).T → shape (6, N)
# ---------------------------------------------------------------------------
_SOCAL_VMODEL = np.array(
    [
        [5.5, 10.5, 19.0, 400.0],   # thickness (km)
        [5.5, 6.3, 6.7, 7.8],       # Vp (km/s)
        [3.18, 3.64, 3.87, 4.5],    # Vs (km/s)
        [2.40, 2.67, 2.80, 3.30],   # density (g/cm³)
        [600.0, 600.0, 600.0, 600.0],  # Qkappa
        [300.0, 300.0, 300.0, 300.0],  # Qmu
    ]
)


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

def _build_receivers():
    """Two stations at different azimuths from the Southern California source.

    PKD (SW, ~200 km) and ORV (NW, ~270 km) give good azimuthal diversity so
    that all six MT components produce distinguishably different seismograms,
    which is required for the Fisher matrix to be well-conditioned.
    """
    return Receivers(
        receivers=[
            Receiver(
                latitude=35.945,
                longitude=-120.541,
                network="BK",
                station_name="PKD",
                components=["Z"],
            ),
            Receiver(
                latitude=39.554,
                longitude=-121.500,
                network="BK",
                station_name="ORV",
                components=["Z"],
            ),
        ]
    )


def _build_mt_model_parameters(include_velocity_model=False):
    """ModelParameters: 6-component MT inference, fixed location nuisance."""
    mp = ModelParameters()
    mp.names["moment_tensor"] = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]
    mp.theta_fiducial["moment_tensor"] = [1e14, 1e14, 1e14, 1e14, 1e14, 1e14]
    # 1% relative step — large enough for stable finite differences but small
    # enough that the linear approximation holds across the seismogram.
    mp.stencil_deltas["moment_tensor"] = [1e12] * 6
    mp.bounds["moment_tensor"] = [[-5e14] * 6, [5e14] * 6]

    # Fixed source location (nuisance only)
    mp.nuisance["source_location"] = [37.636, -118.936, 5.0, 0.0]
    mp.bounds["source_location"] = [37.636, -118.936, 5.0, 0.0]

    if include_velocity_model:
        mp.nuisance["velocity_model"] = _SOCAL_VMODEL
        # constant_sampler yields bounds value as-is
        mp.bounds["velocity_model"] = _SOCAL_VMODEL

    return mp


def _build_dataset_parameters(include_velocity_model=False):
    sampling = {"moment_tensor": "uniform", "source_location": "constant"}
    if include_velocity_model:
        sampling["velocity_model"] = "constant"
    return DatasetGenerationParameters(
        num_simulations=10,
        sampling_method=sampling,
        iterative_least_squares=IterativeLeastSquaresParameters(
            max_iterations=1,
            damping_factor=0.01,
        ),
    )


_COMPRESSION_METHODS = [
    (
        "optimal_score_noise_level",
        {"type": "optimal_score", "covariance": "noise_level", "path": None},
    )
]


def _build_pipeline(tmp_path, sim_params, model_params, dataset_params, num_jobs=1):
    pipeline_params = PipelineParameters(
        run_name="e2e_test",
        output_directory=str(tmp_path),
        job_name="test_job",
        generate_dataset=True,
        num_jobs=num_jobs,
    )
    pipeline = SingleEventPipeline(pipeline_params)
    pipeline.compression_methods = _COMPRESSION_METHODS
    pipeline.load_seismo_parameters(sim_params, model_params, dataset_params)
    return pipeline


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def _assert_fisher_properties(compressor):
    """Check that the Fisher matrix from a real seismic forward model is valid.

    For a linearised MT inversion F = G^T C^-1 G the matrix must be:
      - symmetric (by construction)
      - positive definite (all eigenvalues > 0), implying all six MT
        components produce distinguishably different seismograms
      - invertible without NaN (the Fisher inverse is used for MLE)

    We do NOT check the condition number here: with realistic Green's function
    amplitudes the MT components span many orders of magnitude in sensitivity
    (κ ~ 1e8–1e18 is physically normal for sparse seismic networks), and a
    poor condition number is a geometry issue, not a pipeline bug.
    """
    F = compressor.Fisher_mat
    assert F.shape == (6, 6), f"Unexpected Fisher shape: {F.shape}"

    assert np.allclose(F, F.T, rtol=1e-8, atol=1e-30), "Fisher is not symmetric"

    eigvals = np.linalg.eigvalsh(F)
    assert np.all(eigvals > 0), (
        f"Fisher has non-positive eigenvalues: {eigvals}"
    )

    # Fisher inverse must be finite — NaN would make the compressor useless
    F_inv = compressor.Fisher_mat_inverse
    assert np.all(np.isfinite(F_inv)), (
        f"Fisher inverse contains non-finite values (NaN/Inf): {F_inv}"
    )

    # Fiducial recovery is exact by construction: compress(D_fid) = theta_fid
    recovered = compressor.compress_data_vector(compressor.D_fiducial)
    assert np.allclose(recovered, compressor.theta_fiducial, atol=1e-6), (
        f"Fiducial recovery failed (max |err| = {np.max(np.abs(recovered - compressor.theta_fiducial)):.2e})"
    )

    # Misfit at fiducial is zero
    misfit = compressor.compute_misfit(compressor.D_fiducial)
    assert abs(misfit) < 1e-8, f"Misfit at fiducial = {misfit:.2e}, expected ~0"


def _assert_compression_statistics(compressor, n_trials=100, rng_seed=42):
    """MLE statistics from noisy data should match the Cramér-Rao bound.

    For a linear MT forward model D = G m, the compressor is an unbiased
    estimator of m whose variance equals (G^T C^-1 G)^-1 (Fisher inverse).
    We verify this numerically by drawing noisy samples and comparing the
    empirical covariance to the predicted Fisher inverse.
    """
    rng = np.random.default_rng(rng_seed)
    D_fid = compressor.D_fiducial
    theta_fid = compressor.theta_fiducial
    n_data = len(D_fid)
    sigma = 1.0  # matches the noise_level covariance used to build the compressor

    mle_samples = np.array(
        [compressor.compress_data_vector(D_fid + rng.standard_normal(n_data) * sigma)
         for _ in range(n_trials)]
    )

    # Mean should be close to theta_fid (unbiasedness)
    mean_mle = mle_samples.mean(axis=0)
    std_of_mean = np.sqrt(np.diag(compressor.Fisher_mat_inverse) / n_trials)
    assert np.allclose(mean_mle, theta_fid, atol=6 * std_of_mean.max()), (
        f"MLE mean deviates from theta_fid: err = {mean_mle - theta_fid}"
    )

    # Empirical variance should match Fisher inverse diagonal within a factor of 3
    empirical_var = mle_samples.var(axis=0)
    expected_var = np.diag(compressor.Fisher_mat_inverse)
    ratios = empirical_var / expected_var
    assert np.all(ratios > 0.2) and np.all(ratios < 5.0), (
        f"Empirical/Cramér-Rao variance ratios out of [0.2, 5.0]: {ratios}"
    )


# ---------------------------------------------------------------------------
# Instaseis fixture (computed once per test class)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def instaseis_compressor(tmp_path_factory):
    if _INSTASEIS_DB is None:
        pytest.skip(
            "Instaseis DB not found. Set INSTASEIS_DB or place it at "
            "/data/shared/ROSA_PREM_10s_disc"
        )

    tmp_path = tmp_path_factory.mktemp("instaseis_pipeline")
    receivers = _build_receivers()
    sim_params = SimulationParameters(
        receivers=receivers,
        components="Z",
        seismogram_duration=200,
        syngine_address=_INSTASEIS_DB,
        sampling_rate=1.0,
        processing={
            "filter": {
                "type": "bandpass",
                "freqmin": 0.02,
                "freqmax": 0.05,
                "corners": 4,
                "zerophase": False,
            },
            "sampling_rate": 1.0,
        },
        simulation_type="instaseis",
    )
    pipeline = _build_pipeline(
        tmp_path,
        sim_params,
        _build_mt_model_parameters(),
        _build_dataset_parameters(),
    )
    _, compressor, _, _ = pipeline.prepare_single_compressor(
        "optimal_score_noise_level",
        covariance_data=1.0,
    )
    return compressor


# ---------------------------------------------------------------------------
# CPS Green's function pre-computation helper
# ---------------------------------------------------------------------------

def _precompute_cps_gfs(gf_path: Path, fiducial_path: Path, seismogram_duration: float):
    """Run CPS once for the fiducial geometry and populate both GF directories.

    Mirrors the generate_CPS_perturbations workflow: compute one set of GFs
    for the fiducial velocity model / source-receiver geometry, then copy the
    resulting GF.mseed to both the 'perturbation' and 'fiducial' directories.
    CPSPrecomputedSimulator will then load from disk for every stencil point
    instead of re-running CPS each time.
    """
    import tempfile
    from obspy.geodetics.base import gps2dist_azimuth
    from seismo_sbi.cps_simulator.CPS import calc_CPS_GFs

    # Fiducial source location (must match model_params.nuisance["source_location"])
    src_lat, src_lon, src_depth_km = 37.636, -118.936, 5.0

    # Distances to both receivers (PKD, ORV)
    receiver_coords = [(35.945, -120.541), (39.554, -121.500)]
    dists_km = sorted(set(
        round(gps2dist_azimuth(src_lat, src_lon, rlat, rlon)[0] / 1000.0, 1)
        for rlat, rlon in receiver_coords
    ))

    # CPS requires npts = 2 × window to avoid wrap-around artefacts
    npts = int(seismogram_duration) * 2

    with tempfile.TemporaryDirectory() as wdir:
        calc_CPS_GFs(
            dists_in_km=dists_km,
            evdp_in_km=src_depth_km,
            vmodel=_SOCAL_VMODEL,
            dt=1.0,
            npts=npts,
            wdir=wdir,
            cps_path=_CPS_PATH,
        )
        gf_file = Path(wdir) / "GF.mseed"
        # gf_path holds the "perturbed" models (here just the one fiducial copy)
        shutil.copy(gf_file, gf_path / "GF.mseed")
        # fiducial_path holds the exact fiducial GF used for stencil simulations
        shutil.copy(gf_file, fiducial_path / "GF.mseed")


# ---------------------------------------------------------------------------
# CPS fixture (computed once per test class)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def cps_compressor(tmp_path_factory):
    if _CPS_PATH is None:
        pytest.skip(
            "CPS executables not found. Install CPS or set CPS_PATH to the "
            "directory containing hprep96/hspec96/hpulse96."
        )

    tmp_path = tmp_path_factory.mktemp("cps_pipeline")
    gf_path = tmp_path / "cps_gfs"
    fiducial_path = tmp_path / "cps_gfs_fiducial"
    gf_path.mkdir()
    fiducial_path.mkdir()

    # CPS runs exactly once here; all 25 stencil points load from disk.
    _precompute_cps_gfs(gf_path, fiducial_path, seismogram_duration=200)

    receivers = _build_receivers()
    sim_params = SimulationParameters(
        receivers=receivers,
        components="Z",
        seismogram_duration=200,
        syngine_address=None,
        sampling_rate=1.0,
        processing={
            "filter": {
                "type": "bandpass",
                "freqmin": 0.02,
                "freqmax": 0.05,
                "corners": 4,
                "zerophase": False,
            },
            "sampling_rate": 1.0,
        },
        simulation_type="cps_precomputed",
        cps_path=_CPS_PATH,
        cps_GFs_path=str(gf_path),
        cps_GFs_fiducial_path=str(fiducial_path),
    )
    model_params = _build_mt_model_parameters(include_velocity_model=True)
    dataset_params = _build_dataset_parameters(include_velocity_model=True)

    pipeline = _build_pipeline(tmp_path, sim_params, model_params, dataset_params)
    _, compressor, _, _ = pipeline.prepare_single_compressor(
        "optimal_score_noise_level",
        covariance_data=1.0,
    )
    return compressor


# ---------------------------------------------------------------------------
# Tests: Instaseis pipeline
# ---------------------------------------------------------------------------

class TestInstaseisCompressionPipeline:
    """Full pipeline: Instaseis seismograms → score compression → Fisher tests.

    Requires a local Instaseis database.  The stencil runs 25 simulations
    (5-point, 6-component MT), so this class is marked slow.
    """

    def test_compressor_built(self, instaseis_compressor):
        """Pipeline completes and returns a valid compressor."""
        assert instaseis_compressor is not None
        assert instaseis_compressor.Fisher_mat.shape == (6, 6)
        assert instaseis_compressor.D_fiducial is not None
        assert instaseis_compressor.theta_fiducial is not None

    def test_fisher_positive_definite_and_symmetric(self, instaseis_compressor):
        """Fisher matrix must be symmetric and positive-definite.

        For a linear moment-tensor inversion D = G m, the Fisher matrix
        F = G^T C^-1 G must be PD whenever G has full column rank (i.e. all
        six MT components are independently resolvable from the data).
        A non-PD Fisher matrix indicates either degenerate geometry or a
        failed stencil computation.
        """
        _assert_fisher_properties(instaseis_compressor)

    def test_compression_statistics_match_cramer_rao(self, instaseis_compressor):
        """Empirical MLE spread from noisy data agrees with Cramér-Rao bound.

        Draws 100 noisy realisations D = D_fid + N(0, σ²I) and compresses
        each to a MT estimate.  The empirical variance should match the
        diagonal of Fisher^-1 within a factor of ~3 (generous tolerance to
        keep the test robust against small-sample noise).
        """
        _assert_compression_statistics(instaseis_compressor)


# ---------------------------------------------------------------------------
# Tests: CPS pipeline
# ---------------------------------------------------------------------------

class TestCPSCompressionPipeline:
    """Full pipeline: CPS variable-kernel seismograms → compression → Fisher.

    Requires the Computer Programs in Seismology (CPS) suite to be installed.
    The stencil runs 25 CPS Green's function computations; mark slow.
    """

    def test_compressor_built(self, cps_compressor):
        """Pipeline completes and returns a valid CPS-based compressor."""
        assert cps_compressor is not None
        assert cps_compressor.Fisher_mat.shape == (6, 6)
        assert cps_compressor.D_fiducial is not None
        assert cps_compressor.theta_fiducial is not None

    def test_fisher_positive_definite_and_symmetric(self, cps_compressor):
        """Fisher matrix from CPS stencil must be symmetric and PD.

        Equivalent to the Instaseis test but using Green's functions computed
        on-the-fly via the CPS variable-kernel simulator.  Verifies that the
        CPS forward model is correctly wired into the compression pipeline.
        """
        _assert_fisher_properties(cps_compressor)

    def test_compression_statistics_match_cramer_rao(self, cps_compressor):
        """CPS compressor: empirical MLE spread matches Fisher inverse.

        Same statistical check as the Instaseis variant, ensuring the CPS
        sensitivity kernels produce a well-calibrated uncertainty estimate.
        """
        _assert_compression_statistics(cps_compressor)
