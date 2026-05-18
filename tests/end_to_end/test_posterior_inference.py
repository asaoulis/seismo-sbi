"""End-to-end tests for the full SBI inference pipeline.

Runs the complete pipeline (stencil → compression → inference) for both:

  1. NPE/SBI  – simulate 400 training examples with the linearised kernel
                simulator, train a masked autoregressive flow via
                pipeline.run_single_sbi_inversion, sample the posterior.
  2. Gaussian likelihood MCMC – use pipeline.run_single_gaussian_likelihood_inversion
                (ensemble=False: 20 independent GaussianMove chains via
                joblib/loky, which uses cloudpickle and avoids Pool OOM).

For a linear moment-tensor inversion with i.i.d. Gaussian noise N(0, σ²I)
the analytical posterior is Gaussian:

    mean        = θ_MLE = compress(D_obs)    (optimal score compressor)
    covariance  = F⁻¹                        (Fisher information inverse)

The noise level σ is chosen adaptively (10 : 1 SNR on the fiducial seismogram)
so the posterior is well-conditioned for MCMC and NPE without artificially
constraining the prior bounds.  The original bounds (±5×10¹⁴ Nm) are used
throughout.

The MCMC chains are initialised at the analytical MLE (θ_MLE) with a
GaussianMove step scale derived from the Fisher diagonal.  Starting at the
posterior mode with an appropriate step size gives fast mixing even without
constraining the prior.

Marked @pytest.mark.slow.
"""

import numpy as np
import pytest
from copy import deepcopy

from tests.end_to_end.test_pipeline_simulators import (
    _INSTASEIS_DB,
    _CPS_PATH,
    _COMPRESSION_METHODS,
    _build_receivers,
    _build_mt_model_parameters,
    _build_dataset_parameters,
    _build_pipeline,
    _precompute_cps_gfs,
)

from seismo_sbi.instaseis_simulator.utils import compute_data_vector_length

from seismo_sbi.sbi.compression.gaussian import ScoreCompressionData
from seismo_sbi.sbi.types.parameters import (
    DatasetGenerationParameters,
    SimulationParameters,
)
from seismo_sbi.sbi.types.results import JobData

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SNR_TARGET = 10.0    # σ chosen so RMS(D_fid) / σ = SNR_TARGET
N_TRAIN = 400         # NPE training simulations
N_MCMC_SAMPLES = 8000 # total samples (= N_WALKERS × steps per walker)
N_MCMC_BURN_IN = 500  # burn-in per chain; starting at MLE needs very few
N_WALKERS = 20        # independent chains == num_jobs
RNG_SEED = 42

_PROCESSING = {
    "filter": {
        "type": "bandpass",
        "freqmin": 0.02,
        "freqmax": 0.05,
        "corners": 4,
        "zerophase": False,
    },
    "sampling_rate": 1.0,
}


# ---------------------------------------------------------------------------
# Adaptive noise level
# ---------------------------------------------------------------------------

def _get_sigma(D_fid):
    """σ = RMS(D_fid) / SNR_TARGET gives a 10:1 SNR.

    Works correctly for both Instaseis (μm-scale displacements) and CPS
    (different amplitude normalisation), so the test does not need to know
    the simulator's internal units.
    """
    rms = np.std(D_fid)
    return rms / SNR_TARGET if rms > 1e-30 else 1.0


# ---------------------------------------------------------------------------
# Analytical posterior
# ---------------------------------------------------------------------------

def _analytical_posterior(compressor, D_obs):
    """Return (θ_MLE, F⁻¹) — the exact Gaussian posterior under the kernel model."""
    return compressor.compress_data_vector(D_obs), compressor.Fisher_mat_inverse


def _make_test_observation(compressor, sigma, rng_seed=RNG_SEED):
    n = len(compressor.D_fiducial)
    rng = np.random.default_rng(rng_seed)
    return compressor.D_fiducial + rng.standard_normal(n) * sigma


# ---------------------------------------------------------------------------
# Step size for the Gaussian likelihood MCMC
# ---------------------------------------------------------------------------

def _mcmc_move_size(compressor, parameters):
    """Anisotropic RWMH proposal covariance in [0,1]-scaled parameter space.

    Returns the full covariance matrix for emcee.moves.GaussianMove:

        cov = (2.38² / d) × S F⁻¹ S

    where F⁻¹ is the Fisher inverse (posterior covariance) and S = diag(1/range)
    maps from physical to [0,1] space.  This is the optimal RWMH scaling —
    acceptance ≈ 0.234 regardless of the posterior anisotropy.
    """
    F_inv = compressor.Fisher_mat_inverse
    bounds = parameters.bounds["moment_tensor"]
    mt_range = np.abs(np.array(bounds[1]) - np.array(bounds[0]))
    S_inv = np.diag(1.0 / mt_range)
    F_inv_scaled = S_inv @ F_inv @ S_inv
    d = F_inv.shape[0]
    return (2.38 ** 2 / d) * F_inv_scaled


# ---------------------------------------------------------------------------
# Gaussian likelihood via pipeline
# ---------------------------------------------------------------------------

def _run_gaussian_likelihood_inversion(
    pipeline, compressor, compression_data, D_obs, sigma, dataset_params
):
    """Call pipeline.run_single_gaussian_likelihood_inversion.

    Uses ensemble=False so inference runs as N_WALKERS independent
    Metropolis-Hastings chains via joblib/loky (cloudpickle serialisation —
    no Pool OOM, no pickle issues with CPS internals).

    Chains start at θ_MLE; GaussianMove step size is derived from F⁻¹ so
    they explore the posterior efficiently without constraining the prior.
    """
    pipeline = deepcopy(pipeline)

    theta_mle = compressor.compress_data_vector(D_obs)
    inference_cd = compression_data._replace(theta_fiducial=theta_mle)

    # Switch to fast linearised kernel simulator for likelihood evaluations
    pipeline.use_kernel_simulator_if_possible(
        inference_cd, dataset_params.sampling_method
    )

    move_size = _mcmc_move_size(compressor, pipeline.parameters)

    job = JobData(
        job_name="synthetic_test",
        noise_type="gaussian",
        data_vector=D_obs,
        theta0=None,
    )

    likelihood_config = {
        "run": True,
        "ensemble": False,       # joblib/loky: no Pool, cloudpickle-safe
        "covariance": "empirical",  # → ScalarEmpiricalCovariance(σ)
        "walker_burn_in": N_MCMC_BURN_IN,
        "num_samples": N_MCMC_SAMPLES,
        "move_size": move_size,
    }

    results = list(
        pipeline.run_single_gaussian_likelihood_inversion(
            job, likelihood_config,
            "optimal_score_noise_level",
            deepcopy(pipeline.parameters),
            priors=(None, None),
            mle_start=theta_mle,
        )
    )
    _, inversion_result = results[0]
    return inversion_result.inversion_data.samples


# ---------------------------------------------------------------------------
# SBI / NPE via pipeline
# ---------------------------------------------------------------------------

def _run_sbi(pipeline, compressor, compression_data, D_obs, sigma, dataset_params):
    """Call pipeline.run_single_sbi_inversion.

    The kernel simulator is activated automatically (source_location constant).
    Fisher-constrained bounds focus training data near the true solution for
    the NPE.  Training noise uses the same σ as the compressor covariance.
    """
    pipeline = deepcopy(pipeline)

    theta_mle = compressor.compress_data_vector(D_obs)
    inference_cd = compression_data._replace(theta_fiducial=theta_mle)

    train_params = DatasetGenerationParameters(
        num_simulations=N_TRAIN,
        sampling_method=deepcopy(dataset_params.sampling_method),
        iterative_least_squares=dataset_params.iterative_least_squares,
        use_fisher_to_constrain_bounds=5,
    )

    inversion_data, _, _ = pipeline.run_single_sbi_inversion(
        sbi_method="posterior",
        dataset_details=train_params,
        theta0=None,
        compression_data=inference_cd,
        priors=(None, None),
        compressor_name="optimal_score_noise_level",
    )
    return inversion_data.samples


# ---------------------------------------------------------------------------
# Assertion
# ---------------------------------------------------------------------------

def _assert_posterior_matches_analytic(
    samples, theta_mle, F_inv,
    mean_tol_sigmas=6,
    abs_mean_tol=1e14,
    var_ratio_lo=0.15, var_ratio_hi=6.0,
    label="",
):
    """Check posterior mean and marginal variances against the exact Gaussian.

    mean : max|emp_mean − θ_MLE| < max(mean_tol_sigmas × SE(mean), abs_mean_tol)
    var  : each ratio emp_var_i / F⁻¹_ii ∈ [var_ratio_lo, var_ratio_hi]
    """
    n = samples.shape[0]
    emp_mean = samples.mean(axis=0)
    emp_var = samples.var(axis=0)
    analytic_var = np.diag(F_inv)

    std_of_mean = np.sqrt(analytic_var / n)
    max_tol = max(mean_tol_sigmas * std_of_mean.max(), abs_mean_tol)
    max_err = np.max(np.abs(emp_mean - theta_mle))
    assert max_err < max_tol, (
        f"{label}: posterior mean deviates from analytical MLE\n"
        f"  max|err|  = {max_err:.3e}  (tol = {max_tol:.3e})\n"
        f"  emp_mean  = {emp_mean}\n  theta_mle = {theta_mle}"
    )

    ratios = emp_var / analytic_var
    assert np.all(ratios > var_ratio_lo) and np.all(ratios < var_ratio_hi), (
        f"{label}: marginal variance ratios out of [{var_ratio_lo}, {var_ratio_hi}]\n"
        f"  ratios       = {ratios}\n"
        f"  emp_var      = {emp_var}\n"
        f"  analytic_var = {analytic_var}"
    )


# ---------------------------------------------------------------------------
# Shared fixture builder
# ---------------------------------------------------------------------------

def _build_inference_setup(tmp, sim_params, model_params, dataset_params, data_vector_change=None):
    """Run the stencil once, then rebuild the compressor with the adaptive σ.

    Two-pass: the stencil (sensitivity kernels + fiducial seismogram) is
    computed with placeholder σ=1, then the compressor is rebuilt using the
    correct noise level without re-running the expensive stencil.
    """
    pipeline = _build_pipeline(
        tmp, sim_params, model_params, dataset_params, num_jobs=N_WALKERS
    )

    # Mirror compute_data_vector_properties without needing real test jobs.
    data_length = compute_data_vector_length(
        sim_params.seismogram_duration, sim_params.sampling_rate
    ) + 1
    if data_vector_change is not None:
        data_length += data_vector_change
    num_traces = sum(len(r.components) for r in sim_params.receivers.iterate())
    pipeline.data_vector_length = data_length * num_traces
    pipeline.trace_length = data_length

    # Pass 1: compute stencil (sensitivity kernels and fiducial seismogram)
    _, _, cd_raw, _ = pipeline.prepare_single_compressor(
        "optimal_score_noise_level", covariance_data=1.0,
    )

    # Adaptive σ: 10:1 SNR on the fiducial seismogram
    sigma = _get_sigma(cd_raw.data_fiducial)

    # Pass 2: rebuild compressor with correct σ — no stencil re-run
    _, compressor, compression_data, _ = pipeline.prepare_single_compressor(
        "optimal_score_noise_level",
        covariance_data=sigma,
        compression_data_extras=(cd_raw, None),
    )

    pipeline.training_noise_sampler = lambda: np.random.normal(
        0, sigma, pipeline.data_vector_length
    )

    D_obs = _make_test_observation(compressor, sigma)
    return dict(
        pipeline=pipeline,
        compressor=compressor,
        compression_data=compression_data,
        dataset_params=dataset_params,
        sigma=sigma,
        D_obs=D_obs,
    )


# ---------------------------------------------------------------------------
# Instaseis fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def instaseis_inference(tmp_path_factory):
    if _INSTASEIS_DB is None:
        pytest.skip(
            "Instaseis DB not found. Set INSTASEIS_DB or place it at "
            "/data/shared/ROSA_PREM_10s_disc"
        )
    tmp = tmp_path_factory.mktemp("instaseis_inference")
    sim_params = SimulationParameters(
        receivers=_build_receivers(),
        components="Z",
        seismogram_duration=200,
        syngine_address=_INSTASEIS_DB,
        sampling_rate=1.0,
        processing=_PROCESSING,
        simulation_type="instaseis",
    )
    return _build_inference_setup(
        tmp, sim_params,
        _build_mt_model_parameters(),
        _build_dataset_parameters(),
    )


# ---------------------------------------------------------------------------
# CPS fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def cps_inference(tmp_path_factory):
    if _CPS_PATH is None:
        pytest.skip(
            "CPS executables not found. Install CPS or set CPS_PATH to the "
            "directory containing hprep96/hspec96/hpulse96."
        )
    tmp = tmp_path_factory.mktemp("cps_inference")
    gf_path = tmp / "cps_gfs"
    fiducial_path = tmp / "cps_gfs_fiducial"
    gf_path.mkdir()
    fiducial_path.mkdir()
    _precompute_cps_gfs(gf_path, fiducial_path, seismogram_duration=200)

    sim_params = SimulationParameters(
        receivers=_build_receivers(),
        components="Z",
        seismogram_duration=200,
        syngine_address=None,
        sampling_rate=1.0,
        processing=_PROCESSING,
        simulation_type="cps_precomputed",
        cps_path=_CPS_PATH,
        cps_GFs_path=str(gf_path),
        cps_GFs_fiducial_path=str(fiducial_path),
    )
    return _build_inference_setup(
        tmp, sim_params,
        _build_mt_model_parameters(include_velocity_model=True),
        _build_dataset_parameters(include_velocity_model=True),
        data_vector_change=-1,
    )


# ---------------------------------------------------------------------------
# Tests: Instaseis
# ---------------------------------------------------------------------------

class TestInstaseisInference:
    """Full inference pipeline tests using the Instaseis simulator."""

    def test_gaussian_likelihood_mcmc_matches_analytical(self, instaseis_inference):
        """Gaussian likelihood MCMC recovers the exact analytical posterior.

        pipeline.run_single_gaussian_likelihood_inversion runs 20 independent
        GaussianMove chains (ensemble=False) on the kernel log-likelihood.
        Chains start at θ_MLE; step size is derived from F⁻¹.
        The posterior is analytically Gaussian: mean = MLE, cov = F⁻¹.
        """
        s = instaseis_inference
        samples = _run_gaussian_likelihood_inversion(
            s["pipeline"], s["compressor"], s["compression_data"],
            s["D_obs"], s["sigma"], s["dataset_params"],
        )
        theta_mle, F_inv = _analytical_posterior(s["compressor"], s["D_obs"])
        _assert_posterior_matches_analytic(
            samples, theta_mle, F_inv,
            mean_tol_sigmas=5, var_ratio_lo=0.3, var_ratio_hi=3.0,
            label="Instaseis MCMC",
        )

    def test_sbi_posterior_matches_analytical(self, instaseis_inference):
        """NPE trained on 400 kernel simulations reproduces the analytical Gaussian.

        Tolerances are looser than for MCMC: SBI is an approximation and
        400 training simulations are not enough for sub-percent accuracy.
        """
        s = instaseis_inference
        samples = _run_sbi(
            s["pipeline"], s["compressor"], s["compression_data"],
            s["D_obs"], s["sigma"], s["dataset_params"],
        )
        theta_mle, F_inv = _analytical_posterior(s["compressor"], s["D_obs"])
        _assert_posterior_matches_analytic(
            samples, theta_mle, F_inv,
            mean_tol_sigmas=8, var_ratio_lo=0.15, var_ratio_hi=6.0,
            label="Instaseis SBI",
        )


# ---------------------------------------------------------------------------
# Tests: CPS
# ---------------------------------------------------------------------------

class TestCPSInference:
    """Full inference pipeline tests using the CPS precomputed simulator."""

    def test_gaussian_likelihood_mcmc_matches_analytical(self, cps_inference):
        """Gaussian likelihood MCMC recovers the analytical posterior (CPS kernels).

        Same structure as the Instaseis test; sensitivity kernels come from
        the CPS finite-difference stencil.
        """
        s = cps_inference
        samples = _run_gaussian_likelihood_inversion(
            s["pipeline"], s["compressor"], s["compression_data"],
            s["D_obs"], s["sigma"], s["dataset_params"],
        )
        theta_mle, F_inv = _analytical_posterior(s["compressor"], s["D_obs"])
        _assert_posterior_matches_analytic(
            samples, theta_mle, F_inv,
            mean_tol_sigmas=5, var_ratio_lo=0.3, var_ratio_hi=3.0,
            label="CPS MCMC",
        )

    def test_sbi_posterior_matches_analytical(self, cps_inference):
        """NPE trained on CPS kernel simulations reproduces the analytical Gaussian."""
        s = cps_inference
        samples = _run_sbi(
            s["pipeline"], s["compressor"], s["compression_data"],
            s["D_obs"], s["sigma"], s["dataset_params"],
        )
        theta_mle, F_inv = _analytical_posterior(s["compressor"], s["D_obs"])
        _assert_posterior_matches_analytic(
            samples, theta_mle, F_inv,
            mean_tol_sigmas=8, var_ratio_lo=0.15, var_ratio_hi=6.0,
            label="CPS SBI",
        )
