"""Recorded outputs of every covariance class on one fixed synthetic input.

Each test builds one covariance on two stations (Z, E, N; the E/N data keyed 1/2), measures its
loss, per-element loss, C⁻¹r, loss and matmul closures, sampler draws and the ``GaussianCompressor``
Fisher matrix, and compares them with ``tests/fixtures/covariance_characterisation.npz`` to 1e-12.
Run this file as a script to re-record the fixture.
"""
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from seismo_sbi.sbi.compression.gaussian import GaussianCompressor, ScoreCompressionData
from seismo_sbi.sbi.noises.covariance_estimation import (
    BlockDiagonalEmpiricalCovariance,
    BlockDiagonalFilteredCovariance,
    BlockDiagonalKolbCovariance,
    BlockGaussianSampler,
    DiagonalEmpiricalCovariance,
    EmpiricalCovarianceEstimator,
    GaussianNoiseSampler,
    RunningStandardDeviations,
    ScalarEmpiricalCovariance,
    TheoryBlockDiagonalEmpiricalCovariance,
    build_cov_sigma2_dict,
)
from seismo_sbi.simulators.receivers import Receiver, Receivers

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "covariance_characterisation.npz"
BLOCK_SIZE = 40
N_PARAMS = 3
STATIONS = ["STA1", "STA2"]
FILTER_BAND_HZ = {"freqmin": 0.05, "freqmax": 0.2}


def make_receivers():
    return Receivers(receivers=[Receiver(float(k), float(k), "XX", station, ["Z", "E", "N"])
                                for k, station in enumerate(STATIONS)])


def autocovariances(length):
    """Decaying-cosine autocovariance per station and component (keys Z, 1, 2); PSD as Toeplitz."""
    lags = np.arange(length)
    return {station: {component: (1.0 + 0.3 * k + 0.1 * j) * np.exp(-lags / 8.0) * np.cos(lags / 5.0)
                      for j, component in enumerate(["Z", "1", "2"])}
            for k, station in enumerate(STATIONS)}


def variances():
    return {station: {component: np.float64(0.5 + 0.2 * k + 0.1 * j)
                      for j, component in enumerate(["Z", "1", "2"])}
            for k, station in enumerate(STATIONS)}


def synthetic_problem(n_total):
    rng = np.random.default_rng(1234)
    residual = rng.standard_normal(n_total)
    compression_data = ScoreCompressionData(
        theta_fiducial=rng.standard_normal(N_PARAMS),
        data_fiducial=rng.standard_normal(n_total),
        data_parameter_gradients=rng.standard_normal((N_PARAMS, n_total)),
        second_order_gradients=None,
    )
    return residual, compression_data


def measure(covariance, residual, compression_data, per_element=True):
    values = {"loss": covariance.compute_loss(residual),
              "inverse_times_residual": covariance.matmul_inverse_covariance(residual),
              "fisher": GaussianCompressor(compression_data, covariance).Fisher_mat}
    if per_element:
        values["per_element_loss"] = covariance.compute_loss(residual, reduce=False)
    return values


def measure_closures(covariance, residual, inverse_metadata, block_size):
    loss_callable = covariance.create_loss_callable(inverse_metadata, block_size)
    matmul_callable = covariance.create_matmul_inverse_covariance(inverse_metadata, block_size)
    return {"closure_loss": loss_callable(residual), "closure_inverse_times_residual": matmul_callable(residual)}


def draws(sampler, n_draws=3):
    np.random.seed(7)
    return np.stack([sampler()[0] for _ in range(n_draws)])


def measure_scalar():
    covariance = ScalarEmpiricalCovariance(0.7)
    residual, compression_data = synthetic_problem(6 * BLOCK_SIZE)
    values = measure(covariance, residual, compression_data, per_element=False)
    values.update(measure_closures(covariance, residual, covariance.inverse_metadata, 1))
    return values


def measure_diagonal():
    covariance = DiagonalEmpiricalCovariance(autocovariances(BLOCK_SIZE), make_receivers(), BLOCK_SIZE)
    residual, compression_data = synthetic_problem(6 * BLOCK_SIZE)
    values = measure(covariance, residual, compression_data)
    values.update(measure_closures(covariance, residual, covariance.inverse_metadata, BLOCK_SIZE))
    values["covariance_matrix"] = covariance.covariance_matrix
    return values


def measure_block(covariance):
    residual, compression_data = synthetic_problem(6 * BLOCK_SIZE)
    values = measure(covariance, residual, compression_data)
    values.update(measure_closures(covariance, residual, covariance.inverse_metadata, BLOCK_SIZE))
    values["covariance_first_rows"] = covariance.covariance_matrix_arrays[:, 0, :]
    values["covariance_block_norms"] = np.linalg.norm(covariance.covariance_matrix_arrays, axis=(1, 2))
    values["sampler_draws"] = draws(covariance.create_sampler())
    return values


def measure_empirical_block(tapered):
    return measure_block(BlockDiagonalEmpiricalCovariance(
        autocovariances(BLOCK_SIZE if tapered else 2 * BLOCK_SIZE), make_receivers(), BLOCK_SIZE,
        block_exp_tapering=tapered, num_jobs=1))


def measure_filtered(per_station):
    noise_level = variances() if per_station else 0.8
    return measure_block(BlockDiagonalFilteredCovariance(
        noise_level, FILTER_BAND_HZ, make_receivers(), BLOCK_SIZE, num_jobs=1))


def measure_kolb(per_station):
    noise_level = variances() if per_station else 0.8
    return measure_block(BlockDiagonalKolbCovariance(
        noise_level, receivers=make_receivers(), data_vector_length=BLOCK_SIZE, num_jobs=1))


def theory_covariance_blocks():
    rng = np.random.default_rng(99)
    factors = rng.standard_normal((6, BLOCK_SIZE, BLOCK_SIZE)) / BLOCK_SIZE
    blocks = np.einsum("bij,bkj->bik", factors, factors)
    gradients = rng.standard_normal((N_PARAMS, 6 * BLOCK_SIZE * BLOCK_SIZE)) * 1e-3
    return ScoreCompressionData(None, blocks.reshape(-1), gradients, None)


def measure_theory():
    data_covariance = BlockDiagonalKolbCovariance(
        variances(), receivers=make_receivers(), data_vector_length=BLOCK_SIZE, num_jobs=1)
    covariance = TheoryBlockDiagonalEmpiricalCovariance(
        theory_covariance_blocks(), data_covariance.covariance_matrix_arrays, make_receivers(), BLOCK_SIZE,
        diag_regularisation=0.01, num_jobs=1)
    return measure_block(covariance)


def measure_samplers():
    blocks = BlockDiagonalKolbCovariance(
        variances(), receivers=make_receivers(), data_vector_length=BLOCK_SIZE, num_jobs=1).covariance_matrix_arrays
    sampler = GaussianNoiseSampler(make_receivers(), BLOCK_SIZE, cov_blocks=blocks)
    values = {"gaussian_draws": draws(sampler)}
    sampler.set_adaptive_covariance_with_misc_data(
        {station: {component: np.array([2.0 + j]) for j, component in enumerate(["Z", "1", "2"])}
         for station in STATIONS})
    values["adapted_first_rows"] = sampler.cov_blocks[:, 0, :]
    values["adapted_draws"] = draws(sampler)
    values["block_draws"] = draws(BlockGaussianSampler(sampler.Ls, sampler.block_sizes, None))
    return values


def write_noise_windows(directory, n_windows=4):
    rng = np.random.default_rng(5)
    for window in range(n_windows):
        with h5py.File(directory / f"noise_{window}.h5", "w") as noise_file:
            for station in STATIONS:
                for component in ["Z", "1", "2"]:
                    noise_file.create_dataset(f"outputs/{station}/{component}", data=rng.standard_normal(BLOCK_SIZE))


def measure_estimator(directory):
    write_noise_windows(directory)
    estimator = EmpiricalCovarianceEstimator(directory, make_receivers(), "ZEN", covariance_exp_tapering=False)
    covariances = estimator.compute_stationwise_covariances()
    tapered = EmpiricalCovarianceEstimator.taper_covariances(deepcopy(covariances), BLOCK_SIZE)
    fits = EmpiricalCovarianceEstimator.taper_covariances(deepcopy(covariances), BLOCK_SIZE, ols_fit=False,
                                                          return_fit=True)
    running = RunningStandardDeviations()
    running.update(np.arange(12.0).reshape(3, 4))
    running.update(np.arange(8.0).reshape(2, 4) ** 1.5)
    return {"autocovariances": np.stack([covariances[s][c] for s in STATIONS for c in "Z12"]),
            "tapered": np.stack([tapered[s][c] for s in STATIONS for c in "Z12"]),
            "fits": np.array([fits[s][c] for s in STATIONS for c in "Z12"]),
            "sigma2": np.array([build_cov_sigma2_dict(covariances)[s][c] for s in STATIONS for c in "Z12"]),
            "running_mean": running.mean, "running_std": running.std}


MEASUREMENTS = {
    "scalar": measure_scalar,
    "diagonal": measure_diagonal,
    "empirical_block_tapered": lambda: measure_empirical_block(True),
    "empirical_block_untapered": lambda: measure_empirical_block(False),
    "filtered_per_station": lambda: measure_filtered(True),
    "filtered_scalar": lambda: measure_filtered(False),
    "kolb_per_station": lambda: measure_kolb(True),
    "kolb_scalar": lambda: measure_kolb(False),
    "theory_block": measure_theory,
    "samplers": measure_samplers,
}


def assert_matches_recording(name, values):
    with np.load(FIXTURE) as recorded:
        keys = sorted(key for key in recorded.files if key.startswith(name + "/"))
        assert keys == sorted(f"{name}/{key}" for key in values)
        for key in keys:
            np.testing.assert_allclose(values[key.split("/", 1)[1]], recorded[key], rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("name", MEASUREMENTS)
def test_covariance_matches_recorded_values(name):
    assert_matches_recording(name, MEASUREMENTS[name]())


def test_covariance_estimator_matches_recorded_values(tmp_path):
    assert_matches_recording("estimator", measure_estimator(tmp_path))


if __name__ == "__main__":
    import tempfile

    arrays = {}
    for name, measurement in MEASUREMENTS.items():
        arrays.update({f"{name}/{key}": np.asarray(value) for key, value in measurement().items()})
    with tempfile.TemporaryDirectory() as directory:
        arrays.update({f"estimator/{key}": np.asarray(value)
                       for key, value in measure_estimator(Path(directory)).items()})
    np.savez_compressed(FIXTURE, **arrays)
    print(f"recorded {len(arrays)} arrays to {FIXTURE}")
