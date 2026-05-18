"""Shared pytest fixtures for seismo-sbi tests."""

import numpy as np
import pytest
import h5py

from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
from seismo_sbi.sbi.compression.gaussian import ScoreCompressionData, GaussianCompressor
from seismo_sbi.sbi.noises.covariance_estimation import ScalarEmpiricalCovariance

TRACE_LEN = 60   # samples per component per station
N_PARAMS = 3     # number of inference parameters in toy models
RNG_SEED = 42


@pytest.fixture
def single_receiver():
    return Receiver(
        latitude=0.0, longitude=0.0,
        network="XX", station_name="STA1",
        components=["Z"],
    )


@pytest.fixture
def receivers(single_receiver):
    return Receivers(receivers=[single_receiver])


@pytest.fixture
def two_receivers():
    r1 = Receiver(0.0, 0.0, "XX", "STA1", ["Z"])
    r2 = Receiver(1.0, 1.0, "XX", "STA2", ["Z"])
    return Receivers(receivers=[r1, r2])


@pytest.fixture
def scalar_covariance():
    return ScalarEmpiricalCovariance(sigma_noise_level=1.0)


@pytest.fixture
def score_compression_data():
    rng = np.random.default_rng(RNG_SEED)
    theta_fid = rng.standard_normal(N_PARAMS)
    D_fid = rng.standard_normal(TRACE_LEN)
    # Gradients: shape (n_params, trace_len)
    gradients = rng.standard_normal((N_PARAMS, TRACE_LEN))
    return ScoreCompressionData(
        theta_fiducial=theta_fid,
        data_fiducial=D_fid,
        data_parameter_gradients=gradients,
        second_order_gradients=None,
    )


@pytest.fixture
def gaussian_compressor(score_compression_data, scalar_covariance):
    return GaussianCompressor(score_compression_data, scalar_covariance)


@pytest.fixture
def noise_h5_file(tmp_path, receivers):
    """Write a minimal noise catalogue H5 compatible with RealNoiseSampler / SimulationDataLoader."""
    rng = np.random.default_rng(RNG_SEED)
    path = tmp_path / "noise_0001.h5"
    variance = 1.0
    with h5py.File(path, "w") as f:
        grp_out = f.create_group("outputs")
        grp_misc = f.create_group("misc")
        for rec in receivers.iterate():
            sta = grp_out.create_group(rec.station_name)
            sta_misc = grp_misc.create_group(rec.station_name)
            for comp in rec.components:
                trace = rng.standard_normal(TRACE_LEN) * np.sqrt(variance)
                sta.create_dataset(comp, data=trace)
                sta_misc.create_dataset(comp, data=np.array([variance]))
    return path


@pytest.fixture
def noise_catalogue_dir(tmp_path, receivers):
    """Write 5 noise H5 files into a directory (used by RealNoiseSampler)."""
    rng = np.random.default_rng(RNG_SEED)
    variance = 1.0
    for i in range(5):
        path = tmp_path / f"noise_{i:04d}.h5"
        with h5py.File(path, "w") as f:
            grp_out = f.create_group("outputs")
            grp_misc = f.create_group("misc")
            for rec in receivers.iterate():
                sta = grp_out.create_group(rec.station_name)
                sta_misc = grp_misc.create_group(rec.station_name)
                for comp in rec.components:
                    trace = rng.standard_normal(TRACE_LEN) * np.sqrt(variance)
                    sta.create_dataset(comp, data=trace)
                    sta_misc.create_dataset(comp, data=np.array([variance]))
    return tmp_path
