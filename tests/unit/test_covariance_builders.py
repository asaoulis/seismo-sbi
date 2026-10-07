"""Covariances built by name from a compression option match the classes built directly."""
import numpy as np
import pytest

from seismo_sbi.sbi.noises.covariances import CovarianceLayout, build_covariance_matrix, build_theory_covariance
from tests.unit import test_covariance_characterisation as cc

LAYOUT = CovarianceLayout(cc.make_receivers(), cc.BLOCK_SIZE, 6 * cc.BLOCK_SIZE, cc.FILTER_BAND_HZ, num_jobs=1)
DIRECT = {
    "empirical_block": lambda: cc.BlockDiagonalEmpiricalCovariance(cc.autocovariances(cc.BLOCK_SIZE), cc.make_receivers(),
                                                                   cc.BLOCK_SIZE, num_jobs=1),
    "filtered_block": lambda: cc.BlockDiagonalFilteredCovariance(cc.variances(), cc.FILTER_BAND_HZ, cc.make_receivers(),
                                                                 cc.BLOCK_SIZE, num_jobs=1),
    "kolb": lambda: cc.BlockDiagonalKolbCovariance(cc.variances(), receivers=cc.make_receivers(),
                                                   data_vector_length=cc.BLOCK_SIZE, num_jobs=1),
    "empirical_diagonal": lambda: cc.DiagonalEmpiricalCovariance(cc.autocovariances(cc.BLOCK_SIZE), cc.make_receivers(),
                                                                 cc.BLOCK_SIZE),
    "noise_level": lambda: cc.ScalarEmpiricalCovariance(0.7, data_vector_length=6 * cc.BLOCK_SIZE),
}
DATA = {"empirical_block": cc.autocovariances(cc.BLOCK_SIZE), "empirical_diagonal": cc.autocovariances(cc.BLOCK_SIZE),
        "filtered_block": cc.variances(), "kolb": cc.variances(), "noise_level": 0.7}


@pytest.mark.parametrize("option", sorted(DIRECT))
def test_a_covariance_built_by_name_is_the_class_built_directly(option):
    residual = np.random.default_rng(1).standard_normal(6 * cc.BLOCK_SIZE)
    by_name, direct = build_covariance_matrix(option, DATA[option], LAYOUT), DIRECT[option]()
    assert type(by_name) is type(direct)
    assert np.array_equal(by_name.matmul_inverse_covariance(residual), direct.matmul_inverse_covariance(residual))


def test_an_unknown_covariance_name_is_rejected():
    with pytest.raises(NotImplementedError, match="bogus"):
        build_covariance_matrix("bogus", 1.0, LAYOUT)


def test_the_theory_covariance_adds_the_data_covariance_blocks():
    kolb = DIRECT["kolb"]()
    theory = build_theory_covariance(cc.theory_covariance_blocks(), kolb, 0.0, LAYOUT)
    assert theory.data_covariance is kolb
    assert all(np.array_equal(data, kolb_block)
               for data, kolb_block in zip(theory.data_covariance_arrays, kolb.covariance_matrix_arrays))


def test_a_theory_compressor_without_a_data_covariance_is_rejected():
    from seismo_sbi.sbi.compression.compressor_options import TheoryOptimalScoreOptions
    from seismo_sbi.sbi.compression.compressors import build_compressor
    from seismo_sbi.utils.errors import InvalidConfiguration

    with pytest.raises(InvalidConfiguration, match="needs data_covariance"):
        build_compressor(TheoryOptimalScoreOptions(noise_level=1e-8), None, None, LAYOUT,
                         extra_gradients=cc.theory_covariance_blocks())


def test_a_theory_compressor_without_a_noise_level_or_an_event_is_rejected():
    from seismo_sbi.sbi.compression.compressor_options import TheoryOptimalScoreOptions
    from seismo_sbi.sbi.compression.compressors import build_compressor
    from seismo_sbi.utils.errors import InvalidConfiguration

    with pytest.raises(InvalidConfiguration, match="noise_level: null"):
        build_compressor(TheoryOptimalScoreOptions(data_covariance="filtered_block"), None, None, LAYOUT,
                         extra_gradients=cc.theory_covariance_blocks())


@pytest.mark.parametrize("noise_level, expected", [(None, "event1.h5"), (1e-8, None)])
def test_compressors_get_the_first_event_noise_only_when_their_noise_level_comes_from_it(noise_level, expected):
    from types import SimpleNamespace
    from seismo_sbi.sbi.compression.compressor_options import TheoryOptimalScoreOptions
    from seismo_sbi.sbi.datasets.training_data import event_noise_for_compressors

    config = SimpleNamespace(
        compression_methods=[("theory_optimal_score", TheoryOptimalScoreOptions("filtered_block", noise_level))],
        real_event_jobs={"first": "event1.h5", "second": {"path": "event2.h5"}})
    pipeline = SimpleNamespace(data_manager=SimpleNamespace(data_loader=SimpleNamespace(load_misc_data=lambda path: path)))
    assert event_noise_for_compressors(pipeline, config) == expected
