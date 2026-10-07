"""The noise covariance a compressor and the Gaussian likelihood use, built by name.

:func:`build_covariance_matrix` builds the covariance named by a ``compression`` option from its
covariance data on the data-vector layout of a :class:`CovarianceLayout`;
:func:`build_theory_covariance` adds the theory covariance of a Green's-function ensemble to a data
covariance.
"""
from copy import deepcopy
from typing import NamedTuple, Optional

from seismo_sbi.sbi.noises.diagonal_covariances import DiagonalEmpiricalCovariance, ScalarEmpiricalCovariance
from seismo_sbi.sbi.noises.theory_block_covariance import TheoryBlockDiagonalEmpiricalCovariance
from seismo_sbi.sbi.noises.toeplitz_covariances import (
    BlockDiagonalEmpiricalCovariance,
    BlockDiagonalFilteredCovariance,
    BlockDiagonalKolbCovariance,
)
from seismo_sbi.simulators.receivers import Receivers


class CovarianceLayout(NamedTuple):
    """The data vector a covariance describes: its ``receivers``, ``trace_length`` samples per trace,
    ``data_vector_length`` samples in all, the band-pass ``filter`` applied to the data (a processing
    ``filter`` block) and the number of parallel jobs that factorise the blocks."""

    receivers: Receivers
    trace_length: int
    data_vector_length: int
    filter: Optional[dict] = None
    num_jobs: int = 1


COVARIANCE_BUILDERS = {
    "empirical_block": lambda covariances, layout: BlockDiagonalEmpiricalCovariance(
        covariances, layout.receivers, layout.trace_length, num_jobs=layout.num_jobs),
    "filtered_block": lambda noise_level, layout: BlockDiagonalFilteredCovariance(
        noise_level, layout.filter, layout.receivers, layout.trace_length, num_jobs=layout.num_jobs),
    "kolb": lambda noise_level, layout: BlockDiagonalKolbCovariance(
        noise_level, receivers=layout.receivers, data_vector_length=layout.trace_length, num_jobs=layout.num_jobs),
    "empirical_diagonal": lambda covariances, layout: DiagonalEmpiricalCovariance(
        covariances, layout.receivers, layout.trace_length),
    "noise_level": lambda noise_level, layout: ScalarEmpiricalCovariance(
        noise_level, data_vector_length=layout.data_vector_length),
}


def build_covariance_matrix(option, covariance_data, layout: CovarianceLayout):
    """The covariance ``option`` (a key of ``COVARIANCE_BUILDERS``) from ``covariance_data``.

    ``covariance_data`` is ``{station: {component: autocovariance}}`` for the empirical covariances,
    and a noise level for the others: a standard deviation (m), or ``{station: {component: variance}}``.
    """
    if option not in COVARIANCE_BUILDERS:
        raise NotImplementedError(f"covariance matrix option {option} not implemented")
    return COVARIANCE_BUILDERS[option](deepcopy(covariance_data), layout)


def build_theory_covariance(theory_covariance, data_covariance, diag_regularisation_magnitude,
                            layout: CovarianceLayout):
    """The theory covariance ``theory_covariance`` (the ensemble's ``ScoreCompressionData``) plus the
    blocks of ``data_covariance``, its diagonal regularised by ``diag_regularisation_magnitude`` times
    its largest variance."""
    return TheoryBlockDiagonalEmpiricalCovariance(
        deepcopy(theory_covariance), data_covariance.covariance_matrix_arrays, layout.receivers, layout.trace_length,
        diag_regularisation=diag_regularisation_magnitude, num_jobs=layout.num_jobs)
