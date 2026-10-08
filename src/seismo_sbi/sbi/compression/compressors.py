"""The compressor a ``compression`` entry names, built from its options and the score-compression data.

:func:`build_compressor` takes the typed options of :mod:`~seismo_sbi.sbi.compression.compressor_options`,
the fiducial data and its gradients, and the covariance data of the noise, and returns the compressor
with its noise covariance as ``compressor.C``. A theory compressor's covariance holds its data covariance
as ``compressor.C.data_covariance``.
"""
import numpy as np

from seismo_sbi.sbi.compression.gaussian import GaussianCompressor, MultiPointGaussianCompressor, SecondOrderCompressor
from seismo_sbi.sbi.noises.covariance_estimator import build_cov_sigma2_dict
from seismo_sbi.sbi.noises.covariances import build_covariance_matrix, build_theory_covariance
from seismo_sbi.sbi.noises.real_noise import RealNoiseSampler
from seismo_sbi.utils.errors import InvalidConfiguration


def build_compressor(options, score_compression_data, simulation_parameters, layout, covariance_data=None,
                     extra_gradients=None, prior=None):
    """The compressor of ``options`` around ``score_compression_data``.

    ``covariance_data`` is the job's noise ``{station: {component: autocovariance}}``, or None to estimate
    an ``optimal_score`` covariance from the mean recorded autocovariance of the noise windows of
    ``options.path``; ``extra_gradients`` is the
    theory covariance of a ``theory_optimal_score`` compressor, or the Hessian of a second-order one;
    ``layout`` is the data vector's :class:`~seismo_sbi.sbi.noises.covariances.CovarianceLayout`.
    """
    ctype = options.type

    if ctype == "optimal_score":
        if covariance_data is None:
            covariance_data = RealNoiseSampler(simulation_parameters, options.path, layout.trace_length).mean_covariance_data()
        covariance = build_covariance_matrix(options.covariance, covariance_data, layout)
        return GaussianCompressor(score_compression_data, covariance, prior=prior)

    if ctype == "theory_optimal_score":
        noise_level = options.noise_level
        if noise_level is None:
            if covariance_data is None:
                raise InvalidConfiguration(
                    "compression.theory_optimal_score with noise_level: null takes its noise level from an event's "
                    "pre-event variances, and none was given: pass the event's covariance data, or set noise_level.")
            noise_level = build_cov_sigma2_dict(covariance_data)
        if options.data_covariance is None:
            raise InvalidConfiguration("compression.theory_optimal_score needs data_covariance, the noise "
                                       "covariance model the theory covariance is added to.")
        data_covariance = build_covariance_matrix(options.data_covariance, noise_level, layout)
        covariance = build_theory_covariance(extra_gradients, data_covariance, options.diag_regularisation_magnitude,
                                             layout)
        return GaussianCompressor(score_compression_data, covariance, prior=prior)

    cov_mat = np.diag(options.noise_level**2 * np.ones((layout.data_vector_length)))
    if ctype == "multi_optimal_score":
        return MultiPointGaussianCompressor(score_compression_data, cov_mat)
    return SecondOrderCompressor(score_compression_data, extra_gradients, cov_mat)
