"""Compressors: score compression under a Gaussian likelihood.

:class:`GaussianCompressor` turns a data vector into the score (or the quasi-maximum-likelihood
estimate) from the fiducial data, its parameter gradients and a noise covariance.
:class:`MultiPointGaussianCompressor` adds the estimates from fixed other sources, and
:class:`SecondOrderCompressor` the data part of the log-likelihood Hessian.
"""

import numpy as np
from abc import ABC, abstractclassmethod
from typing import List, NamedTuple, Optional

from ..inversion.gaussian_prior import GaussianPrior
from ..noises.covariance_base import EmpiricalCovariance

class ScoreCompressionData(NamedTuple):

    theta_fiducial : np.ndarray
    data_fiducial : np.ndarray
    data_parameter_gradients : np.ndarray
    second_order_gradients : np.ndarray


class Compressor(ABC):

    #: True when the summary estimates the parameters themselves, so it is scaled like them.
    summary_is_parameter_estimate = False

    @abstractclassmethod
    def compress_data_vector(self, D, *args, **kwargs):
        pass

class GaussianCompressor(Compressor):
    """Score compression under a Gaussian likelihood.

    Built from the fiducial data, its parameter gradients (``ScoreCompressionData``) and a noise
    covariance. Compresses a data vector to the quasi-maximum-likelihood estimate
    ``theta_fiducial + F^-1 score``, with :meth:`compute_score` giving the score itself. A
    :class:`~seismo_sbi.sbi.inversion.gaussian_prior.GaussianPrior` ``prior`` adds its precision to
    ``F`` and its score to the score, so the estimate is the maximum of the posterior.
    """

    summary_is_parameter_estimate = True

    def __init__(self, score_compression_data : ScoreCompressionData, covariance_matrix : EmpiricalCovariance,
                 prior: Optional[GaussianPrior] = None):

        self.num_params = score_compression_data.data_parameter_gradients.shape[0]
        self.prior = prior

        self.C = covariance_matrix

        self.theta_fiducial = None 
        self.D_fiducial = None
        self.dD_Dtheta_gradients = None
        self.Fisher_mat = None
        self.Fisher_mat_inverse = None
        self.scaling = None

        self.set_compression_variables(score_compression_data)

    def set_compression_variables(self, score_compression_data):
        self.theta_fiducial = np.copy(score_compression_data.theta_fiducial)
        self.D_fiducial = np.copy(score_compression_data.data_fiducial)
        self.dD_Dtheta_gradients = np.copy(score_compression_data.data_parameter_gradients)

        self.Fisher_mat = self._compute_Fisher_matrix()
        self.Fisher_mat_inverse = np.linalg.inv(self.Fisher_mat)

    def set_extra_gradients(self, extra_gradients):
        """Give the noise covariance the theory covariance ``extra_gradients`` (the ensemble's
        ``ScoreCompressionData``) of a theory compressor."""
        self.C.set_covariance(extra_gradients)

    def _compute_Fisher_matrix(self):
        # assume a special case of dC/dtheta = 0 throughout

        F = np.zeros((self.num_params, self.num_params))

        for a in range(0, self.num_params):
            for b in range(0, self.num_params):
                F[a, b] += 0.5*(np.dot(self.dD_Dtheta_gradients[a,:], self.C.matmul_inverse_covariance(self.dD_Dtheta_gradients[b,:])) \
                                + np.dot(self.dD_Dtheta_gradients[b,:], self.C.matmul_inverse_covariance(self.dD_Dtheta_gradients[a,:])))
                if self.C.C_derivative is not None:
                    F[a, b] += 0.5 * self.C.compute_trace(self.C.matrix_matrix_product(
                        self.C.matrix_matrix_product(self.C.C_inverse, self.C.C_derivative[a]),
                        self.C.matrix_matrix_product(self.C.C_inverse, self.C.C_derivative[b])
                    ))

        
        if self.prior is not None:
            F += self.prior.precision()

        return F


    def compute_theta_MLE(self, D, damping = None, matmul_callable = None):

        score = self.compute_score(D, matmul_callable=matmul_callable)

        t = self.convert_score_to_theta_MLE(score, damping)
        return t
    
    def convert_score_to_theta_MLE(self, score, damping = None):
        if damping not in [0, None]:
            applied_Fisher_mat_inverse = np.linalg.inv(self.Fisher_mat + damping*np.diag(np.diag(self.Fisher_mat)))
        else:
            applied_Fisher_mat_inverse = self.Fisher_mat_inverse
        return self.theta_fiducial + np.dot(applied_Fisher_mat_inverse, score)
    
    def compress_data_vector(self, D, matmul_callable = None):
        return self.compute_theta_MLE(D, matmul_callable=matmul_callable)

    def compute_score(self, D, matmul_callable = None, reduce=True):
        residual = D - self.D_fiducial
        if matmul_callable is not None:
            covariance_residual_product = matmul_callable(residual)
        else:
            covariance_residual_product = self.C.matmul_inverse_covariance(residual)
        if reduce:
            # first term: vectorised over parameters
            dL_dtheta = self.dD_Dtheta_gradients @ covariance_residual_product

            if self.C.C_derivative is not None:
                # keep existing loop for the expensive derivative terms
                for a in range(self.num_params):
                    covariance_products = self.C.kernels[a]
                    first_cov_term = 0.5 * self.C.vector_vector_dot_product(
                        residual,
                        self.C.matrix_vector_product(covariance_products, residual)
                    )
                    second_cov_term = -0.5 * self.C.traces[a]
                    dL_dtheta[a] += first_cov_term + second_cov_term
        else:
            d = self.dD_Dtheta_gradients.shape[1]
            dL_dtheta = np.empty((d, self.num_params))
            # broadcast multiply and transpose instead of Python loop
            # shape: (num_params, d) * (d,) -> (num_params, d) then transpose
            dL_dtheta[:, :] = (self.dD_Dtheta_gradients * covariance_residual_product).T
        if self.prior is not None:
            dL_dtheta += self.prior.score(self.theta_fiducial)
        return dL_dtheta

    def compute_misfit(self, D):
        return np.dot((D - self.D_fiducial), self.C.matmul_inverse_covariance(D - self.D_fiducial)) / len(self.D_fiducial)
    
    


    
class MultiPointGaussianCompressor(GaussianCompressor):
    """Score compression at the run's fiducial source and at fixed other sources: the
    quasi-maximum-likelihood estimate from each, ``n (1 + n_points)`` numbers for ``n`` parameters.

    The maximum-likelihood search moves the run's fiducial only. ``point_compression_datas`` are the
    ``ScoreCompressionData`` at the fixed sources.
    """

    summary_is_parameter_estimate = False

    def __init__(self, score_compression_data, point_compression_datas : List[ScoreCompressionData],
                 covariance_matrix, prior=None):
        super().__init__(score_compression_data, covariance_matrix, prior=prior)
        self.set_extra_gradients(point_compression_datas)

    def set_extra_gradients(self, point_compression_datas):
        """Build the score compression at each fixed source from its ``ScoreCompressionData``."""
        self._compressors = [GaussianCompressor(point_compression_data, self.C, prior=self.prior)
                             for point_compression_data in point_compression_datas]

    def compress_data_vector(self, D, matmul_callable = None):
        estimates = [super().compress_data_vector(D, matmul_callable=matmul_callable)]
        estimates += [compressor.compress_data_vector(D, matmul_callable=matmul_callable)
                      for compressor in self._compressors]
        return np.hstack(estimates)
    

class SecondOrderCompressor(GaussianCompressor):
    """Second-order score compression: the score ``G C^-1 r`` and the data part of the Hessian of the
    log-likelihood ``H C^-1 r``, ``n + n^2`` numbers for ``n`` parameters.

    ``r = D - D_fiducial``. ``G`` ``(n, n_data)`` are the data's parameter gradients and ``hessian``
    ``H`` ``(n, n, n_data)`` its second derivatives at the fiducial source. The maximum-likelihood
    search uses the first-order compression it inherits.
    """

    summary_is_parameter_estimate = False

    def __init__(self, score_compression_data, hessian, covariance_matrix, prior=None):
        super().__init__(score_compression_data, covariance_matrix, prior=prior)
        self.hessian = hessian

    def set_extra_gradients(self, hessian):
        """Take the Hessian ``(n, n, n_data)`` of the data at the current fiducial source."""
        self.hessian = hessian

    def compress_data_vector(self, data_vector, matmul_callable=None):
        residual = data_vector - self.D_fiducial
        if matmul_callable is not None:
            weighted_residual = matmul_callable(residual)
        else:
            weighted_residual = self.C.matmul_inverse_covariance(residual)
        score = self.dD_Dtheta_gradients @ weighted_residual
        hessian_term = self.hessian @ weighted_residual
        return np.concatenate([score, hessian_term.flatten()])
