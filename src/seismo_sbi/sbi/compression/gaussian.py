"""Compressors: score compression under a Gaussian likelihood.

:class:`GaussianCompressor` turns a data vector into the score (or the quasi-maximum-likelihood
estimate) from the fiducial data, its parameter gradients and a noise covariance.
"""

import numpy as np
from abc import ABC, abstractclassmethod
from typing import List

from typing import NamedTuple


from ..noises.covariance_base import EmpiricalCovariance

class ScoreCompressionData(NamedTuple):

    theta_fiducial : np.ndarray
    data_fiducial : np.ndarray
    data_parameter_gradients : np.ndarray
    second_order_gradients : np.ndarray


class Compressor(ABC):

    @abstractclassmethod
    def compress_data_vector(self, D, *args, **kwargs):
        pass

class GaussianCompressor(Compressor):
    """Score compression under a Gaussian likelihood.

    Built from the fiducial data, its parameter gradients (``ScoreCompressionData``) and a noise
    covariance; compresses a data vector to the quasi-maximum-likelihood estimate
    ``theta_fiducial + F^-1 score``, with :meth:`compute_score` giving the score itself.
    """

    def __init__(self, score_compression_data : ScoreCompressionData, covariance_matrix : EmpiricalCovariance, prior = (None, None)):

        self.num_params = score_compression_data.data_parameter_gradients.shape[0]
        self.prior_mean, self.prior_covariance = prior

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

    def set_priors(self, prior):
        self.prior_mean, self.prior_covariance = prior
        self.Fisher_mat = self._compute_Fisher_matrix()
        self.Fisher_mat_inverse = np.linalg.inv(self.Fisher_mat)

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

        
        if self.prior_covariance is not None:
            F +=  np.linalg.inv(np.diag(self.prior_covariance))

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
        if self.prior_mean is not None:
            theta_diff = self.prior_mean - self.theta_fiducial
            # The sign of theta_diff here is unverified.
            dL_dtheta += np.dot(np.linalg.inv(np.diag(self.prior_covariance)), theta_diff)
        return dL_dtheta

    def compute_misfit(self, D):
        return np.dot((D - self.D_fiducial), self.C.matmul_inverse_covariance(D - self.D_fiducial)) / len(self.D_fiducial)
    
    


    
class MultiPointGaussianCompressor(Compressor):

    def __init__(self, score_compression_datas : List[ScoreCompressionData], covariance_matrix, is_diag = True):

        self._compressors  = []
        for score_compression_data in score_compression_datas:
            self._compressors.append(
                GaussianCompressor(score_compression_data, covariance_matrix, is_diag)
            )
    
    def compress_data_vector(self, D):
        return np.hstack([compressor.compress_data_vector(D) for compressor in self._compressors])
    

class SecondOrderCompressor(GaussianCompressor):

    def __init__(self, score_compression_data, hessian, covariance_matrix, is_diag = True):

        self.num_params = score_compression_data.data_parameter_gradients.shape[0]

        self.theta_fiducial = np.copy(score_compression_data.theta_fiducial)
        self.D_fiducial = np.copy(score_compression_data.data_fiducial)
        self.dD_Dtheta_gradients = np.copy(score_compression_data.data_parameter_gradients)

        self.C = covariance_matrix
        self.is_diag = is_diag
        if is_diag:
            self.C_inverse = np.diag(1/np.diag(covariance_matrix))
        else:
            self.C_inverse = np.linalg.inv(covariance_matrix)

        
        self.efficient_dot_prod = self._select_efficient_dot_product_op(self.C)

        self.hessian = hessian.transpose(2, 0, 1)

        self.F = np.dot(self.dD_Dtheta_gradients, self.efficient_dot_prod(self.C_inverse, self.D_fiducial))
        self.S = np.einsum("mkv,m->kv", self.hessian, self.efficient_dot_prod(self.C_inverse, self.D_fiducial))

    def compress_data_vector(self, data_vector):

        F_hat  = np.dot(self.dD_Dtheta_gradients, self.efficient_dot_prod(self.C_inverse, data_vector))

        S_hat  = np.einsum("mkv,m->kv", self.hessian, self.efficient_dot_prod(self.C_inverse, data_vector))

        delta_F = (F_hat - self.F)
        delta_S = S_hat - self.S

        return np.concatenate([delta_F, delta_S.flatten()])
