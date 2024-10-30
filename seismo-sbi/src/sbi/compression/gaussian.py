import numpy as np
import time
from abc import ABC, abstractclassmethod
from typing import List

from typing import NamedTuple, Callable

import torch

from .ML.seismogram_transformer import LightningModel
from .ML.utils import get_best_model
from ..noises.covariance_estimation import EmpiricalCovariance, DiagonalEmpiricalCovariance

class ScoreCompressionData(NamedTuple):

    theta_fiducial : np.ndarray
    data_fiducial : np.ndarray
    data_parameter_gradients : np.ndarray
    second_order_gradients : np.ndarray


class Compressor(ABC):

    @abstractclassmethod
    def compress_data_vector(self, D, *args, **kwargs):
        pass

class MachineLearningCompressor(Compressor):

    def __init__(self, model_type, model_name, seismogram_preprocessor : Callable, scaler, **model_kwargs):

        self.trained_ml_compressor = get_best_model(model_type, model_name, checkpoint_path="ml_models", **model_kwargs)
        self.seismogram_preprocessor = seismogram_preprocessor

        self.scaler = scaler

    def compress_data_vector(self, D):
        with torch.no_grad():
            processed_seismogram = self.seismogram_preprocessor(D)
            parameters_prediction = self.trained_ml_compressor.forward(processed_seismogram.unsqueeze(0)).detach()
            parameters_prediction = self.scaler.inverse_transform(parameters_prediction).squeeze(0)
            return parameters_prediction

class GaussianCompressor(Compressor):

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

    def clear_priors(self):
        self.prior_mean = None
        self.prior_covariance = None
        self.Fisher_mat = self._compute_Fisher_matrix()
        self.Fisher_mat_inverse = np.linalg.inv(self.Fisher_mat)

    def set_priors(self, prior):
        print('setting priors')
        self.prior_mean, self.prior_covariance = prior
        self.Fisher_mat = self._compute_Fisher_matrix()
        self.Fisher_mat_inverse = np.linalg.inv(self.Fisher_mat)

    def create_covariance_matrix_sampler(self):
        # TODO: move implementation to EmpiricalCovariance classes (need to specify a valid covariance matrix)
        if self.is_diag:
            return lambda : np.random.normal(0, np.diag(self.C.covariance_matrix))
        else:
            return lambda : np.random.multivariate_normal(self.C.covariance_matrix)

    def _compute_Fisher_matrix(self):
        # assume a special case of dC/dtheta = 0 throughout

        F = np.zeros((self.num_params, self.num_params))

        for a in range(0, self.num_params):
            for b in range(0, self.num_params):
                F[a, b] += 0.5*(np.dot(self.dD_Dtheta_gradients[a,:], self.C.matmul_inverse_covariance(self.dD_Dtheta_gradients[b,:])) \
                                + np.dot(self.dD_Dtheta_gradients[b,:], self.C.matmul_inverse_covariance(self.dD_Dtheta_gradients[a,:])))
        
        if self.prior_covariance is not None:
            F +=  np.linalg.inv(np.diag(self.prior_covariance))

        return F


    def compute_theta_MLE(self, D, damping = None, matmul_callable = None):

        score = self.compute_score(D, matmul_callable=matmul_callable)

        t = self.convert_score_to_theta_MLE(score, damping)
        # if self.prior_mean is not None:
        #     theta_diff = self.prior_mean - self.theta_fiducial
        #     print('pre prior', t)
        #     t += np.dot(self.Fisher_mat_inverse, 
        #                 )# + damping*np.diag(np.diag(self.Fisher_mat))
        #     print('post prior', t)
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

        if matmul_callable is not None:
            covariance_residual_product = matmul_callable(D - self.D_fiducial)
        else:
            covariance_residual_product = self.C.matmul_inverse_covariance(D - self.D_fiducial)
        if reduce:
            dL_dtheta = np.zeros(self.num_params)
            for a in range(self.num_params):
                dL_dtheta[a] += np.dot(self.dD_Dtheta_gradients[a,:], covariance_residual_product)
        else:
            dL_dtheta = np.zeros((self.dD_Dtheta_gradients[0,:].shape[0], self.num_params))
            for a in range(self.num_params):
                dL_dtheta[:,a] = np.multiply(self.dD_Dtheta_gradients[a,:], covariance_residual_product)
        if self.prior_mean is not None:
            theta_diff = self.prior_mean - self.theta_fiducial
            # np.dot(np.linalg.inv(np.diag(self.prior_covariance)), theta_diff)
            # or maybe theta_diff should be negative ?
            dL_dtheta += np.dot(np.linalg.inv(np.diag(self.prior_covariance)), theta_diff)
        return dL_dtheta

    def compute_misfit(self, D):
        return np.dot((D - self.D_fiducial), self.C.matmul_inverse_covariance(D - self.D_fiducial)) / len(self.D_fiducial)
    
    def compute_log_likelihood(self, D, theta):

        dL_dtheta = self.compute_score(D)
        delta_theta = theta - self.theta_fiducial
        return self.L_fiducial + np.dot(delta_theta, dL_dtheta)
    


    
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
            self.L_fiducial = -1/2 * np.log(self.C.diagonal().prod())
        else:
            self.C_inverse = np.linalg.inv(covariance_matrix)
            self.L_fiducial = -1/2 * np.log(np.linalg.det(self.C))

        
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
    
    def compute_observed_information(self, data_vector):
        return np.einsum("mkv,m->kv", self.hessian, self.efficient_dot_prod(self.C_inverse, data_vector - self.D_fiducial))
    
    # def compress_data_vector(self, data_vector):

    #     F_hat  = np.dot(self.dD_Dtheta_gradients, self.efficient_dot_prod(self.C_inverse, data_vector))

    #     S_hat  = np.einsum("mkv,m->kv", self.hessian, self.efficient_dot_prod(self.C_inverse, data_vector))

    #     U = np.dot(self.dD_Dtheta_gradients, self.efficient_dot_prod(self.C_inverse,self.dD_Dtheta_gradients.T))
    #     U_inverse = np.linalg.inv(U)

    #     p1 = U_inverse @ (F_hat - self.F)
    #     # return self.theta_fiducial + p1# would be first order
    #     delta_S = S_hat - self.S

    #     G_step = self.efficient_dot_prod(self.C_inverse, self.hessian)
    #     G = np.einsum('ij, jkl -> ikl', self.dD_Dtheta_gradients, G_step)

    #     p2 = U_inverse @ ( np.dot(delta_S, p1) - 1/2 * np.einsum('kmn,m,n->k', G, p1, p1) - np.einsum('mkn,m,n->k', G, p1, p1))
    #     return self.theta_fiducial + (p1 + p2)

    
    def check_compression(self, data_vector, p):
        F_hat  = np.dot(self.dD_Dtheta_gradients, self.efficient_dot_prod(self.C_inverse, data_vector))

        S_hat  = np.einsum("mkv,m->kv", self.hessian, self.efficient_dot_prod(self.C_inverse, data_vector))

        delta_F = (F_hat - self.F)
        delta_S = S_hat - self.S

        lhs = delta_F + np.dot(delta_S, p)

        left_multiplier = self.dD_Dtheta_gradients.T  +np.dot(self.hessian, p)
        right_multiplier = np.dot(self.dD_Dtheta_gradients.T, p) + 1/2 * np.einsum('kmn,m,n->k', self.hessian, p, p)

        rhs = np.dot(left_multiplier.T, self.efficient_dot_prod(self.C_inverse, right_multiplier))

        return np.linalg.norm((rhs - lhs)**2)
    
    def get_delta_statistics(self, data_vector):

        F_hat  = np.dot(self.dD_Dtheta_gradients, self.efficient_dot_prod(self.C_inverse, data_vector))

        S_hat  = np.einsum("mkv,m->kv", self.hessian, self.efficient_dot_prod(self.C_inverse, data_vector))

        delta_F = (F_hat - self.F)
        delta_S = S_hat - self.S

        return delta_F, delta_S

    def compression_optimisation(self, data_vector, scaling, p_scaled):
        p = p_scaled * scaling
        return self.check_compression(data_vector, p)

    def check_first_order_compression(self, data_vector, p):
        F_hat  = np.dot(self.dD_Dtheta_gradients, self.efficient_dot_prod(self.C_inverse, data_vector))

        lhs = (F_hat - self.F)

        U = np.dot(self.dD_Dtheta_gradients, self.efficient_dot_prod(self.C_inverse,self.dD_Dtheta_gradients.T))
        rhs = np.dot(U, p)

        return rhs, lhs