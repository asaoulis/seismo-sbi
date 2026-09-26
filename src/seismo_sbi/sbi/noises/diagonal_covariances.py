"""Uncorrelated-noise covariances: one variance for all samples, or one per trace.

``ScalarEmpiricalCovariance`` takes a single noise level; ``DiagonalEmpiricalCovariance`` takes
a variance per station and component, repeated over the samples of each trace.
"""
from functools import partial

import numpy as np

from seismo_sbi.sbi.noises.covariance_base import EmpiricalCovariance, station_component_value
from seismo_sbi.sbi.noises.noise_samplers import GaussianNoiseSampler


class ScalarEmpiricalCovariance(EmpiricalCovariance):

    inverse_metadata = None

    def __init__(self, sigma_noise_level, data_vector_length=1):
        """``data_vector_length`` is the number of samples each noise draw has."""
        self.noise_level = sigma_noise_level
        self.set_C_inverse(1/sigma_noise_level**2)
        self.inverse_metadata = self.C_inverse
        self.data_vector_length = data_vector_length

    def generic_loss_callable(self, residuals, reduce=True):
        if reduce:
            return -0.5 * np.sum(residuals**2) * self.C_inverse
        return -0.5 * residuals**2 * self.C_inverse

    @staticmethod
    def create_loss_callable(C_inverse, data_vector_length):
        """Loss closed over ``C_inverse``, so worker processes receive it with the function."""
        c_inv = C_inverse
        def _loss(residuals):
            return -0.5 * np.sum(residuals ** 2) * c_inv
        return _loss

    def matmul_inverse_covariance(self, data_vector):
        return data_vector / self.noise_level**2

    @staticmethod
    def callable_matmul_inverse_covariance(data_vector, C_inverse):
        return data_vector * C_inverse

    @staticmethod
    def create_matmul_inverse_covariance(C_inverse, data_vector_length):
        return partial(ScalarEmpiricalCovariance.callable_matmul_inverse_covariance,
                       C_inverse=C_inverse)

    def create_sampler(self):
        return GaussianNoiseSampler(
            receivers=None,
            data_vector_length=1,
            cov_blocks=np.full((self.data_vector_length, 1, 1), self.noise_level ** 2),
        )


class DiagonalEmpiricalCovariance(EmpiricalCovariance):
    inverse_metadata = None

    def __init__(self,station_component_covariances, receivers, data_vector_length):
        self.receivers = receivers
        self.data_vector_length = data_vector_length
        self.station_component_covariances = station_component_covariances
        self.covariance_matrix = self.create_covariance_matrix(station_component_covariances, data_vector_length)
        self.covariance_matrix_arrays = self.create_covariance_matrix(station_component_covariances, data_vector_length, stack=True)
        self.set_C_inverse(1/self.covariance_matrix)
        self.inverse_metadata = self.C_inverse
    

    def create_covariance_matrix(self, station_component_covariances, data_vector_length, stack=False):

        covariance_matrix_diagonals = []

        for receiver in  self.receivers.iterate():
            for component in receiver.components:
                component_data = station_component_value(station_component_covariances, receiver.station_name, component)
                if len(component_data.shape) != 0:
                    component_data = component_data[0]
                covariance_matrix_diagonals.append(component_data * np.ones(data_vector_length))
        if not stack:
            return np.concatenate(covariance_matrix_diagonals) if len(covariance_matrix_diagonals) > 1 else covariance_matrix_diagonals[0][np.newaxis]
        else:
            STACKED = np.stack([np.diag(diag) for diag in covariance_matrix_diagonals], axis=0)
            return STACKED
    def generic_loss_callable(self, residuals, reduce=True):
        if reduce:
            return DiagonalEmpiricalCovariance.loss_callable(residuals, self.C_inverse)
        else:
            elementwise_losses = -0.5 * np.einsum('i,i,i->i', residuals, self.C_inverse, residuals)
            return elementwise_losses

    @staticmethod
    def loss_callable(residuals, C_inverse):
        return -0.5 * residuals @ np.multiply(C_inverse, residuals)

    @staticmethod
    def create_loss_callable(C_inverse, data_vector_length):
        return partial(DiagonalEmpiricalCovariance.loss_callable,
                        C_inverse = C_inverse)
        
    def matmul_inverse_covariance(self, data_vector):
        return self.callable_matmul_inverse_covariance(data_vector, self.C_inverse)
    
    @staticmethod
    def callable_matmul_inverse_covariance(data_vector, C_inverse):
        return np.multiply(C_inverse, data_vector)
    @staticmethod
    def create_matmul_inverse_covariance(C_inverse, data_vector_length):
        return partial(DiagonalEmpiricalCovariance.callable_matmul_inverse_covariance, 
                        C_inverse = C_inverse)
    
    
    def create_sampler(self):
        return GaussianNoiseSampler(
            receivers=self.receivers,
            data_vector_length=self.data_vector_length,
            cov_blocks=self.covariance_matrix_arrays,
            station_component_covariances=self.station_component_covariances,
        )
