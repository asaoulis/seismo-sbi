"""Uncorrelated-noise covariances: one variance for all samples, or one per trace.

``ScalarEmpiricalCovariance`` takes a single noise level; ``DiagonalEmpiricalCovariance`` takes
a variance per station and component, repeated over the samples of each trace.
"""
from functools import partial

import numpy as np

from seismo_sbi.sbi.noises.covariance_base import EmpiricalCovariance
from seismo_sbi.sbi.noises.noise_samplers import GaussianNoiseSampler


class ScalarEmpiricalCovariance(EmpiricalCovariance):

    inverse_metadata = None

    def __init__(self, sigma_noise_level):
        self.noise_level = sigma_noise_level
        self.set_C_inverse(1/sigma_noise_level**2)
        self.inverse_metadata = self.C_inverse
        self.data_vector_length = 1

    @classmethod
    def generic_loss_callable(cls, residuals):
        return -0.5 * np.sum(residuals**2) * cls.C_inverse

    @staticmethod
    def create_loss_callable(C_inverse, data_vector_length):
        """Compatible with the ensemble=False pipeline path.

        Captures C_inverse in a closure so joblib/loky workers receive the
        correct value via cloudpickle rather than reading the class attribute
        (which would be None in freshly-imported worker processes).
        """
        c_inv = ScalarEmpiricalCovariance.C_inverse
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
        def sampler(*args, **kwargs):
            return np.random.randn(1) * self.noise_level, None
        sampling_object = GaussianNoiseSampler(  # type: ignore[arg-type]
            receivers=None,  # not used in scalar case
            data_vector_length=1,
            cov_blocks=[np.array([[self.noise_level ** 2]])],
        )
        return sampling_object


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
            components_dict = station_component_covariances[receiver.station_name]
            for component in receiver.components:
                try:
                    component_data = components_dict[component]
                except KeyError:
                    component = component.replace('E', '1').replace('N', '2')
                    component_data = components_dict[component]
                
                if len(component_data.shape) != 0:
                    component_data = component_data[0]
                covariance_matrix_diagonals.append(component_data * np.ones(data_vector_length))
        # if len(covariance_matrix_diagonals) > 1 add new axis at start
        if not stack:
            return np.concatenate(covariance_matrix_diagonals) if len(covariance_matrix_diagonals) > 1 else covariance_matrix_diagonals[0][np.newaxis]
        else:
            STACKED = np.stack([np.diag(diag) for diag in covariance_matrix_diagonals], axis=0)
            return STACKED
    @classmethod
    def generic_loss_callable(cls, residuals, reduce=True):
        if reduce:
            return DiagonalEmpiricalCovariance.loss_callable(residuals, cls.C_inverse)
        else:
            elementwise_losses = -0.5 * np.einsum('i,i,i->i', residuals, cls.C_inverse, residuals)
            return elementwise_losses

    @staticmethod
    def loss_callable(residuals, C_inverse):
        return -0.5 * residuals @ np.multiply(C_inverse, residuals)

    @staticmethod
    def create_loss_callable(C_inverse, data_vector_length):
        return partial(DiagonalEmpiricalCovariance.loss_callable,
                        C_inverse = C_inverse)
        
    @classmethod
    def matmul_inverse_covariance(cls, data_vector):
        return cls.callable_matmul_inverse_covariance(data_vector, cls.C_inverse)
    
    @staticmethod
    def callable_matmul_inverse_covariance(data_vector, C_inverse):
        return np.multiply(C_inverse, data_vector)
    @staticmethod
    def create_matmul_inverse_covariance(C_inverse, data_vector_length):
        return partial(DiagonalEmpiricalCovariance.callable_matmul_inverse_covariance, 
                        C_inverse = C_inverse)
    
    
    def create_sampler(self):
        def sampler(*args, **kwargs):
            return np.random.randn(self.covariance_matrix.shape[0]) * np.sqrt(self.covariance_matrix), self.station_component_covariances
        # Diagonal case keeps previous simple sampler behaviour.
        sampling_object = GaussianNoiseSampler(  # type: ignore[arg-type]
            receivers=None,
            data_vector_length=self.covariance_matrix.shape[0],
            cov_blocks=[np.diag(self.covariance_matrix)],
            station_component_covariances=self.station_component_covariances,
        )
        sampling_object.sampler = sampler
        return sampling_object
