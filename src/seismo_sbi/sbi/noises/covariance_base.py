"""Base class shared by the Gaussian-likelihood noise covariances.

``EmpiricalCovariance`` fixes the interface every covariance offers: the log-likelihood of a
residual, C⁻¹ times a vector, closures carrying their data for worker processes, and a noise
sampler. ``station_component_value`` reads one trace's entry from a ``{station: {component: value}}``
dict; ``parallel_execution`` builds per-block quantities with joblib.
"""
from abc import ABC, abstractmethod

import joblib


class EmpiricalCovariance(ABC):
    # The sampler uses multiprocessing, which would pickle a large object per call, so the
    # heavy data is kept on the class instead and inherited by the pool.
    C_inverse = None
    data_vector_length = None
    C_derivative = None

    @abstractmethod
    def create_sampler(self):
        pass

    @staticmethod
    @abstractmethod
    def generic_loss_callable(residuals):
        return 

    @classmethod
    def create_loss_callable(cls):
        return cls.generic_loss_callable

    def compute_loss(self, residuals, *args, **kwargs):
        return self.generic_loss_callable(residuals, *args, **kwargs)

    @abstractmethod
    def matmul_inverse_covariance(self, data_vector):
        pass

    @classmethod
    def set_C_inverse(cls, C_inverse):
        cls.C_inverse = C_inverse

    @classmethod
    def set_data_vector_length(cls, data_vector_length):
        cls.data_vector_length = data_vector_length


def parallel_execution(inputs, func, num_jobs = 20):
    if num_jobs in [None, 0 , 1]:
        return [func(block) for block in inputs]
    return joblib.Parallel(n_jobs=num_jobs)(joblib.delayed(func)(block) for block in inputs)


def station_component_value(station_component_values, station_name, component):
    """Entry of ``station_component_values[station_name]`` for a component; E and N fall back to 1 and 2."""
    try:
        return station_component_values[station_name][component]
    except KeyError:
        return station_component_values[station_name][component.replace('E', '1').replace('N', '2')]
