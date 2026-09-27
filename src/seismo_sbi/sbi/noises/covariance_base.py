"""Base class shared by the Gaussian-likelihood noise covariances.

``EmpiricalCovariance`` fixes the interface every covariance offers: the log-likelihood of a
residual, C⁻¹ times a vector, closures carrying their data for worker processes, and a noise
sampler. ``station_component_value`` reads one trace's entry from a ``{station: {component: value}}``
dict.
"""
from abc import ABC, abstractmethod

from seismo_sbi.simulators.simulation_io import component_alias


class EmpiricalCovariance(ABC):
    """Noise covariance interface: likelihood, inverse products, worker closures, a sampler."""

    C_inverse = None
    data_vector_length = None
    C_derivative = None

    @abstractmethod
    def create_sampler(self):
        pass

    @abstractmethod
    def generic_loss_callable(self, residuals):
        return

    def create_loss_callable(self):
        return self.generic_loss_callable

    def compute_loss(self, residuals, *args, **kwargs):
        return self.generic_loss_callable(residuals, *args, **kwargs)

    @abstractmethod
    def matmul_inverse_covariance(self, data_vector):
        pass

    def set_C_inverse(self, C_inverse):
        self.C_inverse = C_inverse

    def set_data_vector_length(self, data_vector_length):
        self.data_vector_length = data_vector_length


def station_component_value(station_component_values, station_name, component):
    """Entry of ``station_component_values[station_name]`` for a component; E and N fall back to 1 and 2."""
    try:
        return station_component_values[station_name][component]
    except KeyError:
        return station_component_values[station_name][component_alias(component)]
