"""Block-diagonal covariances with one stationary (Toeplitz) block per trace.

``BlockDiagonalCovariance`` solves the Toeplitz systems; its subclasses build each block's first
column from a measured autocovariance (empirical), a band-limited white-noise model (filtered)
or the Kolb damped-cosine model.
"""
from copy import deepcopy
from functools import partial

import numpy as np
from scipy.linalg import solve_toeplitz, toeplitz

from seismo_sbi.sbi.noises.covariance_base import EmpiricalCovariance, parallel_execution, station_component_value
from seismo_sbi.sbi.noises.covariance_estimator import EmpiricalCovarianceEstimator
from seismo_sbi.sbi.noises.noise_samplers import GaussianNoiseSampler


class BlockDiagonalCovariance(EmpiricalCovariance):
        inverse_metadata = None

        def __init__(self, receivers, data_vector_length, block_exp_tapering = True, covariance_gradients = None, num_jobs = 20):

            self.set_data_vector_length(data_vector_length)
            self.block_exp_tapering = block_exp_tapering
            self.receivers = receivers
            self.covariance_gradients = covariance_gradients
            self.num_jobs = num_jobs

        def set_toeplitz_cols(self, inverse_metadata):
            self.inverse_metadata = inverse_metadata

        def generic_loss_callable(self, residuals, reduce=True):
            if reduce:
                return self.quadratic_form(residuals, self.inverse_metadata, self.data_vector_length)
            return self.quadratic_form_per_block(residuals, self.inverse_metadata, self.data_vector_length)

        @staticmethod
        def quadratic_form(residuals, toeplitz_cols, block_size):
            reshaped = residuals.reshape(-1, block_size)
            total = 0.0
            for c, x in zip(toeplitz_cols, reshaped):
                y = solve_toeplitz((c, c), x)
                total += x @ y
            return -0.5 * total

        @staticmethod
        def quadratic_form_per_block(residuals, toeplitz_cols, block_size):
            reshaped = residuals.reshape(-1, block_size)
            vals = []
            for c, x in zip(toeplitz_cols, reshaped):
                y = solve_toeplitz((c, c), x)
                vals.append(-0.5 * (x @ y))
            return np.repeat(vals, block_size)

        @staticmethod
        def loss_callable(residuals, toeplitz_cols, data_vector_length):
            return BlockDiagonalCovariance.quadratic_form(residuals, toeplitz_cols, data_vector_length)
        
        @staticmethod
        def create_loss_callable(toeplitz_cols, data_vector_length):
            return partial(BlockDiagonalCovariance.loss_callable,
                            toeplitz_cols = toeplitz_cols, data_vector_length = data_vector_length)
        
        def matmul_inverse_covariance(self, data_vector):
            return self.callable_matmul_inverse_covariance_toeplitz(data_vector, self.inverse_metadata, self.data_vector_length)
        
        @staticmethod
        def callable_matmul_inverse_covariance_toeplitz(data_vector, toeplitz_cols, data_vector_length):
            reshaped = data_vector.reshape(-1, data_vector_length)
            out = []
            for c, x in zip(toeplitz_cols, reshaped):
                out.append(solve_toeplitz((c, c), x))
            return np.concatenate(out)

        def matrix_vector_product(self, matrix, data_vector):
            reshaped_vector = data_vector.reshape(-1, self.data_vector_length)
            return np.einsum('ijk,ik->ij', matrix, reshaped_vector).reshape(-1)
        
        def matrix_matrix_product(self, matrix1, matrix2):
            return np.einsum('ijk,ikl->ijl', matrix1, matrix2)

        def vector_vector_dot_product(self, vector1, vector2):
            return np.dot(vector1, vector2)
    
        def compute_trace(self, matrix):
            return np.trace(matrix, axis1=1, axis2=2).sum()

        @staticmethod
        def create_matmul_inverse_covariance(toeplitz, data_vector_length):
            return partial(BlockDiagonalCovariance.callable_matmul_inverse_covariance_toeplitz, 
                           toeplitz_cols = toeplitz, data_vector_length = data_vector_length)
        
        def create_toeplitz_cols(self, station_component_covariances):
            """First column of every trace's block, shape (n_traces, data_vector_length).

            ``station_component_covariances`` is a ``{station: {component: value}}`` dict passed to
            ``toeplitz_column``, or one noise level σ used for every trace as σ².
            """
            receiver_components_list = [(receiver.station_name, component) for receiver in self.receivers.iterate() for component in receiver.components]
            if isinstance(station_component_covariances, dict):
                def compute_covariance(station_name_components):
                    station_name, component = station_name_components
                    return self.toeplitz_column(station_component_value(station_component_covariances, station_name, component))
                covariance_blocks = parallel_execution(receiver_components_list, compute_covariance, self.num_jobs)
            else:
                sigma_sqr = station_component_covariances**2
                builder = lambda _: self.toeplitz_column(sigma_sqr)
                covariance_blocks = parallel_execution(receiver_components_list, builder, self.num_jobs)
            return np.array(covariance_blocks)

        def create_sampler(self, scale=1e9):

            if hasattr(self, 'covariance_matrix_arrays') and self.covariance_matrix_arrays is not None:
                cov_blocks = [block for block in self.covariance_matrix_arrays]
                sampler = GaussianNoiseSampler(
                    receivers=self.receivers,
                    data_vector_length=self.data_vector_length,
                    cov_blocks=cov_blocks,
                    station_component_covariances=getattr(self, 'station_component_covariances', None),
                )
            elif getattr(self, 'toeplitz_cols_list', None) is not None:
                toeplitz_cols = self.toeplitz_cols_list
                cov_blocks = [toeplitz(c) for c in toeplitz_cols]
                sampler = GaussianNoiseSampler(
                    receivers=self.receivers,
                    data_vector_length=self.data_vector_length,
                    toeplitz_cols=toeplitz_cols,
                    cov_blocks=cov_blocks,
                    station_component_covariances=getattr(self, 'station_component_covariances', None),
                )
            else:
                raise ValueError("No covariance matrix available for sampling")

            return sampler


class BlockDiagonalFilteredCovariance(BlockDiagonalCovariance):
    """
    Block diagonal covariance with filtered noise.
    This is used for the NPE training with filtered noise.
    """
    
    def __init__(self, station_component_covariances, filter, *args, **kwargs):
        super().__init__(
            *args, **kwargs
        )
        self.station_component_covariances = station_component_covariances
        self.freqs = filter['freqmin'], filter['freqmax']

        self.toeplitz_cols_list = self.create_toeplitz_cols(station_component_covariances)
        self.set_toeplitz_cols(self.toeplitz_cols_list)
        self.covariance_matrix_arrays = np.array([toeplitz(c) for c in self.toeplitz_cols_list])
    
    def gamma_bandpass(self, tau, sigma_sqr, freqs):
        fmin, fmax = freqs
        if tau == 0:
            return 2 * sigma_sqr * (fmax - fmin)
        else:
            return sigma_sqr * (np.sin(2*np.pi*fmax*tau) - np.sin(2*np.pi*fmin*tau)) / (np.pi * tau)

    def build_single_covariance_column(self, sigma_sqr, data_vector_length, freqs):
        # sigma_sqr is actually 2*sigma^2*(fmax - fmin)
        sigma_sqr_internal = sigma_sqr / (2 * (freqs[1] - freqs[0]))
        lags = np.arange(data_vector_length)
        gamma_vals = np.array([self.gamma_bandpass(t,sigma_sqr_internal, freqs) for t in lags])
        
        gamma_vals[0] += 0.01 * gamma_vals[0]
        
        return gamma_vals

    def toeplitz_column(self, sigma_sqr):
        return self.build_single_covariance_column(sigma_sqr, self.data_vector_length, self.freqs)


class BlockDiagonalKolbCovariance(BlockDiagonalCovariance):
    """
    Block diagonal covariance with Kolb structure.

    Each block is::

        C_ij = e^(-lambda * |t_j - t_i|) * cos(lambda * omega_0 * |t_j - t_i|)
    """
    def __init__(self, station_component_covariances, omega_0=4.4, lam=1./20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.omega_0 = omega_0
        self.lam = lam
        self.station_component_covariances = station_component_covariances
        
        self.toeplitz_cols_list = self.create_toeplitz_cols(station_component_covariances)
        self.set_toeplitz_cols(self.toeplitz_cols_list)
        self.covariance_matrix_arrays = np.array([toeplitz(c) for c in self.toeplitz_cols_list])

    def build_single_covariance_column(self, sigma_sqr, data_vector_length):
        lags = np.arange(data_vector_length)
        # C_ij = e^(-lambda * |tau|) * cos(lambda * omega_0 * |tau|)
        # scaled by sigma_sqr (variance)
        gamma_vals = sigma_sqr * np.exp(-self.lam * lags) * np.cos(self.lam * self.omega_0 * lags)
        
        gamma_vals[0] += 1e-2 *  gamma_vals[0]
        
        return gamma_vals

    def toeplitz_column(self, val):
        """Column for a variance, or for an autocovariance array whose lag-0 value is the variance."""
        if isinstance(val, np.ndarray):
            sigma_sqr = val[0] if val.ndim > 0 else val.item()
        else:
            sigma_sqr = val
        return self.build_single_covariance_column(sigma_sqr, self.data_vector_length)


class BlockDiagonalEmpiricalCovariance(BlockDiagonalCovariance):
        
        data_vector_length = None
    
        def __init__(self, station_component_covariances, *args, **kwargs):
            """
            station_component_covariances: dict of dicts, where keys are station names and values are dicts with components as keys and covariance data as values
            """
            super().__init__(
                *args, **kwargs
            )
            self.station_component_covariances = deepcopy(station_component_covariances)
            self.toeplitz_cols_list = self.create_toeplitz_cols(station_component_covariances)
            self.set_toeplitz_cols(self.toeplitz_cols_list)
            self.covariance_matrix_arrays = np.array([toeplitz(c) for c in self.toeplitz_cols_list])

        def create_toeplitz_cols(self, station_component_covariances):
            if self.block_exp_tapering:
                station_component_covariances = EmpiricalCovarianceEstimator.taper_covariances(station_component_covariances, self.data_vector_length, fit_length=20, ols_fit=False)
            return super().create_toeplitz_cols(station_component_covariances)

        def toeplitz_column(self, covar_data):
            return covar_data[:self.data_vector_length].copy()
