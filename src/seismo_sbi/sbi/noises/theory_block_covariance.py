"""Block-diagonal covariance of data noise plus forward-model (theory) error.

Each block is the theory covariance from a Green's-function ensemble plus the data-noise block
and a diagonal regularisation, factorised once by Cholesky.
"""
from functools import partial

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from seismo_sbi.sbi.noises.toeplitz_covariances import BlockDiagonalCovariance
from seismo_sbi.utils.parallel import parallel_execution


class TheoryBlockDiagonalEmpiricalCovariance(BlockDiagonalCovariance):
    """Theory plus data covariance, one block per trace.

    ``station_component_covariances`` carries the flattened theory blocks as ``data_fiducial``,
    shape (n_traces * L * L,), and their parameter derivatives as ``data_parameter_gradients``,
    shape (n_params, n_traces * L * L). With ``covariance_gradients=True`` the derivatives enter
    the Fisher matrix and the score; by default the covariance is treated as parameter-free.
    """
    inverse_metadata = None
    def __init__(
        self,
        station_component_covariances,
        data_covariance_arrays,
        *args,
        diag_regularisation=0.001,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.station_component_covariances = station_component_covariances
        self.data_covariance_arrays = data_covariance_arrays
        self.diag_regularisation_magnitude = diag_regularisation

        self.covariance_matrix_arrays = None
        self.C_derivative = None
        self.kernels = None
        self.traces = None

        self.set_covariance(station_component_covariances)
        if self.covariance_gradients:
            self.set_C_derivative(station_component_covariances)

    # Covariance construction

    def set_cholesky_factors(self, cholesky_factors):
        self.inverse_metadata = cholesky_factors

    def set_covariance(self, station_component_covariances):
        covs = self.create_covariance_matrix(
            station_component_covariances.data_fiducial
        )

        def cholesky_block(C):
            return cho_factor(C, check_finite=False)

        self.covariance_matrix_arrays = covs
        cholesky_factors = parallel_execution(
            covs, cholesky_block, self.num_jobs
        )
        self.set_cholesky_factors(cholesky_factors)


    def create_covariance_matrix(self, station_component_covariances):
        theory_covs = station_component_covariances.reshape(
            -1, self.data_vector_length, self.data_vector_length
        )

        diag_regs = np.array([
            np.eye(self.data_vector_length)
            * (np.max(np.diag(theory_covs[i])) * self.diag_regularisation_magnitude)
            for i in range(theory_covs.shape[0])
        ])

        return theory_covs + self.data_covariance_arrays + diag_regs

    def create_C_derivative(self, station_component_covariances):
        grads = station_component_covariances.data_parameter_gradients
        return grads.reshape(
            -1,
            self.covariance_matrix_arrays.shape[0],
            self.data_vector_length,
            self.data_vector_length
        )
    @staticmethod
    def create_matmul_inverse_covariance(cholesky_factors, data_vector_length):
        return partial(TheoryBlockDiagonalEmpiricalCovariance.callable_matmul_inverse_covariance,
                       cholesky_factors = cholesky_factors, block_size = data_vector_length)
    # Quadratic forms
    @staticmethod
    def callable_matmul_inverse_covariance(data_vector, cholesky_factors, block_size):
        reshaped = data_vector.reshape(-1, block_size)
        out = []
        for cf, x in zip(cholesky_factors, reshaped):
            out.append(cho_solve(cf, x, check_finite=False))
        return np.concatenate(out)
    @staticmethod
    def quadratic_form(residuals, cholesky_factors, block_size):
        reshaped = residuals.reshape(-1, block_size)
        total = 0.0
        for cf, x in zip(cholesky_factors, reshaped):
            y = cho_solve(cf, x, check_finite=False)
            total += x @ y
        return -0.5 * total

    @staticmethod
    def quadratic_form_per_block(residuals, cholesky_factors, block_size):
        reshaped = residuals.reshape(-1, block_size)
        vals = []
        for cf, x in zip(cholesky_factors, reshaped):
            y = cho_solve(cf, x, check_finite=False)
            vals.append(-0.5 * (x @ y))
        return np.repeat(vals, block_size)

    def generic_loss_callable(self, residuals, reduce=True):
        if reduce:
            return self.quadratic_form(
                residuals, self.inverse_metadata, self.data_vector_length
            )
        return self.quadratic_form_per_block(
            residuals, self.inverse_metadata, self.data_vector_length
        )
    
    @staticmethod
    def loss_callable(residuals, toeplitz_cols, data_vector_length):
        return TheoryBlockDiagonalEmpiricalCovariance.quadratic_form(residuals, toeplitz_cols, data_vector_length)

    @staticmethod
    def create_loss_callable(toeplitz_cols, data_vector_length):
        return partial(TheoryBlockDiagonalEmpiricalCovariance.loss_callable,
                        toeplitz_cols = toeplitz_cols, data_vector_length = data_vector_length)
    # Inverse covariance × vector

    def matmul_inverse_covariance(self, data_vector):
        reshaped = data_vector.reshape(-1, self.data_vector_length)
        out = []
        for cf, x in zip(self.inverse_metadata, reshaped):
            out.append(cho_solve(cf, x, check_finite=False))
        return np.concatenate(out)

    # Gradient kernels & traces

    def set_C_derivative(self, station_component_covariances):
        identity = np.eye(self.data_vector_length)
        self.C_inverse = np.array([cho_solve(cf, identity, check_finite=False) for cf in self.inverse_metadata])
        self.C_derivative = self.create_C_derivative(station_component_covariances)
        self.set_covariance_constants()

    def set_covariance_constants(self):
        self.kernels = []
        self.traces = []
        for a in range(self.C_derivative.shape[0]):
            cov_kernel = self.matrix_matrix_product(self.C_inverse, self.matrix_matrix_product(self.C_derivative[a], self.C_inverse))
            trace = self.compute_trace(self.matrix_matrix_product(self.C_inverse, self.C_derivative[a]))
            self.kernels.append(cov_kernel)
            self.traces.append(trace)
