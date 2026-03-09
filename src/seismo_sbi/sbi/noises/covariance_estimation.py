import numpy as np
import statsmodels.api as sm
import joblib
import os
import torch

EPS = 1e-20

class RunningStandardDeviations:
    def __init__(self, data=None, track = False):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.std  = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            self.track = track
            self.hist = []
            self.nobservations = 0


    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError(f"Data dims don't match prev observations - {data.shape[1]} != {self.ndimensions}")

            newmean = data.mean(axis=0)
            newstd  = data.std(axis=0)
            if self.track:
                self.hist.append(newmean)

            m = self.nobservations * 1
            n = data.shape[0]

            tmp = self.mean

            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = np.sqrt(self.std)

            self.nobservations += n

from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader
import h5py
from abc import ABC, abstractmethod
from functools import partial
from copy import deepcopy
from scipy.linalg import toeplitz, solve_toeplitz

class GaussianNoiseSampler:
    """Gaussian sampler that owns covariance and can be adapted.

    This class is intended mainly for the block-diagonal covariances where each
    block is Toeplitz. It keeps track of

    - receivers: layout and component ordering
    - toeplitz_cols: first column per block (shape [n_blocks, block_size])
    - cov_blocks: full covariance blocks (shape [n_blocks, block_size, block_size])
    - Ls: Cholesky factors for each block, recomputed when covariance changes

    set_adaptive_covariance_with_misc_data(misc_data) expects misc_data to be a
    nested dict misc_data[station][component] with *actual* variance per
    station-component. It rescales each Toeplitz column so that col[0] matches
    the new variance, and updates cov_blocks and Ls accordingly.
    """

    def __init__(self, receivers, data_vector_length,
                 toeplitz_cols=None,
                 cov_blocks=None,
                 station_component_covariances=None):
        self.receivers = receivers
        self.data_vector_length = data_vector_length
        self.station_component_covariances = station_component_covariances

        # Build (station, component) list in the same order as BlockDiagonal*.
        self.receiver_components = [
            (receiver.station_name, component)
            for receiver in self.receivers.iterate()
            for component in receiver.components
        ]

        # Normalise internal representation.
        if toeplitz_cols is not None:
            self.toeplitz_cols = np.asarray(toeplitz_cols, dtype=float)
            self.cov_blocks = np.asarray(
                cov_blocks
                if cov_blocks is not None
                else [toeplitz(c) for c in self.toeplitz_cols],
                dtype=float,
            )
        elif cov_blocks is not None:
            self.cov_blocks = np.asarray(cov_blocks, dtype=float)
            # Treat diagonal as Toeplitz first column if nothing else given.
            self.toeplitz_cols = np.array(
                [block[0, :].copy() for block in self.cov_blocks],
                dtype=float,
            )
        else:
            raise ValueError("Either toeplitz_cols or cov_blocks must be provided")

        self._build_cholesky()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_cholesky(self):
        """Build Cholesky factors for all covariance blocks."""
        Ls = []
        for block in self.cov_blocks:
            # Assume blocks are already regularised / psd further up.
            Ls.append(np.linalg.cholesky(block))
        self.Ls = Ls
        self.block_sizes = [L.shape[0] for L in self.Ls]

    def _get_target_variance(self, misc_data, station, component):
        """Extract scalar variance from misc_data[station][component]."""
        try:
            val = misc_data[station][component]
        except KeyError:
            # Handle "E"/"N" vs "1"/"2" naming.
            mapped = component.replace("E", "1").replace("N", "2")
            val = misc_data[station][mapped]
        if hasattr(val, "size") and val.size > 1:
            return float(val.ravel()[0])
        return float(val)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        """Draw a sample from the current covariance.

        Returns
        -------
        noise_vector : np.ndarray, shape (n_total,)
        meta : any
            For compatibility with previous interfaces we return
            station_component_covariances as the second value,
            if available, else None.
        """
        noise_vectors = []
        for L, n in zip(self.Ls, self.block_sizes):
            z = np.random.randn(n)
            noise_vectors.append(L @ z)
        noise_vector = np.concatenate(noise_vectors)
        return noise_vector, self.station_component_covariances

    def set_adaptive_covariance_with_misc_data(self, misc_data):
        """Adapt Toeplitz columns and covariance blocks using misc_data.

        For each block (station, component), let c be its Toeplitz first
        column. We compute a scale factor s such that

            (s * c)[0] == target_variance

        where target_variance is read from misc_data[station][component]. The
        entire Toeplitz column is scaled by s, which scales the full block and
        hence preserves its correlation structure while adjusting its marginal
        variance.
        """
        if self.toeplitz_cols is None or self.cov_blocks is None:
            return

        scaled_cols = []
        for (station, comp), col in zip(self.receiver_components, self.toeplitz_cols):
            c0 = col[0]
            if c0 == 0:
                scaled_cols.append(col)
                continue
            target_var = self._get_target_variance(misc_data, station, comp)
            scale = target_var / c0
            scaled_cols.append(col * scale)
        self.toeplitz_cols = np.asarray(scaled_cols, dtype=float)

        # Rebuild covariance blocks from scaled Toeplitz columns.
        self.cov_blocks = np.asarray([toeplitz(c) for c in self.toeplitz_cols], dtype=float)
        print(self.toeplitz_cols)
        # Rebuild Cholesky factors to update the sampler dynamically.
        self._build_cholesky()


class EmpiricalCovariance(ABC):
    # emcee uses multiprocessing, which hates huge objects being passed around
    # instead we force the important data to persist in the class
    # which acts as a global variable for the multiprocessing pool
    # https://thelaziestprogrammer.com/python/multiprocessing-pool-a-global-solution
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

class ScalarEmpiricalCovariance(EmpiricalCovariance):

    def __init__(self, sigma_noise_level):
        self.noise_level = sigma_noise_level
        self.set_C_inverse(1/sigma_noise_level**2)

    @classmethod
    def generic_loss_callable(cls, residuals):
        return -0.5 * np.sum(residuals**2) * cls.C_inverse
        
    def matmul_inverse_covariance(self, data_vector):
        return data_vector / self.noise_level**2
    
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

def parallel_execution(inputs, func, num_jobs = 20):
    if num_jobs in [None, 0 , 1]:
        return [func(block) for block in inputs]
    return joblib.Parallel(n_jobs=num_jobs)(joblib.delayed(func)(block) for block in inputs)

class BlockGaussianSampler:
    def __init__(self, Ls, block_sizes, station_component_covariances):
        self.Ls = Ls
        self.block_sizes = block_sizes
        self.station_component_covariances = station_component_covariances

    def __call__(self, *args, **kwargs):
        noise_vectors = []
        for L, n in zip(self.Ls, self.block_sizes):
            z = np.random.randn(n)
            noise_vectors.append(L @ z)
        noise_vector = np.concatenate(noise_vectors)
        return noise_vector, self.station_component_covariances


class BlockDiagonalCovariance(EmpiricalCovariance):
        inverse_metadata = None

        def __init__(self, receivers, data_vector_length, block_exp_tapering = True, covariance_gradients = None, num_jobs = 20):

            self.set_data_vector_length(data_vector_length)
            self.block_exp_tapering = block_exp_tapering
            self.receivers = receivers
            self.covariance_gradients = covariance_gradients
            self.num_jobs = num_jobs

        @classmethod
        def set_toeplitz_cols(cls, inverse_metadata):
            cls.inverse_metadata = inverse_metadata

        @classmethod
        def set_data_vector_length(cls, data_vector_length):
            cls.data_vector_length = data_vector_length

        @classmethod
        def generic_loss_callable(cls, residuals, reduce=True):
            if cls.inverse_metadata is not None:
                if reduce:
                    return cls.quadratic_form(residuals, cls.inverse_metadata, cls.data_vector_length)
                else:
                    return cls.quadratic_form_per_block(residuals, cls.inverse_metadata, cls.data_vector_length)
            
            if reduce:
                 return cls.quadratic_form(residuals, cls.inverse_metadata, cls.data_vector_length)
            else:
                 return cls.quadratic_form_per_block(residuals, cls.inverse_metadata, cls.data_vector_length)

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
            # This static method was previously using C_inverse.
            # Now it delegates to quadratic_form which uses toeplitz solves.
            return BlockDiagonalCovariance.quadratic_form(residuals, toeplitz_cols, data_vector_length)
        
        @staticmethod
        def create_loss_callable(toeplitz_cols, data_vector_length):
            return partial(BlockDiagonalCovariance.loss_callable,
                            toeplitz_cols = toeplitz_cols, data_vector_length = data_vector_length)
        
        @classmethod
        def matmul_inverse_covariance(cls, data_vector):
            return cls.callable_matmul_inverse_covariance_toeplitz(data_vector, cls.inverse_metadata, cls.data_vector_length)
        
        @staticmethod
        def callable_matmul_inverse_covariance_toeplitz(data_vector, toeplitz_cols, data_vector_length):
            reshaped = data_vector.reshape(-1, data_vector_length)
            out = []
            for c, x in zip(toeplitz_cols, reshaped):
                out.append(solve_toeplitz((c, c), x))
            return np.concatenate(out)

        def matrix_vector_product(self, matrix, data_vector):
            reshaped_vector = data_vector.reshape(-1, self.data_vector_length)
            # return np.matvec(matrix, reshaped_vector).reshape(-1)
            return np.einsum('ijk,ik->ij', matrix, reshaped_vector).reshape(-1)
        
        def matrix_matrix_product(self, matrix1, matrix2):
            # return np.matmul(matrix1, matrix2)
            return np.einsum('ijk,ikl->ijl', matrix1, matrix2)

        def vector_vector_dot_product(self, vector1, vector2):
            return np.dot(vector1, vector2)
            # reshaped_vector1 = vector1.reshape(-1, self.data_vector_length)
            # reshaped_vector2 = vector2.reshape(-1, self.data_vector_length)
            # return np.einsum('ij,ij->', reshaped_vector1, reshaped_vector2)
    
        def compute_trace(self, matrix):
            return np.trace(matrix, axis1=1, axis2=2).sum()

        @staticmethod
        def create_matmul_inverse_covariance(toeplitz, data_vector_length):
            return partial(BlockDiagonalCovariance.callable_matmul_inverse_covariance_toeplitz, 
                           toeplitz_cols = toeplitz, data_vector_length = data_vector_length)
        
        def create_sampler(self, scale=1e9):

            if hasattr(self, 'covariance_matrix_arrays') and self.covariance_matrix_arrays is not None:
                # Full covariance blocks already present (e.g. empirical/theory).
                cov_blocks = [block for block in self.covariance_matrix_arrays]
                sampler = GaussianNoiseSampler(
                    receivers=self.receivers,
                    data_vector_length=self.data_vector_length,
                    cov_blocks=cov_blocks,
                    station_component_covariances=getattr(self, 'station_component_covariances', None),
                )
            elif getattr(self, 'toeplitz_cols_list', None) is not None:
                # Build from stored Toeplitz columns.
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
        print([self.toeplitz_cols_list[i][0] for i in range(len(self.toeplitz_cols_list))], flush=True)
    
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
        
        # Jitter / shrinkage
        gamma_vals[0] += 0.01 * gamma_vals[0]
        
        return gamma_vals
    
    def create_toeplitz_cols(self, station_component_covariances):
        # build full list with comphrension
        receiver_components_list = [(receiver.station_name, component) for receiver in self.receivers.iterate() for component in receiver.components]
        data_vector_length = self.data_vector_length
        freqs= self.freqs
        if isinstance(station_component_covariances, dict):
            def compute_covariance(station_name_components):
                station_name, component = station_name_components
                try:
                    sigma_sqr = station_component_covariances[station_name][component]
                except KeyError:
                    component = component.replace('E', '1').replace('N', '2')
                    sigma_sqr = station_component_covariances[station_name][component]
                cov_y = self.build_single_covariance_column(sigma_sqr, data_vector_length, freqs)
                return cov_y
            covariance_blocks = parallel_execution(receiver_components_list, compute_covariance, self.num_jobs)
        else:
            sigma_sqr = station_component_covariances**2
            builder = lambda _: self.build_single_covariance_column(sigma_sqr=sigma_sqr, data_vector_length=data_vector_length, freqs=freqs)
            covariance_blocks = parallel_execution(receiver_components_list, builder, self.num_jobs)
        return np.array(covariance_blocks)

class BlockDiagonalKolbCovariance(BlockDiagonalCovariance):
    """
    Block diagonal covariance with Kolb structure:
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
        
        # Jitter for stability
        gamma_vals[0] += 1e-2 *  gamma_vals[0]
        
        return gamma_vals

    def create_toeplitz_cols(self, station_component_covariances):
        receiver_components_list = [(receiver.station_name, component) for receiver in self.receivers.iterate() for component in receiver.components]
        data_vector_length = self.data_vector_length

        if isinstance(station_component_covariances, dict):
            def compute_covariance(station_name_components):
                station_name, component = station_name_components
                try:
                    # We assume the value in the dict is the variance (sigma^2) or raw data from which we take variance
                    val = station_component_covariances[station_name][component]
                except KeyError:
                    component = component.replace('E', '1').replace('N', '2')
                    val = station_component_covariances[station_name][component]
                
                # If val is an array (like raw noise or autocorrelation), take its value at lag 0 as variance
                if isinstance(val, np.ndarray):
                    sigma_sqr = val[0] if val.ndim > 0 else val.item()
                else:
                    sigma_sqr = val

                return self.build_single_covariance_column(sigma_sqr, data_vector_length)
            
            covariance_blocks = parallel_execution(receiver_components_list, compute_covariance, self.num_jobs)
        else:
            # If passed as a single scalar or array of scalars
            sigma_sqr = station_component_covariances**2
            builder = lambda _: self.build_single_covariance_column(sigma_sqr=sigma_sqr, data_vector_length=data_vector_length)
            covariance_blocks = parallel_execution(receiver_components_list, builder, self.num_jobs)
            
        return np.array(covariance_blocks)

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
            # build full list with comphrension
            receiver_components_list = [(receiver.station_name, component) for receiver in self.receivers.iterate() for component in receiver.components]
            if self.block_exp_tapering:
                station_component_covariances = EmpiricalCovarianceEstimator.taper_covariances(station_component_covariances, self.data_vector_length, fit_length=20, ols_fit=False)
            def compute_covariance(station_name_components):
                    station_name, component = station_name_components
                    try:
                        covar_data = station_component_covariances[station_name][component]
                    except KeyError:
                        component = component.replace('E', '1').replace('N', '2')
                        covar_data = station_component_covariances[station_name][component]
                    covar_data = covar_data[:self.data_vector_length]
                    
                    # Jitter
                    c = covar_data.copy()
                    # c[0] += 1e-8 * max(1.0, c[0])
                    
                    return c
            covariance_blocks = parallel_execution(receiver_components_list, compute_covariance, self.num_jobs)
            return np.array(covariance_blocks)
    
from scipy.linalg import cho_factor, cho_solve

# def stable_inverse(C, eps=1e-18):
#     # ensure symmetry
#     C = (C + C.T) / 2
#     # add jitter for numerical stability
#     C = C + np.eye(C.shape[0]) * eps
#     c, low = cho_factor(C, lower=True, check_finite=False)
#     return cho_solve((c, low), np.eye(C.shape[0]), check_finite=False)

def stable_inverse(C, eps=1e-18):
    # ensure symmetry
    return np.linalg.inv(C)

import numpy as np
from scipy.linalg import cho_factor, cho_solve

class TheoryBlockDiagonalEmpiricalCovariance(BlockDiagonalCovariance):
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

    # ------------------------------------------------------------------
    # Covariance construction
    # ------------------------------------------------------------------

    @classmethod
    def set_cholesky_factors(cls, cholesky_factors):
        cls.inverse_metadata = cholesky_factors

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

        # # Optional derivatives
        # if hasattr(station_component_covariances, "data_parameter_gradients"):
        #     self.C_derivative = self.create_C_derivative(
        #         station_component_covariances
        #     )
        #     # self.set_covariance_constants()

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
    # ------------------------------------------------------------------
    # Quadratic forms
    # ------------------------------------------------------------------
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

    @classmethod
    def generic_loss_callable(cls, residuals, reduce=True):
        if reduce:
            return cls.quadratic_form(
                residuals, cls.inverse_metadata, cls.data_vector_length
            )
        return cls.quadratic_form_per_block(
            residuals, cls.inverse_metadata, cls.data_vector_length
        )
    
    @staticmethod
    def loss_callable(residuals, toeplitz_cols, data_vector_length):
        # This static method was previously using C_inverse.
        # Now it delegates to quadratic_form which uses toeplitz solves.
        return TheoryBlockDiagonalEmpiricalCovariance.quadratic_form(residuals, toeplitz_cols, data_vector_length)

    @staticmethod
    def create_loss_callable(toeplitz_cols, data_vector_length):
        return partial(TheoryBlockDiagonalEmpiricalCovariance.loss_callable,
                        toeplitz_cols = toeplitz_cols, data_vector_length = data_vector_length)
    # ------------------------------------------------------------------
    # Inverse covariance × vector
    # ------------------------------------------------------------------

    @classmethod
    def matmul_inverse_covariance(cls, data_vector):
        reshaped = data_vector.reshape(-1, cls.data_vector_length)
        out = []
        for cf, x in zip(cls.inverse_metadata, reshaped):
            out.append(cho_solve(cf, x, check_finite=False))
        return np.concatenate(out)

    # ------------------------------------------------------------------
    # Gradient kernels & traces
    # ------------------------------------------------------------------

    def set_covariance_constants(self):
        self.kernels = []
        self.traces = []
        for a in range(self.C_derivative.shape[0]):
            cov_kernel = self.matrix_matrix_product(self.C_inverse, self.matrix_matrix_product(self.C_derivative[a], self.C_inverse))
            trace = self.compute_trace(self.matrix_matrix_product(self.C_inverse, self.C_derivative[a]))
            self.kernels.append(cov_kernel)
            self.traces.append(trace)

class EmpiricalCovarianceEstimator:

    def __init__(self, data_directory, receivers, components, track = False, covariance_exp_tapering = True):
        self.data_directory = data_directory
        self.receivers = receivers
        self.components = components.replace('E', '1').replace('N', '2')
        self.covariance_exp_tapering = covariance_exp_tapering

        self.track = track

        self.data_loader = SimulationDataLoader(self.components, receivers)

        self._precomputed_covariance_path = self.data_directory / f'_{self.components}_covariance_matrix.npy'
    
    def compute_stationwise_covariances(self, reload = False):
        if self._precomputed_covariance_path.exists() and not reload:
            print("Loading precomputed covariance matrix...", end=' ')
            station_component_covariances =  np.load(self._precomputed_covariance_path, allow_pickle=True)[()]
            print("Done.")
        else:
            print("Computing empirical covariance matrix...", end=' ')
            station_component_deviations = self._compute_standard_deviation_online()
                                # compute autocorrelation
                                # auto_correlate = np.correlate(noise_window_data, noise_window_data, mode='full')
            station_component_covariances = self.convert_to_covariance(station_component_deviations)
            if self.covariance_exp_tapering:
                station_component_covariances = self.taper_covariances(station_component_covariances)
            np.save(self._precomputed_covariance_path, station_component_covariances)
            print("Done.")

        return station_component_covariances

    def _compute_standard_deviation_online(self):
        station_component_covariances = {receiver.station_name:{component:RunningStandardDeviations(track=self.track) for component in self.components} for receiver in self.receivers.iterate()}
        for noise_file in self.data_directory.glob('*.h5'):
            with h5py.File(noise_file) as f:
                for receiver in self.receivers.iterate():
                    receiver_name = receiver.station_name
                    for component in self.components:
                            try:
                                noise_window_data = f["outputs"][receiver_name][component][:]
                            except KeyError:
                                print(f"Warning: Could not find {receiver_name} {component} in {noise_file}.")
                                continue
                            data_length = noise_window_data.shape[0]

                            # compute autocorrelation
                            auto_correlate = np.correlate(noise_window_data, noise_window_data, mode='full')
                            averaged_auto_correlations = auto_correlate[:data_length][::-1]/np.arange(data_length, 0, -1)

                            auto_cov = averaged_auto_correlations.reshape(1,-1)

                            station_component_covariances[receiver_name][component].update(auto_cov)
        return station_component_covariances
    

    def convert_to_covariance(self, tracked_covariances):

        station_component_covariances = {}
        for receiver in tracked_covariances.keys():
            station_component_covariances[receiver] = {}
            for component in tracked_covariances[receiver].keys():
                try:
                    covar_data = tracked_covariances[receiver][component].mean
                except AttributeError:
                    print(receiver, component)
                station_component_covariances[receiver][component] = covar_data

        return station_component_covariances
    @staticmethod
    def taper_covariances(station_component_covariances, data_len, fit_length=30, ols_fit=True, return_fit= False):
        for receiver in station_component_covariances.keys():
            for component in station_component_covariances[receiver].keys():
                covar_data = station_component_covariances[receiver][component]
                x = np.arange(0, data_len)

                if ols_fit:
                    scaled_data = np.log(np.abs(covar_data))
                    scaled_data -= scaled_data[0]

                    model = sm.OLS(scaled_data[:fit_length], x[:fit_length])
                    results = model.fit()
                    gradient = results.params[0]
                else:
                    gradient = -1/fit_length
                covar_data = covar_data * np.exp(gradient * x)
                if not return_fit:
                    station_component_covariances[receiver][component] = covar_data
                else:
                    station_component_covariances[receiver][component] = (covar_data[0], -gradient)

        return station_component_covariances

def build_cov_sigma2_dict(station_component_covariances):
    """
    For filtered covariance, BlockDiagonalFilteredCovariance expects either a dict with sigma^2
    per station-component or a scalar. We pass the lag-0 variance.
    """
    sigma2_dict = {}
    for station, comp_dict in station_component_covariances.items():
        sigma2_dict[station] = {}
        for component, cov_vec in comp_dict.items():
            if hasattr(cov_vec, "__len__") and len(cov_vec) > 0:
                sigma2 = float(cov_vec[0])
            else:
                sigma2 = float(cov_vec)
            sigma2_dict[station][component] = sigma2
    return sigma2_dict

