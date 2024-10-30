import numpy as np
import statsmodels.api as sm
import joblib
import os


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

from src.instaseis_simulator.dataloader import SimulationDataLoader
import h5py
from abc import ABC, abstractmethod
from functools import partial
from copy import deepcopy

class GaussianNoiseSampler:
    def __init__(self, sampler):
        self.sampler = sampler
    
    def __call__(self, *args, **kwds):
        return self.sampler(*args, **kwds)
    
    def set_adaptive_covariance_with_misc_data(self, misc_data):
        pass

class EmpiricalCovariance(ABC):
    # emcee uses multiprocessing, which hates huge objects being passed around
    # instead we force the important data to persist in the class
    # which acts as a global variable for the multiprocessing pool
    # https://thelaziestprogrammer.com/python/multiprocessing-pool-a-global-solution
    C_inverse = None
    data_vector_length = None

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
        sampling_object = GaussianNoiseSampler(sampler)
        return sampling_object

class DiagonalEmpiricalCovariance(EmpiricalCovariance):

    def __init__(self,station_component_covariances, receivers, data_vector_length):
        self.receivers = receivers
        self.data_vector_length = data_vector_length
        self.station_component_covariances = station_component_covariances
        self.covariance_matrix = self.create_covariance_matrix(station_component_covariances, data_vector_length)
        self.set_C_inverse(1/self.covariance_matrix)

    def create_covariance_matrix(self, station_component_covariances, data_vector_length):

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
        return np.concatenate(covariance_matrix_diagonals) if len(covariance_matrix_diagonals) > 1 else covariance_matrix_diagonals[0][np.newaxis]

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
        sampling_object = GaussianNoiseSampler(sampler)
        return sampling_object

from scipy.linalg import toeplitz

def parallel_execution(inputs, func, num_jobs = 20):
    if num_jobs in [None, 0 , 1]:
        return [func(block) for block in inputs]
    return joblib.Parallel(n_jobs=num_jobs)(joblib.delayed(func)(block) for block in inputs)

class BlockDiagonalEmpiricalCovariance(EmpiricalCovariance):
        
        data_vector_length = None
    
        def __init__(self, station_component_covariances, receivers, data_vector_length, block_exp_tapering = True, num_jobs = 20):

            self.set_data_vector_length(data_vector_length)
            self.block_exp_tapering = block_exp_tapering
            self.receivers = receivers
            self.num_jobs = num_jobs
            self.station_component_covariances = deepcopy(station_component_covariances)
            self.covariance_matrix_arrays = self.create_covariance_matrix(station_component_covariances)
            self.num_jobs = num_jobs
            self.set_C_inverse(np.array(parallel_execution(self.covariance_matrix_arrays, np.linalg.inv, num_jobs)))

        @classmethod
        def set_data_vector_length(cls, data_vector_length):
            cls.data_vector_length = data_vector_length
    
        def create_covariance_matrix(self, station_component_covariances):
            # build full list with comphrension
            receiver_components_list = [(receiver.station_name, component) for receiver in self.receivers.iterate() for component in receiver.components]
            if self.block_exp_tapering:
                station_component_covariances = EmpiricalCovarianceEstimator.taper_covariances(station_component_covariances, fit_length=25, ols_fit=False)
            def compute_covariance(station_name_components):
                    station_name, component = station_name_components
                    try:
                        covar_data = station_component_covariances[station_name][component]
                    except KeyError:
                        component = component.replace('E', '1').replace('N', '2')
                        covar_data = station_component_covariances[station_name][component]
                    covar_data = covar_data[:self.data_vector_length]
                    if self.block_exp_tapering:
                        cov_matrix_block = toeplitz(covar_data)

                    else:
                        eigvals, eigvecs = np.linalg.eig(toeplitz(covar_data))
                        eigvals[np.where(eigvals < 0)]=0
                        cov_matrix_block = np.dot(eigvecs, np.dot(np.diag(eigvals), np.linalg.inv(eigvecs)))
                    
                    if np.any(np.linalg.eigvals(cov_matrix_block) < 0):
                        print("Warning: Covariance matrix block is not positive definite.", np.linalg.cond(cov_matrix_block))
                        
                    return cov_matrix_block
            covariance_blocks = parallel_execution(receiver_components_list, compute_covariance, self.num_jobs)
            return covariance_blocks
        
        @classmethod
        def generic_loss_callable(cls, residuals, reduce=True):
            if reduce:
                loss = BlockDiagonalEmpiricalCovariance.loss_callable(residuals, cls.C_inverse, cls.data_vector_length)
            else:
                reshaped_residuals = residuals.reshape(-1, cls.data_vector_length)
                loss = -0.5 * np.einsum('ij,ij->i', reshaped_residuals, np.einsum('ijk,ik->ij', cls.C_inverse, reshaped_residuals))
                loss = loss.repeat(cls.data_vector_length).reshape(-1)
            return loss
        
        @staticmethod
        def loss_callable(residuals, C_inverse, data_vector_length):
            reshaped_residuals = residuals.reshape(-1, data_vector_length)
            loss = -0.5 * np.einsum('ij,ij->', reshaped_residuals, np.einsum('ijk,ik->ij', C_inverse, reshaped_residuals))
            return loss
        
        @staticmethod
        def create_loss_callable(C_inverse, data_vector_length):
            return partial(BlockDiagonalEmpiricalCovariance.loss_callable,
                            C_inverse = C_inverse, data_vector_length = data_vector_length)
        
        @classmethod
        def matmul_inverse_covariance(cls, data_vector):
            return cls.callable_matmul_inverse_covariance(data_vector, cls.C_inverse, cls.data_vector_length)
        
        @staticmethod
        def callable_matmul_inverse_covariance(data_vector, C_inverse, data_vector_length):
            reshaped_vector = data_vector.reshape(-1, data_vector_length)
            return np.einsum('ijk,ik->ij', C_inverse, reshaped_vector).reshape(-1)

        
        @staticmethod
        def create_matmul_inverse_covariance(C_inverse, data_vector_length):
            return partial(BlockDiagonalEmpiricalCovariance.callable_matmul_inverse_covariance, 
                           C_inverse = C_inverse, data_vector_length = data_vector_length)
        
        def create_sampler(self):
            def sampler(*args, **kwargs):
                # extremely slow so not being used
                noise_vectors = []
                for block in self.covariance_matrix_arrays:
                    noise_vectors.append(
                        np.random.multivariate_normal(np.zeros(block.shape[0]), block, size=1).flatten()
                    ) 
                noise_vector = np.concatenate(noise_vectors)
                return noise_vector, self.station_component_covariances
            sampling_object = GaussianNoiseSampler(sampler)
            return sampling_object
            


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
    def taper_covariances(station_component_covariances, fit_length=30, ols_fit=True, return_fit= False):
        for receiver in station_component_covariances.keys():
            for component in station_component_covariances[receiver].keys():
                covar_data = station_component_covariances[receiver][component]
                x = np.arange(0, len(covar_data))

                if ols_fit:
                    scaled_data = np.log(np.abs(covar_data))
                    scaled_data -= scaled_data[0]

                    model = sm.OLS(scaled_data[:fit_length], x[:fit_length])
                    results = model.fit()
                    gradient = results.params[0]
                else:
                    gradient = -1/fit_length
                covar_data = covar_data[0] * np.exp(gradient * x)
                if not return_fit:
                    station_component_covariances[receiver][component] = covar_data
                else:
                    station_component_covariances[receiver][component] = (covar_data[0], -gradient)

        return station_component_covariances