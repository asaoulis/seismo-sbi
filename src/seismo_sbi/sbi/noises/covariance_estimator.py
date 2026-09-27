"""Empirical noise autocovariances estimated from windows of recorded noise.

``EmpiricalCovarianceEstimator`` averages per-trace autocovariances over a directory of noise
windows (h5), optionally with an exponential taper; ``build_cov_sigma2_dict`` reduces them to
lag-0 variances.
"""
import h5py
import numpy as np
import statsmodels.api as sm

from seismo_sbi.simulators.simulation_io import SimulationDataLoader, component_alias


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


def stable_inverse(C, eps=1e-18):
    return np.linalg.inv(C)


class EmpiricalCovarianceEstimator:
    """Estimate per-trace noise autocovariances from a directory of recorded noise windows."""


    def __init__(self, data_directory, receivers, components, track = False, covariance_exp_tapering = True):
        self.data_directory = data_directory
        self.receivers = receivers
        self.components = component_alias(components)
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
    def taper_covariances(station_component_covariances, data_len=None, fit_length=30, ols_fit=True, return_fit= False):
        """Exponentially taper each autocovariance in place; ``data_len`` defaults to each one's length."""
        for receiver in station_component_covariances.keys():
            for component in station_component_covariances[receiver].keys():
                covar_data = station_component_covariances[receiver][component]
                x = np.arange(0, len(covar_data) if data_len is None else data_len)

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
