
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from seismo_sbi.sbi.configuration import ModelParameters


class SymmetricLogScaler:
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def transform(self, X):
        sign = np.sign(X)
        X_abs = np.abs(X)
        X_clipped = np.clip(X_abs, self.lower_bound, self.upper_bound)
        X_log = np.log10(X_clipped)
        X_scaled = sign * (X_log - np.log10(self.lower_bound)) / (np.log10(self.upper_bound) - np.log10(self.lower_bound))
        X_scaled = X_scaled * 0.5 + 0.5
        return X_scaled

    def inverse_transform(self, X_scaled):
        X_scaled = (X_scaled - 0.5 ) / 0.5
        sign = np.sign(X_scaled)
        X_abs_scaled = np.abs(X_scaled)
        X_unscaled = np.power(10, (X_abs_scaled * (np.log10(self.upper_bound) - np.log10(self.lower_bound))) + np.log10(self.lower_bound))
        X_clipped = sign* np.clip(X_unscaled, self.lower_bound, self.upper_bound)
        return X_clipped
    
class LinearSymmetricLogScaler:
    def __init__(self, lower_bound, upper_bound, linear_range = 0.05):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.linear_range = linear_range

    def transform(self, X):
        sign = np.sign(X)
        X_abs = np.abs(X)
        X_clipped = np.clip(X_abs, 0, self.upper_bound)
        
        X_linear = np.where(X_clipped <= self.lower_bound, 
                            (X_clipped / self.lower_bound) * self.linear_range, 
                            (0.5 - self.linear_range) * (np.log10(X_clipped) - np.log10(self.lower_bound)) / (np.log10(self.upper_bound) - np.log10(self.lower_bound)))
        
        X_scaled = sign * X_linear + 0.5
        return X_scaled

    def inverse_transform(self, X_scaled):
        X_scaled = (X_scaled - 0.5)
        sign = np.sign(X_scaled)
        X_abs_scaled = np.abs(X_scaled)
        # print(np.max((((X_abs_scaled / (0.5 - self.linear_range) )* (np.log10(self.upper_bound) - np.log10(self.lower_bound))) + np.log10(self.lower_bound))))
        
        X_inverse = np.where(X_abs_scaled <= self.linear_range, 
                             (X_abs_scaled / self.linear_range) * self.lower_bound, 
                             10 ** (((X_abs_scaled / (0.5 - self.linear_range) )* (np.log10(self.upper_bound) - np.log10(self.lower_bound))) + np.log10(self.lower_bound)))

        X_unscaled = sign * np.clip(X_inverse, 0, self.upper_bound)
        return X_unscaled


class ZeroOneScaler:
    def __init__(self, bounds):
        self.bounds = bounds
        self.lower_bound = bounds[0]
        self.upper_bound = bounds[1]
        self.range = self.upper_bound - self.lower_bound

    def transform(self, X):
        X_scaled = (X - self.lower_bound) / self.range
        return X_scaled

    def inverse_transform(self, X_scaled):
        X = X_scaled * self.range + self.lower_bound
        return X

class FlexibleScaler:

    def __init__(self, parameters : ModelParameters):

        self.indices = []
        self.scalers = []
        self.index_to_param_type = {}
        index = 0
        self.n_features_in_ = parameters.parameter_to_vector('theta_fiducial').shape[0]

        for param_type, params in parameters.theta_fiducial.items():
            # if param_type in ["moment_tensor"] and len(np.array(parameters.bounds["moment_tensor"]).shape) == 1:
            #     bounds = parameters.bounds["moment_tensor"]
            #     # self.scalers.append( SymmetricLogScaler(bounds[0]*0.01, bounds[1]))
            #     self.scalers.append( LinearSymmetricLogScaler(bounds[0] * 0.1, bounds[1]))

            #     self.indices.append((index, index + len(params)))
            #     for i in range(len(params)):
            #         self.index_to_param_type[index + i] = param_type
            # else:
            scaler = ZeroOneScaler(np.array(parameters.bounds[param_type]))
            # scaler.fit(compressed_dataset[:, index:index + len(params)])
            self.scalers.append(scaler )
            self.indices.append((index, index + len(params)))
            for i in range(len(params)):
                self.index_to_param_type[index + i] = param_type
            index += len(params)
                
    
    def transform(self, X):
        X_scaled = np.zeros_like(X)
        for (start, end), scaler in zip(self.indices, self.scalers):
            X_scaled[:, start:end] = scaler.transform(X[:, start:end])
        
        return X_scaled
    
    def inverse_transform(self, X_scaled):
        X = np.zeros_like(X_scaled)
        for (start, end), scaler in zip(self.indices, self.scalers):
            X[:, start:end] = scaler.inverse_transform(X_scaled[:, start:end])

        return X
    

class GeneralScaler:
    def __init__(self, raw_compressed_dataset):
        self.log_scaler = SymmetricLogScaler(5*np.min(np.abs(raw_compressed_dataset), axis=0), np.max(np.abs(raw_compressed_dataset),axis=0))
 
        first_transform = self.log_scaler.transform(raw_compressed_dataset)
        self.scaler = MinMaxScaler()
        self.scaler.fit(first_transform)

    def transform(self, X):
        first_transform = self.log_scaler.transform(X)
        return self.scaler.transform(first_transform)
    
    # def inverse_transform(self, X):
    #     # pad X to length 35
    #     X = np.concatenate([X, np.zeros((X.shape[0], self.num_statistics))], axis=1)
    #     first_transform = self.scaler.inverse_transform(X)
    #     return self.log_scaler.inverse_transform(first_transform)[:, :self.n_features_in_]