from pathlib import Path
import numpy as np
import obspy
import os

from seismo_sbi.sbi.configuration import SimulationParameters
from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader


class RealNoiseSampler:

    # TODO: add components implementation

    def __init__(self, simulation_parameters : SimulationParameters, directory, data_length = None, adaptive_covariance= None):
        receivers = simulation_parameters.receivers
        self.num_stations = len(receivers.receivers)
        self.components = simulation_parameters.components
        self.components = self.components.replace('E', '1').replace('N', '2')
        self.vector_length = round(simulation_parameters.seismogram_duration * simulation_parameters.sampling_rate)

        self.data_loader = SimulationDataLoader(self.components, simulation_parameters.receivers, data_length)

        self.noise_paths = self._find_noise_paths(directory)
        np.random.shuffle(self.noise_paths)

        self.adaptive_covariance = adaptive_covariance
        if self.adaptive_covariance is not None:
            for receiver in self.adaptive_covariance.keys():
                for component in self.adaptive_covariance[receiver].keys():
                    self.adaptive_covariance[receiver][component] = self.adaptive_covariance[receiver][component][0]

        print(f"Found {len(self.noise_paths)} noise realisations.")
    
    def _find_noise_paths(self, directory):
        return np.array(list(Path(directory).glob('*.h5')))

    def __call__(self, noise_path = None, no_rescale = False, noise_index = None):
        
        # if self.noise_index_counter == len(self.noise_paths):
        #     self.noise_index_counter = 0
        #     np.random.shuffle(self.noise_paths)
        if noise_index is not None:
            noise_path = self.noise_paths[noise_index]
        if noise_path is None:
            noise_index = np.random.randint(0, len(self.noise_paths))
            noise_path = self.noise_paths[noise_index]
        try:
            noise_realisations = self._load_noise_file(noise_path)
        except KeyError:
            return self.__call__(noise_path = None, no_rescale = no_rescale, noise_index=noise_index +1)
        # self.noise_index_counter += 1

        if no_rescale:
            misc_data = self.data_loader.load_misc_data(noise_path)
            noise_realisations = self._load_noise_file(noise_path)
            return noise_realisations, misc_data
        elif self.adaptive_covariance is  None:
            noise_realisations = self._load_noise_file(noise_path)
            return noise_realisations
        else:
            misc_data = self.data_loader.load_misc_data(noise_path)
            scales = self.calculate_scales(misc_data)
            noise_realisations = self._load_noise_file(noise_path, scale_dict = scales)
            return noise_realisations, misc_data


    
    def _load_noise_file(self, path : Path, *args, **kwargs):
        return self.data_loader.load_flattened_simulation_vector(path, *args, **kwargs)
    
    def calculate_scales(self, misc_data):
        scales = {}
        for receiver in misc_data.keys():
            scales[receiver] = {}
            for component in misc_data[receiver].keys():
                noise_instance_variance = misc_data[receiver][component] if misc_data[receiver][component].size == 1 else misc_data[receiver][component][0]
                try:
                    data_noise_variance = self.adaptive_covariance[receiver][component]
                except KeyError:
                    new_comp = component.replace('E', '1').replace('N', '2')
                    data_noise_variance = self.adaptive_covariance[receiver][new_comp]
  

                ratio = noise_instance_variance / data_noise_variance
                scales[receiver][component] = ratio

        return scales
    
    def set_adaptive_covariance_with_misc_data(self, misc_data):
        adaptive_covariance = {}
        for receiver in misc_data.keys():
            adaptive_covariance[receiver] = {}
            for component in misc_data[receiver].keys():
                noise_instance_variance = misc_data[receiver][component] if misc_data[receiver][component].size == 1 else misc_data[receiver][component][0]
                adaptive_covariance[receiver][component] = [noise_instance_variance]
            
        self.adaptive_covariance = adaptive_covariance