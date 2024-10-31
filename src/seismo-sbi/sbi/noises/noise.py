import numpy as np

class NoisySimulator:

    def __init__(self, simulator_function, noise_source):

        self.simulator_function = simulator_function
        self.noise_source = noise_source
    
    def simulate(self, simulation_args, noise_args, seed):
        return self.simulator_function(simulation_args) + self.noise_source(noise_args)

def gaussian_noise(scale, size, seed):
    np.random.seed(seed)
    return np.random.normal(loc=0, scale = scale, size=size)