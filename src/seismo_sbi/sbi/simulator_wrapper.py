from functools import partial
from sbi import utils as utils
from sbi import analysis as analysis
from copy import deepcopy

from seismo_sbi.instaseis_simulator.simulator import InstaseisSourceSimulator, FixedLocationKernelSimulator
from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader
from seismo_sbi.sbi.configuration import  ModelParameters, SimulationParameters

class GeneralSimulatorWrapper:

    def __init__(self, simulation_parameters: SimulationParameters,  parameters, data_loader, samplers):

        # in future, this can be extended for aribitrary forward models
        default_config = ('instaseis', None)
        self.set_simulation_objects(default_config, simulation_parameters, parameters, data_loader, samplers)

        self.generic_simulation_callable = deepcopy(self.simulation_callable)
        self.generic_simulation_save_callable = deepcopy(self.simulation_save_callable)


    def set_simulation_objects(self, simulator_config, simulation_parameters, parameters, data_loader, samplers):

        self.simulator = self.select_and_initialise_simulator(simulator_config, simulation_parameters)

        self.simulation_save_callable = self.simulator.execute_sim_and_save_outputs

        self.simulation_callable = partial(self.input_output_simulation, parameters, data_loader, samplers, self.simulator)

    def select_and_initialise_simulator(self, simulator_config, simulation_parameters):
        if simulator_config[0] == 'instaseis':
            simulator = InstaseisSourceSimulator(simulation_parameters.syngine_address, 
                                        components= simulation_parameters.components, 
                                        receivers = simulation_parameters.receivers,
                                        seismogram_duration_in_s = simulation_parameters.seismogram_duration,
                                        synthetics_processing = simulation_parameters.processing)
        elif simulator_config[0] == 'kernel':
            score_compression_data = simulator_config[1]
            simulator = FixedLocationKernelSimulator(score_compression_data,
                            components= simulation_parameters.components, 
                            receivers = simulation_parameters.receivers,
                            seismogram_duration_in_s = simulation_parameters.seismogram_duration,
                            synthetics_processing = simulation_parameters.processing)
        else:
            raise NotImplementedError(f"Simulator {simulator_config[0]} not implemented")
        return simulator
    
    def create_input_output_simulation_callable(self, parameters, data_loader, samplers):
        return partial(self.input_output_simulation, parameters, data_loader, samplers)

    def input_output_simulation(self, parameters : ModelParameters, data_loader : SimulationDataLoader, samplers, simulator, theta):
        if len(theta.shape) == 1:
            theta = theta.reshape(1,-1)
        theta_fiducial_map = parameters.vector_to_parameters(theta[0], 'theta_fiducial')
        sampled_nuisance = {key: next(samplers[key](1)) for key in parameters.nuisance.keys()}
        inputs_map = {**theta_fiducial_map, **sampled_nuisance}
        return data_loader.convert_sim_data_to_array(
                    {"outputs": simulator.run_simulation(inputs_map)[1]}
                ).flatten()