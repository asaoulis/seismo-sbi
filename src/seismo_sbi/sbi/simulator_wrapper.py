"""Wire a configured forward model into the inference pipeline.

:class:`GeneralSimulatorWrapper` builds the simulator the configuration asks for, folds in the
nuisance effects staged at simulation time, and exposes the two callables the pipeline uses: one
that returns a flat data vector for a parameter vector, and one that writes a simulation to disk.
"""

from functools import partial
from copy import deepcopy

from seismo_sbi.simulators.post_processing import build_post_processing_chain
from seismo_sbi.simulators.registry import build_simulator
from seismo_sbi.simulators.simulation_io import SimulationDataLoader
from seismo_sbi.sbi.configuration import ModelParameters, SimulationParameters


class GeneralSimulatorWrapper:

    def __init__(self, simulation_parameters: SimulationParameters,  parameters, data_loader, samplers):

        default_config = (simulation_parameters.simulation_type, None)
        self.set_simulation_objects(default_config, simulation_parameters, parameters, data_loader, samplers)
        self.data_loader_callable = data_loader.convert_sim_data_to_array
        self.generic_simulation_callable = deepcopy(self.simulation_callable)
        self.generic_simulation_save_callable = deepcopy(self.simulation_save_callable)


    def set_simulation_objects(self, simulator_config, simulation_parameters, parameters, data_loader, samplers):

        # Copied so the parsed configuration is not mutated.
        effect_configs = dict(getattr(parameters, 'nuisance_effect_config', {}))

        # Nuisances staged "training_augmentation" are folded in per batch by the dataloader.
        nuisance_stage = getattr(parameters, 'nuisance_stage', {})
        sim_staged_keys = [
            key for key in parameters.nuisance.keys()
            if nuisance_stage.get(key, "simulation") == "simulation"
        ]

        # Shift-based effects need the sampling rate to turn seconds into samples.
        for _shift_key in ('time_shift_error', 'azimuthal_anisotropy',
                           'shear_wave_splitting', 'dispersion_spread'):
            if _shift_key in sim_staged_keys:
                effect_configs[_shift_key] = dict(
                    effect_configs.get(_shift_key, {})
                )
                effect_configs[_shift_key]['sampling_rate'] = (
                    simulation_parameters.sampling_rate
                )

        post_processing_effects = list(
            build_post_processing_chain(sim_staged_keys, effect_configs).effects
        )
        self.simulator = self.select_and_initialise_simulator(
            simulator_config, simulation_parameters, post_processing_effects=post_processing_effects
        )

        self.simulation_save_callable = self.simulator.execute_sim_and_save_outputs

        self.simulation_callable = partial(self.input_output_simulation, parameters, data_loader, samplers, self.simulator)

    def select_and_initialise_simulator(self, simulator_config, simulation_parameters, post_processing_effects=None):
        return build_simulator(simulator_config, simulation_parameters, post_processing_effects,
                               data_flattening=getattr(self, "data_loader_callable", None))

    def create_input_output_simulation_callable(self, parameters, data_loader, samplers):
        return partial(self.input_output_simulation, parameters, data_loader, samplers)

    def input_output_simulation(self, parameters : ModelParameters, data_loader : SimulationDataLoader, samplers, simulator, theta, **kwargs):
        if len(theta.shape) == 1:
            theta = theta.reshape(1,-1)
        theta_fiducial_map = parameters.vector_to_parameters(theta[0], 'theta_fiducial')
        sampled_nuisance = {key: next(samplers[key](1)) for key in parameters.nuisance.keys()}
        inputs_map = {**theta_fiducial_map, **sampled_nuisance, **kwargs}
        return data_loader.convert_sim_data_to_array(
                    {"outputs": simulator.run_simulation(inputs_map)[1]}
                ).flatten()