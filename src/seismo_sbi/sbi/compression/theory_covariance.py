import numpy as np

from ...instaseis_simulator.simulator import Simulator, CPSPrecomputedSimulator

class CPSTheoryCovarianceEstimationSimulator(Simulator):
    """
    A class to simulate the covariance matrix based on theoretical models.
    This class is designed
    to work with theoretical covariance models and can be extended
    to include more complex models in the future.
    """

    def __init__(self, simulator : CPSPrecomputedSimulator, data_flattening, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulator = simulator
        self.num_realisations = simulator.num_models
        self.data_flattening_callable = data_flattening
        self.num_traces = len([comp for rec in self.simulator.receivers.iterate() for comp in rec.components])
    
    def generic_point_source_simulation(self, source, **kwargs):
        simulations = []
        for _ in range(self.num_realisations):
            simulation = self.simulator.generic_point_source_simulation(source, **kwargs)
            simulations.append(self.data_flattening_callable({"outputs":simulation}))
        
        simulations = np.array(simulations)
        seismograms = simulations.reshape(self.num_realisations,self.num_traces, -1)
        seismograms = seismograms.transpose(1, 0, 2)  # Rearranging to (num_traces, num_realisations, trace_length)

        # now compute covariance arrays of each trace such that we have (num_traces, trace_length, trace_length)
        demeaned = seismograms - seismograms.mean(axis=1, keepdims=True)
        cov_blocks = np.einsum('nrt,nru->ntu', demeaned, demeaned) / (demeaned.shape[1] - 1)
        cov_blocks = cov_blocks.reshape(self.num_traces, -1)
        # repack into map of maps in same format as simulation dict
        all_cov_blocks_map = {}
        counter = 0
        for receiver in self.simulator.receivers.iterate():
            all_cov_blocks_map[receiver.station_name] = {}
            for component in self.components:
                all_cov_blocks_map[receiver.station_name][component] = cov_blocks[counter]
                counter += 1

        return all_cov_blocks_map
