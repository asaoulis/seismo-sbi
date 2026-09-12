"""Theory-error covariance measured from an ensemble of Earth models.

Wraps an ensemble simulator so that one "simulation" is a whole set of draws for the same
source, and returns the covariance of their residuals about the reference member. The result
travels through the pipeline in the shape of a simulation, one covariance block per trace.
"""

import numpy as np
import joblib

from seismo_sbi.simulators.base import Simulator
from seismo_sbi.simulators.ensemble import GFEnsembleSimulator
from seismo_sbi.utils.seismograms import apply_station_time_shifts

def parallel_execution(inputs, func, num_jobs = 20):
    if num_jobs in [None, 0 , 1]:
        return [func(block) for block in inputs]
    return joblib.Parallel(n_jobs=num_jobs)(joblib.delayed(func)(block) for block in inputs)

class EnsembleTheoryCovarianceEstimationSimulator(Simulator):
    """Per-trace theory-error covariance estimated from an ensemble simulator.

    One draw per ensemble member, demeaned against the reference member or the ensemble mean;
    the result is a ``(n_samples, n_samples)`` covariance per trace, returned in the shape of a
    simulation map.
    """

    def __init__(self, simulator: GFEnsembleSimulator, data_flattening, *args, internal_jobs=20, covariance_mean='fiducial', **kwargs):
        super().__init__(*args, **kwargs)
        self.simulator = simulator
        self.num_realisations = simulator.num_models
        self.data_flattening_callable = data_flattening
        self.num_traces = len([comp for rec in self.receivers.iterate() for comp in rec.components])
        self.covariance_mean = covariance_mean
        self.num_jobs = internal_jobs
        self.receivers.set_time_shifts({rec.station_name: 0 for rec in self.simulator.receivers.iterate()})

    def generic_point_source_simulation(self, source, **kwargs):
        sim_func  = lambda _: self.data_flattening_callable({"outputs":apply_station_time_shifts(self.simulator.receivers, self.simulator.generic_point_source_simulation(source, **kwargs))})
        simulations = parallel_execution(range(self.num_realisations), sim_func, num_jobs=self.num_jobs)

        
        simulations = np.array(simulations)
        seismograms = simulations.reshape(self.num_realisations,self.num_traces, -1)
        seismograms = seismograms.transpose(1, 0, 2)

        if self.covariance_mean == 'fiducial':
            fiducial_data = self.data_flattening_callable({"outputs":apply_station_time_shifts(self.simulator.receivers, self.simulator.generic_point_source_simulation(source, **{**kwargs, **{'use_fiducial': True}} ))})
            fiducial_data = fiducial_data.reshape(self.num_traces, -1)
            mean_obs = fiducial_data[:, None, :]
        elif self.covariance_mean == 'ensemble':
            mean_obs = seismograms.mean(axis=1, keepdims=True)
        demeaned = seismograms - mean_obs
        cov_blocks = np.einsum('nrt,nru->ntu', demeaned, demeaned) / (demeaned.shape[1] - 1)
        cov_blocks = cov_blocks.reshape(self.num_traces, -1)
        all_cov_blocks_map = {}
        counter = 0
        for receiver in self.simulator.receivers.iterate():
            all_cov_blocks_map[receiver.station_name] = {}
            for component in receiver.components:
                all_cov_blocks_map[receiver.station_name][component] = cov_blocks[counter]
                counter += 1

        return all_cov_blocks_map


CPSTheoryCovarianceEstimationSimulator = EnsembleTheoryCovarianceEstimationSimulator