"""Linearised forward model: seismograms from precomputed moment-tensor sensitivity kernels.

The kernels come from a :class:`~seismo_sbi.sbi.compression.gaussian.ScoreCompressionData` and are
contracted with the six moment-tensor components, so one simulation is a matrix product rather
than a wavefield calculation.
"""

from typing import TYPE_CHECKING

import numpy as np

from .base import Simulator
from .sources import GenericPointSource

if TYPE_CHECKING:
    from seismo_sbi.sbi.compression.gaussian import ScoreCompressionData


class FixedLocationKernelSimulator(Simulator):

    def __init__(self, score_compression_data: "ScoreCompressionData" = None,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        # score_compression_data may be None when the simulator is constructed before the
        # kernels are known (e.g. as the initial simulator in a pipeline that will swap in
        # real kernels via use_kernel_simulator_if_possible). Kernels are required before
        # any simulation is actually run.
        if score_compression_data is None:
            self.sensitivity_kernels = None
            self.trace_length = None
        else:
            self.sensitivity_kernels = score_compression_data.data_parameter_gradients
            num_traces = len([comp for rec in self.receivers.iterate() for comp in rec.components])
            self.trace_length = self.sensitivity_kernels.shape[1] // num_traces

    def generic_point_source_simulation(self, source: GenericPointSource, *, stf_duration=None, **kwargs):
        
        all_seismograms_map = {}

        seismograms = self._compute_seismograms_from_kernels(source)

        seismograms = seismograms.reshape(-1, self.trace_length)

        trace_counter = 0
        for rec_idx, receiver in enumerate(self.receivers.iterate()):
            all_seismograms_map[receiver.station_name] = {}
            for comp_idx, component in enumerate(receiver.components):
                all_seismograms_map[receiver.station_name][component] = seismograms[trace_counter]
                trace_counter +=1
            
        return all_seismograms_map
    
    def _compute_seismograms_from_kernels(self, source: GenericPointSource):

        moment_tensor_components = source.moment_tensor.components
        seismograms = np.dot(self.sensitivity_kernels.T, moment_tensor_components)
        return seismograms
