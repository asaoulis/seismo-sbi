"""Linearised forward model: seismograms from precomputed moment-tensor sensitivity kernels.

The kernels come from a :class:`~seismo_sbi.sbi.compression.gaussian.ScoreCompressionData` and are
contracted with the six moment-tensor components, so one simulation is a matrix product rather
than a wavefield calculation.
"""

from typing import TYPE_CHECKING

import numpy as np

from .base import Simulator
from .simulation_io import seismogram_array_to_map
from .sources import GenericPointSource

if TYPE_CHECKING:
    from seismo_sbi.sbi.compression.gaussian import ScoreCompressionData


class FixedLocationKernelSimulator(Simulator):

    def __init__(self, score_compression_data: "ScoreCompressionData" = None,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        # None when the simulator is built before the kernels are known; they are required
        # before any simulation runs.
        if score_compression_data is None:
            self.sensitivity_kernels = None
            self.trace_length = None
        else:
            self.sensitivity_kernels = score_compression_data.data_parameter_gradients
            num_traces = len([comp for rec in self.receivers.iterate() for comp in rec.components])
            self.trace_length = self.sensitivity_kernels.shape[1] // num_traces

    def generic_point_source_simulation(self, source: GenericPointSource, *, stf_duration=None, **kwargs):
        seismograms = self._compute_seismograms_from_kernels(source)
        seismograms = seismograms.reshape(-1, self.trace_length)
        return seismogram_array_to_map(seismograms, self.receivers)
    
    def _compute_seismograms_from_kernels(self, source: GenericPointSource):

        moment_tensor_components = source.moment_tensor.components
        seismograms = np.dot(self.sensitivity_kernels.T, moment_tensor_components)
        return seismograms
