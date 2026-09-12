"""Instaseis forward model for a point source at any location.

Opens the Instaseis database once per simulation and reads one seismogram per receiver through
:class:`~seismo_sbi.simulators.instaseis.querier.InstaseisDBQuerier`.
"""

from ..base import Simulator
from ..sources import GenericPointSource
from .querier import InstaseisDBQuerier


class InstaseisSourceSimulator(Simulator):

    def __init__(self, instaseis_model_loc, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.instaseis_model_loc = instaseis_model_loc
        self.sampling_rate = float(InstaseisDBQuerier(self.instaseis_model_loc,
                                                      self.synthetics_processing,
                                                       self.seismogram_length,
                                                       self.source_depth_offset_km).sampling_rate)

    def generic_point_source_simulation(self, source: GenericPointSource, *, stf_duration=None, **kwargs):

        instaseis_db_querier = InstaseisDBQuerier(self.instaseis_model_loc,
                                                  self.synthetics_processing,
                                                    self.seismogram_length,
                                                    self.source_depth_offset_km)

        all_seismograms_map = {}
        for receiver in self.receivers.iterate():
            all_seismograms_map[receiver.station_name] = {}
            receiver_results = instaseis_db_querier.get_seismograms(
                source, receiver, self.components, stf_duration=stf_duration
            )

            for component in self.components:
                all_seismograms_map[receiver.station_name][component] = receiver_results[component]

        return all_seismograms_map
