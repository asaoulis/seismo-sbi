"""A forward model registered under a simulation_type is the one build_simulator returns."""
import numpy as np
import pytest
from obspy.geodetics import gps2dist_azimuth

from seismo_sbi.sbi.types.parameters import SimulationParameters
from seismo_sbi.simulators.base import Simulator
from seismo_sbi.simulators.receivers import Receiver, Receivers
from seismo_sbi.simulators.registry import SIMULATOR_REGISTRY, build_simulator, register_simulator

SAMPLING_RATE_HZ = 1.0
SOURCE_DEPTH_KM = 10.0


class HomogeneousPWaveSimulator(Simulator):
    """Far-field P wave in a homogeneous whole space, as displacement in metres."""

    p_velocity_m_s = 6000.0
    density_kg_m3 = 2700.0
    pulse_width_s = 4.0

    def generic_point_source_simulation(self, source, **kwargs):
        location = source.source_location
        m_rr, m_tt, m_pp, m_rt, m_rp, m_tp = source.moment_tensor.components
        moment_tensor = np.array([[m_rr, m_rt, m_rp], [m_rt, m_tt, m_tp], [m_rp, m_tp, m_pp]])
        time_s = np.arange(int(self.seismogram_length * SAMPLING_RATE_HZ) + 1) / SAMPLING_RATE_HZ
        seismograms = {}
        for receiver in self.receivers.iterate():
            epicentral_m, azimuth_deg, _ = gps2dist_azimuth(location.latitude, location.longitude,
                                                            receiver.latitude, receiver.longitude)
            azimuth = np.radians(azimuth_deg)
            ray = np.array([location.depth * 1e3, -epicentral_m * np.cos(azimuth),
                            epicentral_m * np.sin(azimuth)])
            distance_m = np.linalg.norm(ray)
            gamma = ray / distance_m
            arrival_s = distance_m / self.p_velocity_m_s
            moment_rate = np.exp(-0.5 * ((time_s - arrival_s) / self.pulse_width_s) ** 2) \
                / (self.pulse_width_s * np.sqrt(2 * np.pi))
            amplitude = gamma @ moment_tensor @ gamma / (
                4 * np.pi * self.density_kg_m3 * self.p_velocity_m_s ** 3 * distance_m)
            up, south, east = amplitude * gamma[:, None] * moment_rate
            seismograms[receiver.station_name] = {"Z": up, "N": -south, "E": east}
        return seismograms


def build_homogeneous_p_wave(simulation_parameters, *, post_processing_effects):
    return HomogeneousPWaveSimulator(
        components=simulation_parameters.components, receivers=simulation_parameters.receivers,
        seismogram_duration_in_s=simulation_parameters.seismogram_duration,
        synthetics_processing=simulation_parameters.processing, post_processing_effects=post_processing_effects)


@pytest.fixture
def homogeneous_p_wave():
    register_simulator("test_homogeneous_p_wave", build_homogeneous_p_wave)
    receivers = Receivers(receivers=[Receiver(37.6, -118.9, "XX", "ABOVE"), Receiver(38.0, -118.5, "XX", "AWAY")])
    parameters = SimulationParameters(receivers, "ZEN", 60.0, None, SAMPLING_RATE_HZ, {},
                                      simulation_type="test_homogeneous_p_wave")
    yield build_simulator(parameters.simulation_type, parameters)
    SIMULATOR_REGISTRY.pop("test_homogeneous_p_wave")


def test_registered_builder_is_selected_by_simulation_type(homogeneous_p_wave):
    moment_tensor = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) * 1e16
    source = {"source_location": [37.6, -118.9, SOURCE_DEPTH_KM, 0.0], "moment_tensor": list(moment_tensor)}

    positive = homogeneous_p_wave.run_simulation(source)[1]
    negative = homogeneous_p_wave.run_simulation({**source, "moment_tensor": list(-moment_tensor)})[1]

    assert isinstance(homogeneous_p_wave, HomogeneousPWaveSimulator)
    arrival_s = SOURCE_DEPTH_KM * 1e3 / 6000.0
    moment_rate_at_2_s = np.exp(-0.5 * ((2.0 - arrival_s) / 4.0) ** 2) / (4.0 * np.sqrt(2 * np.pi))
    expected_up_m = 1e16 / (4 * np.pi * 2700.0 * 6000.0 ** 3 * SOURCE_DEPTH_KM * 1e3) * moment_rate_at_2_s
    np.testing.assert_allclose(positive["ABOVE"]["Z"][2], expected_up_m, rtol=1e-9)
    np.testing.assert_allclose(negative["AWAY"]["Z"], -positive["AWAY"]["Z"])


def test_unknown_simulation_type_raises():
    parameters = SimulationParameters(Receivers(receivers=[]), "Z", 10.0, None, 1.0, {}, simulation_type="no_such")

    with pytest.raises(NotImplementedError, match="no_such"):
        build_simulator(parameters.simulation_type, parameters)


def test_a_configured_kernel_type_builds_before_its_kernels_exist():
    from seismo_sbi.simulators.kernel import FixedLocationKernelSimulator

    parameters = SimulationParameters(Receivers(receivers=[Receiver(37.6, -118.9, "XX", "ABOVE")]), "Z", 10.0, None,
                                      1.0, {"sampling_rate": 1.0}, simulation_type="kernel")

    assert isinstance(build_simulator(parameters.simulation_type, parameters), FixedLocationKernelSimulator)
