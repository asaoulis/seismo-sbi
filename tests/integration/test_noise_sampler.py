"""Integration tests: RealNoiseSampler loads H5 catalogue and returns correctly-shaped vectors."""

import numpy as np
import pytest

from seismo_sbi.sbi.noises.real_noise import RealNoiseSampler
from seismo_sbi.sbi.types.parameters import SimulationParameters

from tests.conftest import TRACE_LEN


def _make_sim_params(receivers):
    """Build a minimal SimulationParameters for a 1-station, Z-only setup."""
    return SimulationParameters(
        receivers=receivers,
        components="Z",
        seismogram_duration=TRACE_LEN,  # 1 Hz × TRACE_LEN s = TRACE_LEN samples
        sampling_rate=1.0,
        syngine_address="syngine://prem_i_2s",
        processing={},
    )


class TestRealNoiseSampler:

    @pytest.fixture(autouse=True)
    def setup(self, receivers, noise_catalogue_dir):
        sim_params = _make_sim_params(receivers)
        self.sampler = RealNoiseSampler(
            simulation_parameters=sim_params,
            directory=noise_catalogue_dir,
        )
        self.expected_len = TRACE_LEN

    def test_finds_noise_files(self):
        assert len(self.sampler.noise_paths) == 5

    def test_call_returns_array(self):
        result = self.sampler()
        # Without no_rescale or adaptive_covariance, returns only the noise vector
        assert isinstance(result, np.ndarray)
        assert result.shape == (self.expected_len,)

    def test_call_multiple_times(self):
        for _ in range(3):
            result = self.sampler()
            assert result.shape == (self.expected_len,)

    def test_no_rescale_returns_tuple(self):
        result = self.sampler(no_rescale=True)
        assert isinstance(result, tuple)
        noise, misc = result
        assert noise.shape == (self.expected_len,)

    def test_misc_data_contains_station(self):
        noise, misc = self.sampler(no_rescale=True)
        assert "STA1" in misc

    def test_misc_data_contains_component(self):
        noise, misc = self.sampler(no_rescale=True)
        assert "Z" in misc["STA1"]

    def test_misc_data_variance_positive(self):
        noise, misc = self.sampler(no_rescale=True)
        variance = misc["STA1"]["Z"]
        assert float(np.squeeze(variance)) > 0

    def test_noise_is_finite(self):
        result = self.sampler()
        assert np.all(np.isfinite(result))

    def test_specific_noise_index(self):
        """Requesting a specific index should return a fixed noise realisation."""
        noise_a = self.sampler(noise_index=0)
        noise_b = self.sampler(noise_index=0)
        assert np.allclose(noise_a, noise_b)

    def test_different_indices_differ(self):
        """Different noise files should produce different realisations (with high prob.)."""
        n0 = self.sampler(noise_index=0)
        n1 = self.sampler(noise_index=1)
        assert not np.allclose(n0, n1)
