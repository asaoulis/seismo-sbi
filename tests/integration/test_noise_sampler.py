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


class TestRealNoiseSamplerFreezeScale:
    """freeze_scale=True is the generic-event mode: set_adaptive_covariance_with_misc_data is a
    no-op so draws are never rescaled to a single event's pre-event variance."""

    def _sampler(self, receivers, noise_catalogue_dir, freeze_scale):
        return RealNoiseSampler(
            simulation_parameters=_make_sim_params(receivers),
            directory=noise_catalogue_dir,
            freeze_scale=freeze_scale,
        )

    def test_default_is_unfrozen(self, receivers, noise_catalogue_dir):
        sampler = self._sampler(receivers, noise_catalogue_dir, freeze_scale=False)
        assert sampler.freeze_scale is False

    def test_frozen_ignores_set_adaptive_covariance(self, receivers, noise_catalogue_dir):
        sampler = self._sampler(receivers, noise_catalogue_dir, freeze_scale=True)
        _, misc = sampler(no_rescale=True)
        sampler.set_adaptive_covariance_with_misc_data(misc)
        # No-op: adaptive_covariance stays None, so __call__ takes the un-rescaled branch and
        # returns a plain array (not the rescaled (noise, misc) tuple).
        assert sampler.adaptive_covariance is None
        result = sampler()
        assert isinstance(result, np.ndarray)
        assert result.shape == (TRACE_LEN,)

    def test_unfrozen_sets_adaptive_covariance(self, receivers, noise_catalogue_dir):
        sampler = self._sampler(receivers, noise_catalogue_dir, freeze_scale=False)
        _, misc = sampler(no_rescale=True)
        sampler.set_adaptive_covariance_with_misc_data(misc)
        # Legacy single-event behaviour: covariance is set and __call__ returns the rescaled tuple.
        assert sampler.adaptive_covariance is not None
        result = sampler()
        assert isinstance(result, tuple)
