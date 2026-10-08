"""Integration tests: RealNoiseSampler loads H5 catalogue and returns correctly-shaped vectors."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from seismo_sbi.sbi.noises.noise_model import NoiseModelConfiguration
from seismo_sbi.sbi.noises.real_noise import RealNoiseSampler
from seismo_sbi.sbi.pipeline import SBIPipeline
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

    def test_draw_returns_the_noise_alone(self):
        result = self.sampler.draw()
        assert isinstance(result.noise, np.ndarray)
        assert result.noise.shape == (self.expected_len,)
        assert result.present is None and result.covariance_data is None

    def test_call_multiple_times(self):
        for _ in range(3):
            result = self.sampler.draw().noise
            assert result.shape == (self.expected_len,)

    def test_draw_with_covariance_carries_the_window_covariance(self):
        noise, _, misc = self.sampler.draw_with_covariance()
        assert noise.shape == (self.expected_len,)
        assert misc is not None

    def test_misc_data_contains_station(self):
        noise, _, misc = self.sampler.draw_with_covariance()
        assert "STA1" in misc

    def test_misc_data_contains_component(self):
        noise, _, misc = self.sampler.draw_with_covariance()
        assert "Z" in misc["STA1"]

    def test_misc_data_variance_positive(self):
        noise, _, misc = self.sampler.draw_with_covariance()
        variance = misc["STA1"]["Z"]
        assert float(np.squeeze(variance)) > 0

    def test_noise_is_finite(self):
        result = self.sampler.draw().noise
        assert np.all(np.isfinite(result))

    def test_specific_noise_index(self):
        """Requesting a specific index should return a fixed noise realisation."""
        noise_a = self.sampler.draw_with_covariance(window_index=0).noise
        noise_b = self.sampler.draw_with_covariance(window_index=0).noise
        assert np.allclose(noise_a, noise_b)

    def test_different_indices_differ(self):
        """Different noise files should produce different realisations (with high prob.)."""
        n0 = self.sampler.draw_with_covariance(window_index=0).noise
        n1 = self.sampler.draw_with_covariance(window_index=1).noise
        assert not np.allclose(n0, n1)

    def test_preload_cache_draws_from_same_pool(self):
        """The opt-in in-RAM cache returns ONLY genuine catalogue windows (distribution-identical
        to the on-disk random draw) and correctly-shaped, finite vectors."""
        # Gather every on-disk window by index for a membership check.
        on_disk = [self.sampler.draw_with_covariance(window_index=i).noise for i in range(len(self.sampler.noise_paths))]
        self.sampler.preload_cache(max_workers=4)
        assert self.sampler._noise_cache is not None
        assert self.sampler._noise_cache.shape[1] == self.expected_len
        # Every cached draw must EQUAL one of the on-disk windows (drawn from the same pool).
        for _ in range(12):
            v = self.sampler.draw().noise
            assert v.shape == (self.expected_len,)
            assert np.all(np.isfinite(v))
            assert any(np.allclose(v, w, rtol=1e-5, atol=0) for w in on_disk), (
                "cached noise draw is not a genuine catalogue window"
            )

    def test_draw_with_covariance_reads_disk_after_preloading(self):
        """The cache only serves the training draw; a draw with its covariance still reads the window."""
        self.sampler.preload_cache(max_workers=4)
        noise, _, misc = self.sampler.draw_with_covariance()
        assert noise.shape == (self.expected_len,)
        assert "STA1" in misc


class TestTrainingNoiseFollowsTheEvent:
    """The noise model decides whether the pipeline rescales its training noise to an event's
    pre-event noise: recorded noise by default, never with ``rescale: false`` or white noise."""

    @staticmethod
    def _pipeline(receivers, noise_model, noise_catalogue_dir):
        stub = SimpleNamespace(simulation_parameters=_make_sim_params(receivers), trace_length=TRACE_LEN,
                               data_vector_length=TRACE_LEN, test_noises={},
                               noise_covariances=lambda: dict(data_covariance=None, empirical_covariance=None))
        SBIPipeline.load_test_noises(stub, NoiseModelConfiguration.from_yaml_block(
            {"noise_catalogue_path": noise_catalogue_dir, **noise_model}), [])
        return stub

    def _event_covariance(self, stub):
        return RealNoiseSampler(stub.simulation_parameters, None).data_loader.load_misc_data(
            sorted(stub.training_noise_sampler.noise_paths)[0])

    @pytest.mark.parametrize("noise_model, follows", [({"type": "real_noise"}, True),
                                                      ({"type": "real_noise", "rescale": False}, False)])
    def test_recorded_noise_follows_the_event_unless_rescale_is_off(self, receivers, noise_catalogue_dir,
                                                                    noise_model, follows):
        stub = self._pipeline(receivers, noise_model, noise_catalogue_dir)
        SBIPipeline.rescale_training_noise(stub, self._event_covariance(stub))
        assert (stub.training_noise_sampler.target_variances is not None) is follows
        assert (stub.training_noise_sampler.draw().covariance_data is not None) is follows

    def test_white_noise_keeps_its_level(self, receivers, noise_catalogue_dir):
        stub = self._pipeline(receivers, {"type": "gaussian", "noise_level": 2.0}, noise_catalogue_dir)
        SBIPipeline.rescale_training_noise(stub, {"STA1": {"Z": np.array([1.0])}})
        assert stub.training_noise_sampler.noise_level == 2.0


class TestRealNoiseSamplerShortWindowSkip:
    """A noise window sitting on a station data gap has a trace shorter than data_length;
    SimulationDataLoader anchors the length to the first receiver and truncates, silently
    returning a sub-length vector that will not broadcast against the full-length data
    vector D and crashes the compression. When data_length is known the sampler
    must SKIP such windows — the variable-length analogue of the missing-station KeyError
    skip."""

    @staticmethod
    def _write_window(path, receivers, length):
        with h5py.File(path, "w") as f:
            grp_out = f.create_group("outputs")
            grp_misc = f.create_group("misc")
            for rec in receivers.iterate():
                sta = grp_out.create_group(rec.station_name)
                sta_misc = grp_misc.create_group(rec.station_name)
                for comp in rec.components:
                    sta.create_dataset(comp, data=np.random.standard_normal(length))
                    sta_misc.create_dataset(comp, data=np.array([1.0]))

    def test_expected_length_computed_only_with_data_length(self, tmp_path, receivers):
        self._write_window(tmp_path / "good.h5", receivers, TRACE_LEN)
        with_len = RealNoiseSampler(_make_sim_params(receivers), tmp_path, data_length=TRACE_LEN)
        assert with_len._expected_length == TRACE_LEN  # 1 receiver × 1 comp × TRACE_LEN
        # legacy back-compat: no data_length -> no length check (behaviour unchanged)
        without_len = RealNoiseSampler(_make_sim_params(receivers), tmp_path)
        assert without_len._expected_length is None

    def test_random_draws_skip_short_windows(self, tmp_path, receivers):
        self._write_window(tmp_path / "good.h5", receivers, TRACE_LEN)
        self._write_window(tmp_path / "short.h5", receivers, TRACE_LEN // 2)
        sampler = RealNoiseSampler(_make_sim_params(receivers), tmp_path, data_length=TRACE_LEN)
        for _ in range(50):
            noise = sampler.draw().noise
            assert noise.shape == (TRACE_LEN,)   # never the truncated short window

    def test_explicit_short_path_falls_back(self, tmp_path, receivers):
        self._write_window(tmp_path / "good.h5", receivers, TRACE_LEN)
        self._write_window(tmp_path / "short.h5", receivers, TRACE_LEN // 2)
        sampler = RealNoiseSampler(_make_sim_params(receivers), tmp_path, data_length=TRACE_LEN)
        short_index = [path.name for path in sampler.noise_paths].index("short.h5")
        noise = sampler.draw_with_covariance(window_index=short_index).noise
        assert noise.shape == (TRACE_LEN,)       # skipped to the good window

    def test_all_short_raises(self, tmp_path, receivers):
        # No usable window -> bounded retry raises instead of recursing forever.
        self._write_window(tmp_path / "short0.h5", receivers, TRACE_LEN // 2)
        self._write_window(tmp_path / "short1.h5", receivers, TRACE_LEN // 3)
        sampler = RealNoiseSampler(_make_sim_params(receivers), tmp_path, data_length=TRACE_LEN)
        with pytest.raises(RuntimeError):
            sampler.draw()


def test_from_receivers_draws_what_the_simulation_parameters_constructor_draws(receivers, noise_catalogue_dir):
    np.random.seed(3)
    from_parameters = RealNoiseSampler(_make_sim_params(receivers), noise_catalogue_dir)
    np.random.seed(3)
    from_receivers = RealNoiseSampler.from_receivers(receivers, "Z", TRACE_LEN, 1.0, noise_catalogue_dir)

    assert list(from_receivers.noise_paths) == list(from_parameters.noise_paths)
    np.random.seed(4)
    expected = from_parameters.draw().noise
    np.random.seed(4)
    np.testing.assert_array_equal(from_receivers.draw().noise, expected)


def test_from_windows_draws_whole_rows_of_the_given_windows(receivers):
    windows = np.arange(4 * TRACE_LEN, dtype=float).reshape(4, TRACE_LEN)
    sampler = RealNoiseSampler.from_windows(windows, receivers, "Z")
    assert sampler.noise_paths.size == 0
    np.random.seed(0)
    draws = [sampler.draw().noise for _ in range(20)]
    assert all(draw.shape == (TRACE_LEN,) for draw in draws)
    assert {int(draw[0]) // TRACE_LEN for draw in draws} <= {0, 1, 2, 3}
    assert all(np.array_equal(draw, windows[int(draw[0]) // TRACE_LEN]) for draw in draws)


def test_from_windows_with_presence_returns_the_window_and_its_stations(receivers):
    windows = np.ones((3, TRACE_LEN))
    present = np.array([[True], [False], [True]])
    sampler = RealNoiseSampler.from_windows(windows, receivers, "Z", present=present)
    assert sampler.allow_incomplete
    noise, mask, _ = sampler.draw()
    assert noise.shape == (TRACE_LEN,) and mask.shape == (1,)
    assert sampler.subset_window_count([0]) == 2


def test_a_pool_of_mostly_unusable_windows_still_finds_the_usable_one(tmp_path, receivers):
    for window in range(1500):
        with h5py.File(tmp_path / f"empty_{window:04d}.h5", "w") as noise_file:
            noise_file.create_group("outputs")
    TestRealNoiseSamplerShortWindowSkip._write_window(tmp_path / "good.h5", receivers, TRACE_LEN)
    sampler = RealNoiseSampler(_make_sim_params(receivers), tmp_path, data_length=TRACE_LEN)
    good_index = [path.name for path in sampler.noise_paths].index("good.h5")
    noise = sampler.draw_with_covariance(window_index=good_index + 1).noise
    assert noise.shape == (TRACE_LEN,)


def test_the_pool_covariance_is_the_mean_of_the_recorded_autocovariances(tmp_path, receivers):
    variances = [1.0, 4.0, 9.0, 0.25]
    for index, variance in enumerate(variances):
        with h5py.File(tmp_path / f"noise_{index}.h5", "w") as window:
            for receiver in receivers.iterate():
                for component in receiver.components:
                    window.create_dataset(f"outputs/{receiver.station_name}/{component}", data=np.zeros(TRACE_LEN))
                    window.create_dataset(f"misc/{receiver.station_name}/{component}",
                                          data=variance * np.exp(-np.arange(4) / 2.0))
    np.random.seed(1)
    first = RealNoiseSampler(_make_sim_params(receivers), tmp_path, TRACE_LEN).mean_covariance_data()
    np.random.seed(2)
    second = RealNoiseSampler(_make_sim_params(receivers), tmp_path, TRACE_LEN).mean_covariance_data()

    for station, components in first.items():
        for component, autocovariance in components.items():
            np.testing.assert_allclose(autocovariance, np.mean(variances) * np.exp(-np.arange(4) / 2.0), rtol=1e-12)
            np.testing.assert_array_equal(autocovariance, second[station][component])


def test_the_pool_covariance_skips_the_windows_a_draw_skips(tmp_path, two_receivers):
    def write(name, variance, lengths):
        with h5py.File(tmp_path / f"{name}.h5", "w") as window:
            window.create_group("outputs")
            for station, length in lengths.items():
                window.create_dataset(f"outputs/{station}/Z", data=np.zeros(length))
                window.create_dataset(f"misc/{station}/Z", data=variance * np.ones(3))

    write("complete", 1.0, {"STA1": TRACE_LEN, "STA2": TRACE_LEN})
    write("long", 3.0, {"STA1": TRACE_LEN + 5, "STA2": TRACE_LEN + 5})
    write("short", 50.0, {"STA1": TRACE_LEN, "STA2": TRACE_LEN - 4})
    write("one_station", 70.0, {"STA1": TRACE_LEN})
    sampler = RealNoiseSampler(_make_sim_params(two_receivers), tmp_path, TRACE_LEN)

    usable = {path.name for path in sampler.noise_paths if sampler._read_window(path)[0] is not None}
    assert usable == {path.name for path in sampler.noise_paths if sampler._window_is_usable(path)}
    assert usable == {"complete.h5", "long.h5"}
    for components in sampler.mean_covariance_data().values():
        np.testing.assert_array_equal(components["Z"], 2.0 * np.ones(3))
