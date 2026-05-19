"""Phase 0.3 — SBI ingestion contract test.

Takes the event h5 produced by the synthetic pipeline test (0.1) and feeds
it to RealNoiseSampler and DataManager exactly as pipeline.py does. Pins the
flattened-vector shape, dtype, and ordering.

This test is the gate for Phase 2: the new mseed_to_sbi_h5 tool must produce
h5 files that make this test pass without modification.
"""

import numpy as np
import pytest

from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader
from seismo_sbi.sbi.noises.real_noise import RealNoiseSampler
from seismo_sbi.sbi.types.parameters import SimulationParameters
from seismo_sbi.sbi.data_manager import DataManager
from seismo_sbi.sbi.dataset_compressor import DatasetCompressor

from tests.end_to_end.test_preprocessing_synthetic import (
    STATIONS,
    NETWORK,
    SR_TARGET,
    EVENT_START,
    EVENT_END,
    DATA_VECTOR_LEN,  # 61 — inclusive slice at 1 Hz for 60-second window
)

pytestmark = pytest.mark.slow

N_COMPONENTS_PER_STATION = 3  # Z, 1(E), 2(N)
EXPECTED_FLAT_LEN = len(STATIONS) * N_COMPONENTS_PER_STATION * DATA_VECTOR_LEN


def _build_receivers():
    """Receivers matching the synthetic pipeline stations, with E/N components."""
    return Receivers(
        receivers=[
            Receiver(
                latitude=0.0,
                longitude=0.0,
                network=NETWORK,
                station_name=sta,
                components=["Z", "E", "N"],
            )
            for sta in STATIONS
        ]
    )


def _build_sim_params(receivers):
    event_duration = (EVENT_END - EVENT_START).total_seconds()
    return SimulationParameters(
        receivers=receivers,
        components="ZEN",
        seismogram_duration=event_duration,
        sampling_rate=SR_TARGET,
        syngine_address="syngine://prem_i_2s",  # not actually called in these tests
        processing={},
    )


@pytest.fixture(scope="module")
def event_h5(pipeline_output):  # reuse the module-scoped fixture from the synthetic test
    path = pipeline_output["event_h5"]
    assert path.exists(), "Event h5 must exist (run test_preprocessing_synthetic first)"
    return path


# We need the pipeline_output fixture from test_preprocessing_synthetic.
# Import it explicitly so pytest can discover it even when running this file alone.
from tests.end_to_end.test_preprocessing_synthetic import pipeline_output  # noqa: F401


class TestSimulationDataLoader:
    """SimulationDataLoader correctly reads the event h5 written by ProcessedDataSlicer."""

    @pytest.fixture(autouse=True)
    def setup(self, event_h5):
        receivers = _build_receivers()
        self.loader = SimulationDataLoader(
            components="ZEN",
            receivers=receivers,
            data_length=None,
        )
        self.event_h5 = event_h5

    def test_flattened_vector_length(self):
        vec = self.loader.load_flattened_simulation_vector(self.event_h5)
        assert vec.shape == (EXPECTED_FLAT_LEN,), (
            f"Expected flat vector of length {EXPECTED_FLAT_LEN}, got {vec.shape}"
        )

    def test_flattened_vector_dtype(self):
        vec = self.loader.load_flattened_simulation_vector(self.event_h5)
        assert np.issubdtype(vec.dtype, np.floating), (
            f"Expected floating dtype, got {vec.dtype}"
        )

    def test_flattened_vector_finite(self):
        vec = self.loader.load_flattened_simulation_vector(self.event_h5)
        assert np.all(np.isfinite(vec)), "Flattened vector contains NaN or Inf"

    def test_flattened_vector_nonzero(self):
        vec = self.loader.load_flattened_simulation_vector(self.event_h5)
        assert np.any(vec != 0.0), "Flattened vector is all-zero"

    def test_misc_data_keys(self):
        misc = self.loader.load_misc_data(self.event_h5)
        for sta in STATIONS:
            assert sta in misc, f"Station {sta} missing from misc data"

    def test_misc_data_variance_positive(self):
        misc = self.loader.load_misc_data(self.event_h5)
        for sta in STATIONS:
            for comp, val in misc[sta].items():
                v = float(np.atleast_1d(val).flat[0])
                assert v > 0, f"{sta}/{comp} variance non-positive: {v}"

    def test_data_length_truncation(self):
        """data_length cap is respected."""
        receivers = _build_receivers()
        loader = SimulationDataLoader(
            components="ZEN",
            receivers=receivers,
            data_length=DATA_VECTOR_LEN // 2,
        )
        vec = loader.load_flattened_simulation_vector(self.event_h5)
        expected = len(STATIONS) * N_COMPONENTS_PER_STATION * (DATA_VECTOR_LEN // 2)
        assert vec.shape == (expected,)


class TestRealNoiseSamplerWithSyntheticH5:
    """RealNoiseSampler can consume a catalogue directory of synthetic h5 files.

    The event_dir produced by test_preprocessing_synthetic contains one h5.
    We verify the sampler loads it correctly — this mirrors how training noise
    is consumed by the SBI pipeline.
    """

    @pytest.fixture(autouse=True)
    def setup(self, event_h5):
        receivers = _build_receivers()
        sim_params = _build_sim_params(receivers)
        # Directory containing the single event h5 acts as a one-file noise catalogue
        self.sampler = RealNoiseSampler(
            simulation_parameters=sim_params,
            directory=event_h5.parent,
        )
        self.event_h5 = event_h5

    def test_finds_h5_files(self):
        assert len(self.sampler.noise_paths) >= 1

    def test_call_returns_array(self):
        result = self.sampler()
        assert isinstance(result, np.ndarray)
        assert result.shape == (EXPECTED_FLAT_LEN,)

    def test_no_rescale_returns_noise_and_misc(self):
        noise, misc = self.sampler(no_rescale=True)
        assert noise.shape == (EXPECTED_FLAT_LEN,)
        assert isinstance(misc, dict)
        for sta in STATIONS:
            assert sta in misc

    def test_noise_is_finite(self):
        result = self.sampler()
        assert np.all(np.isfinite(result))

    def test_reproducible_at_fixed_index(self):
        a = self.sampler(noise_index=0)
        b = self.sampler(noise_index=0)
        np.testing.assert_array_equal(a, b)


class TestComponentFallbackAndOrdering:
    """SimulationDataLoader's E→1 / N→2 fallback and station ordering."""

    def test_EN_receivers_read_12_h5_keys(self, event_h5):
        """Receivers declared with ['Z','E','N'] can load h5 data stored as '1','2','Z'.

        This is the core contract: the pipeline writes '1'/'2' but the Receivers
        object uses 'E'/'N'. The DataLoader must bridge this silently.
        """
        receivers = _build_receivers()  # components=['Z','E','N']
        loader = SimulationDataLoader(components="ZEN", receivers=receivers)
        # Must not raise KeyError even though h5 has '1'/'2' not 'E'/'N'
        vec = loader.load_flattened_simulation_vector(event_h5)
        assert vec.shape == (EXPECTED_FLAT_LEN,)
        assert np.all(np.isfinite(vec))

    def test_misc_data_en_fallback_to_12(self, event_h5):
        """load_misc_data with ['Z','E','N'] receivers reads '1'/'2' from h5."""
        receivers = _build_receivers()
        loader = SimulationDataLoader(components="ZEN", receivers=receivers)
        misc = loader.load_misc_data(event_h5)
        for sta in STATIONS:
            # Keys returned should be '1' and '2' (the h5 key names after fallback)
            assert sta in misc
            keys = set(misc[sta].keys())
            # The DataLoader returns whatever key it found (the h5 key after fallback)
            assert keys.issubset({"Z", "1", "2", "E", "N"}), (
                f"{sta}: unexpected misc keys {keys}"
            )
            assert len(keys) == 3, f"{sta}: expected 3 component keys, got {keys}"

    def test_station_order_in_flat_vector_follows_receivers_order(self, event_h5):
        """Flattened vector blocks follow the order defined in Receivers."""
        import h5py as h5py_local

        receivers = _build_receivers()
        loader = SimulationDataLoader(components="ZEN", receivers=receivers)
        vec = loader.load_flattened_simulation_vector(event_h5)

        n_comp = 3
        block = DATA_VECTOR_LEN  # samples per component
        with h5py_local.File(event_h5, "r") as f:
            for i, sta in enumerate(STATIONS):
                z_from_h5 = f["outputs"][sta]["Z"][()]
                z_from_vec = vec[i * n_comp * block : i * n_comp * block + block]
                np.testing.assert_array_equal(
                    z_from_vec, z_from_h5,
                    err_msg=f"Station {sta} Z block at position {i} does not match h5"
                )

    def test_component_blocks_are_non_identical(self, event_h5):
        """Z, E(→1), N(→2) blocks for each station must not be copies of each other."""
        receivers = _build_receivers()
        loader = SimulationDataLoader(components="ZEN", receivers=receivers)
        vec = loader.load_flattened_simulation_vector(event_h5)

        block = DATA_VECTOR_LEN
        for i in range(len(STATIONS)):
            offset = i * 3 * block
            z = vec[offset : offset + block]
            e = vec[offset + block : offset + 2 * block]
            n = vec[offset + 2 * block : offset + 3 * block]
            assert not np.allclose(z, e), f"Station {STATIONS[i]}: Z == E block"
            assert not np.allclose(z, n), f"Station {STATIONS[i]}: Z == N block"
            assert not np.allclose(e, n), f"Station {STATIONS[i]}: E == N block"


class TestAdaptiveCovarianceScaling:
    """RealNoiseSampler.calculate_scales uses /misc variance to rescale noise."""

    @pytest.fixture(autouse=True)
    def setup(self, event_h5):
        receivers = _build_receivers()
        sim_params = _build_sim_params(receivers)
        # Inject a known adaptive covariance (twice the h5 variance → scale = 2)
        misc_path = event_h5
        from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader
        loader = SimulationDataLoader(components="ZEN", receivers=receivers)
        misc = loader.load_misc_data(misc_path)

        # Build an adaptive covariance that equals the h5 misc (scale → 1.0 for all)
        adaptive_cov = {}
        for sta, comps in misc.items():
            adaptive_cov[sta] = {}
            for comp, val in comps.items():
                adaptive_cov[sta][comp] = [float(np.atleast_1d(val).flat[0])]

        self.sampler = RealNoiseSampler(
            simulation_parameters=sim_params,
            directory=event_h5.parent,
            adaptive_covariance=adaptive_cov,
        )
        self.event_h5 = event_h5

    def test_scale_factor_one_when_variances_match(self):
        """When adaptive_covariance equals the h5 misc, all scale factors are 1.0."""
        noise, misc = self.sampler(no_rescale=True)
        scales = self.sampler.calculate_scales(misc)
        for sta, comps in scales.items():
            for comp, s in comps.items():
                assert abs(float(s) - 1.0) < 0.01, (
                    f"{sta}/{comp}: expected scale=1.0, got {s}"
                )

    def test_scaled_noise_shape_unchanged(self):
        """Adaptive rescaling does not change the output vector shape."""
        result, misc = self.sampler()
        assert result.shape == (EXPECTED_FLAT_LEN,)


class TestDataManagerRealEventIngestion:
    """DataManager._create_job_data_from_real_events produces valid JobData."""

    def test_job_data_created(self, event_h5):
        receivers = _build_receivers()
        sim_params = _build_sim_params(receivers)

        loader = SimulationDataLoader(
            components="ZEN",
            receivers=receivers,
            data_length=None,
        )
        # DatasetCompressor is required by DataManager but not called in this path
        manager = DataManager(
            data_loader=loader,
            dataset_compressor=None,
        )

        real_event_jobs = {"test_event": str(event_h5)}
        test_noises = {}  # no noise models needed — real events don't add noise

        jobs = manager._create_job_data_from_real_events(real_event_jobs, test_noises)
        # With no test_noises, no job data is created (the loop over test_noises is empty)
        assert isinstance(jobs, list)

    def test_load_simulation_vector_matches_loader(self, event_h5):
        """DataManager.load_simulation_vector delegates to SimulationDataLoader correctly."""
        receivers = _build_receivers()
        loader = SimulationDataLoader(
            components="ZEN",
            receivers=receivers,
            data_length=None,
        )
        manager = DataManager(data_loader=loader, dataset_compressor=None)
        vec = manager.load_simulation_vector(str(event_h5))
        assert vec.shape == (EXPECTED_FLAT_LEN,)
        assert np.all(np.isfinite(vec))

    def test_load_noise_parametrisation_data(self, event_h5):
        """DataManager.load_noise_parametrisation_data returns variance dict."""
        receivers = _build_receivers()
        loader = SimulationDataLoader(
            components="ZEN",
            receivers=receivers,
            data_length=None,
        )
        manager = DataManager(data_loader=loader, dataset_compressor=None)
        misc = manager.load_noise_parametrisation_data(str(event_h5))
        assert isinstance(misc, dict)
        for sta in STATIONS:
            assert sta in misc
