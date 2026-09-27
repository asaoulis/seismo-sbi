"""The theory covariance visits each ensemble member once; covariances accept arrays and numbers."""
from copy import deepcopy

import numpy as np
import pytest

from seismo_sbi.sbi.noises.covariance_estimator import EmpiricalCovarianceEstimator
from seismo_sbi.sbi.noises.diagonal_covariances import ScalarEmpiricalCovariance
from seismo_sbi.sbi.noises.theory_block_covariance import TheoryBlockDiagonalEmpiricalCovariance
from seismo_sbi.sbi.noises.toeplitz_covariances import BlockDiagonalCovariance
from seismo_sbi.sbi.pipeline import likelihood_covariance
from seismo_sbi.simulators.gf_ensemble import GFEnsembleSimulator
from seismo_sbi.simulators.receivers import Receiver, Receivers
from seismo_sbi.simulators.simulation_io import SimulationSaver, seismogram_map_to_array
from seismo_sbi.simulators.sources import GeneralMomentTensor, GenericPointSource, SourceLocation
from seismo_sbi.simulators.theory_covariance import EnsembleTheoryCovarianceEstimationSimulator
from seismo_sbi.utils.errors import InvalidConfiguration

TRACE_LEN = 8
PROCESSING = {"sampling_rate": 1.0, "filter": {"type": "bandpass", "freqmin": 0.01, "freqmax": 0.1}}


class RecordingEnsemble(GFEnsembleSimulator):
    """Four members whose seismograms are constant at the member's value; records every member used."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.members_used = []

    @property
    def members(self):
        return [10, 20, 30, 40]

    @property
    def fiducial_member(self):
        return 0

    def generic_point_source_simulation(self, source, *, use_fiducial=False, seed=None, member=None, **kwargs):
        member = self.select_member(use_fiducial=use_fiducial, seed=seed, member=member)
        self.members_used.append(member)
        return {receiver.station_name: {component: np.full(TRACE_LEN, float(member))
                                        for component in receiver.components}
                for receiver in self.receivers.iterate()}


@pytest.fixture
def receivers():
    return Receivers(receivers=[Receiver(0.0, 0.0, "XX", "STA1", ["Z"]), Receiver(1.0, 1.0, "XX", "STA2", ["Z"])])


@pytest.fixture
def ensemble(receivers):
    return RecordingEnsemble(components=["Z"], receivers=receivers, seismogram_duration_in_s=TRACE_LEN,
                             synthetics_processing=PROCESSING)


def theory_estimator(ensemble, receivers, **kwargs):
    return EnsembleTheoryCovarianceEstimationSimulator(
        simulator=ensemble, data_flattening=lambda d: seismogram_map_to_array(d["outputs"], receivers),
        internal_jobs=1, components=["Z"], receivers=deepcopy(receivers),
        seismogram_duration_in_s=TRACE_LEN, synthetics_processing=PROCESSING, **kwargs)


def source():
    return GenericPointSource(SourceLocation(0.0, 0.0, 10.0, 0.0), GeneralMomentTensor([1e14] * 6))


def test_theory_covariance_visits_each_member_exactly_once_in_order(ensemble, receivers):
    theory_estimator(ensemble, receivers).generic_point_source_simulation(source())
    assert ensemble.members_used == [10, 20, 30, 40, 0]


def test_theory_covariance_member_loop_is_the_same_seeded_or_not(ensemble, receivers):
    unseeded = theory_estimator(ensemble, receivers).generic_point_source_simulation(source())
    seeded_estimator = theory_estimator(ensemble, receivers)
    seeded_estimator.seed = 7
    seeded = seeded_estimator.generic_point_source_simulation(source())
    assert ensemble.members_used == [10, 20, 30, 40, 0] * 2
    assert np.array_equal(unseeded["STA1"]["Z"], seeded["STA1"]["Z"])


def test_theory_covariance_is_the_ensemble_variance_about_the_fiducial(ensemble, receivers):
    blocks = theory_estimator(ensemble, receivers).generic_point_source_simulation(source())
    members = np.array([10.0, 20.0, 30.0, 40.0])
    expected = np.sum((members - 0.0) ** 2) / (len(members) - 1)
    assert np.allclose(blocks["STA2"]["Z"].reshape(TRACE_LEN, TRACE_LEN), expected)


def test_select_member_still_draws_with_replacement(ensemble):
    np.random.seed(0)
    draws = [ensemble.select_member() for _ in range(12)]
    assert len(set(draws)) < len(draws)
    assert set(draws) <= set(ensemble.members)


def test_select_member_returns_a_named_member(ensemble):
    assert ensemble.select_member(member=30) == 30
    assert ensemble.select_member(use_fiducial=True, member=30) == 0


def test_theory_block_covariance_inherits_the_block_diagonal_loss():
    assert (TheoryBlockDiagonalEmpiricalCovariance.generic_loss_callable
            is BlockDiagonalCovariance.generic_loss_callable)


def test_estimate_from_windows_matches_the_directory_path(tmp_path, receivers):
    rng = np.random.default_rng(3)
    n_windows, n_samples = 6, 32
    windows = rng.normal(size=(n_windows, 2, n_samples)) * np.array([1.0, 3.0])[None, :, None]
    for index, window in enumerate(windows):
        waveforms = {receiver.station_name: {"Z": window[trace]}
                     for trace, receiver in enumerate(receivers.iterate())}
        SimulationSaver(output_data=waveforms).dump_data_as_hdf5(tmp_path / f"noise_{index}.h5")

    from_files = EmpiricalCovarianceEstimator(tmp_path, receivers, "Z", covariance_exp_tapering=False,
                                              verbose=False).compute_stationwise_covariances()
    from_arrays = EmpiricalCovarianceEstimator(tmp_path, receivers, "Z", covariance_exp_tapering=False,
                                               verbose=False).estimate_from_windows(windows)
    for station in ("STA1", "STA2"):
        assert np.allclose(from_arrays[station]["Z"], from_files[station]["Z"])
    assert from_arrays["STA2"]["Z"][0] == pytest.approx(9.0, rel=0.5)


def test_numeric_likelihood_covariance_is_white_noise_of_that_sigma():
    covariance = likelihood_covariance(0.5, compressor_covariance=None, data_vector_length=4)
    assert isinstance(covariance, ScalarEmpiricalCovariance)
    residuals = np.ones(4)
    assert covariance.generic_loss_callable(residuals) == pytest.approx(-0.5 * 4 / 0.25)
    closure = covariance.create_loss_callable(covariance.inverse_metadata, covariance.data_vector_length)
    assert closure(residuals) == covariance.generic_loss_callable(residuals)


def test_empirical_likelihood_covariance_is_an_independent_copy():
    compressor_covariance = ScalarEmpiricalCovariance(2.0, data_vector_length=3)
    copied = likelihood_covariance("empirical", compressor_covariance, data_vector_length=3)
    assert copied is not compressor_covariance
    assert copied.C_inverse == compressor_covariance.C_inverse


def test_unknown_likelihood_covariance_option_is_rejected():
    with pytest.raises(InvalidConfiguration):
        likelihood_covariance("kolb", compressor_covariance=None, data_vector_length=3)
