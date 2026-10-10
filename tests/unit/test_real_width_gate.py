"""The real-data width gate: the observation set round trip, the lune-area statistic, and the
callback's cadence and logged numbers on a synthetic set."""
import numpy as np
import pytest

from seismo_sbi.moment_tensor.lune_angles import lune_credible_area
from seismo_sbi.sbi.npe.training import real_width_gate
from seismo_sbi.sbi.npe.training.real_width_gate import (
    RealWidthGate, gate_callbacks, load_observation_set, write_observation)
from seismo_sbi.sbi.training_configuration import TrainingConfiguration


def narrow_samples(n=400, seed=0):
    """Double couples with a little scatter: a tightly resolved source type."""
    generator = np.random.default_rng(seed)
    base = np.array([0.0, 1.0, -1.0, 0.1, 0.0, 0.0])
    return base + 0.02 * generator.standard_normal((n, 6))


def wide_samples(n=400, seed=1):
    """Independent components: the whole lune."""
    return np.random.default_rng(seed).standard_normal((n, 6))


class LogRecorder:
    def __init__(self):
        self.flow, self.device, self.training, self.logged = object(), "cpu", True, {}

    def log(self, name, value, **_):
        self.logged[name] = value

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class FakeTrainer:
    def __init__(self, epoch):
        self.current_epoch, self.sanity_checking, self.global_rank = epoch, False, 0


def test_an_observation_set_round_trips(tmp_path):
    write_observation(tmp_path / "ev1.npz", np.ones((3, 3, 20)), np.zeros((3, 2)), [64.8, -16.9, 6.0, 0.0])
    write_observation(tmp_path / "ev2.npz", np.ones((2, 3, 20)), np.zeros((2, 2)), None, event_id="second")
    ids, items = load_observation_set(tmp_path)
    assert ids == ["ev1", "second"]
    assert items[0][0].shape == (3, 3, 20) and items[0][2].tolist() == pytest.approx([64.8, -16.9, 6.0, 0.0])
    assert items[1][2] is None
    with pytest.raises(FileNotFoundError):
        load_observation_set(tmp_path / "empty")


def test_a_tight_source_type_has_a_small_lune_area_and_a_broad_one_a_large_one():
    from seismo_sbi.moment_tensor.lune_angles import mts6_to_gamma_delta
    tight = lune_credible_area(*mts6_to_gamma_delta(narrow_samples()), 0.95)
    broad = lune_credible_area(*mts6_to_gamma_delta(wide_samples()), 0.95)
    assert tight < 0.05 < 0.5 < broad
    assert np.isnan(lune_credible_area([1.0, 1.0], [2.0, 2.0]))


def test_the_gate_logs_the_kept_fraction_on_its_epochs_only(monkeypatch):
    items = [(np.zeros((2, 3, 20)), np.zeros((2, 2)), None)] * 4
    samples = [narrow_samples(seed=s) for s in range(2)] + [wide_samples(seed=s) for s in range(2)]
    monkeypatch.setattr(real_width_gate, "sample_subsets_batched",
                        lambda posterior, items, scaler, **_: samples)
    gate = RealWidthGate(items, ["a", "b", "c", "d"], data_scaler=None, every_n_epochs=3, num_samples=10)
    module = LogRecorder()
    gate.on_validation_epoch_end(FakeTrainer(epoch=0), module)
    assert module.logged == {} and gate.history == []
    gate.on_validation_epoch_end(FakeTrainer(epoch=2), module)
    assert module.logged["real/kept_fraction"] == pytest.approx(0.5)
    assert 0.0 < module.logged["real/median_lune_area95"] < 1.0
    assert gate.history[0][0] == 2 and module.training is True


def test_the_configuration_block_builds_the_callback_or_nothing(tmp_path):
    write_observation(tmp_path / "ev.npz", np.ones((2, 3, 20)), np.zeros((2, 2)))
    training = TrainingConfiguration.from_yaml_block(
        {"ml_real_width_gate": {"observations": str(tmp_path), "every_n_epochs": 2, "max_lune_area": 0.3}})
    data = type("Data", (), {"data_scaler": None})()
    callbacks = gate_callbacks(training, data)
    assert len(callbacks) == 1 and callbacks[0].every_n_epochs == 2 and callbacks[0].max_lune_area == 0.3
    assert gate_callbacks(TrainingConfiguration.from_yaml_block({}), data) == []
    from seismo_sbi.utils.errors import InvalidConfiguration
    with pytest.raises(InvalidConfiguration):
        TrainingConfiguration.from_yaml_block({"ml_real_width_gate": {"observatons": str(tmp_path)}})


def test_non_finite_draws_are_dropped_and_a_mostly_non_finite_event_is_not_kept(monkeypatch):
    with_nan = narrow_samples()
    with_nan[:5] = np.nan
    with_nan[5, 2] = np.inf
    all_nan = np.full((400, 6), np.nan)
    items = [(np.zeros((2, 3, 20)), np.zeros((2, 2)), None)] * 2
    monkeypatch.setattr(real_width_gate, "sample_subsets_batched",
                        lambda posterior, items, scaler, **_: [with_nan, all_nan])
    gate = RealWidthGate(items, ["a", "b"], data_scaler=None, every_n_epochs=1, num_samples=10)
    module = LogRecorder()
    gate.on_validation_epoch_end(FakeTrainer(epoch=0), module)
    assert module.logged["real/kept_fraction"] == pytest.approx(0.5)
    assert module.logged["real/median_lune_area95"] < 0.05


def test_an_event_whose_eigenvalues_do_not_converge_gets_a_nan_area(monkeypatch):
    def fail(_):
        raise np.linalg.LinAlgError("Eigenvalues did not converge")
    monkeypatch.setattr(real_width_gate, "mts6_to_gamma_delta", fail)
    assert np.isnan(real_width_gate.finite_lune_area(narrow_samples()))
