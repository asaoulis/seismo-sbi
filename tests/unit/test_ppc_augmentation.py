"""Tests for the PosteriorPredictiveChecks nuisance-augmentation path.

The legacy integer-shift `random_shift_distributions` mechanism was migrated to the
unified `SeismogramEffect` chain (`augmentation_chains` / `augmentation_nuisance_params`).
These tests cover the `_apply_augmentation` helper (the per-ensemble chain application),
the vec<->map round-trip it relies on, and the warn-on-failure fallback.
"""

import numpy as np
import pytest

from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
from seismo_sbi.instaseis_simulator.post_processing import (
    PostProcessingChain,
    AmplitudeErrorEffect,
    SeismogramEffect,
)
from seismo_sbi.plotting.posterior_predictive_checks import PosteriorPredictiveChecks

T = 16


def _receivers():
    return Receivers(receivers=[
        Receiver(0.0, 0.0, "XX", "STA1", ["Z"]),
        Receiver(1.0, 1.0, "XX", "STA2", ["Z"]),
    ])


def _ppc(augmentation_chains=None, augmentation_nuisance_params=None):
    return PosteriorPredictiveChecks(
        simulator=lambda p: None,
        receivers=_receivers(),
        n_traces=2,
        n_jobs=1,
        augmentation_chains=augmentation_chains,
        augmentation_nuisance_params=augmentation_nuisance_params,
    )


def _flat_vec():
    # Two stations (Z only) concatenated → length 2*T.
    return np.concatenate([np.arange(T, dtype=float) + 1.0,
                           np.arange(T, dtype=float) + 100.0])


def test_vec_map_roundtrip():
    ppc = _ppc()
    vec = _flat_vec()
    out = ppc._outputs_map_to_vec(ppc._vec_to_outputs_map(vec))
    assert np.allclose(out, vec)


def test_apply_augmentation_scales_synthetics():
    chain = PostProcessingChain([AmplitudeErrorEffect(scale_range=(2.0, 2.0))])
    ppc = _ppc({"ens": chain}, {"ens": {"amplitude_error": 1.0}})
    vec = _flat_vec()
    np.random.seed(0)
    out = ppc._apply_augmentation([vec], ensemble_name="ens")
    assert len(out) == 1
    assert np.allclose(out[0], 2.0 * vec)          # prob=1, deterministic scale ⇒ doubled


def test_apply_augmentation_noop_when_no_chain_for_ensemble():
    chain = PostProcessingChain([AmplitudeErrorEffect(scale_range=(2.0, 2.0))])
    ppc = _ppc({"ens": chain}, {"ens": {"amplitude_error": 1.0}})
    vec = _flat_vec()
    out = ppc._apply_augmentation([vec], ensemble_name="other")   # not in dict
    assert np.allclose(out[0], vec)


def test_apply_augmentation_noop_when_ensemble_name_none():
    chain = PostProcessingChain([AmplitudeErrorEffect(scale_range=(2.0, 2.0))])
    ppc = _ppc({"ens": chain}, {"ens": {"amplitude_error": 1.0}})
    vec = _flat_vec()
    out = ppc._apply_augmentation([vec], ensemble_name=None)
    assert np.allclose(out[0], vec)


class _RaisingEffect(SeismogramEffect):
    def __call__(self, seismograms_map, receivers, **nuisance_params):
        raise RuntimeError("boom")


def test_apply_augmentation_warns_and_falls_back_on_failure():
    """A failing chain must warn (not silently swallow) and keep the un-augmented vec."""
    chain = PostProcessingChain([_RaisingEffect()])
    ppc = _ppc({"ens": chain}, {"ens": {"amplitude_error": 1.0}})
    vec = _flat_vec()
    with pytest.warns(UserWarning, match="augmentation failed"):
        out = ppc._apply_augmentation([vec], ensemble_name="ens")
    assert np.allclose(out[0], vec)                # fell back to original
