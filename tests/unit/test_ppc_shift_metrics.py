#!/usr/bin/env python3
"""Shift-tolerant PPC metrics: ``Shifted corr misfit`` and ``Autocorr misfit``.

A zero-lag correlation cannot tell "wrong mechanism" from "right mechanism, wrong
arrival time" — and a 1-D velocity model plus a catalogue hypocentre guarantee the
latter.  These two metrics judge waveform SHAPE with the arrival time free, so a
solution is no longer penalised for travel-time error it did not cause.
"""
import numpy as np
import pytest

from seismo_sbi.evaluation.ppc_metrics import (
    autocorrelation_misfit, correlation_misfit_per_sample, shifted_correlation_misfit)
from seismo_sbi.plotting.posterior_predictive_checks import PosteriorPredictiveChecks


N_TRACES, TLEN = 4, 256


def _meta(**kw):
    """The metric ``meta`` for ``N_TRACES`` traces, a 30-sample autocorrelation lag and the default shift budget."""
    return {"n_traces": N_TRACES, "max_shift_samples": None, "autocorr_maxlag": 30, **kw}


def _wavelets(seed=0, n_traces=N_TRACES, tlen=TLEN):
    """Band-limited pulses — a realistic stand-in for filtered displacement traces."""
    rng = np.random.default_rng(seed)
    t = np.arange(tlen)
    out = np.empty((n_traces, tlen))
    for i in range(n_traces):
        c = tlen // 2 + rng.integers(-10, 10)
        w = 12.0 + rng.uniform(-2, 2)
        out[i] = np.sin((t - c) / w) * np.exp(-((t - c) / (3 * w)) ** 2)
    return out


def _shift(mat, k):
    """Linear (non-wrapping) shift by k samples — what a travel-time error does."""
    out = np.zeros_like(mat)
    if k >= 0:
        out[:, k:] = mat[:, : mat.shape[1] - k]
    else:
        out[:, :k] = mat[:, -k:]
    return out


@pytest.fixture
def obs():
    return _wavelets(0)


@pytest.mark.parametrize("metric", [shifted_correlation_misfit, autocorrelation_misfit])
def test_identical_traces_score_zero(obs, metric):
    got = metric(obs.ravel(), obs.ravel()[None, :], _meta())
    assert got.shape == (1,)
    assert got[0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("shift", [3, 8, -6])
def test_a_pure_time_shift_is_forgiven(obs, shift):
    """The headline property: the same waveform arriving late must not be scored as wrong."""
    syn = _shift(obs, shift).ravel()[None, :]
    meta = _meta()
    zero_lag = correlation_misfit_per_sample(obs.ravel(), syn, meta)[0]
    shifted = shifted_correlation_misfit(obs.ravel(), syn, meta)[0]
    autocorr = autocorrelation_misfit(obs.ravel(), syn, meta)[0]
    assert shifted < 0.05, "lag-optimised correlation should recover the match"
    assert autocorr < 0.05, "autocorrelation shape is shift-invariant"
    assert zero_lag > 5 * shifted, "the zero-lag metric is the one that punishes the shift"


def test_shifted_corr_still_punishes_a_polarity_flip(obs):
    """Shift tolerance must not become sign blindness — polarity IS the ISO physics.

    It survives only because the lag budget is smaller than half a dominant period: a
    half-period shift would realign an inverted wavelet. Measured here: flip scores 0.57
    with L=26 against a ~75-sample period. Keep L well under half a period.
    """
    syn = (-obs).ravel()[None, :]
    meta = _meta()
    flip = shifted_correlation_misfit(obs.ravel(), syn, meta)[0]
    match = shifted_correlation_misfit(obs.ravel(), _shift(obs, 5).ravel()[None, :], meta)[0]
    assert flip > 0.4, "an inverted waveform must be clearly penalised"
    assert flip > 10 * max(match, 1e-3)


def test_autocorr_is_blind_to_polarity_by_construction(obs):
    """GUARD, not a bug: ACF(-x) == ACF(x), so this metric cannot see the ISO sign.

    It is registered for shape/duration/frequency content only. Any +ISO vs -ISO claim must
    rest on the polarity analysis, the zero-lag correlation or `Shifted corr misfit` — never
    on this metric alone. If this test ever fails the metric changed meaning; do not "fix"
    it by loosening the assertion.
    """
    got = autocorrelation_misfit(obs.ravel(), (-obs).ravel()[None, :], _meta())
    assert got[0] == pytest.approx(0.0, abs=1e-9)


def test_a_genuinely_different_waveform_is_punished(obs):
    """Shift tolerance must not flatter an unrelated synthetic."""
    rng = np.random.default_rng(3)
    unrelated = rng.normal(size=obs.shape)
    meta = _meta()
    good = shifted_correlation_misfit(obs.ravel(), _shift(obs, 5).ravel()[None, :], meta)[0]
    bad = shifted_correlation_misfit(obs.ravel(), unrelated.ravel()[None, :], meta)[0]
    assert bad > 0.5 and bad > good + 0.4


def test_lag_optimisation_forgives_a_similar_wavelet_at_another_time(obs):
    """Documents the cost of the tolerance: a same-family pulse elsewhere scores well.

    Measured 0.02 vs 0.45 zero-lag. That is the intended trade — but it is why the shifted
    metric is reported ALONGSIDE the zero-lag one rather than replacing it.
    """
    meta = _meta()
    other = _wavelets(99)
    assert shifted_correlation_misfit(obs.ravel(), other.ravel()[None, :], meta)[0] < 0.1
    assert correlation_misfit_per_sample(obs.ravel(), other.ravel()[None, :], meta)[0] > 0.3


def test_the_lag_budget_is_bounded(obs):
    """A shift beyond the budget must NOT be recovered — else P could align onto S."""
    far = _shift(obs, 40).ravel()[None, :]
    near = _shift(obs, 3).ravel()[None, :]
    meta = _meta(max_shift_samples=4)
    assert shifted_correlation_misfit(obs.ravel(), near, meta)[0] < 0.05
    assert shifted_correlation_misfit(obs.ravel(), far, meta)[0] > 0.5


def test_the_default_lag_budget_scales_with_trace_length(obs):
    """Default L = 10 % of the trace, so the budget follows the window, not the sample rate."""
    meta = _meta()
    within = _shift(obs, int(0.05 * TLEN)).ravel()[None, :]
    beyond = _shift(obs, int(0.30 * TLEN)).ravel()[None, :]
    assert shifted_correlation_misfit(obs.ravel(), within, meta)[0] < 0.2
    assert shifted_correlation_misfit(obs.ravel(), beyond, meta)[0] > 0.5


def test_autocorr_is_computed_per_trace_not_over_the_concatenation(obs):
    """Run over concatenated traces the lag products straddle trace boundaries.

    Reordering the traces changes the concatenation but not the per-trace content, so a
    per-trace metric must be invariant to it and a whole-vector one must not be.
    """
    perm = obs[::-1]
    meta = _meta()
    a = autocorrelation_misfit(obs.ravel(), obs.ravel()[None, :], meta)[0]
    b = autocorrelation_misfit(perm.ravel(), perm.ravel()[None, :], meta)[0]
    assert a == pytest.approx(b, abs=1e-9)
    # cross-comparison: same trace SET, different order -> per-trace ACFs still pair up
    cross = autocorrelation_misfit(obs.ravel(), perm.ravel()[None, :], meta)[0]
    assert cross > a


def test_metrics_are_registered_by_default():
    p = PosteriorPredictiveChecks(simulator=lambda x: np.zeros(4), n_traces=1)
    assert "Shifted corr misfit" in p.metrics
    assert "Autocorr misfit" in p.metrics


def test_unknown_trace_layout_falls_back_without_crashing(obs):
    """n_traces that doesn't divide the vector must degrade, not raise."""
    got = shifted_correlation_misfit(obs.ravel(), obs.ravel()[None, :], _meta(n_traces=7))
    assert np.isfinite(got).all()
    got2 = autocorrelation_misfit(obs.ravel(), obs.ravel()[None, :], _meta(n_traces=7))
    assert np.isfinite(got2).all()


def test_batch_scoring_matches_one_at_a_time(obs):
    """Vectorised over the ensemble axis — each row must be scored independently."""
    meta = _meta()
    rows = [_shift(obs, k).ravel() for k in (0, 4, -9)]
    batch = shifted_correlation_misfit(obs.ravel(), np.array(rows), meta)
    singles = [shifted_correlation_misfit(obs.ravel(), r[None, :], meta)[0] for r in rows]
    assert batch == pytest.approx(singles, abs=1e-12)


def test_a_failing_metric_gives_nan_and_a_warning(obs):
    p = PosteriorPredictiveChecks(simulator=lambda x: np.zeros(4), n_traces=N_TRACES)

    def broken(obs, synthetics, meta):
        raise ValueError("no layout")

    p.add_metric("Broken", broken)
    with pytest.warns(UserWarning, match="Broken"):
        results = p.evaluate_metrics(obs.ravel(), {"NPE": obs.ravel()[None, :]})
    assert np.isnan(results["NPE"]["Broken"]["raw"]).all()
    assert results["NPE"]["MSE"]["raw"][0] == 0.0
