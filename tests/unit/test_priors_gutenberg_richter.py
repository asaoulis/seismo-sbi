"""Unit tests for the Gutenberg-Richter magnitude statistics."""

import numpy as np
import pytest
from scipy import stats

from seismo_sbi.priors.gutenberg_richter import (
    GutenbergRichterModel,
    estimate_mc_maxcurvature,
    fit_b_value_aki,
)


def test_aki_b_value_recovers_known_b_on_continuous_exponential():
    rng = np.random.default_rng(0)
    b_true = 1.0
    beta = b_true * np.log(10.0)
    mc = 2.0
    # continuous (unbinned) exponential above Mc -> delta_m = 0 recovers b exactly
    mags = mc + rng.exponential(scale=1.0 / beta, size=400_000)
    b_hat = fit_b_value_aki(mags, mc, delta_m=0.0)
    assert abs(b_hat - b_true) < 0.02


def test_truncated_gr_sampling_matches_analytic_cdf():
    rng = np.random.default_rng(1)
    model = GutenbergRichterModel(b_value=0.9, mw_min=1.5, mw_max=6.0)
    samples = model.sample_magnitudes(20_000, rng)
    assert samples.min() >= model.mw_min - 1e-9
    assert samples.max() <= model.mw_max + 1e-9
    # KS test against the analytic truncated-exponential CDF
    res = stats.kstest(samples, model.cdf)
    assert res.pvalue > 0.01


def test_estimate_mc_maxcurvature_finds_mode():
    # frequency-magnitude distribution peaked at 2.5
    mags = np.concatenate([np.full(2000, 2.5), np.linspace(1.0, 5.0, 300)])
    mc = estimate_mc_maxcurvature(mags, delta_m=0.1)
    assert abs(mc - 2.5) <= 0.15


def test_gr_model_validates_inputs():
    with pytest.raises(ValueError):
        GutenbergRichterModel(b_value=1.0, mw_min=5.0, mw_max=5.0)
    with pytest.raises(ValueError):
        GutenbergRichterModel(b_value=-1.0, mw_min=1.0, mw_max=5.0)
