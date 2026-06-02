"""Unit tests for seismo_sbi.data_quality.metrics (deterministic, no Instaseis)."""
import numpy as np
import pytest

from seismo_sbi.data_quality import metrics as M
from seismo_sbi.data_quality.metrics import TraceDescriptor
from seismo_sbi.instaseis_simulator.utils import shift_1d_with_padding


def _wave(n=128, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)
    return np.sin(t) + 0.1 * rng.standard_normal(n)


# --------------------------------------------------------------------- align_best_lag
@pytest.mark.parametrize("k", [-5, -1, 0, 3, 7])
def test_align_best_lag_recovers_known_shift(k):
    syn = _wave()
    obs = shift_1d_with_padding(syn, k)  # obs is syn delayed by k
    xc, lag = M.align_best_lag(obs, syn, max_lag=20)
    assert lag == k
    assert xc > 0.99


def test_align_best_lag_sign_matches_shift_primitive():
    """+lag must mean 'delay syn', i.e. shift_1d_with_padding(syn, +lag) ~ obs."""
    syn = _wave(seed=1)
    obs = shift_1d_with_padding(syn, 4)
    _, lag = M.align_best_lag(obs, syn, max_lag=20)
    assert np.corrcoef(obs[10:-10], shift_1d_with_padding(syn, lag)[10:-10])[0, 1] > 0.99


def test_align_best_lag_zero_variance_guard():
    z = np.zeros(64)
    assert M.align_best_lag(z, z, max_lag=10) == (0.0, 0)
    assert M.align_best_lag(_wave(64), np.full(64, 3.0), max_lag=10) == (0.0, 0)


# ------------------------------------------------------------- variance reduction etc
def test_variance_reduction_identical_is_one():
    o = _wave()
    assert M.variance_reduction(o, o) == pytest.approx(1.0)


def test_variance_reduction_zero_obs_is_nan():
    assert np.isnan(M.variance_reduction(np.zeros(32), _wave(32)))


def test_aligned_vr_at_least_raw_for_pure_shift():
    syn = _wave(seed=2)
    obs = shift_1d_with_padding(syn, 5)
    _, lag = M.align_best_lag(obs, syn, 20)
    assert M.aligned_variance_reduction(obs, syn, lag) >= M.variance_reduction(obs, syn)
    assert M.aligned_variance_reduction(obs, syn, lag) == pytest.approx(1.0, abs=1e-9)


def test_peak_amplitude_ratio():
    o = _wave()
    assert M.peak_amplitude_ratio(o, 2.0 * o) == pytest.approx(0.5)
    assert M.peak_amplitude_ratio(o, np.zeros_like(o)) == float("inf")


# -------------------------------------------------------------- compute_trace_metrics
def test_compute_trace_metrics_shapes_and_fields():
    n_t = 64
    traces = [
        TraceDescriptor("AAA", "Z", 1.0, 1.0),
        TraceDescriptor("AAA", "E", 1.0, 1.0),
        TraceDescriptor("BBB", "Z", 2.0, 2.0),
    ]
    obs = np.stack([_wave(n_t, s) for s in range(3)])
    syn = np.stack([shift_1d_with_padding(obs[i], 2) for i in range(3)])
    rows = M.compute_trace_metrics(obs, syn, traces, src_lat=0.0, src_lon=0.0, max_lag=15)
    assert len(rows) == 3
    assert rows[0].station == "AAA" and rows[0].component == "Z"
    assert all(np.isfinite([r.dist_km, r.azimuth]).all() for r in rows)
    # best lag aligning syn(=obs delayed by 2) back to obs is -2
    assert rows[0].best_lag_samples == -2
    # to_row preserves the legacy CSV column order
    assert list(rows[0].to_row().keys()) == [
        "station", "component", "dist_km", "azimuth", "vr", "aligned_vr",
        "max_xcorr", "best_lag_samples", "amp_ratio_obs_syn", "obs_peak", "syn_peak"]


# ----------------------------------------------------------- PPC-derived per-station
def test_correlation_misfit_identical_and_anticorrelated():
    sta = np.stack([_wave(seed=4), _wave(seed=5)])
    assert M.correlation_misfit(sta, sta) == pytest.approx(0.0, abs=1e-9)
    assert M.correlation_misfit(sta, -sta) == pytest.approx(2.0, abs=1e-9)


def test_envelope_misfit_identical_is_zero_or_nan():
    sta = np.stack([_wave(seed=6), _wave(seed=7)])
    val = M.envelope_misfit(sta, sta)
    if not np.isnan(val):  # scipy present
        assert val == pytest.approx(0.0, abs=1e-9)


def test_station_reduced_chi2_fallback_and_covariance():
    o = np.stack([_wave(seed=8)])
    s = np.stack([_wave(seed=9)])
    r = (o - s).ravel()
    # no covariance -> r.r / dof
    assert M.station_reduced_chi2(o, s) == pytest.approx(np.dot(r, r) / r.size)
    assert M.station_reduced_chi2(o, s, dof=2) == pytest.approx(np.dot(r, r) / 2)

    class _Cov:  # compute_loss = -0.5 * chi2  =>  chi2 = -2 * compute_loss = r.r
        def compute_loss(self, resid, reduce=True):
            return -0.5 * float(np.dot(resid, resid))

    assert M.station_reduced_chi2(o, s, covariance=_Cov()) == pytest.approx(
        np.dot(r, r) / r.size)
