"""Unit tests for the pre-event-noise SNR metrics + gates (Fable-designed).

Covers the SNRMetrics maths (noise floor of 1, debiased signal SNR, synthetic signal
window) and the calibrated gate set (dead / unrecognisable / excess / conditional-fit /
sigma-outlier / event-contamination), plus the default-OFF guarantee, the RETIREMENT of
the old G2 below-noise drop (expected-low-signal traces are kept), and the
station-collapse rule.
"""
import numpy as np

from seismo_sbi.data_quality.metrics import (
    SNRMetrics, TraceDescriptor, signal_window, snr_metrics, TraceMetrics)
from seismo_sbi.data_quality.policy import (
    QAThresholds, decide_component, component_verdicts, event_contamination,
    sigma_outlier_verdicts, snr_station_drop, _snr_component_gate)


def _pulse(n=400, c=200, w=15, amp=1.0):
    t = np.arange(n)
    return amp * np.exp(-0.5 * ((t - c) / w) ** 2) * np.cos((t - c) / 4.0)


def _tr(sta="AAA", comp="Z"):
    return TraceDescriptor(sta, comp, 0.0, 0.0)


# --------------------------------------------------------------------- SNRMetrics maths
def test_signal_window_brackets_synthetic_energy():
    syn = _pulse()
    lo, hi = signal_window(syn, (0.05, 0.95))
    assert 0 < lo < 200 < hi < 400          # centred on the pulse
    # degenerate synthetic -> full window
    assert signal_window(np.zeros(400)) == (0, 400)


def test_snr_clean_signal_high_and_debiased():
    syn = _pulse(amp=1.0)
    obs = syn.copy()
    sigma = 0.01
    m = snr_metrics(obs[None], syn[None], [_tr()], {("AAA", "Z"): sigma})[0]
    assert m.snr_syn > 20 and m.snr_obs > 20
    # obs == syn (noise-free here) -> snr_sig ~ snr_obs (well above the floor)
    assert abs(m.snr_sig - np.sqrt(m.snr_obs ** 2 - 1)) < 1e-6


def test_snr_pure_noise_obs_floor_is_one():
    """A pure-noise observation (RMS ~ sigma) has snr_obs ~ 1 and snr_sig ~ 0."""
    syn = _pulse(amp=1.0)
    sigma = 0.3
    rng = np.random.default_rng(0)
    obs = rng.normal(0, sigma, syn.shape)
    m = snr_metrics(obs[None], syn[None], [_tr()], {("AAA", "Z"): sigma})[0]
    assert 0.6 < m.snr_obs < 1.6            # ~1 within sampling noise
    assert m.snr_sig < 1.0                  # debiased signal ~ 0


def test_snr_dead_channel_zero_and_missing_sigma_nan():
    syn = _pulse(amp=1.0)
    obs = np.zeros_like(syn)
    m = snr_metrics(obs[None], syn[None], [_tr()], {("AAA", "Z"): 0.01})[0]
    assert m.snr_obs == 0.0 and m.snr_sig == 0.0 and m.snr_syn > 20
    # missing sigma -> nan sigma, all SNR 0
    m2 = snr_metrics(obs[None], syn[None], [_tr()], {})[0]
    assert not np.isfinite(m2.sigma) and m2.snr_syn == 0.0


# ------------------------------------------------------------------------- the gates
def _snr(snr_syn, snr_sig, snr_obs_full=1.0, snr_syn_full=1.0, sigma=0.1):
    snr_obs = np.sqrt(snr_sig ** 2 + 1.0)
    return SNRMetrics("AAA", "Z", sigma, snr_obs, snr_syn, snr_sig, snr_obs_full, snr_syn_full)


def test_gate_dead_fires_but_spares_benign_misfit():
    t = QAThresholds(enable_snr_gates=True)
    # KSN-like: strong prediction, ~no observed signal
    assert _snr_component_gate(_snr(snr_syn=10, snr_sig=0.05), t) == "drop-snr-dead"
    # benign 1-D misfit: obs ~0.5x prediction -> snr_sig ~5 vs snr_syn 10 -> KEEP
    assert _snr_component_gate(_snr(snr_syn=10, snr_sig=5.0), t) is None


def test_below_noise_is_KEPT_g2_retired():
    """The old G2 'below-noise' drop is retired: an expected-low-signal trace
    (snr_syn < snr_syn_min) is uninformative, NOT bad — it must be kept."""
    t = QAThresholds(enable_snr_gates=True)
    assert _snr_component_gate(_snr(snr_syn=1.5, snr_sig=1.0), t) is None
    assert _snr_component_gate(_snr(snr_syn=0.01, snr_sig=0.0), t) is None
    tm = TraceMetrics("AAA", "Z", 100.0, 45.0, 0.0, 0.0, 0.05, 0, 1.0, 1.0, 1.0)
    # even with terrible xcorr/amplitude, conditional fit gates keep an expected-quiet trace
    tc = QAThresholds(enable_snr_gates=True, conditional_fit_gates=True)
    assert decide_component(tm, tc, snr=_snr(snr_syn=1.5, snr_sig=1.0)).verdict == "keep"


def test_gate_excess():
    t = QAThresholds(enable_snr_gates=True)
    # excess is OPT-IN (fragile under amplitude bias): off by default even with SNR gates on
    excess_snr = _snr(snr_syn=8, snr_sig=6, snr_obs_full=40, snr_syn_full=5)
    assert _snr_component_gate(excess_snr, t) is None
    te = QAThresholds(enable_snr_gates=True, enable_snr_excess=True)
    assert _snr_component_gate(excess_snr, te) == "drop-snr-excess"
    # NOT conditioned on snr_syn: an interloper at an expected-quiet station still fires
    quiet_interloper = _snr(snr_syn=0.5, snr_sig=30, snr_obs_full=30, snr_syn_full=0.5)
    assert _snr_component_gate(quiet_interloper, te) == "drop-snr-excess"
    # never fires when obs is within the budget (factor 5 -> energy 25x)
    ok = _snr(snr_syn=8, snr_sig=6, snr_obs_full=10, snr_syn_full=5)
    assert _snr_component_gate(ok, te) is None


def test_gate_dead_unrecognisable_branch():
    """G1u: marginal observed energy AND no waveform match at any lag -> dead.
    (Glitch peaks defeat a peak-amplitude guard; energy + coherence do not.)"""
    t = QAThresholds(enable_snr_gates=True, snr_dead_unrecog_ratio=0.25)
    tm_bad = TraceMetrics("AAA", "Z", 100.0, 45.0, 0.0, 0.0, 0.05, 0, 1.4, 1.0, 1.0)
    # snr_sig/snr_syn = 0.15: above the plain dead ratio (0.1), below unrecog (0.25)
    marginal = _snr(snr_syn=10, snr_sig=1.5)
    assert _snr_component_gate(marginal, t, tm_bad) == "drop-snr-dead"
    # same energies but the waveform matches at some lag -> kept
    tm_ok = TraceMetrics("AAA", "Z", 100.0, 45.0, 0.0, 0.0, 0.5, 3, 1.4, 1.0, 1.0)
    assert _snr_component_gate(marginal, t, tm_ok) is None
    # branch is opt-in: default thresholds ignore it even with metrics supplied
    assert _snr_component_gate(marginal, QAThresholds(enable_snr_gates=True), tm_bad) is None
    # and without TraceMetrics the branch cannot run
    assert _snr_component_gate(marginal, t, None) is None


def test_conditional_fit_gates():
    """Classical xcorr/amp gates fire ONLY where signal is expected AND observed."""
    t = QAThresholds(enable_snr_gates=True, conditional_fit_gates=True,
                     xcorr_drop=0.2, amp_lo=0.1, amp_hi=5.0)
    bad_fit = TraceMetrics("AAA", "Z", 100.0, 45.0, 0.0, 0.0, 0.1, 0, 1.0, 1.0, 1.0)
    # expected + observed + incoherent -> drop-corr
    assert decide_component(bad_fit, t, snr=_snr(10, 5.0)).verdict == "drop-corr"
    # expected but NOT observed (snr_sig < 2, above dead ratio) -> keep (never judge noise)
    assert decide_component(bad_fit, t, snr=_snr(10, 1.5)).verdict == "keep"
    # not expected (snr_syn < 5) -> keep regardless of fit
    assert decide_component(bad_fit, t, snr=_snr(3, 2.5)).verdict == "keep"
    # amplitude band still fires when expected + observed + coherent (SBR-Z-style gain error)
    gain_err = TraceMetrics("AAA", "Z", 100.0, 45.0, 0.7, 0.7, 0.7, 0, 0.05, 1.0, 20.0)
    assert decide_component(gain_err, t, snr=_snr(30, 3.0)).verdict == "drop-amp"
    # without SNR metrics the legacy unconditional path applies unchanged
    assert decide_component(bad_fit, t, snr=None).verdict == "drop-corr"


def test_gate_degenerate_sigma_is_dead():
    t = QAThresholds(enable_snr_gates=True)
    bad = SNRMetrics("AAA", "Z", float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0)
    assert _snr_component_gate(bad, t) == "drop-snr-dead"
    zero = SNRMetrics("AAA", "Z", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert _snr_component_gate(zero, t) == "drop-snr-dead"


def test_gate_off_by_default():
    assert _snr_component_gate(_snr(snr_syn=10, snr_sig=0.01), QAThresholds()) is None
    assert _snr_component_gate(None, QAThresholds(enable_snr_gates=True)) is None


def test_decide_component_snr_takes_priority_and_default_noop():
    tm = TraceMetrics("AAA", "Z", 100.0, 45.0, 0.9, 0.9, 0.9, 0, 1.0, 100.0, 100.0)
    # gates off -> classical keep, SNR ignored even if dead
    assert decide_component(tm, QAThresholds(), snr=_snr(10, 0.01)).verdict == "keep"
    # gates on -> dead SNR overrides an otherwise-coherent trace
    v = decide_component(tm, QAThresholds(enable_snr_gates=True), snr=_snr(10, 0.01))
    assert v.verdict == "drop-snr-dead"


def _snr_c(comp, snr_syn, snr_sig):
    snr_obs = np.sqrt(snr_sig ** 2 + 1.0)
    return SNRMetrics("AAA", comp, 0.1, snr_obs, snr_syn, snr_sig, 1.0, 1.0)


def test_component_verdicts_with_snr_and_station_collapse():
    t = QAThresholds(enable_snr_gates=True)
    mets = [TraceMetrics("AAA", c, 1, 1, 0.9, 0.9, 0.9, 0, 1.0, 1.0, 1.0) for c in ("Z", "1", "2")]
    snrs = [_snr_c("Z", 10, 0.01), _snr_c("1", 10, 5.0), _snr_c("2", 10, 5.0)]  # only Z dead
    cv = component_verdicts(mets, t, snr_metrics=snrs)
    assert cv["AAA"]["Z"].verdict == "drop-snr-dead"
    assert cv["AAA"]["1"].verdict == "keep"
    # Z failed -> whole-station SNR drop
    assert snr_station_drop(cv["AAA"]) == "drop-snr-dead"
    # only one non-Z fails -> no station collapse
    cv2 = {"1": type(cv["AAA"]["1"])("AAA", "1", "drop-snr-noise", 0.9, 1.0),
           "2": cv["AAA"]["2"], "Z": cv["AAA"]["1"]}
    assert snr_station_drop(cv2) is None


# ------------------------------------------------------- sigma-outlier channel health
def _snr_sigma(sta, comp, sigma):
    return SNRMetrics(sta, comp, sigma, 1.0, 1.0, 0.0, 1.0, 1.0)


def test_sigma_outlier_flags_broken_channel_and_default_off():
    snrs = [_snr_sigma(s, "Z", 1e-8) for s in ("AAA", "BBB", "CCC", "DDD")]
    snrs.append(_snr_sigma("YMZ", "Z", 1e-5))            # ~1000x the network median
    # opt-in: default thresholds -> no-op
    assert sigma_outlier_verdicts(snrs, QAThresholds()) == {}
    t = QAThresholds(sigma_rel_max=50.0)
    out = sigma_outlier_verdicts(snrs, t)
    assert out == {("YMZ", "Z"): "drop-snr-noisy"}
    # medians are PER COMPONENT: a noisy horizontal population doesn't mask a broken Z
    snrs += [_snr_sigma(s, "E", 5e-7) for s in ("AAA", "BBB", "CCC")]
    assert sigma_outlier_verdicts(snrs, t) == {("YMZ", "Z"): "drop-snr-noisy"}


def test_sigma_outlier_escape_hatch_spares_matching_waveform():
    """A pre-window spike can inflate sigma on a good trace; a visible waveform match
    (xcorr >= 0.4, sane amplitude) escapes the drop."""
    snrs = [_snr_sigma(s, "Z", 1e-8) for s in ("AAA", "BBB", "CCC", "DDD")]
    snrs.append(_snr_sigma("OKW", "Z", 1e-5))
    t = QAThresholds(sigma_rel_max=50.0)
    match = [TraceMetrics("OKW", "Z", 1, 1, 0.6, 0.6, 0.64, 2, 0.7, 1.0, 1.0)]
    assert sigma_outlier_verdicts(snrs, t, metrics=match) == {}
    garbage = [TraceMetrics("OKW", "Z", 1, 1, 0.0, 0.0, 0.35, 47, 700.0, 1.0, 1.0)]
    assert sigma_outlier_verdicts(snrs, t, metrics=garbage) == {("OKW", "Z"): "drop-snr-noisy"}


# ----------------------------------------------------------- event contamination flag
def _make_event(n_sta, n_bad, xc_good=0.55, xc_bad=0.12):
    """Build metrics/snrs/verdicts for one synthetic event: n_bad of n_sta stations
    dropped-incoherent (interloper), the rest kept."""
    mets, snrs, verdicts = [], [], {}
    from seismo_sbi.data_quality.policy import ComponentVerdict
    for i in range(n_sta):
        sta = f"S{i:02d}"
        bad = i < n_bad
        xc = xc_bad if bad else xc_good
        mets.append(TraceMetrics(sta, "Z", 1, 1, 0.5, 0.5, xc, 0, 0.6, 1.0, 1.0))
        snrs.append(SNRMetrics(sta, "Z", 0.1, 10.0, 10.0, 8.0, 10.0, 10.0))
        verdicts.setdefault(sta, {})["Z"] = ComponentVerdict(
            sta, "Z", "drop-corr" if bad else "keep", xc, 0.6)
    return mets, snrs, verdicts


def test_event_contamination_flags_widespread_failure():
    r = event_contamination(*_make_event(20, 10))
    assert r["contaminated"] == 1.0 and r["frac_expected_dropped"] == 0.5
    # soft branch: modest drop fraction but network median xcorr collapsed
    r2 = event_contamination(*_make_event(20, 5, xc_good=0.25))
    assert r2["contaminated"] == 1.0 and r2["median_xcorr_expected"] < 0.3


def test_event_contamination_clean_event_not_flagged():
    r = event_contamination(*_make_event(20, 1))
    assert r["contaminated"] == 0.0
    assert 0.0 < r["frac_expected_dropped"] < 0.1
    assert r["median_xcorr_expected"] > 0.5


def test_event_contamination_needs_enough_expected_traces_and_honours_exclude():
    # too few expected-signal traces -> statistic meaningless, never flagged
    r = event_contamination(*_make_event(4, 4))
    assert r["contaminated"] == 0.0 and r["n_expected"] == 4.0
    # excluding the persistent-bad channels removes them from the statistic
    mets, snrs, verdicts = _make_event(20, 10)
    excl = tuple((f"S{i:02d}", "Z") for i in range(10))
    r2 = event_contamination(mets, snrs, verdicts, exclude=excl)
    assert r2["contaminated"] == 0.0 and r2["frac_expected_dropped"] == 0.0
