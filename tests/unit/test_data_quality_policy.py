"""Unit tests for seismo_sbi.data_quality.policy decision logic."""
import numpy as np
import pytest

from seismo_sbi.data_quality.metrics import TraceMetrics
from seismo_sbi.data_quality.policy import (
    QAThresholds, StationSummary, decide_station, summarise_event, summarise_station)


def _summary(**overrides):
    """A station summary that, by default, passes every gate (-> 'keep')."""
    base = dict(dist_km=10.0, azimuth=45.0, vr={"Z": 0.5}, aligned_vr={"Z": 0.6},
                lag_Z=0, xcorr_Z=0.9, median_amp_ratio=1.0, aligned_coherence=0.9)
    base.update(overrides)
    return StationSummary(**base)


def _tm(station, component, **kw):
    base = dict(dist_km=10.0, azimuth=45.0, vr=0.5, aligned_vr=0.6, max_xcorr=0.9,
                best_lag_samples=0, amp_ratio_obs_syn=1.0, obs_peak=1.0, syn_peak=1.0)
    base.update(kw)
    return TraceMetrics(station=station, component=component, **base)


T = QAThresholds()


def test_keep_when_all_gates_pass():
    assert decide_station(_summary(), T).verdict == "keep"


def test_drop_corr_on_low_z_xcorr():
    assert decide_station(_summary(xcorr_Z=0.4, aligned_coherence=0.4), T).verdict == "drop-corr"


@pytest.mark.parametrize("ratio", [25.0, 0.01])
def test_drop_amp_on_extreme_amplitude(ratio):
    assert decide_station(_summary(median_amp_ratio=ratio), T).verdict == "drop-amp"


def test_time_shift_when_lag_large_and_coherent():
    v = decide_station(_summary(lag_Z=4, xcorr_Z=0.7), T)
    assert v.verdict == "time-shift"
    assert v.suggested_shift == 4


def test_small_lag_stays_keep_with_zero_shift():
    v = decide_station(_summary(lag_Z=1, xcorr_Z=0.7), T)
    assert v.verdict == "keep"
    assert v.suggested_shift == 0


def test_large_lag_but_poor_xcorr_is_not_shifted():
    # |lag|>=2 but xcorr below xcorr_shift_ok -> not a trustworthy shift -> keep
    assert decide_station(_summary(lag_Z=5, xcorr_Z=0.52), T).verdict == "keep"


# --------------------------------------------------------------- precedence ordering
def test_corr_beats_amp():
    # incoherent AND bad amplitude -> drop-corr wins (coherence judged first)
    assert decide_station(_summary(xcorr_Z=0.3, median_amp_ratio=100.0), T).verdict == "drop-corr"


def test_amp_beats_ppc_and_shift():
    assert decide_station(
        _summary(median_amp_ratio=50.0, aligned_coherence=0.1, lag_Z=5), T).verdict == "drop-amp"


# --------------------------------------------------------------- PPC coherence gate
def test_ppc_coherence_gate_drops_when_enabled():
    v = decide_station(_summary(aligned_coherence=0.3), QAThresholds(enable_ppc_drops=True))
    assert v.verdict == "drop-fit"


def test_ppc_gate_off_preserves_legacy_verdict():
    # aligned_coherence below the gate, but gates OFF -> falls through to keep/shift
    assert decide_station(
        _summary(aligned_coherence=0.3), QAThresholds(enable_ppc_drops=False)).verdict == "keep"


def test_optional_zero_lag_misfit_gates_off_by_default():
    s = _summary(corr_misfit=1.9, envelope_misfit=5.0)
    assert decide_station(s, QAThresholds()).verdict == "keep"  # thresholds None -> inactive
    assert decide_station(s, QAThresholds(corr_misfit_drop=1.0)).verdict == "drop-fit"


# ------------------------------------------------------------------ summarise chain
def test_summarise_station_collapses_components():
    ms = [_tm("AAA", "Z", max_xcorr=0.8, best_lag_samples=3, amp_ratio_obs_syn=2.0),
          _tm("AAA", "E", max_xcorr=0.9, amp_ratio_obs_syn=4.0),
          _tm("AAA", "N", max_xcorr=1.0, amp_ratio_obs_syn=6.0)]
    s = summarise_station(ms)
    assert s.lag_Z == 3 and s.xcorr_Z == 0.8
    assert s.median_amp_ratio == pytest.approx(4.0)
    assert s.aligned_coherence == pytest.approx(0.9)
    assert set(s.vr) == {"Z", "E", "N"}


def test_summarise_event_groups_by_station():
    ms = [_tm("AAA", "Z", max_xcorr=0.9, best_lag_samples=4),
          _tm("AAA", "E"), _tm("BBB", "Z", max_xcorr=0.9)]
    out = summarise_event(ms, T)
    assert set(out) == {"AAA", "BBB"}
    assert out["AAA"].verdict == "time-shift" and out["AAA"].suggested_shift == 4
    assert out["BBB"].suggested_shift == 0


# ------------------------------------------------------------- per-component verdicts
from seismo_sbi.data_quality.policy import (  # noqa: E402
    ComponentVerdict, component_verdicts, decide_component)
from seismo_sbi.data_quality.serialization import components_from_verdicts  # noqa: E402


def test_decide_component_keep_drop_corr_drop_amp():
    assert decide_component(_tm("A", "Z", max_xcorr=0.9, amp_ratio_obs_syn=1.0), T).verdict == "keep"
    assert decide_component(_tm("A", "E", max_xcorr=0.1), T).verdict == "drop-corr"
    assert decide_component(_tm("A", "N", amp_ratio_obs_syn=99.0), T).verdict == "drop-amp"
    # coherence is judged before amplitude (matches the station policy priority)
    v = decide_component(_tm("A", "Z", max_xcorr=0.1, amp_ratio_obs_syn=99.0), T)
    assert v.verdict == "drop-corr" and isinstance(v, ComponentVerdict)


def test_component_verdicts_group_by_station_component():
    ms = [_tm("AAA", "Z"), _tm("AAA", "E", max_xcorr=0.1), _tm("BBB", "Z")]
    cv = component_verdicts(ms, T)
    assert set(cv) == {"AAA", "BBB"}
    assert cv["AAA"]["Z"].is_kept and cv["AAA"]["E"].is_dropped
    assert cv["BBB"]["Z"].is_kept


def test_components_from_verdicts_per_component_drops_one_channel():
    # AAA: station kept, but its N channel is incoherent -> keep [Z, E]
    # BBB: station dropped entirely -> []
    metrics = [_tm("AAA", "Z"), _tm("AAA", "E"), _tm("AAA", "N", max_xcorr=0.1),
               _tm("BBB", "Z", max_xcorr=0.1), _tm("BBB", "E", max_xcorr=0.1),
               _tm("BBB", "N", max_xcorr=0.1)]
    sv = summarise_event(metrics, T)
    cv = component_verdicts(metrics, T)
    comp_map = components_from_verdicts(sv, ["AAA", "BBB", "CCC"],
                                        component_verdicts=cv)
    assert comp_map["AAA"] == ["Z", "E"]      # dodgy N dropped
    assert comp_map["BBB"] == []              # whole station dropped
    assert comp_map["CCC"] == []              # absent from verdicts
    # backward-compat: without component_verdicts the kept station keeps all channels
    assert components_from_verdicts(sv, ["AAA"])["AAA"] == ["Z", "E", "N"]
