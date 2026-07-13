"""Unit tests for seismo_sbi.data_quality.guards — the promoted (2026-07 calibrated)
QA presets, model-free guards, and the shared per-component gate composition."""
import datetime as dt

import numpy as np
import pytest

from seismo_sbi.data_quality import (
    QAThresholds,
    TraceDescriptor,
    compose_component_qa,
    compute_trace_metrics,
    data_qa_thresholds,
    neighbour_window_flag,
    obs_dead_components,
    snr_metrics,
)


# ---------------------------------------------------------------- presets ----
def test_data_qa_thresholds_minimal():
    th = data_qa_thresholds("minimal")
    assert isinstance(th, QAThresholds)
    assert th.enable_snr_gates and th.enable_snr_excess
    assert th.snr_dead_ratio == 0.1 and th.snr_dead_min_syn == 5.0
    assert th.snr_dead_unrecog_ratio == 0.25 and th.xcorr_dead == 0.1
    assert th.sigma_rel_max == 50.0 and th.snr_excess_factor == 5.0
    # classical station-level gates neutralised
    assert th.xcorr_drop == 0.0 and th.amp_lo == 0.0 and th.amp_hi == 1e12
    assert not th.enable_ppc_drops and not th.conditional_fit_gates


def test_data_qa_thresholds_full_adds_conditional_fit():
    th = data_qa_thresholds("full")
    assert th.conditional_fit_gates
    assert th.snr_fit_min_syn == 5.0 and th.snr_fit_sig_min == 2.0
    assert th.xcorr_drop == 0.2 and th.amp_lo == 0.1 and th.amp_hi == 5.0


def test_data_qa_thresholds_overrides_and_bad_level():
    th = data_qa_thresholds("minimal", snr_excess_factor=7.0)
    assert th.snr_excess_factor == 7.0
    with pytest.raises(ValueError):
        data_qa_thresholds("mature")


# ------------------------------------------------------------- obs guards ----
def test_obs_dead_components_flags_flatline_and_relative_outlier():
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 1.0, size=(4, 3, 64))
    obs[1, 2, :] = 0.0                        # absolute flatline
    obs[3, 0, :] *= 1e-4                      # gross relative outlier vs peers
    present = ["AAA", "BBB", "CCC", "DDD"]
    dead = obs_dead_components(obs, present, ["Z", "E", "N"])
    assert dead[("BBB", "N")] == "drop-dead"
    assert dead[("DDD", "Z")] == "drop-dead"
    assert len(dead) == 2


def test_neighbour_window_flag():
    t0 = dt.datetime(2026, 1, 12, 0, 58, 0)
    others = [t0 - dt.timedelta(seconds=98), t0 - dt.timedelta(days=2)]
    out = neighbour_window_flag(t0, 800.0, others, pre_s=120.0)
    assert out["neighbour_in_window"] and out["nearest_neighbour_s"] == -98.0
    out2 = neighbour_window_flag(t0, 200.0, [t0 + dt.timedelta(seconds=500)])
    assert not out2["neighbour_in_window"]
    out3 = neighbour_window_flag(t0, 200.0, [])
    assert not out3["neighbour_in_window"]


def test_neighbour_window_flag_magnitude_relative():
    t0 = dt.datetime(2025, 2, 1, 12, 0, 0)
    times = [t0 + dt.timedelta(seconds=61), t0 + dt.timedelta(seconds=3600)]
    mags = [1.8, 4.0]
    # micro-event in window does NOT qualify (4.0 event analysed, delta 0.7)
    out = neighbour_window_flag(t0, 200.0, times, catalogue_mags=mags,
                                event_mag=4.0, delta_mag=0.7)
    assert not out["neighbour_in_window"]
    assert out["nearest_any_s"] == 61.0                  # context still recorded
    assert out["nearest_neighbour_s"] == 3600.0 and out["nearest_neighbour_mag"] == 4.0
    # comparable-size neighbour in window DOES qualify
    out2 = neighbour_window_flag(t0, 200.0, times, catalogue_mags=[3.5, 4.0],
                                 event_mag=4.0, delta_mag=0.7)
    assert out2["neighbour_in_window"] and out2["nearest_neighbour_s"] == 61.0
    assert out2["nearest_neighbour_mag"] == 3.5


# ------------------------------------------------------------ composition ----
def _make_event(n_sta=6, T=128, seed=1):
    """Healthy synthetic event: obs == syn + small noise, flat sigma."""
    rng = np.random.default_rng(seed)
    present = [f"S{i:02d}" for i in range(n_sta)]
    comps = ["Z", "E", "N"]
    traces = [TraceDescriptor(s, c, 36.0 + 0.01 * i, 25.0) for i, s in enumerate(present)
              for c in comps]
    t = np.linspace(0, 8 * np.pi, T)
    syn = np.stack([np.sin(t + 0.1 * i) * 5.0 for i in range(len(traces))])
    obs = syn + rng.normal(0, 0.05, syn.shape)
    sigma = {(s, c): 0.05 for s in present for c in comps}
    return present, comps, traces, obs, syn, sigma


def _qa(obs, syn, traces, sigma, present, comps, th, **kw):
    metrics = compute_trace_metrics(obs, syn, traces, 36.0, 25.0, 16)
    snr = snr_metrics(obs, syn, traces, sigma)
    obs3d = obs.reshape(len(present), len(comps), -1)
    return compose_component_qa(metrics, snr, present, comps, th, obs=obs3d, **kw)


def test_compose_healthy_event_keeps_everything():
    present, comps, traces, obs, syn, sigma = _make_event()
    comp_map, dropped, comp_drops, flags = _qa(obs, syn, traces, sigma, present, comps,
                                               data_qa_thresholds("full"),
                                               min_stations=3)
    assert set(comp_map) == set(present) and not dropped and not comp_drops
    assert not flags.get("contaminated")
    assert all(comp_map[s] == comps for s in present)


def test_compose_dead_channel_dropped_station_survives():
    present, comps, traces, obs, syn, sigma = _make_event()
    obs = obs.copy()
    obs[1] = 1e-13 * np.random.default_rng(2).normal(size=obs.shape[1])  # S00 E dead
    comp_map, dropped, comp_drops, _ = _qa(obs, syn, traces, sigma, present, comps,
                                           data_qa_thresholds("minimal"),
                                           min_stations=3)
    assert comp_drops.get(("S00", "E"), "").startswith("drop")
    assert "S00" in comp_map and "E" not in comp_map["S00"]   # station survives per-component
    assert not dropped


def test_compose_station_drops_only_when_no_component_survives():
    present, comps, traces, obs, syn, sigma = _make_event()
    obs = obs.copy()
    obs[0:3] = 0.0                                            # all of S00 flatlined
    comp_map, dropped, comp_drops, _ = _qa(obs, syn, traces, sigma, present, comps,
                                           data_qa_thresholds("minimal"),
                                           min_stations=3)
    assert "S00" not in comp_map and dropped["S00"].startswith("drop")


def test_compose_blocklist_applied_and_never_rank_filled_back():
    present, comps, traces, obs, syn, sigma = _make_event(n_sta=4)
    block = {("S00", "Z"), ("S00", "E"), ("S00", "N")}
    comp_map, dropped, comp_drops, _ = _qa(obs, syn, traces, sigma, present, comps,
                                           data_qa_thresholds("minimal"),
                                           blocklist=block,
                                           min_stations=4, min_fraction=1.0)
    # keep-floor demands all 4 stations, but S00 is fully blocklisted -> stays out
    assert "S00" not in comp_map
    assert dropped["S00"] == "drop-blocklist"
    assert all(comp_drops[k] == "drop-blocklist" for k in block)


def test_compose_keep_floor_rank_fills_non_blocklisted():
    present, comps, traces, obs, syn, sigma = _make_event(n_sta=4)
    obs = obs.copy()
    obs[3:6] *= 1e-13                                          # S01 all-dead
    comp_map, dropped, _, _ = _qa(obs, syn, traces, sigma, present, comps,
                                  data_qa_thresholds("minimal"),
                                  min_stations=4, min_fraction=1.0)
    # floor of 4 forces the dead S01 back in (rank-fill), full components restored
    assert set(comp_map) == set(present)
    assert comp_map["S01"] == comps and not dropped


def test_compose_contaminated_warn_keeps_all_but_health_drops():
    present, comps, traces, obs, syn, sigma = _make_event(n_sta=6)
    rng = np.random.default_rng(3)
    obs = obs.copy()
    # simulate an interloper: most channels incoherent + energetic vs the synthetic
    obs[3:] = rng.normal(0, 50.0, obs[3:].shape)
    obs[0] = 0.0                                               # S00 Z genuinely dead
    th = data_qa_thresholds("full")
    comp_map, dropped, comp_drops, flags = _qa(obs, syn, traces, sigma, present, comps,
                                               th, min_stations=2, min_fraction=0.0,
                                               contaminated_action="warn")
    assert flags.get("contaminated")
    # warn: only the obs-dead health drop survives the drop list
    assert set(comp_drops) == {("S00", "Z")}
    assert comp_drops[("S00", "Z")] == "drop-dead"
    assert "S00" in comp_map and "Z" not in comp_map["S00"]
    assert all(s in comp_map for s in present)

    comp_map2, _, comp_drops2, flags2 = _qa(obs, syn, traces, sigma, present, comps,
                                            th, min_stations=2, min_fraction=0.0,
                                            contaminated_action="drop")
    assert flags2.get("contaminated")
    assert len(comp_drops2) > len(comp_drops)                  # gates applied as usual
