"""Tests for the moveout record-section renderer and its MisfitsPlotting adapter.

The point of the renderer is that EVERY trace is legible — the old figures normalised 45
traces by one global peak and 13 of 15 stations came out flat. The normalisation maths is
therefore the thing worth pinning down, alongside the layout/ordering plumbing.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from seismo_sbi.plotting.waveform_compare import (
    _norm_table,
    _window_slice,
    moveout_record_section,
)
from seismo_sbi.plotting.seismo_plots import MisfitsPlotting
from seismo_sbi.simulators.receivers import Receiver, Receivers

N, C, T = 4, 3, 64
EVENT = (36.5, 25.5, 10.0)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _obs(seed=0):
    rng = np.random.default_rng(seed)
    cube = rng.normal(size=(N, C, T))
    # station 0 is 1000x louder — the case that flattened everything under global norm
    cube[0] *= 1000.0
    return cube


def _coords():
    return np.array([[36.6, 25.5], [36.5, 25.8], [36.9, 25.5], [36.2, 25.2]])


def _names():
    return ["AAA", "BBB", "CCC", "DDD"]


# ---------------------------------------------------------------- normalisation maths
def test_per_trace_norm_makes_every_trace_unit_scale():
    obs = _obs()
    table = _norm_table(obs, [], "per_trace")
    scaled = obs / table[..., None]
    np.testing.assert_allclose(np.abs(scaled).max(axis=-1), 1.0)


def test_per_station_norm_is_constant_within_a_station_and_varies_across():
    obs = _obs()
    table = _norm_table(obs, [], "per_station")
    for i in range(N):
        assert len(set(np.round(table[i], 12))) == 1        # one scale per station
    assert table[0, 0] > 100 * table[1, 0]                  # loud station keeps its own


def test_global_norm_is_one_number_everywhere():
    table = _norm_table(_obs(), [], "global")
    assert np.allclose(table, table.flat[0])


def test_norm_accounts_for_overlays_not_just_obs():
    """An overlay that overshoots must widen the divisor, else it draws off-axis."""
    obs = np.ones((1, 1, T))
    big = np.full((1, 1, T), 5.0)
    assert _norm_table(obs, [], "per_trace")[0, 0] == pytest.approx(1.0)
    assert _norm_table(obs, [big], "per_trace")[0, 0] == pytest.approx(5.0)


def test_zero_trace_does_not_divide_by_zero():
    obs = np.zeros((2, 1, T))
    assert np.all(_norm_table(obs, [], "per_trace") == 1.0)


def test_unknown_normalise_rejected():
    with pytest.raises(ValueError, match="normalise"):
        _norm_table(_obs(), [], "nonsense")


# ---------------------------------------------------------------- window slicing
def test_window_none_is_the_full_trace():
    assert _window_slice(T, 1.0, None, None) == (0, T, 0.0)


def test_absolute_window_crops_by_sampling_rate():
    assert _window_slice(T, 2.0, (5.0, 10.0), None) == (10, 20, 0.0)


def test_arrival_window_centres_on_the_arrival():
    i0, i1, org = _window_slice(200, 1.0, ("arrival", 20.0, 30.0), 100.0)
    assert (i0, i1, org) == (80, 130, 100.0)


def test_arrival_window_falls_back_when_no_arrival_known():
    assert _window_slice(T, 1.0, ("arrival", 20.0, 30.0), None) == (0, T, 0.0)


def test_arrival_window_clamped_to_trace_and_degenerate_falls_back():
    # arrival far past the end -> degenerate slice -> full trace
    assert _window_slice(T, 1.0, ("arrival", 1.0, 1.0), 500.0) == (0, T, 0.0)


# ---------------------------------------------------------------- renderer plumbing
def test_panels_layout_gives_one_axis_per_component():
    fig = moveout_record_section(_obs(), None, _names(), _coords(), EVENT)
    assert len(fig.axes) == C


def test_interleaved_layout_gives_a_single_axis():
    fig = moveout_record_section(_obs(), None, _names(), _coords(), EVENT, layout="interleaved")
    assert len(fig.axes) == 1


def test_deterministic_overlay_adds_one_line_per_trace():
    obs = _obs()
    base = moveout_record_section(obs, None, _names(), _coords(), EVENT)
    n_base = sum(len(a.lines) for a in base.axes)
    with_syn = moveout_record_section(obs, {"syn": obs * 0.5}, _names(), _coords(), EVENT)
    assert sum(len(a.lines) for a in with_syn.axes) == n_base + N * C


def test_ensemble_band_creates_filled_regions():
    from matplotlib.collections import PolyCollection
    ens = np.stack([_obs(seed=s) for s in range(5)])
    fig = moveout_record_section(_obs(), {"ppc": ens}, _names(), _coords(), EVENT,
                                 ensemble_style="band")
    polys = [c for a in fig.axes for c in a.collections if isinstance(c, PolyCollection)]
    assert len(polys) == N * C


def test_ensemble_spaghetti_draws_lines_not_bands():
    from matplotlib.collections import PolyCollection
    ens = np.stack([_obs(seed=s) for s in range(6)])
    fig = moveout_record_section(_obs(), {"ppc": ens}, _names(), _coords(), EVENT,
                                 ensemble_style="spaghetti", max_lines=3)
    assert not [c for a in fig.axes for c in a.collections if isinstance(c, PolyCollection)]
    # 3 sampled members + the observation, per trace
    assert sum(len(a.lines) for a in fig.axes) == N * C * 4


def test_band_plus_best_adds_a_dashed_line_for_member_zero():
    """'band+best' must draw the best-fitting member (index 0, as returned by
    select_best_synthetics) on top of the band."""
    ens = np.stack([_obs(seed=s) for s in range(5)])
    band = moveout_record_section(_obs(), {"ppc": ens}, _names(), _coords(), EVENT,
                                  ensemble_style="band")
    plus = moveout_record_section(_obs(), {"ppc": ens}, _names(), _coords(), EVENT,
                                  ensemble_style="band+best")
    assert sum(len(a.lines) for a in plus.axes) == sum(len(a.lines) for a in band.axes) + N * C
    dashed = [ln for a in plus.axes for ln in a.lines if ln.get_linestyle() == "--"]
    assert len(dashed) == N * C
    # and it must be member 0, not the median
    i, ci = 0, 0
    d = _norm_table(_obs(), [ens.mean(axis=0)], "per_station")[i, ci]
    y = dashed[0].get_ydata()
    np.testing.assert_allclose(y - y.mean(), (0.42 * ens[0, i, ci] / d) - (0.42 * ens[0, i, ci] / d).mean(),
                               atol=1e-9)


def test_adapter_subsampling_keeps_the_best_member_first():
    """Random subsampling must not shuffle member 0 away — 'band+best' depends on it."""
    mp = MisfitsPlotting(_receivers(), 1.0, None)
    rng = np.random.default_rng(3)
    obs = rng.normal(size=N * C * T)
    ens = rng.normal(size=(50, N * C * T))
    ens[0] = obs * 0.999                      # a recognisable "best" member
    fig = mp.plot_record_section(obs, EVENT, ensembles={"PPC": ens},
                                 ensemble_style="band+best", max_samples=6)
    dashed = [ln for a in fig.axes for ln in a.lines if ln.get_linestyle() == "--"]
    assert len(dashed) == N * C
    obs_line = [ln for a in fig.axes for ln in a.lines
                if ln.get_color() == "black"][0].get_ydata()
    # member 0 ~= 0.999 * obs, so the dashed line tracks the observation closely
    assert np.corrcoef(dashed[0].get_ydata(), obs_line)[0, 1] > 0.999


def test_ordering_by_distance_and_azimuth_differ():
    def ylabels(order_by):
        fig = moveout_record_section(_obs(), None, _names(), _coords(), EVENT,
                                     order_by=order_by)
        base = fig.axes[0].get_yticks()
        return [lbl for _, lbl in sorted(zip(base, _names()))]
    assert ylabels("distance") != ylabels("azimuth")


def test_rank_scale_spaces_stations_uniformly():
    fig = moveout_record_section(_obs(), None, _names(), _coords(), EVENT, y_scale="rank")
    assert sorted(fig.axes[0].get_yticks().tolist()) == [0.0, 1.0, 2.0, 3.0]


def test_true_scale_uses_physical_distance():
    fig = moveout_record_section(_obs(), None, _names(), _coords(), EVENT, y_scale="true")
    ticks = np.sort(fig.axes[0].get_yticks())
    assert ticks.max() > 10.0                         # km, not ranks
    assert not np.allclose(np.diff(ticks), np.diff(ticks)[0])   # non-uniform spacing


def test_channel_mask_greys_the_dropped_trace_and_adds_a_legend_entry():
    from seismo_sbi.plotting.waveform_compare import DROPPED_COLOR, OBS_COLOR
    mask = np.ones((N, C), bool)
    mask[1, 2] = False
    fig = moveout_record_section(_obs(), None, _names(), _coords(), EVENT, channel_mask=mask)
    colors = [ln.get_color() for a in fig.axes for ln in a.lines]
    assert colors.count(DROPPED_COLOR) == 1
    assert colors.count(OBS_COLOR) == N * C - 1
    assert any("QA-dropped" in t.get_text() for t in fig.legends[0].get_texts())


def test_obs_must_be_a_cube():
    with pytest.raises(ValueError, match=r"\(N, C, T\)"):
        moveout_record_section(np.zeros((N, T)), None, _names(), _coords(), EVENT)


def test_two_component_stations_are_supported():
    fig = moveout_record_section(_obs()[:, :2], None, _names(), _coords(), EVENT,
                                 components=("Z", "E"))
    assert len(fig.axes) == 2


def test_reduction_velocity_shifts_the_time_axis_per_station():
    obs = _obs()
    plain = moveout_record_section(obs, None, _names(), _coords(), EVENT)
    reduced = moveout_record_section(obs, None, _names(), _coords(), EVENT,
                                     reduction_velocity=5.0)
    x_plain = plain.axes[0].lines[0].get_xdata()
    x_red = reduced.axes[0].lines[0].get_xdata()
    assert not np.allclose(x_plain, x_red)


# ---------------------------------------------------------------- MisfitsPlotting adapter
def _receivers(components=("Z", "E", "N")):
    return Receivers(receivers=[
        Receiver(36.6, 25.5, "HL", "AAA", list(components)),
        Receiver(36.5, 25.8, "HL", "BBB", list(components)),
        Receiver(36.9, 25.5, "HL", "CCC", list(components)),
        Receiver(36.2, 25.2, "HL", "DDD", list(components)),
    ])


def test_adapter_reshapes_flat_vectors_in_receiver_major_order():
    mp = MisfitsPlotting(_receivers(), 1.0, None)
    flat = np.arange(N * C * T, dtype=float)
    cube, comps = mp._reshape_to_cube(flat)
    assert cube.shape == (1, N, C, T)
    assert comps == ["Z", "E", "N"]
    np.testing.assert_allclose(cube[0, 1, 0], flat[C * T: C * T + T])


def test_adapter_rejects_ragged_component_sets():
    recv = Receivers(receivers=[
        Receiver(36.6, 25.5, "HL", "AAA", ["Z", "E", "N"]),
        Receiver(36.5, 25.8, "HL", "BBB", ["Z"]),
    ])
    with pytest.raises(ValueError, match="uniform component set"):
        MisfitsPlotting(recv, 1.0, None)._reshape_to_cube(np.zeros(4 * T))


def test_adapter_renders_deterministic_and_ensemble_overlays():
    mp = MisfitsPlotting(_receivers(), 1.0, None)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=N * C * T)
    fig = mp.plot_record_section(
        obs, EVENT,
        deterministic={"best fit": obs * 0.9},
        ensembles={"PPC": rng.normal(size=(12, N * C * T))},
        max_samples=5)
    assert len(fig.axes) == C
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert labels == ["observed", "best fit", "PPC"]


def test_adapter_subsamples_large_ensembles():
    mp = MisfitsPlotting(_receivers(), 1.0, None)
    rng = np.random.default_rng(1)
    obs = rng.normal(size=N * C * T)
    fig = mp.plot_record_section(obs, EVENT,
                                 ensembles={"PPC": rng.normal(size=(500, N * C * T))},
                                 ensemble_style="spaghetti", max_samples=4, max_lines=99)
    # 4 kept members + obs per trace
    assert sum(len(a.lines) for a in fig.axes) == N * C * 5
