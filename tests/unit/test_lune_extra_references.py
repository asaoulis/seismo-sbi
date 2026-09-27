"""
Unit tests for the optional ``extra_references`` overlay on the lune plotters
(seismo_sbi.plotting.distributions). These exercise the new plumbing — the
``_scatter_extra_references`` helper and the ``_add_lune_legend`` extra-marker
handles — WITHOUT a real ModelParameters / pyrocko / basemap render (that heavy
path is covered by the e2e smoke). The two model-dependent methods are stubbed
to the identity, which is exactly their effect on the scale/basis-invariant lune
position for already-canonical [Mrr,Mtt,Mpp,Mrt,Mrp,Mtp] inputs.
"""
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from seismo_sbi.plotting.distributions import (
    PosteriorPlotter, _add_lune_legend, LUNE_REFERENCE_STYLES, LUNE_ENSEMBLE_COLORS)


def _fake_bm(gamma, delta):
    """Stand-in for the Basemap projection: pass (γ,δ) through as (x,y)."""
    return np.atleast_1d(gamma), np.atleast_1d(delta)


def _plotter():
    pp = PosteriorPlotter(data_scaler=None, parameters_info=[], parameters=None)
    # Stub the two model-dependent methods (identity; canonical MT in, MT out).
    pp.get_moment_tensors = lambda samples, theta0: (np.empty((0, 6)),
                                                     np.asarray(theta0, dtype=float))
    pp.convert_mt_convention = lambda m: np.asarray(m, dtype=float)
    return pp


def test_scatter_extra_references_styles_and_skips_none():
    pp = _plotter()
    fig, ax = plt.subplots()
    # a clearly non-DC tensor + a near-isotropic one (distinct lune positions)
    refs = OrderedDict([
        ("Lentas", np.array([-2.0, 1.0, 1.0, 0.5, -0.3, 0.2]) * 1e16),
        ("Fountoulakis", np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) * 1e16),
        ("Missing", None),                       # must be skipped
    ])
    specs = pp._scatter_extra_references(ax, _fake_bm, refs)

    assert [s["label"] for s in specs] == ["Lentas", "Fountoulakis"]   # None dropped
    # styles cycle in order
    assert specs[0]["marker"] == LUNE_REFERENCE_STYLES[0]["marker"]
    assert specs[1]["color"] == LUNE_REFERENCE_STYLES[1]["color"]
    # one scatter artist per non-None reference
    assert len(ax.collections) == 2
    plt.close(fig)


def test_scatter_extra_references_empty_is_noop():
    pp = _plotter()
    fig, ax = plt.subplots()
    assert pp._scatter_extra_references(ax, _fake_bm, {}) == []
    assert len(ax.collections) == 0
    plt.close(fig)


def test_add_lune_legend_appends_extra_markers():
    fig, ax = plt.subplots()
    extra = [{"label": "Lentas", "color": "gold", "marker": "*"},
             {"label": "Fountoulakis", "color": "dodgerblue", "marker": "s"}]
    _add_lune_legend(ax, ["NPE ML", "Gaussian"], LUNE_ENSEMBLE_COLORS, extra_markers=extra)
    leg = ax.get_legend()
    assert leg is not None
    texts = [t.get_text() for t in leg.get_texts()]
    assert texts == ["NPE ML", "Gaussian", "Lentas", "Fountoulakis"]
    plt.close(fig)


def test_add_lune_legend_escapes_percent_under_usetex():
    # ChainConsumer can leave text.usetex on; '%' is a LaTeX comment char and must be escaped.
    prev = matplotlib.rcParams["text.usetex"]
    matplotlib.rcParams["text.usetex"] = True
    try:
        fig, ax = plt.subplots()
        _add_lune_legend(ax, ["DC 100%"], LUNE_ENSEMBLE_COLORS,
                         extra_markers=[{"label": "ISO 32%", "color": "gold", "marker": "*"}])
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert texts == [r"DC 100\%", r"ISO 32\%"]
        plt.close(fig)
    finally:
        matplotlib.rcParams["text.usetex"] = prev


def test_plot_lunes_kde_accepts_extra_references_arg():
    # Signature/acceptance guard: the param exists and defaults to None (no behaviour change).
    import inspect
    sig = inspect.signature(PosteriorPlotter.plot_lunes_kde)
    for p in ("extra_references", "reference_label", "primary_reference"):
        assert p in sig.parameters
        assert sig.parameters[p].default is None
    sig2 = inspect.signature(PosteriorPlotter.plot_lunes)
    assert "extra_references" in sig2.parameters and "reference_label" in sig2.parameters


# --------------------------------------------------------------------------- #
# Primary-reference (gold 'truth' diamond) legend entry — the Zahradník marker.
# --------------------------------------------------------------------------- #
def test_primary_reference_legend_labels_existing_truth():
    # When the ensembles carried a theta0 truth (already drawn), we only emit its legend
    # spec — no extra scatter (truth_drawn already True).
    pp = _plotter()
    fig, ax = plt.subplots()
    truth = np.array([-2.0, 1.0, 1.0, 0.5, -0.3, 0.2]) * 1e16
    specs = pp._primary_reference_legend(ax, _fake_bm, truth, "Zahradník", None)
    assert specs == [{"label": "Zahradník", "color": "peru", "marker": "d"}]
    assert len(ax.collections) == 0          # truth already drawn by the main loop
    plt.close(fig)


def test_primary_reference_legend_draws_when_no_truth():
    # No theta0 truth (dropout lune) but a primary_reference given → draw the peru diamond + label.
    pp = _plotter()
    fig, ax = plt.subplots()
    ref = np.array([1.0, -1.0, 0.0, 0.5, 0.5, 0.2]) * 1e16
    specs = pp._primary_reference_legend(ax, _fake_bm, None, "Zahradník", ref)
    assert specs == [{"label": "Zahradník", "color": "peru", "marker": "d"}]
    assert len(ax.collections) == 1          # primary reference drawn here
    plt.close(fig)


def test_primary_reference_legend_empty_when_no_label_or_ref():
    pp = _plotter()
    fig, ax = plt.subplots()
    truth = np.array([-2.0, 1.0, 1.0, 0.5, -0.3, 0.2]) * 1e16
    assert pp._primary_reference_legend(ax, _fake_bm, truth, None, None) == []   # no label
    assert pp._primary_reference_legend(ax, _fake_bm, None, "Zahradník", None) == []  # nothing to draw
    assert len(ax.collections) == 0
    plt.close(fig)
