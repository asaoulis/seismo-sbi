"""Regression test: the new ``seismo_sbi.data_quality`` code must reproduce the
committed Santorini QA artifacts exactly, proving the port from the ad-hoc
``qa_forward_check.py`` is behaviour-preserving.

Uses saved obs/syn ``.npz`` fixtures (captured from the *current* script before the
refactor) so no Instaseis/forward model is needed in CI.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from seismo_sbi.data_quality import (
    QAThresholds, TraceDescriptor, compute_trace_metrics, summarise_event)
from seismo_sbi.data_quality.alignment import nonzero_shifts, optimise_event_shifts

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "data" / "data_quality"
SANTO = REPO / "scripts" / "santorini_pathbreaker"
EVENT = "No14_id3250"

# Original numeric fields that MUST be byte-for-byte preserved by the port.
NUMERIC_FIELDS = ["dist_km", "azimuth", "lag_Z", "xcorr_Z", "median_amp_ratio",
                  "suggested_shift"]

pytestmark = pytest.mark.skipif(
    not (FIXTURES / f"{EVENT}_allstations.npz").exists(),
    reason="QA regression fixtures not present")


def _load_traces(npz):
    f = np.load(npz, allow_pickle=True)
    traces = [TraceDescriptor(str(s), str(c), float(la), float(lo)) for s, c, la, lo in
              zip(f["station"], f["component"], f["latitude"], f["longitude"])]
    return f["obs"], f["syn"], traces, float(f["src_lat"]), float(f["src_lon"])


def test_allstation_verdicts_match_golden():
    obs, syn, traces, slat, slon = _load_traces(FIXTURES / f"{EVENT}_allstations.npz")
    metrics = compute_trace_metrics(obs, syn, traces, slat, slon, max_lag=60)
    verdicts = summarise_event(metrics, QAThresholds())  # default thresholds (gates on)

    golden = json.loads(
        (SANTO / "diagnostics" / EVENT / "station_qa"
         / f"{EVENT}_allstation_verdicts.json").read_text())["present"]

    assert set(verdicts) == set(golden)
    for sta, g in golden.items():
        v = verdicts[sta]
        assert v.verdict == g["verdict"], sta
        d = {"dist_km": v.summary.dist_km, "azimuth": v.summary.azimuth,
             "lag_Z": v.summary.lag_Z, "xcorr_Z": v.summary.xcorr_Z,
             "median_amp_ratio": v.summary.median_amp_ratio,
             "suggested_shift": v.suggested_shift}
        for k in NUMERIC_FIELDS:
            assert d[k] == pytest.approx(g[k], rel=1e-9, abs=1e-12), (sta, k)
        # per-component VR / aligned VR
        for comp, val in g["vr"].items():
            assert v.summary.vr[comp] == pytest.approx(val, rel=1e-9, abs=1e-12)
        for comp, val in g["aligned_vr"].items():
            assert v.summary.aligned_vr[comp] == pytest.approx(val, rel=1e-9, abs=1e-12)


def test_time_shifts_match_golden():
    obs, syn, traces, _, _ = _load_traces(FIXTURES / f"{EVENT}_kept.npz")
    results = optimise_event_shifts(obs, syn, traces, max_shift=8)
    shifts = nonzero_shifts(results)

    golden = json.loads((SANTO / "events" / EVENT / "time_shifts.json").read_text())
    assert shifts == golden
