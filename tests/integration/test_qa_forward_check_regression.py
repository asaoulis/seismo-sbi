"""``seismo_sbi.data_quality`` reproduces one event's recorded QA verdicts and time shifts exactly.

The observed and synthetic traces and the recorded verdicts and shifts are committed fixtures
under ``tests/data/data_quality``, so no forward model is needed.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from seismo_sbi.data_quality.metrics import TraceDescriptor, compute_trace_metrics
from seismo_sbi.data_quality.policy import QAThresholds, summarise_event
from seismo_sbi.data_quality.alignment import nonzero_shifts, optimise_event_shifts

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "data" / "data_quality"
EVENT = "No14_id3250"

# Original numeric fields that MUST be byte-for-byte preserved by the port.
NUMERIC_FIELDS = ["dist_km", "azimuth", "lag_Z", "xcorr_Z", "median_amp_ratio",
                  "suggested_shift"]


def _load_traces(npz):
    f = np.load(npz, allow_pickle=True)
    traces = [TraceDescriptor(str(s), str(c), float(la), float(lo)) for s, c, la, lo in
              zip(f["station"], f["component"], f["latitude"], f["longitude"])]
    return f["obs"], f["syn"], traces, float(f["src_lat"]), float(f["src_lon"])


def test_allstation_verdicts_match_golden():
    obs, syn, traces, slat, slon = _load_traces(FIXTURES / f"{EVENT}_allstations.npz")
    metrics = compute_trace_metrics(obs, syn, traces, slat, slon, max_lag=60)
    verdicts = summarise_event(metrics, QAThresholds())  # default thresholds (gates on)

    golden = json.loads((FIXTURES / f"{EVENT}_allstation_verdicts.json").read_text())["present"]

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

    golden = json.loads((FIXTURES / f"{EVENT}_time_shifts.json").read_text())
    assert shifts == golden
