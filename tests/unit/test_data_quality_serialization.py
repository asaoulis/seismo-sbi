"""Unit tests for seismo_sbi.data_quality.serialization (on-disk contracts)."""
import json
from pathlib import Path

import pytest

from seismo_sbi.data_quality.policy import StationSummary, StationVerdict
from seismo_sbi.data_quality.serialization import (
    QAArtifacts, components_from_verdicts, load_qa_artifacts, verdict_to_json,
    write_components_json, write_time_shifts_json, write_verdicts_json)

ORIGINAL_KEYS = ["dist_km", "azimuth", "vr", "aligned_vr", "lag_Z", "xcorr_Z",
                 "median_amp_ratio", "verdict", "suggested_shift"]


def _verdict(verdict="keep", shift=0, **summary_overrides):
    base = dict(dist_km=10.0, azimuth=45.0, vr={"Z": 0.5}, aligned_vr={"Z": 0.6},
                lag_Z=shift, xcorr_Z=0.9, median_amp_ratio=1.0, aligned_coherence=0.9)
    base.update(summary_overrides)
    return StationVerdict(verdict, shift, StationSummary(**base))


def test_verdict_to_json_preserves_original_schema():
    d = verdict_to_json(_verdict())
    # original nine keys present, in their original order, before any new field
    assert list(d.keys())[:9] == ORIGINAL_KEYS
    assert "aligned_coherence" in d
    # confounded PPC fields omitted unless computed
    assert "corr_misfit" not in d and "reduced_chi2" not in d


def test_verdict_to_json_includes_ppc_fields_when_present():
    d = verdict_to_json(_verdict(corr_misfit=0.3, envelope_misfit=0.8, reduced_chi2=1.2))
    assert d["corr_misfit"] == 0.3 and d["envelope_misfit"] == 0.8 and d["reduced_chi2"] == 1.2


def test_components_from_verdicts():
    verdicts = {"AAA": _verdict("keep"), "BBB": _verdict("time-shift", 3),
                "CCC": _verdict("drop-amp")}
    comp = components_from_verdicts(verdicts, all_stations=["AAA", "BBB", "CCC", "DDD"])
    assert comp == {"AAA": ["Z", "E", "N"], "BBB": ["Z", "E", "N"], "CCC": [], "DDD": []}


def test_time_shifts_roundtrip(tmp_path):
    p = tmp_path / "time_shifts.json"
    write_time_shifts_json(p, {"AAA": 3, "BBB": -2})
    assert json.loads(p.read_text()) == {"AAA": 3, "BBB": -2}


def test_load_qa_artifacts_roundtrip(tmp_path):
    verdicts = {"AAA": _verdict("keep"), "CCC": _verdict("drop-amp")}
    write_components_json(tmp_path / "components.json",
                          components_from_verdicts(verdicts, ["AAA", "CCC"]))
    write_time_shifts_json(tmp_path / "time_shifts.json", {"AAA": 2})
    write_verdicts_json(tmp_path / "verdicts.json", verdicts, absent_from_h5=["ZZZ"])

    art = load_qa_artifacts("EV",
                            components_path=tmp_path / "components.json",
                            time_shifts_path=tmp_path / "time_shifts.json",
                            verdicts_path=tmp_path / "verdicts.json")
    assert isinstance(art, QAArtifacts)
    assert art.kept_stations == ["AAA"]
    assert art.usable_stations == ["AAA"]
    assert art.absent_from_h5 == ["ZZZ"]
    assert art.time_shifts == {"AAA": 2}


def test_load_qa_artifacts_missing_files_are_empty():
    art = load_qa_artifacts("EV")
    assert art.present == {} and art.components == {} and art.time_shifts == {}


def test_golden_verdicts_schema_parity():
    """The committed Santorini verdicts must carry exactly the original nine fields,
    so our serialiser stays a superset of what existing readers expect."""
    golden = Path("scripts/santorini_pathbreaker/diagnostics/No14_id3250/"
                  "station_qa/No14_id3250_allstation_verdicts.json")
    if not golden.exists():
        pytest.skip("santorini golden not present")
    rec = next(iter(json.loads(golden.read_text())["present"].values()))
    assert set(ORIGINAL_KEYS).issubset(rec.keys())
    assert set(rec.keys()).issubset(set(ORIGINAL_KEYS) | {
        "aligned_coherence", "corr_misfit", "envelope_misfit", "reduced_chi2"})
