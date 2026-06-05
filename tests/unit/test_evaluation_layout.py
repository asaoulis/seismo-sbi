"""
Unit tests for seismo_sbi.evaluation.layout and seismo_sbi.evaluation.station_usage.

All tests are pure (no torch / GPU / Instaseis needed) and run in the fast gate.

Coverage
--------
layout:
    OutputLayout.event_dir       — mkdir-on-use, correct path
    OutputLayout.validation_dir  — mkdir-on-use, correct path
    OutputLayout.stage_dir       — mkdir-on-use, correct path
    resolve_output_layout        — builds OutputLayout at correct path
    git_rev                      — returns str or None (never raises)

station_usage:
    write_station_breakdown      — correct JSON keys, kept ⊆ all_available, dropped_by_qa correct
    write_station_usage          — JSON + CSV written, matrix row count == #events, cell codes F/A/-
"""
import csv
import json
from pathlib import Path

import pytest

from seismo_sbi.evaluation.layout import (
    OutputLayout,
    git_rev,
    resolve_output_layout,
)
from seismo_sbi.evaluation.station_usage import (
    write_station_breakdown,
    write_station_usage,
)


# ---------------------------------------------------------------------------
# layout.OutputLayout
# ---------------------------------------------------------------------------

class TestOutputLayout:
    def test_model_root_created_on_init(self, tmp_path):
        model_root = tmp_path / "models" / "mymodel"
        assert not model_root.exists()
        layout = OutputLayout(model_root=model_root)
        assert model_root.is_dir()

    def test_event_dir_correct_path_and_created(self, tmp_path):
        layout = OutputLayout(model_root=tmp_path / "m")
        d = layout.event_dir("No01_id2148")
        assert d == layout.model_root / "No01_id2148"
        assert d.is_dir()

    def test_event_dir_idempotent(self, tmp_path):
        layout = OutputLayout(model_root=tmp_path / "m")
        d1 = layout.event_dir("No01")
        d2 = layout.event_dir("No01")
        assert d1 == d2
        assert d1.is_dir()

    def test_validation_dir_correct_path_and_created(self, tmp_path):
        layout = OutputLayout(model_root=tmp_path / "m")
        d = layout.validation_dir()
        assert d == layout.model_root / "validation"
        assert d.is_dir()

    def test_stage_dir_correct_path_and_created(self, tmp_path):
        layout = OutputLayout(model_root=tmp_path / "m")
        d = layout.stage_dir("my_stage")
        assert d == layout.model_root / "my_stage"
        assert d.is_dir()

    def test_multiple_event_dirs_independent(self, tmp_path):
        layout = OutputLayout(model_root=tmp_path / "m")
        d1 = layout.event_dir("ev1")
        d2 = layout.event_dir("ev2")
        assert d1 != d2
        assert d1.is_dir() and d2.is_dir()

    def test_model_root_accepts_path_or_str(self, tmp_path):
        # model_root can be passed as str; should be converted to Path
        root_str = str(tmp_path / "str_root")
        layout = OutputLayout(model_root=root_str)
        assert isinstance(layout.model_root, Path)
        assert layout.model_root.is_dir()


class TestResolveOutputLayout:
    def test_returns_output_layout_at_correct_path(self, tmp_path):
        layout = resolve_output_layout(tmp_path, "santorini_first_ml_v1")
        assert isinstance(layout, OutputLayout)
        assert layout.model_root == tmp_path / "santorini_first_ml_v1"
        assert layout.model_root.is_dir()

    def test_model_root_is_child_of_output_root(self, tmp_path):
        layout = resolve_output_layout(str(tmp_path), "mymodel")
        assert layout.model_root.parent == tmp_path


# ---------------------------------------------------------------------------
# layout.git_rev
# ---------------------------------------------------------------------------

class TestGitRev:
    def test_returns_str_or_none(self):
        result = git_rev()
        assert result is None or isinstance(result, str)

    def test_returns_str_or_none_with_path(self, tmp_path):
        # A non-git directory: should return None rather than raising.
        result = git_rev(path=tmp_path)
        assert result is None

    def test_does_not_raise_on_bad_path(self):
        # Even a completely bogus path must not raise.
        result = git_rev(path="/nonexistent/path/that/will/never/exist")
        assert result is None


# ---------------------------------------------------------------------------
# station_usage.write_station_breakdown
# ---------------------------------------------------------------------------

class _FakeStationConfig:
    """Minimal stub for StationConfig.as_dict()."""
    def __init__(self, label, kept, dropped):
        self._label = label
        self._kept = kept
        self._dropped = dropped

    def as_dict(self):
        return {
            "label": self._label,
            "n": len(self._kept),
            "keep_indices": list(range(len(self._kept))),
            "kept": self._kept,
            "dropped": self._dropped,
        }


MASTER = ["AAA", "BBB", "CCC", "DDD", "EEE"]
ALLS = ["AAA", "BBB", "CCC", "DDD"]   # subset of master (EEE absent)
FILT = ["AAA", "CCC"]                 # QA-kept subset of alls


class TestWriteStationBreakdown:
    def _breakdown(self, tmp_path, alls=None, filt=None, dropout_configs=None):
        alls = ALLS if alls is None else alls
        filt = FILT if filt is None else filt
        return write_station_breakdown(
            tmp_path, "No01_id2148", MASTER, alls, filt, dropout_configs
        )

    def test_json_file_written(self, tmp_path):
        self._breakdown(tmp_path)
        out = tmp_path / "ml_No01_id2148_stations.json"
        assert out.exists()

    def test_json_keys(self, tmp_path):
        self._breakdown(tmp_path)
        with open(tmp_path / "ml_No01_id2148_stations.json") as f:
            d = json.load(f)
        for key in ("event", "master_stations", "all_available", "filtered",
                    "dropped_by_qa", "n_all", "n_filtered", "dropout_configs"):
            assert key in d, f"missing key '{key}'"

    def test_dropped_by_qa_correct(self, tmp_path):
        self._breakdown(tmp_path)
        with open(tmp_path / "ml_No01_id2148_stations.json") as f:
            d = json.load(f)
        expected_dropped = sorted(set(ALLS) - set(FILT))
        assert d["dropped_by_qa"] == expected_dropped

    def test_filtered_subset_of_all_available(self, tmp_path):
        self._breakdown(tmp_path)
        with open(tmp_path / "ml_No01_id2148_stations.json") as f:
            d = json.load(f)
        assert set(d["filtered"]).issubset(set(d["all_available"]))

    def test_n_counts(self, tmp_path):
        result = self._breakdown(tmp_path)
        assert result["n_all"] == len(ALLS)
        assert result["n_filtered"] == len(FILT)

    def test_dropout_configs_none_gives_empty_list(self, tmp_path):
        result = self._breakdown(tmp_path, dropout_configs=None)
        assert result["dropout_configs"] == []

    def test_dropout_configs_serialised(self, tmp_path):
        cfg = _FakeStationConfig("all (N=4)", ALLS, [])
        result = self._breakdown(tmp_path, dropout_configs=[cfg])
        assert len(result["dropout_configs"]) == 1
        assert result["dropout_configs"][0]["label"] == "all (N=4)"
        assert result["dropout_configs"][0]["kept"] == ALLS

    def test_returns_dict_matching_json(self, tmp_path):
        result = self._breakdown(tmp_path)
        with open(tmp_path / "ml_No01_id2148_stations.json") as f:
            on_disk = json.load(f)
        assert result["event"] == on_disk["event"]
        assert result["all_available"] == on_disk["all_available"]
        assert result["filtered"] == on_disk["filtered"]


# ---------------------------------------------------------------------------
# station_usage.write_station_usage
# ---------------------------------------------------------------------------

class TestWriteStationUsage:
    def _make_per_event(self):
        """Three events with varying station availability."""
        return {
            "No01": {
                "all_available": ["AAA", "BBB", "CCC"],
                "filtered": ["AAA", "CCC"],
                "n_all": 3,
                "n_filtered": 2,
            },
            "No02": {
                "all_available": ["AAA", "DDD"],
                "filtered": ["AAA", "DDD"],
                "n_all": 2,
                "n_filtered": 2,
            },
            "No03": {
                "all_available": ["BBB"],
                "filtered": ["BBB"],
                "n_all": 1,
                "n_filtered": 1,
            },
        }

    def test_json_and_csv_written(self, tmp_path):
        write_station_usage(tmp_path, MASTER, self._make_per_event())
        assert (tmp_path / "station_usage.json").exists()
        assert (tmp_path / "station_usage.csv").exists()

    def test_matrix_row_count_equals_events(self, tmp_path):
        per_event = self._make_per_event()
        write_station_usage(tmp_path, MASTER, per_event)
        with open(tmp_path / "station_usage.json") as f:
            d = json.load(f)
        assert len(d["events"]) == len(per_event)

    def test_csv_row_count_equals_events_plus_header(self, tmp_path):
        per_event = self._make_per_event()
        write_station_usage(tmp_path, MASTER, per_event)
        with open(tmp_path / "station_usage.csv", newline="") as f:
            rows = list(csv.reader(f))
        # header + one row per event
        assert len(rows) == len(per_event) + 1

    def test_cell_codes_correct(self, tmp_path):
        per_event = self._make_per_event()
        write_station_usage(tmp_path, MASTER, per_event)
        with open(tmp_path / "station_usage.json") as f:
            d = json.load(f)
        codes_no01 = d["events"]["No01"]["codes"]
        # AAA ∈ filtered → F
        assert codes_no01["AAA"] == "F"
        # CCC ∈ filtered → F
        assert codes_no01["CCC"] == "F"
        # BBB ∈ alls but NOT filtered → A
        assert codes_no01["BBB"] == "A"
        # DDD absent from No01 → -
        assert codes_no01["DDD"] == "-"
        # EEE absent from No01 → -
        assert codes_no01["EEE"] == "-"

    def test_all_filtered_gives_all_F(self, tmp_path):
        per_event = {"EV": {"all_available": ["AAA", "BBB"],
                            "filtered": ["AAA", "BBB"],
                            "n_all": 2, "n_filtered": 2}}
        write_station_usage(tmp_path, ["AAA", "BBB", "CCC"], per_event)
        with open(tmp_path / "station_usage.json") as f:
            d = json.load(f)
        codes = d["events"]["EV"]["codes"]
        assert codes["AAA"] == "F"
        assert codes["BBB"] == "F"
        assert codes["CCC"] == "-"

    def test_legend_in_json(self, tmp_path):
        write_station_usage(tmp_path, MASTER, self._make_per_event())
        with open(tmp_path / "station_usage.json") as f:
            d = json.load(f)
        assert "legend" in d
        assert "F" in d["legend"] and "A" in d["legend"] and "-" in d["legend"]

    def test_master_stations_in_json(self, tmp_path):
        write_station_usage(tmp_path, MASTER, self._make_per_event())
        with open(tmp_path / "station_usage.json") as f:
            d = json.load(f)
        assert d["master_stations"] == list(MASTER)

    def test_empty_per_event_writes_empty_matrix(self, tmp_path):
        write_station_usage(tmp_path, MASTER, {})
        with open(tmp_path / "station_usage.json") as f:
            d = json.load(f)
        assert d["events"] == {}
