"""Event directory naming and the -ISO label.

The label decides how an event is filed *and* which population it joins in the
summary tables, so it comes from the **conservative quantile** of the posterior
lune latitude (``keep AND c_iso_<lvl> <= -5``), not from a bare sign
probability.  The distinction is not academic: ``P(iso < 0) >= 0.9`` passes
events whose sign is unresolved at 95% confidence, because it ignores how wide
the posterior is.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2]
           / "scripts" / "santorini_pathbreaker" / "lomax_catalogue")
)

import polarity_events as events  # noqa: E402


def write_solution(run_dir, event_id, ml, mw, delta, traces):
    """A minimal solution folder: solution.json plus posterior samples."""
    folder = run_dir / "solutions" / event_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "solution.json").write_text(json.dumps({
        "event_id": event_id, "ml": ml,
        "features": {"Mw": mw, "delta": delta},
    }))
    # Only the trace matters here; put it in Mrr and zero the rest.
    samples = np.zeros((len(traces), 6))
    samples[:, 0] = np.asarray(traces, dtype=float)
    np.save(folder / "samples.npy", samples)
    return folder


def test_dirname_puts_magnitude_first_and_pads_it():
    assert events.event_dirname("20250212T011455", 4.836) == "Mw4.84_20250212T011455"
    assert events.event_dirname("20250212T011455", 4.0) == "Mw4.00_20250212T011455"


def test_dirname_marks_a_missing_magnitude_rather_than_dropping_it():
    """A silently shortened name would look like a different scheme."""
    assert events.event_dirname("20250212T011455") == "Mwxxx_20250212T011455"
    assert events.event_dirname("20250212T011455", float("nan")) == "Mwxxx_20250212T011455"


def test_dirname_appends_the_label():
    name = events.event_dirname("20250224T113307", 4.0, "negISO")
    assert name == "Mw4.00_20250224T113307_negISO"


def test_unknown_labels_do_not_leak_into_the_name():
    """A typo'd label must not silently create a parallel directory scheme."""
    assert events.event_dirname("20250224T113307", 4.0, "nonsense") == "Mw4.00_20250224T113307"


def test_dirnames_sort_by_magnitude():
    names = sorted(events.event_dirname(f"2025021{i}T000000", mw)
                   for i, mw in enumerate([4.84, 3.31, 5.02]))
    assert [n.split("_")[0] for n in names] == ["Mw3.31", "Mw4.84", "Mw5.02"]


def _entry(keep=True, c75=-12.0, c90=-9.0, c95=-7.0, area=0.02, mw=4.0):
    return {"keep": keep, "c_iso_75": c75, "c_iso_90": c90, "c_iso_95": c95,
            "lune_area95": area, "Mw": mw}


@pytest.mark.parametrize("c90,expected", [
    (-16.4, "negISO"), (-5.87, "negISO"), (-5.0, "negISO"),
    (-4.99, "negISO75"), (0.0, "negISO75"),
])
def test_primary_label_uses_the_dead_band_edge(c90, expected):
    assert events.event_label(_entry(c90=c90)) == expected


def test_an_event_failing_the_quality_filter_is_never_labelled():
    """`keep` is the lune-area filter; a rejected event carries no label."""
    assert events.event_label(_entry(keep=False, c90=-20.0)) == ""


def test_loose_only_events_are_tagged_separately():
    """Significant at 0.75 but not 0.90 must not sit in the same bucket."""
    assert events.event_label(_entry(c75=-6.3, c90=0.0, c95=0.0)) == "negISO75"
    assert events.event_dirname("E", 3.42, "negISO75") == "Mw3.42_E_negISO75"


def test_nothing_significant_gives_no_label():
    assert events.event_label(_entry(c75=-1.0, c90=0.0, c95=0.0)) == ""
    assert events.event_label({}) == ""


def test_the_real_catalogue_disagreement_is_reproduced():
    """20250217T110648: P(iso<0)=0.905 but c_iso_95=0 -- must NOT be `negISO`.

    This is the case that motivated using a conservative quantile at all: the
    naive sign probability calls it confidently implosive while its posterior
    does not resolve the sign at 95%.
    """
    assert events.event_label(_entry(c75=-8.82, c90=-0.43, c95=0.0)) == "negISO75"
    assert events.event_label(_entry(c75=-15.11, c90=-12.30, c95=-10.47)) == "negISO"


def test_naive_confidence_still_available_as_a_fallback(tmp_path):
    write_solution(tmp_path, "STRADDLE", 4.0, 3.8, -12.0,
                   traces=[-3.0] * 6 + [1.0] * 4)
    assert events.iso_confidence(tmp_path, "STRADDLE") == pytest.approx(0.6)


def test_event_directory_is_created(tmp_path):
    path = events.event_directory(tmp_path / "figures", "20250212T011455", 4.84)
    assert path.is_dir()
    assert path.name == "Mw4.84_20250212T011455"


def write_maps_table(run_dir, rows):
    """The probabilistic catalogue's maps_table.csv, in its sibling folder."""
    import pandas as pd

    root = run_dir.parent / f"{run_dir.name}_probabilistic" / "maps"
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "maps_table.csv", index=False)


def test_index_round_trip_and_labelling(tmp_path):
    run = tmp_path / "run"
    (run).mkdir()
    write_solution(run, "AAA", 4.37, 4.00, -18.4, traces=[-2.0] * 100)
    write_solution(run, "BBB", 5.28, 4.84, -8.4, traces=[-1.0] * 5 + [1.0] * 5)
    write_maps_table(run, [
        {"event_id": "AAA", "keep": True, "c_iso_75": -15.1, "c_iso_90": -12.3,
         "c_iso_95": -10.5, "lune_area95": 0.015, "Mw": 4.00},
        {"event_id": "BBB", "keep": True, "c_iso_75": -1.0, "c_iso_90": 0.0,
         "c_iso_95": 0.0, "lune_area95": 0.02, "Mw": 4.84},
    ])
    tmp_path = run
    tables = tmp_path / "tables"

    frame = events.build_event_index(tmp_path, ["AAA", "BBB"], tables)
    assert list(frame["event_id"]) == ["BBB", "AAA"], "should sort by Mw descending"
    assert set(frame.loc[frame["is_neg_iso"], "event_id"]) == {"AAA"}
    assert frame.set_index("event_id").loc["AAA", "directory"] == "Mw4.00_AAA_negISO"
    assert frame.set_index("event_id").loc["BBB", "directory"] == "Mw4.84_BBB"

    assert frame.set_index("event_id").loc["AAA", "c_iso_90"] == pytest.approx(-12.3)

    loaded = events.load_event_index(tables)
    assert set(loaded) == {"AAA", "BBB"}
    assert loaded["AAA"]["label"] == "negISO"
    assert loaded["BBB"]["label"] != "negISO"


def test_load_event_index_is_empty_when_absent(tmp_path):
    assert events.load_event_index(tmp_path) == {}


# -- the +ISO mirror ------------------------------------------------------


@pytest.mark.parametrize("c90,expected", [
    (16.4, "posISO"), (5.87, "posISO"), (5.0, "posISO"),
    (4.99, "posISO75"), (0.0, "posISO75"),
])
def test_positive_label_uses_the_same_dead_band_edge(c90, expected):
    """+5 exactly is significant, mirroring -5 exactly on the implosive side.

    ``c75`` is held above the band throughout (the negative fixture's default
    does the same on its side), so this varies only the primary level.
    """
    assert events.event_label(_entry(c90=c90, c75=c90 + 8.0)) == expected


def test_the_dead_band_is_symmetric():
    assert events.POS_DEAD_BAND_DEG == -events.DEAD_BAND_DEG


def test_the_two_iso_labels_are_mutually_exclusive():
    """A single conservative level cannot be <= -5 and >= +5 at once."""
    assert events.event_label(_entry(c90=-12.0, c75=-15.0)) == "negISO"
    assert events.event_label(_entry(c90=12.0, c75=15.0)) == "posISO"


def test_a_positive_event_failing_the_quality_filter_is_never_labelled():
    assert events.event_label(_entry(keep=False, c90=20.0, c75=22.0)) == ""


def test_an_unresolved_event_between_the_bands_gets_no_label():
    assert events.event_label(_entry(c75=3.0, c90=1.0, c95=0.0)) == ""


def test_positive_dirname_suffixes():
    assert events.event_dirname("E", 4.25, "posISO") == "Mw4.25_E_posISO"
    assert events.event_dirname("E", 4.25, "posISO75") == "Mw4.25_E_posISO75"


def test_index_labels_both_signs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_solution(run, "NEG", 4.37, 4.00, -18.4, traces=[-2.0] * 10)
    write_solution(run, "POS", 4.40, 4.25, 21.0, traces=[2.0] * 10)
    write_solution(run, "MID", 3.90, 3.50, 1.0, traces=[-1.0] * 5 + [1.0] * 5)
    write_maps_table(run, [
        {"event_id": "NEG", "keep": True, "c_iso_75": -15.1, "c_iso_90": -12.3,
         "c_iso_95": -10.5, "lune_area95": 0.015, "Mw": 4.00},
        {"event_id": "POS", "keep": True, "c_iso_75": 22.9, "c_iso_90": 21.0,
         "c_iso_95": 19.7, "lune_area95": 0.007, "Mw": 4.25},
        {"event_id": "MID", "keep": True, "c_iso_75": 1.0, "c_iso_90": 0.0,
         "c_iso_95": 0.0, "lune_area95": 0.02, "Mw": 3.50},
    ])
    frame = events.build_event_index(run, ["NEG", "POS", "MID"], run / "tables")
    by_id = frame.set_index("event_id")

    assert set(frame.loc[frame["is_neg_iso"], "event_id"]) == {"NEG"}
    assert set(frame.loc[frame["is_pos_iso"], "event_id"]) == {"POS"}
    assert by_id.loc["POS", "directory"] == "Mw4.25_POS_posISO"
    assert by_id.loc["NEG", "directory"] == "Mw4.00_NEG_negISO"
    assert by_id.loc["MID", "directory"] == "Mw3.50_MID"
    # the two flags must never both be set
    assert not (frame["is_neg_iso"] & frame["is_pos_iso"]).any()
    assert "posISO" in frame.attrs["iso_source"]
