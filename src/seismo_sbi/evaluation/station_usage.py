"""
station_usage.py
================
Station-breakdown writers for the evaluation harness.

Lifted verbatim from ``scripts/santorini_pathbreaker/run_posttrain_eval.py``
(private helpers ``_write_station_breakdown`` / ``_write_station_usage`` made
public by dropping the leading underscore).

Pure JSON/CSV writers operating on in-memory name lists and
``StationConfig.as_dict()`` dicts — no torch or plotting deps, so these are
cheap to unit-test.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def write_station_breakdown(
    out_dir,
    event: str,
    master_names: Sequence[str],
    alls: Sequence[str],
    filt: Sequence[str],
    dropout_configs: Optional[List],
) -> dict:
    """Per-event station-usage artifact.

    Populated entirely from in-memory data already used by the inversions: the
    all-available set, the QA-filtered subset, and each station-dropout
    ``StationConfig.as_dict()`` (which exposes label/n/keep_indices/kept/dropped).
    Writes ``ml_<event>_stations.json`` under ``out_dir``.

    Returns the dict that was written (for in-memory aggregation).

    Parameters
    ----------
    out_dir:
        Destination directory (typically ``layout.event_dir(event)``).
    event:
        Event identifier string (e.g. ``"No01_id2148"``).
    master_names:
        The full model master station list.
    alls:
        All-available stations for this event (subset of master, ordered as in master).
    filt:
        QA-filtered subset used by the traditional inversions (⊆ alls).
    dropout_configs:
        List of ``StationConfig`` objects (or ``None`` / empty) from the dropout
        ensemble; each must expose ``.as_dict()``.
    """
    stations = {
        "event": event,
        "master_stations": list(master_names),
        "all_available": list(alls),
        "filtered": list(filt),
        "dropped_by_qa": sorted(set(alls) - set(filt)),
        "n_all": len(alls),
        "n_filtered": len(filt),
        "dropout_configs": [c.as_dict() for c in (dropout_configs or [])],
    }
    out_dir = Path(out_dir)
    with open(out_dir / f"ml_{event}_stations.json", "w") as f:
        json.dump(stations, f, indent=2)
    return stations


def write_station_usage(
    model_root,
    master_names: Sequence[str],
    per_event_stations: Dict[str, dict],
) -> None:
    """Cross-event station-usage matrix at the model root.

    Cell codes per (event, master station):
    ``F`` = used by both all and filtered posteriors,
    ``A`` = available (all-only, QA-dropped),
    ``-`` = absent for that event.

    Writes ``station_usage.json`` + a human-readable ``station_usage.csv`` under
    ``model_root``.

    Parameters
    ----------
    model_root:
        The per-model output root (``layout.model_root``).
    master_names:
        The full model master station list.
    per_event_stations:
        ``{event: stations_dict}`` where each value is what
        ``write_station_breakdown`` returned (has ``all_available``, ``filtered``,
        ``n_all``, ``n_filtered``).
    """
    master = list(master_names)
    matrix: Dict[str, dict] = {}
    for event, st in per_event_stations.items():
        alls, filt = set(st["all_available"]), set(st["filtered"])
        codes: Dict[str, str] = {}
        for s in master:
            if s in filt:
                codes[s] = "F"
            elif s in alls:
                codes[s] = "A"
            else:
                codes[s] = "-"
        matrix[event] = {
            "codes": codes,
            "n_all": st["n_all"],
            "n_filtered": st["n_filtered"],
        }

    model_root = Path(model_root)
    with open(model_root / "station_usage.json", "w") as f:
        json.dump(
            {
                "legend": {
                    "F": "all+filtered",
                    "A": "all-only (QA-dropped)",
                    "-": "absent",
                },
                "master_stations": master,
                "events": matrix,
            },
            f,
            indent=2,
        )
    with open(model_root / "station_usage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "n_all", "n_filtered"] + master)
        for event, row in matrix.items():
            w.writerow(
                [event, row["n_all"], row["n_filtered"]]
                + [row["codes"][s] for s in master]
            )
