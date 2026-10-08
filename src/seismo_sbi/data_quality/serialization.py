"""The files a data-QA run writes and reads.

``components.json`` maps ``{station: [Z, E, N]}``, with ``[]`` dropping the station, and is read
by ``Receivers._convert_to_instaseis_receivers``. ``time_shifts.json`` maps ``{station: int}``
for the non-zero static shifts in samples, read by ``Receivers.set_time_shifts``.
``*_allstation_verdicts.json`` holds ``{"present": {station: {...}}, "absent_from_h5": [...]}``
as the audit trail; each per-station record lists the nine summary fields first, then the
multi-component coherence and any fidelity field computed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .policy import KEPT_VERDICTS, StationVerdict


def write_components_json(path, station_components: Dict[str, List[str]]) -> None:
    Path(path).write_text(json.dumps(station_components, indent=2))


def write_time_shifts_json(path, shifts: Dict[str, int]) -> None:
    """Write the ``{station: shift}`` map (callers pass non-zero shifts only)."""
    Path(path).write_text(json.dumps({k: int(v) for k, v in shifts.items()}, indent=2))


def verdict_to_json(v: StationVerdict) -> dict:
    """A verdict as a JSON record: the nine summary fields in a fixed order, then the
    multi-component coherence and any posterior-predictive fidelity field computed."""
    s = v.summary
    out = dict(
        dist_km=s.dist_km, azimuth=s.azimuth,
        vr=s.vr, aligned_vr=s.aligned_vr,
        lag_Z=s.lag_Z, xcorr_Z=s.xcorr_Z,
        median_amp_ratio=s.median_amp_ratio,
        verdict=v.verdict, suggested_shift=v.suggested_shift,
    )
    out["aligned_coherence"] = s.aligned_coherence
    for name in ("corr_misfit", "envelope_misfit", "reduced_chi2"):
        val = getattr(s, name)
        if val is not None:
            out[name] = val
    return out


def write_verdicts_json(
    path,
    verdicts: Dict[str, StationVerdict],
    absent_from_h5: List[str],
) -> None:
    payload = dict(
        present={s: verdict_to_json(v) for s, v in verdicts.items()},
        absent_from_h5=list(absent_from_h5),
    )
    Path(path).write_text(json.dumps(payload, indent=2))


def components_from_verdicts(
    verdicts: Dict[str, StationVerdict],
    all_stations: List[str],
    component_verdicts: Dict[str, Dict] = None,
    full_components: List[str] = ("Z", "E", "N"),
) -> Dict[str, List[str]]:
    """Build the ``components.json`` map: ``{station: [kept components] | []}``.

    By default the policy is per station: kept stations get all ``full_components`` and
    everything else (drops, stations absent from the verdicts) gets ``[]``.

    If ``component_verdicts`` (``{station: {component: ComponentVerdict}}``, from
    :func:`policy.component_verdicts`) is supplied, a kept station keeps only the components
    whose per-component verdict is ``keep``, so a station can keep ``[Z, E]`` while a bad ``N``
    channel is dropped (zero-filled at load). A station-level drop still zeroes the whole station (``[]``), and a
    kept station with no surviving component also collapses to ``[]``."""
    kept = {s for s, v in verdicts.items() if v.is_kept}
    full = list(full_components)
    out: Dict[str, List[str]] = {}
    for s in all_stations:
        if s not in kept:
            out[s] = []
            continue
        if component_verdicts and s in component_verdicts:
            cv = component_verdicts[s]
            out[s] = [c for c in full if c in cv and cv[c].is_kept]
        else:
            out[s] = list(full)
    return out


@dataclass
class QAArtifacts:
    """The QA decisions for one event, loaded from disk."""

    event: str
    present: Dict[str, dict] = field(default_factory=dict)       # station -> verdict record
    absent_from_h5: List[str] = field(default_factory=list)
    components: Dict[str, List[str]] = field(default_factory=dict)
    time_shifts: Dict[str, int] = field(default_factory=dict)

    @property
    def kept_stations(self) -> List[str]:
        """Stations the components.json kept (non-empty == used by the inversion)."""
        return [s for s, c in self.components.items() if c]

    @property
    def usable_stations(self) -> List[str]:
        """Stations whose verdict is keep/time-shift (the QA-kept subset)."""
        return [s for s, d in self.present.items()
                if str(d.get("verdict", "")).lower() in KEPT_VERDICTS]


def load_qa_artifacts(
    event: str,
    *,
    components_path=None,
    time_shifts_path=None,
    verdicts_path=None,
) -> QAArtifacts:
    """Load whichever of the three artifacts exist into a :class:`QAArtifacts`.

    Each path is given separately, since a study may keep the three files in different
    directories. A missing file leaves its field empty.
    """
    art = QAArtifacts(event=event)
    if components_path is not None and Path(components_path).exists():
        art.components = json.loads(Path(components_path).read_text())
    if time_shifts_path is not None and Path(time_shifts_path).exists():
        art.time_shifts = {k: int(v) for k, v in
                           json.loads(Path(time_shifts_path).read_text()).items()}
    if verdicts_path is not None and Path(verdicts_path).exists():
        v = json.loads(Path(verdicts_path).read_text())
        art.present = v.get("present") or {}
        art.absent_from_h5 = v.get("absent_from_h5") or []
    return art
