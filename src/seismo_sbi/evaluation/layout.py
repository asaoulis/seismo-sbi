"""
layout.py
=========
Per-model output tree helpers for the evaluation harness.

Lifted from ``scripts/santorini_pathbreaker/run_posttrain_eval.py`` and
generalised: ``OutputLayout`` gains ``validation_dir()`` and ``stage_dir(name)``
alongside the existing ``event_dir(event)``.  ``_git_rev`` is renamed to the
public ``git_rev`` so other modules (and the CLI) can call it without the
leading underscore.

All directory methods are *mkdir-on-use* (lazy), mirroring the original
``event_dir`` behaviour.  No heavy deps — subprocess only (for git_rev).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OutputLayout:
    """Resolved per-model output tree.

    All ``*_dir`` methods lazily ``mkdir(parents=True, exist_ok=True)`` on first
    access, so callers can ask for a dir and write into it without separate mkdir
    calls.

    Layout::

        model_root/
            run_meta.json
            ml_posttrain_summary.json
            station_usage.json / .csv
            <event>/          ← event_dir(event)
            validation/       ← validation_dir()
            <stage_name>/     ← stage_dir(stage_name)
    """

    model_root: Path

    def __post_init__(self):
        self.model_root = Path(self.model_root)
        self.model_root.mkdir(parents=True, exist_ok=True)

    def event_dir(self, event: str) -> Path:
        """Return (and create) the per-event subdirectory."""
        d = self.model_root / event
        d.mkdir(parents=True, exist_ok=True)
        return d

    def validation_dir(self) -> Path:
        """Return (and create) the per-model validation subdirectory."""
        d = self.model_root / "validation"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def stage_dir(self, name: str) -> Path:
        """Return (and create) an arbitrary named stage subdirectory."""
        d = self.model_root / name
        d.mkdir(parents=True, exist_ok=True)
        return d


def resolve_output_layout(output_root, model_name) -> OutputLayout:
    """Build the per-model layout under ``output_root/model_name`` (mkdir on use)."""
    return OutputLayout(model_root=Path(output_root) / model_name)


def git_rev(path=None) -> "str | None":
    """Best-effort short git revision of the checkout (None if unavailable).

    ``path`` is the directory used as the git working tree (``-C`` flag);
    defaults to the current working directory.
    """
    cmd = ["git"]
    if path is not None:
        cmd += ["-C", str(path)]
    cmd += ["rev-parse", "--short", "HEAD"]
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return None
