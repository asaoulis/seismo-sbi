"""Every committed pipeline configuration parses, with the paths it names resolved from its run directory.

Configurations under ``scripts/configs/`` are run from ``scripts/`` and those under
``examples/configs/`` from ``examples/``, so relative paths are resolved from there. The set is
what git tracks, so local-only files beside them are not checked; nor are the directories of the
studies that moved to mt-sbi.
"""
import subprocess
from pathlib import Path

import pytest

from seismo_sbi.sbi.configuration import SBI_Configuration

REPO = Path(__file__).resolve().parents[2]
CHECKED_DIRS = ["scripts/configs/azores", "scripts/configs/croatia", "scripts/configs/JAN",
                "scripts/configs/long_valley", "scripts/configs/ridgecrest"]
NOT_PIPELINE_CONFIGS = {"examples/configs/axisem_ensemble.yaml"}


def committed_configs():
    """Repository-relative paths of the tracked pipeline configurations checked here."""
    listed = subprocess.run(["git", "ls-files", *CHECKED_DIRS, "examples/configs"], cwd=REPO,
                            capture_output=True, text=True)
    return sorted(p for p in listed.stdout.split("\n")
                  if p.endswith(".yaml") and p not in NOT_PIPELINE_CONFIGS)


@pytest.mark.parametrize("config", committed_configs())
def test_committed_config_parses(config, monkeypatch):
    monkeypatch.chdir(REPO / Path(config).parts[0])
    SBI_Configuration().parse_config_file(REPO / config)
