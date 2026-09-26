"""The three LV2 inversions reproduce the golden baseline within its loose tolerances.

Runs the Gaussian-likelihood MCMC and SBI score compression (event_inversion.py) and the ML-NPE
leg (run_ml_inversion.py) on the real LV2 event from ``examples/``, summarises gamma, delta and
Mw per method and compares them with ``baseline_LV2.json``. Inputs are the local LV2 data, the
LV2 checkpoint, ``INSTASEIS_DB`` and ``CPS_PATH``; outputs, stencils included, are cached under
``LV2_CONTINUITY_OUTPUT``. Run with ``pytest tests/continuity -m slow``.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.continuity.compare_to_baseline import compare
from tests.continuity.extract_source_summary import summarise_pkl

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXAMPLES = REPO / "examples"
INSTASEIS_DB = Path(os.environ.get("INSTASEIS_DB", "/data/shared/ROSA_PREM_10s_disc"))
CPS_PATH = Path(os.environ.get("CPS_PATH", "/nonexistent"))
OUTPUT = Path(os.environ.get("LV2_CONTINUITY_OUTPUT", EXAMPLES / "data" / "pipeline_outputs" / "continuity"))
REQUIRED = [EXAMPLES / "data" / "preprocessed" / "LV2" / "events", EXAMPLES / "data" / "models" / "LV2_perturbations",
            EXAMPLES / "ml-checkpoints" / "checkpoints", EXAMPLES / "configs" / "stations.txt", INSTASEIS_DB, CPS_PATH]


def write_runtime_config(directory):
    """The committed config with this machine's database, CPS binaries and output directory."""
    config = yaml.safe_load((HERE / "LV2_continuity.yaml").read_text())
    config["output_directory"] = str(OUTPUT)
    config["seismic_context"]["syngine_address"] = str(INSTASEIS_DB)
    config["seismic_context"]["cps_path"] = str(CPS_PATH)
    path = Path(directory) / "LV2_continuity.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def run_from_examples(script, *args):
    environment = dict(os.environ, PATH=f"{CPS_PATH}{os.pathsep}{os.environ.get('PATH', '')}")
    subprocess.run([sys.executable, str(script), *map(str, args)], cwd=EXAMPLES, env=environment, check=True)


@pytest.mark.slow
@pytest.mark.requires_data
def test_lv2_inversions_match_the_golden_baseline(tmp_path):
    missing = [str(path) for path in REQUIRED if not path.exists()]
    if missing:
        pytest.skip(f"LV2 continuity inputs missing: {missing}")
    config = write_runtime_config(tmp_path)
    run_from_examples(REPO / "scripts" / "event_inversion.py", "-c", config)
    run_from_examples(HERE / "run_ml_inversion.py", "--config", config, "--ckpt_dir", "./ml-checkpoints",
                      "--output_dir", tmp_path / "ml")
    summary = {}
    summarise_pkl(OUTPUT / "jobs" / "continuity" / "results" / "inversion_results.pkl", summary, False)
    summarise_pkl(tmp_path / "ml" / "inversion_results_ml.pkl", summary, False)
    all_pass, n_methods = compare(summary, json.loads((HERE / "baseline_LV2.json").read_text()))
    assert n_methods == 3
    assert all_pass
