"""The example notebooks execute headless and print the numbers they printed before.

Each notebook runs in a copy of ``examples/`` under a temporary directory, with the rest of
the repository linked in, so nothing it writes lands in the checkout. Every number a code cell
prints is compared with ``notebook_outputs.json``, as is which cells raise. A notebook that is
broken today is recorded broken, so the test also notices the day it is repaired. After a
deliberate change, rewrite the reference with ``python tests/examples/test_notebooks_execute.py``.
"""
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).with_name("notebook_outputs.json")
INSTASEIS_DB = Path(os.environ.get("INSTASEIS_DB", "/data/shared/ROSA_PREM_10s_disc"))
CPS_PROGRAM = Path(os.environ.get("CPS_PATH", ""), "hprep96")
INSTASEIS_DB_20S = Path(os.environ.get("INSTASEIS_DB_20S", "/data/shared/prem_a_20s"))
#: Relative tolerance on every printed number.
RTOL = 1e-6
#: Notebook to the local inputs it needs; a missing one skips the notebook.
NOTEBOOKS = {
    "ridgecrest_obspy": [INSTASEIS_DB_20S, REPO / "examples" / "data" / "ridgecrest"],
    "nuisances": [INSTASEIS_DB_20S, REPO / "examples" / "data" / "ridgecrest"],
    "npe_flagship": [INSTASEIS_DB_20S, REPO / "examples" / "data" / "ridgecrest"],
    "01_forward_models_and_receivers": [INSTASEIS_DB, CPS_PROGRAM],
    "02_noise_covariances_and_likelihood": [INSTASEIS_DB],
    "03_npe_training_and_evaluation": [INSTASEIS_DB],
    "04_source_conventions": [INSTASEIS_DB],
    "05_resolution_and_tradeoffs": [CPS_PROGRAM],
    "azores_inversion": [INSTASEIS_DB, REPO / "examples" / "data" / "azores"],
    "theory_errors_LV2": [REPO / "examples" / "data", REPO / "examples" / "ml-checkpoints"],
}
#: Notebook to about three times its usual running time in seconds; a cell still running after
#: that long fails the notebook, so a hung worker pool fails fast.
TIMEOUT_S = {"ridgecrest_obspy": 120, "nuisances": 1800, "npe_flagship": 2400, "01_forward_models_and_receivers": 300, "02_noise_covariances_and_likelihood": 600,
             "03_npe_training_and_evaluation": 2400, "04_source_conventions": 300,
             "05_resolution_and_tradeoffs": 900, "azores_inversion": 2400, "theory_errors_LV2": 5400}
#: Code cells whose printed numbers change run to run (a subprocess's partly captured output,
#: unseeded noise draws, network training, MCMC convergence warnings, git output); only whether
#: they raise is compared.
STOCHASTIC_CELLS = {"nuisances": {5, 6}, "npe_flagship": {5, 6, 8, 9, 10, 11}, "theory_errors_LV2": {0, 2, 3, 8, 10, 11, 13, 17}, "azores_inversion": {2, 8, 11, 14},
                    "02_noise_covariances_and_likelihood": {7}, "03_npe_training_and_evaluation": {3, 4, 5}}
MASKS = [re.compile(r"[^\n\r]*(it/s|s/it|\?it)[^\n\r]*"), re.compile(r"/tmp/\S+"), re.compile(r"\d{4}-\d\d-\d\d[ T][\d:.,]+"),
         re.compile(r"\d+(\.\d+)?\s*(s|ms|seconds|it/s|s/it)\b"),
         re.compile(r"\d\d:\d\d(:\d\d)?"), re.compile(r"0x[0-9a-f]+"),
         re.compile(r"\d+%\|[^\n]*")]
NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def mirror_repository(root: Path) -> Path:
    """Link the repository into ``root``, copying ``examples/`` and ``scripts/configs/``.

    ``.git`` is left out, so a notebook's git commands cannot touch the checkout's hooks.
    """
    for entry in REPO.iterdir():
        if entry.name not in ("examples", "scripts", ".git"):
            (root / entry.name).symlink_to(entry)
    (root / "scripts").mkdir()
    for entry in (REPO / "scripts").iterdir():
        if entry.name == "configs":
            shutil.copytree(entry, root / "scripts" / "configs", symlinks=True)
        else:
            (root / "scripts" / entry.name).symlink_to(entry)
    shutil.copytree(REPO / "examples", root / "examples", symlinks=True,
                    ignore=shutil.ignore_patterns("wandb", "sbi-logs", "__pycache__",
                                                  ".ipynb_checkpoints"))
    return root / "examples"


def input_checksums() -> dict:
    """``{path: sha256}`` of every file under ``examples/data`` except the pipeline outputs."""
    data = REPO / "examples" / "data"
    return {str(path.relative_to(data)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(data.rglob("*"))
            if path.is_file() and "pipeline_outputs" not in path.relative_to(data).parts}


def execute(name: str, root: Path):
    """Run ``examples/<name>.ipynb`` to the end, keeping going past a raising cell."""
    import nbformat
    from nbclient import NotebookClient

    cwd = mirror_repository(root)
    notebook = nbformat.read(REPO / "examples" / f"{name}.ipynb", as_version=4)
    NotebookClient(notebook, timeout=TIMEOUT_S[name], kernel_name="python3", allow_errors=True,
                   resources={"metadata": {"path": str(cwd)}}).execute()
    return notebook


def printed_text(cell) -> str:
    """What a code cell printed or returned, with paths, timings and progress bars masked."""
    parts = [output.get("text", "") for output in cell.get("outputs", [])
             if output.get("output_type") == "stream"]
    parts += [output["data"].get("text/plain", "") for output in cell.get("outputs", [])
              if output.get("output_type") == "execute_result"]
    text = "".join(parts)
    for mask in MASKS:
        text = mask.sub("#", text)
    return text


def summarise(notebook) -> list:
    """Per code cell: the exception it raised (or ``None``) and the numbers it printed."""
    summary = []
    for cell in (c for c in notebook.cells if c.cell_type == "code"):
        errors = [o["ename"] for o in cell.get("outputs", []) if o.get("output_type") == "error"]
        summary.append({"error": errors[0] if errors else None,
                        "numbers": [float(n) for n in NUMBER.findall(printed_text(cell))]})
    return summary


def mismatches(summary: list, reference: list, stochastic=()) -> list:
    """One line per code cell whose exception or printed numbers differ from the reference.

    Cells listed in ``stochastic`` are checked for their exception only.
    """
    if len(summary) != len(reference):
        return [f"{len(summary)} code cells, the reference has {len(reference)}"]
    found = []
    for index, (cell, expected) in enumerate(zip(summary, reference)):
        numbers, wanted = np.array(cell["numbers"]), np.array(expected["numbers"])
        if cell["error"] != expected["error"]:
            found.append(f"cell {index}: raised {cell['error']}, expected {expected['error']}")
        elif index in stochastic:
            continue
        elif numbers.shape != wanted.shape:
            found.append(f"cell {index}: printed {numbers.size} numbers, expected {wanted.size}")
        elif not np.allclose(numbers, wanted, rtol=RTOL, atol=0.0, equal_nan=True):
            worst = np.max(np.abs(numbers - wanted) / np.maximum(np.abs(wanted), 1e-300))
            found.append(f"cell {index}: numbers differ, worst relative {worst:.3g}")
    return found


@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.parametrize("name", list(NOTEBOOKS))
def test_the_notebook_prints_what_it_printed_before(name, tmp_path, monkeypatch):
    monkeypatch.setenv("INSTASEIS_DB", str(INSTASEIS_DB))
    monkeypatch.setenv("INSTASEIS_DB_20S", str(INSTASEIS_DB_20S))
    missing = [str(path) for path in NOTEBOOKS[name] if not path.exists()]
    if missing:
        pytest.skip(f"needs {missing}")
    reference = json.loads(REFERENCE.read_text())[name]
    inputs_before = input_checksums()
    summary = summarise(execute(name, tmp_path))
    assert mismatches(summary, reference, STOCHASTIC_CELLS.get(name, ())) == []
    assert input_checksums() == inputs_before, "the notebook changed the checkout's examples/data"


if __name__ == "__main__":
    import tempfile

    os.environ["INSTASEIS_DB"] = str(INSTASEIS_DB)
    os.environ["INSTASEIS_DB_20S"] = str(INSTASEIS_DB_20S)
    names = sys.argv[1:] or list(NOTEBOOKS)
    stored = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else {}
    for name in names:
        with tempfile.TemporaryDirectory() as scratch:
            stored[name] = summarise(execute(name, Path(scratch)))
        print(name, [cell["error"] for cell in stored[name]])
    REFERENCE.write_text(json.dumps(stored, indent=1) + "\n")
