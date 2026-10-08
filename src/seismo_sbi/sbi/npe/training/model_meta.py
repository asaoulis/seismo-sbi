"""The ``model_meta.json`` sidecar a training run writes beside its checkpoints.

The sidecar records the architecture, the station locations, the trace length and the parameter
scaling, so a run can be rebuilt without its configuration. :func:`read_model_meta` reads it from a
run directory.
"""
import json
from pathlib import Path

#: File name of the sidecar inside a run directory.
MODEL_META_FILENAME = "model_meta.json"


def read_model_meta(run_directory) -> dict:
    """The parsed sidecar of the run in ``run_directory``; raises ``FileNotFoundError`` when it is missing."""
    meta_path = Path(run_directory) / MODEL_META_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} is missing; a run is rebuilt from its sidecar.")
    return json.loads(meta_path.read_text())
