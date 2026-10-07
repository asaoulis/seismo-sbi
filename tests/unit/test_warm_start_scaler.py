"""A warm-started run keeps the scalar-moment convention of the run it resumes."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from seismo_sbi.sbi.scalers import MomentTensorScaler, ScalerConfiguration, scaler_provenance
from seismo_sbi.sbi.datasets.training_data import training_scaler
from seismo_sbi.sbi.types.parameters import ModelParameters

MAX_ABS_NM = 2e18
ML_SCALER = ScalerConfiguration(moment_tensor="scale_shape", mt_log_decades=9.0)
#: The scale-shape window ``ML_SCALER`` resolves to, recorded before the M0 convention was.
LEGACY_RECORD = {"moment_tensor": "scale_shape", "log10_m0_min": np.log10(MAX_ABS_NM / np.sqrt(2)) - 9.0,
                 "log10_m0_max": np.log10(MAX_ABS_NM / np.sqrt(2))}


def moment_tensor_parameters():
    p = ModelParameters()
    p.names = {"moment_tensor": ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]}
    p.theta_fiducial = {"moment_tensor": [1e15] * 6}
    p.bounds = {"moment_tensor": [[-MAX_ABS_NM] * 6, [MAX_ABS_NM] * 6]}
    return p


def source_run(models_path, record):
    """A resumable run directory whose sidecar holds ``record`` as its theta_scaler (none if None)."""
    run = models_path / "source_run"
    run.mkdir()
    if record is not None:
        (run / "model_meta.json").write_text(json.dumps({"model_config": {"theta_scaler": record}}))
    return SimpleNamespace(warm_start_run_name="source_run")


def m0_convention(scaler):
    return next(block for block in scaler.scalers if isinstance(block, MomentTensorScaler)).m0_convention


def test_a_cold_start_uses_the_full_tensor_moment(tmp_path):
    cold_start = SimpleNamespace(warm_start_run_name=None)
    scaler = training_scaler(moment_tensor_parameters(), ML_SCALER, None, cold_start, tmp_path)
    assert m0_convention(scaler) == "full_tensor"


def test_warm_starting_a_legacy_run_keeps_the_six_component_moment(tmp_path):
    training = source_run(tmp_path, dict(LEGACY_RECORD))
    scaler = training_scaler(moment_tensor_parameters(), ML_SCALER, None, training, tmp_path)
    assert m0_convention(scaler) == "six_components"
    assert scaler_provenance(scaler)["m0_convention"] == "six_components"


def test_warm_starting_a_run_without_a_sidecar_keeps_the_six_component_moment(tmp_path):
    training = source_run(tmp_path, None)
    assert m0_convention(training_scaler(moment_tensor_parameters(), ML_SCALER, None, training, tmp_path)) \
        == "six_components"


def test_warm_starting_a_full_tensor_run_keeps_the_full_tensor_moment(tmp_path):
    training = source_run(tmp_path, {**LEGACY_RECORD, "m0_convention": "full_tensor"})
    assert m0_convention(training_scaler(moment_tensor_parameters(), ML_SCALER, None, training, tmp_path)) \
        == "full_tensor"


def test_warm_starting_with_a_different_scaling_raises(tmp_path):
    training = source_run(tmp_path, {**LEGACY_RECORD, "log10_m0_max": LEGACY_RECORD["log10_m0_max"] + 1.0})
    with pytest.raises(ValueError, match="MISMATCH"):
        training_scaler(moment_tensor_parameters(), ML_SCALER, None, training, tmp_path)
