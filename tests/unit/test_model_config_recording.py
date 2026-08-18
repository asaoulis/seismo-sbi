#!/usr/bin/env python3
"""Settings resolved AFTER the trainer is built must still reach model_meta.json.

`CompressionTrainer.__init__` merges the caller's ``model_config`` into a NEW dict, so a
caller that mutates its own dict afterwards silently loses the entry from the sidecar.
That is how the MMD auxiliary-loss block went unrecorded: every MMD-trained checkpoint was
metadata-identical to a non-MMD one, leaving a lambda sweep unattributable after the fact.
`record_model_config` is the supported way to register such late-resolved settings.
"""
import inspect

from seismo_sbi.sbi.compression.ML.train import CompressionTrainer


def _trainer_config_after(initial):
    """Build only the merged-config state, without constructing the (heavy) model."""
    obj = CompressionTrainer.__new__(CompressionTrainer)
    from seismo_sbi.sbi.compression.ML.train import DEFAULT_MODEL_CONFIG
    obj._model_config = {**DEFAULT_MODEL_CONFIG, "channels": 128, **(initial or {})}
    return obj


def test_the_merged_config_is_a_new_dict_not_the_callers():
    """The root cause: mutating the caller's dict after construction cannot reach the sidecar."""
    callers = {"station_encoder": "tcn"}
    obj = _trainer_config_after(callers)
    callers["mmd"] = {"lambda_mmd": 50.0}          # what train_NPE used to rely on
    assert "mmd" not in obj._model_config, (
        "if this ever passes by reference, the recording bug is masked rather than fixed"
    )


def test_record_model_config_reaches_the_recorded_config():
    obj = _trainer_config_after({"station_encoder": "tcn"})
    CompressionTrainer.record_model_config(obj, mmd={"lambda_mmd": 50.0, "warmup_epochs": 5})
    assert obj._model_config["mmd"]["lambda_mmd"] == 50.0
    assert obj._model_config["station_encoder"] == "tcn"   # merges, does not replace


def test_record_model_config_returns_the_config_and_accepts_several_entries():
    obj = _trainer_config_after({})
    out = CompressionTrainer.record_model_config(obj, mmd={"lambda_mmd": 5.0}, extra=1)
    assert out is obj._model_config
    assert out["mmd"]["lambda_mmd"] == 5.0 and out["extra"] == 1


def test_the_sidecar_serialises_the_recorded_model_config():
    """meta['model_config'] is self._model_config, so recorded entries land in the JSON.

    The dict lives in ``write_model_meta`` (which ``train`` delegates to, and which the
    wall-kill recovery path calls directly) — both are asserted so the guarantee cannot be
    lost by the sidecar being built somewhere that only one entry point reaches.
    """
    src = inspect.getsource(CompressionTrainer.write_model_meta)
    assert '"model_config": self._model_config' in src, (
        "sidecar must serialise the trainer's own merged config, not a caller-supplied one"
    )
    assert "self.write_model_meta(" in inspect.getsource(CompressionTrainer.train), (
        "train must write the sidecar through write_model_meta so a training run and the "
        "recovery path emit the same metadata"
    )


def test_train_npe_registers_mmd_through_the_recorder():
    """Regression guard on the call site itself — a plain dict mutation is not enough."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[2] / "scripts" / "train_NPE.py"
    text = src.read_text()
    assert "trainer.record_model_config(mmd=" in text, (
        "train_NPE must register the MMD block via record_model_config, or MMD checkpoints "
        "lose it from model_meta.json"
    )


def test_every_record_model_config_call_site_uses_keywords():
    """`record_model_config(self, **entries)` is KEYWORD-ONLY — a positional dict raises
    TypeError at runtime.

    Regression guard for a real failure: `record_model_config({"warm_start_checkpoint": ...})`
    reached the cluster and killed a 2-GPU training job ~2 minutes in. Neither the unit tests
    (which call the trainer method directly) nor the smoke gate (which never invokes
    `train_NPE.main`) executed that line, so a source-level pin is the cheap guard — the same
    reason `test_train_npe_registers_mmd_through_the_recorder` exists.
    """
    import re
    from pathlib import Path
    src = Path(__file__).resolve().parents[2] / "scripts" / "train_NPE.py"
    calls = re.findall(r"record_model_config\(([^)]*)", src.read_text())
    assert calls, "expected at least one record_model_config call site in train_NPE.py"
    for arg in calls:
        arg = arg.strip()
        assert not arg.startswith("{"), (
            f"record_model_config takes **entries, not a positional dict: "
            f"record_model_config({arg}...) raises TypeError at runtime"
        )
        assert re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", arg), (
            f"record_model_config call site must pass keyword arguments, got: {arg}"
        )
