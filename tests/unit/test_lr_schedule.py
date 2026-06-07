"""Unit tests for the selectable second-stage LR schedule (warmup -> cosine|constant).

The follow-up NDE-head/LR experiment (ml-improvements/positional-encodings) adds a
``constant`` option so the LR is held flat at the base lr after warmup instead of
cosine-decaying to ``lr*0.1``. These tests pin that behaviour and the plumbing of
``lr_second_stage`` from ``CompressionTrainer`` into ``NPELightningModule``.

Pure CPU, no Instaseis/CPS, no disk I/O.
"""

import types

import pytest
import torch

from seismo_sbi.sbi.compression.ML.maf import build_nsf
from seismo_sbi.sbi.compression.ML.seismogram_transformer import NPELightningModule
from seismo_sbi.sbi.compression.ML.train import CompressionTrainer

pytestmark = pytest.mark.unit


def _tiny_module(stage, lr=1e-4):
    """A minimal NPELightningModule with a real (tiny) flow and a mocked trainer."""
    flow = build_nsf(dim=6, conditional_dim=8, hidden_features=32, num_transforms=2)
    mod = NPELightningModule(flow, lr=lr, lr_second_stage=stage)
    return mod


def _lr_curve(stage, max_epochs=40, lr=1e-4):
    mod = _tiny_module(stage, lr=lr)
    # configure_optimizers reads only trainer.max_epochs (epoch schedule); mock it.
    mod.trainer = types.SimpleNamespace(
        max_epochs=max_epochs,
        estimated_stepping_batches=max_epochs * 10,
        num_training_batches=10,
    )
    (opt,), (sch_cfg,) = mod.configure_optimizers()
    sch = sch_cfg["scheduler"]
    lrs = []
    for _ in range(max_epochs):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    return lrs


def test_constant_schedule_is_flat_at_base_lr_after_warmup():
    base_lr = 1e-4
    max_epochs = 40
    warmup_epochs = max(1, int(0.05 * max_epochs))  # mirrors configure_optimizers
    lrs = _lr_curve("constant", max_epochs=max_epochs, lr=base_lr)

    # During/after warmup the LR ramps up to base_lr and then stays exactly flat.
    post_warmup = lrs[warmup_epochs:]
    assert post_warmup, "expected epochs after warmup"
    assert all(abs(x - base_lr) < 1e-12 for x in post_warmup), (
        f"constant schedule should hold {base_lr}; got {post_warmup}")
    # Warmup never exceeds base_lr.
    assert max(lrs) <= base_lr + 1e-12


def test_cosine_schedule_decays_below_constant():
    base_lr = 1e-4
    const = _lr_curve("constant", lr=base_lr)
    cos = _lr_curve("cosine", lr=base_lr)
    # Cosine anneals toward eta_min = lr*0.1; its final LR is well below the flat one.
    assert cos[-1] < const[-1]
    assert cos[-1] == pytest.approx(base_lr * 0.1, rel=0.2)
    # The constant final LR is the base lr, NOT decayed.
    assert const[-1] == pytest.approx(base_lr, abs=1e-12)


def test_default_schedule_is_cosine():
    """Absent an explicit choice the schedule stays cosine (legacy, byte-identical)."""
    mod = _tiny_module("cosine")
    assert mod.lr_second_stage == "cosine"
    # NPELightningModule default arg is cosine too.
    default_mod = NPELightningModule(
        build_nsf(dim=6, conditional_dim=8, hidden_features=16, num_transforms=2), lr=1e-4)
    assert default_mod.lr_second_stage == "cosine"


def test_compression_trainer_forwards_lr_second_stage():
    """CompressionTrainer stores lr_second_stage and hands it to the Lightning module."""
    station_locations = torch.tensor(
        [[10.0, 20.0], [11.0, 21.0], [9.0, 19.0]], dtype=torch.float32).numpy()
    trainer = CompressionTrainer(
        components="Z",
        station_locations=station_locations,
        channels=16, latent_dim=16, num_dims=6,
        architecture="seismogram_transformer",
        trace_length=200,
        lr=1e-4,
        lr_second_stage="constant",
    )
    assert trainer.lr_second_stage == "constant"
    assert trainer.model.lr_second_stage == "constant"


def test_compression_trainer_lr_second_stage_defaults_cosine():
    station_locations = torch.tensor([[10.0, 20.0]], dtype=torch.float32).numpy()
    trainer = CompressionTrainer(
        components="Z",
        station_locations=station_locations,
        channels=16, latent_dim=16, num_dims=6,
        architecture="seismogram_transformer",
        trace_length=200,
    )
    assert trainer.lr_second_stage == "cosine"
    assert trainer.model.lr_second_stage == "cosine"
