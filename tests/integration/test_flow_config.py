"""Integration tests for the configurable NDE head (normalising-flow) sizing.

The follow-up NDE-head/LR experiment exposes ``ml_flow`` -> ``flow_config`` so the flow's
coupling-transform depth (``num_transforms``) and, optionally, its hidden width
(``hidden_features``) can be set from the YAML. These tests pin:

* num_transforms flows through CompressionTrainer into the assembled flow (default 5, override 8);
* the flow hidden width defaults to the embedding channel width (``channels``) and can be
  decoupled via an explicit ``hidden_features`` override;
* flow_config persists to model_meta.json so a reloaded checkpoint rebuilds the SAME-sized head.

Builds the real CompressionTrainer flow (seismogram_transformer embedding). CPU-fast, no
Instaseis/CPS, no training.
"""

import numpy as np
import pytest
import torch

from pyknos.nflows import transforms

from seismo_sbi.sbi.compression.ML.train import CompressionTrainer

pytestmark = pytest.mark.integration

_C, _T, _DM = "Z", 200, 16


def _locations():
    return torch.tensor(
        [[10.0, 20.0], [11.0, 21.0], [9.0, 19.0]], dtype=torch.float32).numpy()


def _count_coupling_transforms(flow):
    return sum(
        isinstance(t, transforms.PiecewiseRationalQuadraticCouplingTransform)
        for t in flow._transform._transforms
    )


def _conditioner_hidden_widths(flow):
    """Hidden Linear widths inside the first coupling transform's conditioner net."""
    coup = [t for t in flow._transform._transforms
            if isinstance(t, transforms.PiecewiseRationalQuadraticCouplingTransform)]
    net = coup[0].transform_net
    return {m.out_features for m in net.modules() if isinstance(m, torch.nn.Linear)}


def _trainer(channels=_DM, flow_config=None):
    return CompressionTrainer(
        components=_C,
        station_locations=_locations(),
        channels=channels, latent_dim=channels, num_dims=6,
        architecture="seismogram_transformer",
        trace_length=_T,
        flow_config=flow_config,
    )


def test_default_flow_has_five_coupling_transforms():
    trainer = _trainer()
    assert _count_coupling_transforms(trainer.flow) == 5


def test_num_transforms_override_changes_flow_depth():
    trainer = _trainer(flow_config={"num_transforms": 8})
    assert _count_coupling_transforms(trainer.flow) == 8


def test_flow_hidden_width_defaults_to_channels():
    channels = 24
    trainer = _trainer(channels=channels, flow_config={"num_transforms": 8})
    widths = _conditioner_hidden_widths(trainer.flow)
    # The residual conditioner's hidden layers are sized to `channels` when not overridden.
    assert channels in widths


def test_flow_hidden_width_can_be_decoupled_from_channels():
    channels = 24
    trainer = _trainer(channels=channels,
                       flow_config={"num_transforms": 8, "hidden_features": 40})
    widths = _conditioner_hidden_widths(trainer.flow)
    assert 40 in widths
    assert channels not in widths  # decoupled: channel width no longer drives the head


def test_flow_config_round_trips_through_model_meta(tmp_path):
    """Saving + reloading via model_meta.json rebuilds the same-sized NDE head."""
    import json
    from seismo_sbi.sbi.compression.ML.train import CompressionTrainer as CT

    trainer = _trainer(flow_config={"num_transforms": 8})
    # Mimic train()'s sidecar dump (without running a full epoch).
    out = tmp_path / "run"
    (out / "checkpoints").mkdir(parents=True)
    meta = {
        "architecture": trainer.architecture,
        "model_config": trainer._model_config,
        "flow_config": trainer._flow_config,
        "trace_length": trainer.trace_length,
        "num_seismic_components": trainer.num_seismic_components,
        "num_dims": trainer.num_dims,
        "latent_dim": trainer.latent_dim,
        "feature_length": trainer._feature_length,
        "station_locations_shape": trainer._station_locations_shape,
        "station_locations": np.asarray(trainer._station_locations).tolist(),
    }
    (out / "model_meta.json").write_text(json.dumps(meta, default=str))

    # A fresh DEFAULT trainer (num_transforms=5) must rebuild to 8 from the sidecar.
    reloader = CT(components=_C, station_locations=_locations(),
                  channels=_DM, latent_dim=_DM, num_dims=6,
                  architecture="seismogram_transformer", trace_length=_T)
    assert _count_coupling_transforms(reloader.flow) == 5  # before reload
    # load_best needs a checkpoint; emulate just the flow-rebuild half it performs.
    with open(out / "model_meta.json") as f:
        loaded = json.load(f)
    reloader._flow_config = loaded["flow_config"]
    reloader.flow = CT._assemble_flow(
        architecture=loaded["architecture"],
        num_seismic_components=loaded["num_seismic_components"],
        model_config=loaded["model_config"],
        flow_config=loaded["flow_config"],
        feature_length=loaded["feature_length"],
        latent_dim=loaded["latent_dim"],
        num_dims=loaded["num_dims"],
        station_locations=np.asarray(loaded["station_locations"]),
        trace_length=loaded["trace_length"],
        device=reloader.device,
    )
    assert _count_coupling_transforms(reloader.flow) == 8  # after reload
