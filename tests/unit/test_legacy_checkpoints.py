"""Checkpoints from before the pluggable station encoder still load, weight for weight."""
from pathlib import Path

import numpy as np
import pytest
import torch

from seismo_sbi.sbi.compression.ML.legacy_checkpoints import (
    is_legacy_state_dict, remap_legacy_state_dict,
)

LV2_CHECKPOINTS = Path(__file__).resolve().parents[2] / "examples" / "ml-checkpoints"


def test_old_encoder_keys_are_renamed_and_the_unused_head_dropped():
    old = {
        "flow._embedding_net.CNN_feature_extractor.seismic_trace_CNN.conv_stack.0.weight": 1,
        "flow._embedding_net.CNN_feature_extractor.feedforward_net.layers.0.weight": 2,
        "flow._transform._transforms.0.weight": 3,
    }
    assert remap_legacy_state_dict(old) == {
        "flow._embedding_net.station_encoder._cnn.conv_stack.0.weight": 1,
        "flow._transform._transforms.0.weight": 3,
    }


def test_a_current_state_dict_passes_through_untouched():
    current = {"flow._embedding_net.station_encoder._cnn.conv_stack.0.weight": 1}
    assert not is_legacy_state_dict(current)
    assert remap_legacy_state_dict(current) is current


@pytest.mark.skipif(not (LV2_CHECKPOINTS / "checkpoints" / "best_model-LV2.ckpt").exists()
                    or (LV2_CHECKPOINTS / "checkpoints" / "best_model-LV2.ckpt").stat().st_size < 10_000,
                    reason="the LV2 checkpoint is not pulled from LFS")
def test_the_lv2_checkpoint_loads_strictly_into_the_current_model():
    from seismo_sbi.sbi.compression.ML.train import CompressionTrainer

    path = LV2_CHECKPOINTS / "checkpoints" / "best_model-LV2.ckpt"
    state = torch.load(path, map_location="cpu")["state_dict"]
    assert is_legacy_state_dict(state)
    coords = state["flow._embedding_net.all_station_transformer.station_coords"].numpy()
    trainer = CompressionTrainer(["Z", "E", "N"], coords.reshape(-1, 2), 256, 256)
    trainer.load_best(LV2_CHECKPOINTS)
    loaded = trainer.model.state_dict()
    for key, value in remap_legacy_state_dict(state).items():
        assert np.array_equal(loaded[key].cpu().numpy(), value.numpy()), key
