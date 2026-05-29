"""Unit tests for per-station encoders in the ML compression stack.

Verifies:
- Each registered encoder's output shape is (B*N, L, D).
- ``output_length`` and ``output_dim`` attributes match the actual forward-pass shape.
- The encoder registry contains the expected names.

These tests are dependency-free (no Instaseis/CPS, no large models).
"""

import pytest
import torch
from seismo_sbi.sbi.compression.ML.station_encoders import (
    PER_STATION_ENCODER_REGISTRY,
    build_station_encoder,
)

# (encoder_name, encoder_config, input_length)
ENCODER_CASES = [
    ("cnn", {}, 201),
    ("pno", {"width": 8, "modes": 4, "n_blocks": 2, "downsample": 4}, 201),
    ("tcn", {"channels": 8, "n_blocks": 2, "kernel_size": 3, "downsample": 4}, 201),
]

_BN = 6    # batch*stations
_C = 3     # seismic components (ZNE)
_D_MODEL = 16


def test_registry_contains_expected_encoders():
    assert "cnn" in PER_STATION_ENCODER_REGISTRY
    assert "pno" in PER_STATION_ENCODER_REGISTRY
    assert "tcn" in PER_STATION_ENCODER_REGISTRY


@pytest.mark.parametrize("enc_name,enc_cfg,T", ENCODER_CASES, ids=[c[0] for c in ENCODER_CASES])
def test_encoder_output_shape(enc_name, enc_cfg, T):
    """Encoder output must be (B*N, L, D) with L == output_length and D == output_dim."""
    encoder = build_station_encoder(
        enc_name,
        num_seismic_components=_C,
        input_length=T,
        d_model=_D_MODEL,
        **enc_cfg,
    )
    x = torch.randn(_BN, _C, T)
    with torch.no_grad():
        out = encoder(x)

    assert out.ndim == 3, f"{enc_name}: expected 3-D output, got shape {out.shape}"
    BN_out, L_out, D_out = out.shape

    assert BN_out == _BN, f"{enc_name}: batch dim mismatch: got {BN_out}, expected {_BN}"
    assert L_out == encoder.output_length, (
        f"{enc_name}: output_length attr ({encoder.output_length}) != actual L ({L_out})"
    )
    assert D_out == encoder.output_dim, (
        f"{enc_name}: output_dim attr ({encoder.output_dim}) != actual D ({D_out})"
    )


@pytest.mark.parametrize("enc_name,enc_cfg,T", ENCODER_CASES, ids=[c[0] for c in ENCODER_CASES])
def test_encoder_output_finite(enc_name, enc_cfg, T):
    """Encoder output must be finite for random input."""
    encoder = build_station_encoder(
        enc_name,
        num_seismic_components=_C,
        input_length=T,
        d_model=_D_MODEL,
        **enc_cfg,
    )
    x = torch.randn(_BN, _C, T)
    with torch.no_grad():
        out = encoder(x)
    assert torch.isfinite(out).all(), f"{enc_name}: non-finite values in encoder output"


@pytest.mark.parametrize("enc_name,enc_cfg,T", ENCODER_CASES, ids=[c[0] for c in ENCODER_CASES])
def test_encoder_handles_zero_input(enc_name, enc_cfg, T):
    """Encoder must not crash or produce NaN/inf on an all-zero trace (tests amplitude clip)."""
    encoder = build_station_encoder(
        enc_name,
        num_seismic_components=_C,
        input_length=T,
        d_model=_D_MODEL,
        **enc_cfg,
    )
    x = torch.zeros(_BN, _C, T)
    with torch.no_grad():
        out = encoder(x)
    assert torch.isfinite(out).all(), f"{enc_name}: non-finite output on zero-valued input"
