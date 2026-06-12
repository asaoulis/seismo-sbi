"""Unit tests for per-station encoders in the ML compression stack.

Verifies:
- Each registered encoder's output shape is (B*N, L, D).
- ``output_length`` and ``output_dim`` attributes match the actual forward-pass shape.
- The encoder registry contains the expected names.

These tests are dependency-free (no Instaseis/CPS, no large models).
"""

import math

import pytest
import torch
from seismo_sbi.sbi.compression.ML.station_encoders import (
    PER_STATION_ENCODER_REGISTRY,
    build_station_encoder,
    normalize_trace,
    log_amp_channel,
    station_amplitudes,
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


def test_input_decimator_shapes_and_band_limited_exactness():
    """InputDecimator: output length = ceil(T/factor); on a band-limited signal the
    anti-aliased decimation matches ideal striding in the interior (no information loss)."""
    from seismo_sbi.sbi.compression.ML.station_encoders import InputDecimator

    T, factor, C = 600, 3, 3
    dec = InputDecimator(factor, C, antialias=True)
    assert dec.output_length(T) == (T - 1) // factor + 1 == 200

    # Band-limited mix (0.03-0.08 Hz at 1 Hz sampling) — well below the post-decimation
    # Nyquist (0.5/3 ≈ 0.167 Hz), so striding is information-preserving.
    t = torch.arange(T, dtype=torch.float64)
    sig = sum(torch.sin(2 * math.pi * f * t + p) for f, p in [(0.03, 0.3), (0.05, 1.1), (0.08, 2.0)])
    x = sig.to(torch.float32).view(1, 1, 1, -1).repeat(2, 4, C, 1)   # (B,N,C,T)

    y = dec(x)
    assert y.shape == (2, 4, C, 200)
    ref = x[..., ::factor]
    interior = slice(5, -5)   # edges taper from the FIR; compare the interior
    rel = (y[..., interior] - ref[..., interior]).abs().max() / ref.abs().max()
    assert rel < 1e-2, f"band-limited anti-aliased decimation drifted from ideal stride: {rel:.2e}"


def test_input_decimator_flat_layouts_agree_and_finite():
    """4-D (B,N,C,T) and flat (B*N,C,T) inputs decimate identically; output is finite."""
    from seismo_sbi.sbi.compression.ML.station_encoders import InputDecimator

    B, N, C, T = 2, 3, 3, 201
    dec = InputDecimator(2, C, antialias=True)
    x4 = torch.randn(B, N, C, T)
    y4 = dec(x4)
    y_flat = dec(x4.reshape(B * N, C, T)).reshape(B, N, C, -1)
    assert torch.allclose(y4, y_flat, atol=1e-6)
    assert torch.isfinite(y4).all()
    assert y4.shape[-1] == dec.output_length(T)

    with pytest.raises(ValueError):
        InputDecimator(1, C)   # factor < 2 is rejected


def test_input_decimation_train_inference_symmetric():
    """A model built with ml_encoder input_decimate sizes its encoder to the decimated
    length and applies the SAME decimation in embed() on both the 4-D and packed paths,
    so train (packed) and inference (4-D) see the same model entry."""
    from seismo_sbi.sbi.compression.ML.seismogram_transformer import SeismogramTransformer

    T, N, B = 201, 2, 3
    model = SeismogramTransformer(
        num_seismic_components=1,
        transformer_config={
            "channels": _D_MODEL, "nheads": 2, "layers": 1,
            "station_encoder": "tcn",
            "encoder_config": {"channels": 8, "n_blocks": 2, "downsample": 4},
            "input_decimate": {"factor": 3, "antialias": True},
        },
        feature_length=_D_MODEL,
        num_outputs=6,
        noise_model=None,
        seismogram_locations=torch.tensor([[10.0, 20.0], [11.0, 21.0]]),
        device=torch.device("cpu"),
        input_length=T,
    )
    # The decimator is active and the encoder/token count was sized for the DECIMATED
    # length (T/3), not the raw T: a tcn downsample=4 over 67 samples gives fewer tokens
    # than over 201.
    assert model.input_decimator is not None and model.input_decimator.factor == 3
    dec_T = (T - 1) // 3 + 1
    undec = SeismogramTransformer(
        num_seismic_components=1,
        transformer_config={
            "channels": _D_MODEL, "nheads": 2, "layers": 1,
            "station_encoder": "tcn",
            "encoder_config": {"channels": 8, "n_blocks": 2, "downsample": 4},
        },
        feature_length=_D_MODEL, num_outputs=6, noise_model=None,
        seismogram_locations=torch.tensor([[10.0, 20.0], [11.0, 21.0]]),
        device=torch.device("cpu"), input_length=T,
    )
    assert model.L < undec.L, "decimation should reduce the transformer token count"
    model.eval()
    with torch.no_grad():
        out = model.embed(torch.randn(B, N, 1, T))   # 4-D inference path
    assert out.shape == (B, _D_MODEL)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("enc_name", ["cnn", "tcn", "pno"])
def test_encoder_amp_safe_bf16_autocast(enc_name):
    """Every encoder must produce a finite output under a bf16 autocast (the production AMP
    path). Regression: the pno SpectralConv1d FFT has no bf16 kernel and used to raise
    'Unsupported dtype BFloat16'; it now runs its FFT in fp32 inside the autocast region."""
    T = 201
    enc = build_station_encoder(enc_name, num_seismic_components=_C, input_length=T,
                                d_model=_D_MODEL, downsample=2)
    x = torch.randn(_BN, _C, T)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = enc(x)
    assert torch.isfinite(out.float()).all(), f"{enc_name}: non-finite under bf16 autocast"


@pytest.mark.parametrize("enc_name", ["cnn", "tcn", "pno"])
def test_encoder_uniform_downsample(enc_name):
    """All three encoders accept a uniform `downsample` (cnn gained it) and produce a token
    count of ceil(T/downsample) for cnn (same-padded) / T//downsample for tcn/pno — short-input
    safe (the model-entry Nyquist decimation feeds them a reduced T)."""
    T = 67   # e.g. 201 after input_decimate=3
    enc = build_station_encoder(enc_name, num_seismic_components=_C, input_length=T,
                                d_model=_D_MODEL, downsample=2)
    x = torch.randn(_BN, _C, T)
    with torch.no_grad():
        out = enc(x)
    assert out.shape[1] == enc.output_length
    assert 25 <= enc.output_length <= 40, f"{enc_name}: unexpected token count {enc.output_length}"
    assert torch.isfinite(out).all()


def test_transformer_handles_long_token_sequence():
    """Regression: an encoder emitting L > 60 tokens must not break the transformer's
    time-embedding add (max_time_steps used to default to 60). PNO with downsample=2 on a
    200-sample trace yields L=100, which pre-fix overflowed the fixed time_embed buffer.
    """
    from seismo_sbi.sbi.compression.ML.seismogram_transformer import SeismogramTransformer

    T, N, B = 200, 2, 3
    model = SeismogramTransformer(
        num_seismic_components=1,
        transformer_config={
            "channels": _D_MODEL, "nheads": 2, "layers": 1,
            "station_encoder": "pno",
            "encoder_config": {"width": 8, "modes": 4, "n_blocks": 1, "downsample": 2},
        },
        feature_length=_D_MODEL,
        num_outputs=6,
        noise_model=None,
        seismogram_locations=torch.tensor([[10.0, 20.0], [11.0, 21.0]]),
        device=torch.device("cpu"),
        input_length=T,
    )
    assert model.L > 60, f"expected long token sequence to exercise the fix, got L={model.L}"
    model.eval()
    with torch.no_grad():
        out = model.embed(torch.randn(B, N, 1, T))
    assert out.shape == (B, _D_MODEL)
    assert torch.isfinite(out).all()


def test_tcn_even_kernel_size_rejected():
    """DilatedTCNEncoder only preserves length for odd kernels; an even kernel must raise
    a clear error at construction rather than failing the residual add at forward time."""
    with pytest.raises(ValueError, match="odd"):
        build_station_encoder(
            "tcn", num_seismic_components=_C, input_length=201, d_model=_D_MODEL,
            kernel_size=4,
        )


# ---------------------------------------------------------------------------
# Shared amplitude primitives — numerics / behavioural contract
# ---------------------------------------------------------------------------

def test_normalize_trace_unit_peak_and_max_val():
    x = torch.randn(_BN, _C, 50) * 7.0
    x_norm, max_val = normalize_trace(x)
    assert max_val.shape == (_BN, 1, 1)
    # Each trace's max-abs over (C, T) is 1 after normalisation.
    peaks = x_norm.abs().amax(dim=(1, 2))
    assert torch.allclose(peaks, torch.ones(_BN), atol=1e-5)
    # max_val matches the raw peak.
    assert torch.allclose(max_val.squeeze(), x.abs().amax(dim=(1, 2)), atol=1e-5)


def test_log_amp_channel_broadcasts_log_of_max():
    max_val = torch.tensor([[[3.0]], [[10.0]]])     # (2,1,1)
    ch = log_amp_channel(max_val, length=4)
    assert ch.shape == (2, 1, 4)
    assert torch.allclose(ch[0], torch.full((1, 4), math.log(3.0)))
    assert torch.allclose(ch[1], torch.full((1, 4), math.log(10.0)))


def test_station_amplitudes_per_station_and_per_component():
    # (B=1, N=2, C=2, T=3) with known peaks.
    x = torch.tensor([[
        [[1.0, -2.0, 0.5], [0.1, 0.2, 0.3]],     # station 0: comp maxima 2.0, 0.3
        [[-4.0, 1.0, 2.0], [5.0, -1.0, 0.0]],    # station 1: comp maxima 4.0, 5.0
    ]])
    per_station = station_amplitudes(x, per_component=False)
    assert per_station.shape == (1, 2, 1)
    assert torch.allclose(per_station[0, 0, 0], torch.tensor(math.log(2.0)), atol=1e-6)
    assert torch.allclose(per_station[0, 1, 0], torch.tensor(math.log(5.0)), atol=1e-6)

    per_comp = station_amplitudes(x, per_component=True)
    assert per_comp.shape == (1, 2, 2)
    expected = torch.log(torch.tensor([[[2.0, 0.3], [4.0, 5.0]]]))
    assert torch.allclose(per_comp, expected, atol=1e-6)


def test_cnn_amplitude_channel_equals_log_max():
    """The CNN's LAST output channel must be log(max|x|) broadcast over time."""
    from seismo_sbi.sbi.compression.ML.cnn_feature_extractor import SeismicTraceCNN
    cnn = SeismicTraceCNN(_C, input_length=201, final_layer=_D_MODEL).eval()
    x = torch.randn(_BN, _C, 201) * 3.0
    with torch.no_grad():
        out = cnn(x)                              # (B, D, L)
    expected_log = x.abs().amax(dim=(1, 2)).clamp_min(1e-12).log()   # (B,)
    amp_channel = out[:, -1, :]                   # (B, L)
    assert torch.allclose(amp_channel, expected_log[:, None].expand_as(amp_channel), atol=1e-5)


def test_cnn_conv_body_scale_invariant_amp_channel_tracks_scale():
    """Scaling the input by c>0 leaves the conv body unchanged (max-abs normalisation) and
    shifts only the log-amplitude channel by log(c) — the contract the new amplitude path
    relies on (scale lives in the amplitude feature, shape in the body)."""
    from seismo_sbi.sbi.compression.ML.cnn_feature_extractor import SeismicTraceCNN
    cnn = SeismicTraceCNN(_C, input_length=201, final_layer=_D_MODEL).eval()
    x = torch.randn(_BN, _C, 201)
    c = 10.0
    with torch.no_grad():
        out1 = cnn(x)
        out2 = cnn(c * x)
    # Conv body (all but the appended amplitude channel) is identical.
    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)
    # Amplitude channel differs by exactly log(c).
    delta = (out2[:, -1, :] - out1[:, -1, :])
    assert torch.allclose(delta, torch.full_like(delta, math.log(c)), atol=1e-5)
