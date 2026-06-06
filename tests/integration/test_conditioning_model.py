"""Integration tests for source-location conditioning wired into the model.

Builds tiny SeismogramTransformers (and a full CompressionTrainer flow) and feeds
packed contexts to exercise each injection option. No Instaseis/CPS, no disk I/O.
"""

import numpy as np
import pytest
import torch

from seismo_sbi.sbi.compression.ML.seismogram_transformer import SeismogramTransformer
from seismo_sbi.sbi.compression.ML.source_conditioning import pack_context
from seismo_sbi.sbi.compression.ML.train import CompressionTrainer

pytestmark = pytest.mark.integration

# Tiny, CPU-fast dimensions. T must be long enough for the default CNN conv stack.
_C, _T, _N, _DM = 1, 200, 3, 16
_NCOND, _DCOND = 3, 8

INJECTIONS = ["relative_posemb", "token_add", "film", "concat_context"]


def _locations():
    return torch.tensor([[10.0, 20.0], [11.0, 21.0], [9.0, 19.0]], dtype=torch.float32)


def _base_config(**extra):
    cfg = {"channels": _DM, "nheads": 2, "layers": 1}
    cfg.update(extra)
    return cfg


def _build_model(conditioning=None):
    cfg = _base_config()
    if conditioning is not None:
        cfg["conditioning"] = conditioning
    return SeismogramTransformer(
        num_seismic_components=_C,
        transformer_config=cfg,
        feature_length=_DM,
        num_outputs=6,
        noise_model=None,
        seismogram_locations=_locations(),
        device=torch.device("cpu"),
        input_length=_T,
    )


def _packed_batch(B=4):
    seis = torch.randn(B, _N, _C, _T)
    source = torch.randn(B, _NCOND) * 10.0
    return pack_context(seis, source), seis, source


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def test_unconditioned_model_takes_4d_path():
    """No conditioning block ⇒ n_cond==0 and the 4-D embed path is used."""
    model = _build_model(conditioning=None)
    assert model._n_cond == 0
    x = torch.randn(4, _N, _C, _T)
    out = model.embed(x)
    assert out.shape == (4, _DM)
    assert torch.isfinite(out).all()


def test_unknown_injection_rejected():
    with pytest.raises(ValueError):
        _build_model(conditioning={"n_cond": _NCOND, "d_cond": _DCOND, "inject": ["bogus"]})


# ---------------------------------------------------------------------------
# Each injection option
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inject", INJECTIONS)
def test_single_injection_finite(inject):
    model = _build_model(conditioning={
        "n_cond": _NCOND, "d_cond": _DCOND, "coord_mode": "geographic", "inject": [inject],
    })
    model.eval()
    ctx, _, _ = _packed_batch()
    with torch.no_grad():
        out = model.embed(ctx)
    assert out.shape == (4, _DM)
    assert torch.isfinite(out).all()


def test_all_injections_combined_finite():
    model = _build_model(conditioning={
        "n_cond": _NCOND, "d_cond": _DCOND, "coord_mode": "geographic", "inject": INJECTIONS,
    })
    model.eval()
    ctx, _, _ = _packed_batch()
    with torch.no_grad():
        out = model.embed(ctx)
    assert out.shape == (4, _DM)
    assert torch.isfinite(out).all()


def test_cartesian_coord_mode_finite():
    model = _build_model(conditioning={
        "n_cond": _NCOND, "d_cond": _DCOND, "coord_mode": "cartesian",
        "inject": ["relative_posemb", "concat_context"],
    })
    model.eval()
    ctx, _, _ = _packed_batch()
    with torch.no_grad():
        out = model.embed(ctx)
    assert torch.isfinite(out).all()


def test_conditioning_actually_changes_output():
    """Different source locations must produce different embeddings (conditioning is wired)."""
    model = _build_model(conditioning={
        "n_cond": _NCOND, "d_cond": _DCOND, "inject": INJECTIONS,
    })
    # Push FiLM/token weights off their zero init so modulation is non-trivial.
    for p in model.parameters():
        if p.requires_grad and p.dim() >= 1:
            torch.nn.init.normal_(p, std=0.1)
    model.eval()
    seis = torch.randn(2, _N, _C, _T)
    src_a = torch.zeros(2, _NCOND)
    src_b = torch.ones(2, _NCOND) * 30.0
    with torch.no_grad():
        out_a = model.embed(pack_context(seis, src_a))
        out_b = model.embed(pack_context(seis, src_b))
    assert not torch.allclose(out_a, out_b, atol=1e-5)


# ---------------------------------------------------------------------------
# Full flow via CompressionTrainer
# ---------------------------------------------------------------------------

def test_compression_trainer_log_prob_on_packed_context():
    """The conditioned embedding net plugs into the NSF flow and yields a finite log-prob."""
    station_locations = _locations().numpy()
    trainer = CompressionTrainer(
        components="Z",
        station_locations=station_locations,
        channels=_DM, latent_dim=_DM, num_dims=6,
        architecture="seismogram_transformer",
        trace_length=_T,
        model_config={"conditioning": {
            "n_cond": _NCOND, "d_cond": _DCOND, "coord_mode": "geographic",
            "inject": ["token_add", "film", "relative_posemb", "concat_context"],
        }},
    )
    B = 5
    seis = torch.randn(B, _N, _C, _T)
    source = torch.randn(B, _NCOND) * 5.0
    ctx = pack_context(seis, source).to(trainer.device)
    theta = torch.randn(B, 6).to(trainer.device)
    trainer.model.to(trainer.device)   # Lightning moves the module in real training; do it here.
    trainer.model.eval()
    with torch.no_grad():
        log_prob = trainer.model(ctx, theta)
    assert log_prob.shape == (B,)
    assert torch.isfinite(log_prob).all()


# ---------------------------------------------------------------------------
# Data plumbing (3c): dataloader conditioning extraction + inference packing
# ---------------------------------------------------------------------------

def test_dataset_load_conditioning_extraction():
    """_load_conditioning concatenates the requested raw attrs in order."""
    from seismo_sbi.sbi.compression.ML.dataloading import TorchSimulationDataset

    class _StubLoader:
        def load_input_data(self, path):
            return {"source_location": {"latitude": 12.0, "longitude": -34.0,
                                        "depth": 7.0, "time_shift": 1.0}}

    ds = TorchSimulationDataset.__new__(TorchSimulationDataset)
    ds.data_loader = _StubLoader()
    ds.conditioning_param_map = {"source_location": ["latitude", "longitude", "depth"]}
    vec = ds._load_conditioning("dummy.h5")
    assert np.allclose(vec, [12.0, -34.0, 7.0])           # order preserved, time_shift excluded


def test_perturb_conditioning_applies_per_coordinate_gaussian():
    """v3 source-location uncertainty: _perturb_conditioning adds per-coordinate Gaussian noise
    (mean≈clean, std≈configured) and is a no-op when conditioning_noise_std is None."""
    import torch
    from seismo_sbi.sbi.compression.ML.dataloading import TorchSimulationDataset

    ds = TorchSimulationDataset.__new__(TorchSimulationDataset)
    base = torch.tensor([36.5, 25.6, 8.0])

    # No-op when unset.
    ds.conditioning_noise_std = None
    assert torch.equal(ds._perturb_conditioning(base), base)

    # Per-coordinate Gaussian with the configured std (lat°, lon°, depth km).
    std = torch.tensor([0.010, 0.013, 1.5])
    ds.conditioning_noise_std = std
    torch.manual_seed(0)
    samples = torch.stack([ds._perturb_conditioning(base) for _ in range(8000)])
    assert torch.allclose(samples.mean(0), base, atol=0.1)        # mean ≈ clean location
    assert torch.allclose(samples.std(0), std, rtol=0.12)          # std ≈ configured per axis
    # Fresh draw each call (not a fixed offset).
    assert not torch.equal(ds._perturb_conditioning(base), ds._perturb_conditioning(base))


# ---------------------------------------------------------------------------
# Amplitude embedding (opt-in)
# ---------------------------------------------------------------------------

def _build_amp_model(amplitude, conditioning=None):
    cfg = _base_config()
    cfg["amplitude_embedding"] = amplitude
    if conditioning is not None:
        cfg["conditioning"] = conditioning
    return SeismogramTransformer(
        num_seismic_components=_C,
        transformer_config=cfg,
        feature_length=_DM,
        num_outputs=6,
        noise_model=None,
        seismogram_locations=_locations(),
        device=torch.device("cpu"),
        input_length=_T,
    )


def test_no_amplitude_key_is_backward_compatible():
    """Absent config ⇒ no amplitude module ⇒ unchanged embed path."""
    assert _build_model().amplitude_embedding is None


@pytest.mark.parametrize("mode", ["array_relative", "absolute"])
def test_amplitude_model_finite_embedding(mode):
    model = _build_amp_model({"mode": mode}).eval()
    x = torch.randn(4, _N, _C, _T)
    with torch.no_grad():
        out = model.embed(x)
    assert out.shape == (4, _DM)
    assert torch.isfinite(out).all()


def test_amplitude_path_changes_pooled_embedding():
    """Ablation: switching the amplitude embedding off on the SAME model + input changes the
    pooled output ⇒ amplitude information demonstrably reaches the flow context."""
    model = _build_amp_model({"mode": "array_relative"}).eval()
    x = torch.randn(4, _N, _C, _T)
    with torch.no_grad():
        out_with = model.embed(x)
        saved, model.amplitude_embedding = model.amplitude_embedding, None
        out_without = model.embed(x)
        model.amplitude_embedding = saved
    assert not torch.allclose(out_with, out_without, atol=1e-5)


def test_amplitude_model_sensitive_to_cross_station_ratios():
    """Changing inter-station amplitude ratios changes the embedding (the diagnostic signal)."""
    model = _build_amp_model({"mode": "array_relative"}).eval()
    x = torch.randn(4, _N, _C, _T)
    x2 = x.clone()
    x2[:, 0] *= 5.0       # boost station 0, suppress station 1 → new radiation-pattern ratios
    x2[:, 1] *= 0.2
    with torch.no_grad():
        out1, out2 = model.embed(x), model.embed(x2)
    assert not torch.allclose(out1, out2, atol=1e-5)


def test_amplitude_with_conditioning_finite():
    """Amplitude embedding coexists with source-location conditioning."""
    model = _build_amp_model(
        {"mode": "array_relative"},
        conditioning={"n_cond": _NCOND, "d_cond": _DCOND, "inject": ["token_add", "film"]},
    ).eval()
    ctx, _, _ = _packed_batch()
    with torch.no_grad():
        out = model.embed(ctx)
    assert out.shape == (4, _DM)
    assert torch.isfinite(out).all()


def test_distance_correction_requires_conditioning():
    """distance_correction needs a source location ⇒ building it without conditioning errors."""
    with pytest.raises(ValueError, match="distance_correction needs a source location"):
        _build_amp_model({"mode": "array_relative", "distance_correction": True})


def test_amplitude_distance_correction_with_conditioning_finite():
    """Distance-corrected amplitude embedding runs under source-location conditioning, where the
    per-station epicentral distance is available."""
    model = _build_amp_model(
        {"mode": "array_relative", "distance_correction": True},
        conditioning={"n_cond": _NCOND, "d_cond": _DCOND, "coord_mode": "geographic",
                      "inject": ["relative_posemb"]},
    ).eval()
    ctx, _, _ = _packed_batch()
    with torch.no_grad():
        out = model.embed(ctx)
    assert out.shape == (4, _DM)
    assert torch.isfinite(out).all()


def test_amplitude_snr_weighting_finite():
    """SNR-weighted reference is self-contained (no conditioning needed) and stays finite."""
    model = _build_amp_model({"mode": "array_relative", "snr_weighting": True}).eval()
    x = torch.randn(4, _N, _C, _T)
    with torch.no_grad():
        out = model.embed(x)
    assert out.shape == (4, _DM)
    assert torch.isfinite(out).all()


def test_inference_path_packs_source_location():
    """MachineLearningCompressor packs a known source location into the model input."""
    from seismo_sbi.sbi.compression.gaussian import MachineLearningCompressor

    model = _build_model(conditioning={
        "n_cond": _NCOND, "d_cond": _DCOND, "inject": ["token_add", "concat_context"],
    })
    model.eval()

    captured = {}
    orig_forward = model.forward

    def spy_forward(inp):
        captured["shape"] = tuple(inp.shape)
        return orig_forward(inp)
    model.forward = spy_forward

    class _IdentityScaler:
        def inverse_transform(self, x):
            return x

    comp = MachineLearningCompressor.__new__(MachineLearningCompressor)
    comp.trained_ml_compressor = model
    comp.seismogram_preprocessor = lambda D: D            # already a tensor (N, C, T)
    comp.scaler = _IdentityScaler()
    comp.source_location = torch.tensor([10.0, 20.0, 5.0])

    D = torch.randn(_N, _C, _T)
    out = comp.compress_data_vector(D)
    # Model received a packed 2-D context of width N*C*T + n_cond.
    assert captured["shape"] == (1, _N * _C * _T + _NCOND)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# RFF station positional encoding (opt-in; review §3.2)
# ---------------------------------------------------------------------------

def _build_pe_model(positional_encoding=None, conditioning=None, **cfg_extra):
    cfg = _base_config(**cfg_extra)
    if positional_encoding is not None:
        cfg["positional_encoding"] = positional_encoding
    if conditioning is not None:
        cfg["conditioning"] = conditioning
    return SeismogramTransformer(
        num_seismic_components=_C,
        transformer_config=cfg,
        feature_length=_DM,
        num_outputs=6,
        noise_model=None,
        seismogram_locations=_locations(),
        device=torch.device("cpu"),
        input_length=_T,
    )


def test_no_posenc_key_is_backward_compatible():
    """Absent config ⇒ no RFF posenc module ⇒ legacy sinusoid path."""
    assert _build_model().all_station_transformer.station_posenc is None


def test_sinusoidal_mode_is_legacy():
    assert _build_pe_model({"mode": "sinusoidal"}).all_station_transformer.station_posenc is None


def test_fourier_absolute_fixed_station_finite():
    """Unconditioned model + fourier posenc (absolute coords, no depth) ⇒ finite embedding."""
    model = _build_pe_model({"mode": "fourier"}).eval()
    sp = model.all_station_transformer.station_posenc
    assert sp is not None and sp.coords_kind == "absolute" and sp.in_dim == 2
    x = torch.randn(4, _N, _C, _T)
    with torch.no_grad():
        out = model.embed(x)
    assert out.shape == (4, _DM) and torch.isfinite(out).all()


def test_fourier_relative_depth_finite_and_coords_kind():
    """Primary path: conditioned relative geometry + depth, two layers, every-layer injection."""
    model = _build_pe_model(
        {"mode": "fourier", "include_depth": True},
        conditioning={"n_cond": _NCOND, "d_cond": _DCOND, "coord_mode": "geographic",
                      "inject": ["relative_posemb"]},
        layers=2,
    ).eval()
    sp = model.all_station_transformer.station_posenc
    assert sp is not None and sp.coords_kind == "relative" and sp.include_depth and sp.in_dim == 4
    ctx, _, _ = _packed_batch()
    with torch.no_grad():
        out = model.embed(ctx)
    assert out.shape == (4, _DM) and torch.isfinite(out).all()


def test_fourier_posenc_is_active_vs_legacy():
    """Ablation: nulling the RFF posenc (⇒ legacy sinusoid) on the SAME model + input changes
    the pooled embedding, proving the RFF positional path is wired and reaches the flow context."""
    model = _build_pe_model(
        {"mode": "fourier", "include_depth": True},
        conditioning={"n_cond": _NCOND, "d_cond": _DCOND, "inject": ["relative_posemb"]},
        layers=2,
    ).eval()
    ctx, _, _ = _packed_batch()
    with torch.no_grad():
        out_fourier = model.embed(ctx)
        saved = model.all_station_transformer.station_posenc
        model.all_station_transformer.station_posenc = None
        out_legacy = model.embed(ctx)
        model.all_station_transformer.station_posenc = saved
    assert not torch.allclose(out_fourier, out_legacy, atol=1e-5)


def test_fourier_posenc_sensitive_to_source_location():
    """Different source locations ⇒ different source-relative geometry ⇒ different embedding."""
    model = _build_pe_model(
        {"mode": "fourier", "include_depth": True},
        conditioning={"n_cond": _NCOND, "d_cond": _DCOND, "coord_mode": "geographic",
                      "inject": ["relative_posemb"]},
    ).eval()
    seis = torch.randn(2, _N, _C, _T)
    src_a = torch.tensor([[10.0, 20.0, 5.0], [10.0, 20.0, 5.0]])
    src_b = torch.tensor([[40.0, 50.0, 5.0], [40.0, 50.0, 5.0]])
    with torch.no_grad():
        oa = model.embed(pack_context(seis, src_a))
        ob = model.embed(pack_context(seis, src_b))
    assert not torch.allclose(oa, ob, atol=1e-5)


def test_fourier_posenc_depth_reaches_model():
    """Changing ONLY source depth changes the embedding iff include_depth — proving the depth
    path. inject=['relative_posemb'] uses only lat/lon, so depth reaches the output SOLELY via
    the positional encoder; include_depth=False is the control where depth must be ignored."""
    cond = {"n_cond": _NCOND, "d_cond": _DCOND, "coord_mode": "geographic",
            "inject": ["relative_posemb"]}
    seis = torch.randn(2, _N, _C, _T)
    src_a = torch.tensor([[10.0, 20.0, 5.0], [10.0, 20.0, 5.0]])
    src_b = torch.tensor([[10.0, 20.0, 80.0], [10.0, 20.0, 80.0]])  # only depth differs

    with_depth = _build_pe_model({"mode": "fourier", "include_depth": True}, cond).eval()
    with torch.no_grad():
        oa = with_depth.embed(pack_context(seis, src_a))
        ob = with_depth.embed(pack_context(seis, src_b))
    assert not torch.allclose(oa, ob, atol=1e-5)            # depth reaches the model

    without_depth = _build_pe_model({"mode": "fourier", "include_depth": False}, cond).eval()
    with torch.no_grad():
        ca = without_depth.embed(pack_context(seis, src_a))
        cb = without_depth.embed(pack_context(seis, src_b))
    assert torch.allclose(ca, cb, atol=1e-5)                # control: depth ignored when off


@pytest.mark.parametrize("every", [True, False])
def test_inject_every_layer_finite(every):
    model = _build_pe_model(
        {"mode": "fourier", "include_depth": True, "inject_every_layer": every},
        conditioning={"n_cond": _NCOND, "d_cond": _DCOND, "inject": ["relative_posemb"]},
        layers=2,
    ).eval()
    assert model.all_station_transformer.inject_every_layer is every
    ctx, _, _ = _packed_batch()
    with torch.no_grad():
        out = model.embed(ctx)
    assert out.shape == (4, _DM) and torch.isfinite(out).all()


def test_include_depth_without_conditioning_raises():
    with pytest.raises(ValueError, match="include_depth needs a source depth"):
        _build_pe_model({"mode": "fourier", "include_depth": True})


def test_cls_token_with_fourier_raises():
    with pytest.raises(ValueError, match="use_cls_token"):
        _build_pe_model({"mode": "fourier"}, use_cls_token=True)


def test_invalid_posenc_mode_raises():
    with pytest.raises(ValueError, match="must be 'fourier' or 'sinusoidal'"):
        _build_pe_model({"mode": "bogus"})
