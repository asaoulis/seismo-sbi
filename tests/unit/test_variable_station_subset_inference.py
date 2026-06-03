"""Unit tests for variable-station SUBSET inference primitives.

These exercise the pieces that let a model trained with ``variable_stations=True`` be
applied to an arbitrary subset of its master station set:

* ``pack_subset_observation`` — packs one (subset) sample into the 2-D context the model
  expects, round-tripping through ``unpack_variable_context``.
* ``SeismogramTransformer.forward`` on a packed subset context (N' < N_master) — finite,
  correctly-shaped output.
* ``SimulationDataLoader.load_event_subset`` — name-keyed subset slicing of an event H5.
"""
import numpy as np
import h5py
import pytest
import torch

from seismo_sbi.sbi.compression.ML.source_conditioning import (
    pack_subset_observation,
    unpack_variable_context,
)


def test_pack_subset_observation_roundtrip_unconditioned():
    N, C, T = 3, 3, 8
    stacked = np.random.randn(N, C, T).astype(np.float32)
    coords = np.array([[45.0, 16.0], [46.0, 17.0], [44.0, 15.0]], dtype=np.float32)

    ctx = pack_subset_observation(stacked, coords)  # (1, W)
    assert ctx.dim() == 2 and ctx.shape[0] == 1

    seis, out_coords, mask, source_vec = unpack_variable_context(
        ctx, n_components=C, trace_length=T, n_cond=0
    )
    assert seis.shape == (1, N, C, T)
    assert torch.allclose(seis[0], torch.as_tensor(stacked))
    assert torch.allclose(out_coords[0], torch.as_tensor(coords))
    assert mask.shape == (1, N) and bool(mask.all())
    assert source_vec is None


def test_pack_subset_observation_mask_and_source():
    N, C, T, n_cond = 2, 3, 8, 3
    stacked = np.random.randn(N, C, T).astype(np.float32)
    coords = np.array([[45.0, 16.0], [46.0, 17.0]], dtype=np.float32)
    present = [True, False]
    src = np.array([45.8, 16.0, 5.0], dtype=np.float32)

    ctx = pack_subset_observation(stacked, coords, present_mask=present, source_vec=src)
    seis, out_coords, mask, source_vec = unpack_variable_context(
        ctx, n_components=C, trace_length=T, n_cond=n_cond
    )
    assert mask[0].tolist() == [True, False]
    assert source_vec is not None and torch.allclose(source_vec[0], torch.as_tensor(src))


def test_pack_subset_observation_validates_shapes():
    with pytest.raises(ValueError):
        pack_subset_observation(np.zeros((3, 8)), np.zeros((3, 2)))          # not (N,C,T)
    with pytest.raises(ValueError):
        pack_subset_observation(np.zeros((3, 3, 8)), np.zeros((2, 2)))       # coords mismatch
    with pytest.raises(ValueError):
        pack_subset_observation(np.zeros((3, 3, 8)), np.zeros((3, 2)),
                                present_mask=[True, False])                  # mask length


def _build_variable_station_net(n_master=4, C=3, T=16, latent=8):
    from seismo_sbi.sbi.compression.ML.train import (
        _build_seismogram_transformer, DEFAULT_MODEL_CONFIG,
    )
    station_locations = torch.tensor(
        [[45.0, 16.0], [46.0, 17.0], [44.0, 15.0], [43.0, 18.0]][:n_master],
        dtype=torch.float32,
    )
    model_config = {
        **DEFAULT_MODEL_CONFIG,
        "channels": 8, "nheads": 2, "layers": 2,
        "station_encoder": "tcn",
        "encoder_config": {"channels": 8, "n_blocks": 2, "kernel_size": 3, "downsample": 4},
        "variable_stations": True,
        "station_coords_mode": "absolute",
    }
    net = _build_seismogram_transformer(
        num_seismic_components=C, model_config=model_config,
        feature_length=latent, latent_dim=latent,
        station_locations=station_locations, device="cpu", trace_length=T,
    )
    net.eval()
    return net, station_locations, C, T, latent


def test_variable_station_net_accepts_subset_context():
    """A variable-station model produces a finite, correctly-shaped output for a station
    SUBSET (N' < N_master) packed via pack_subset_observation."""
    net, station_locations, C, T, latent = _build_variable_station_net()

    # subset: keep stations 0 and 2 of the 4-station master set
    keep = [0, 2]
    stacked = np.random.randn(len(keep), C, T).astype(np.float32)
    coords = station_locations.numpy()[keep]
    ctx = pack_subset_observation(stacked, coords)

    with torch.no_grad():
        out = net.forward(ctx)
    assert out.shape == (1, latent)
    assert torch.isfinite(out).all()

    # full master set via the 4-D convenience path should also work and differ from subset
    full = torch.as_tensor(np.random.randn(1, 4, C, T), dtype=torch.float32)
    with torch.no_grad():
        out_full = net.forward(full)
    assert out_full.shape == (1, latent) and torch.isfinite(out_full).all()


def test_load_event_subset(tmp_path):
    from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader
    from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers

    T = 10
    recs = Receivers(receivers=[
        Receiver(45.0, 16.0, "CR", "AAA", ["Z", "E", "N"], 0),
        Receiver(46.0, 17.0, "CR", "BBB", ["Z", "E", "N"], 0),
        Receiver(44.0, 15.0, "CR", "CCC", ["Z", "E", "N"], 0),
    ])
    loader = SimulationDataLoader("ZEN", recs)

    h5_path = tmp_path / "event.h5"
    with h5py.File(h5_path, "w") as f:
        out = f.create_group("outputs")
        for s, base in (("AAA", 0.0), ("BBB", 100.0), ("CCC", 200.0)):
            g = out.create_group(s)
            for i, comp in enumerate(("Z", "1", "2")):   # on-disk channel keys are Z/1/2
                g.create_dataset(comp, data=base + i + np.arange(T, dtype=float))

    data, coords = loader.load_event_subset(str(h5_path), ["BBB", "AAA"], stacked=True)
    assert data.shape == (2, 3, T)
    # coords ordered to match the requested station order
    assert np.allclose(coords, np.array([[46.0, 17.0], [45.0, 16.0]]))
    # BBB row should carry the base=100 traces, confirming name-keyed selection + order
    assert np.isclose(data[0, 0, 0], 100.0)
    assert np.isclose(data[1, 0, 0], 0.0)
    # master receiver set is restored after the call
    assert [r.station_name for r in loader.receivers.iterate()] == ["AAA", "BBB", "CCC"]

    with pytest.raises(KeyError):
        loader.load_event_subset(str(h5_path), ["ZZZ"], stacked=True)


# --------------------------------------------------------------------------- #
# Conditioned-model inference path: sample_station_dropout_ensemble must forward a
# per-event source_vec into the packed context (the post-train eval threading).
# --------------------------------------------------------------------------- #
class _CapturePosterior:
    """Stub posterior: records the last context handed to .sample, returns zeros."""
    def __init__(self):
        self.contexts = []

    def sample(self, shape, x, show_progress_bars=False):
        self.contexts.append(torch.as_tensor(x).detach().cpu())
        return torch.zeros((shape[0], 6))


class _IdentityScaler:
    def inverse_transform(self, s):
        return np.asarray(s)


def test_dropout_ensemble_threads_source_vec():
    """A conditioned model's per-event source_vec is packed into every station config's
    inference context (and round-trips out via unpack_variable_context)."""
    from seismo_sbi.sbi.compression.ML.station_dropout import (
        config_from_kept, sample_station_dropout_ensemble,
    )

    names = ["AAA", "BBB", "CCC"]
    N, C, T, n_cond = 3, 3, 8, 3
    obs = np.random.randn(N, C, T).astype(np.float32)
    coords = np.array([[45.0, 16.0], [46.0, 17.0], [44.0, 15.0]], dtype=np.float32)
    src = np.array([45.5, 16.5, 7.0], dtype=np.float32)

    posterior = _CapturePosterior()
    configs = [config_from_kept(names, names, "all"),
               config_from_kept(names, ["AAA", "CCC"], "subset")]
    ensemble, _results = sample_station_dropout_ensemble(
        posterior, obs, coords, configs, _IdentityScaler(),
        num_samples=4, device="cpu", event_name="EV", source_vec=src)

    assert set(ensemble.keys()) == {"all", "subset"}
    assert len(posterior.contexts) == 2
    # The second config keeps 2 stations; its context must carry the SAME source vector.
    seis, _coords, _mask, source_vec = unpack_variable_context(
        posterior.contexts[1], n_components=C, trace_length=T, n_cond=n_cond)
    assert seis.shape[1] == 2
    assert source_vec is not None
    assert torch.allclose(source_vec[0], torch.as_tensor(src))


def test_dropout_ensemble_unconditioned_has_no_source_vec():
    """With source_vec=None (unconditioned model) the packed context carries no source vector."""
    from seismo_sbi.sbi.compression.ML.station_dropout import (
        config_from_kept, sample_station_dropout_ensemble,
    )

    names = ["AAA", "BBB"]
    N, C, T = 2, 3, 8
    obs = np.random.randn(N, C, T).astype(np.float32)
    coords = np.array([[45.0, 16.0], [46.0, 17.0]], dtype=np.float32)

    posterior = _CapturePosterior()
    configs = [config_from_kept(names, names, "all")]
    sample_station_dropout_ensemble(
        posterior, obs, coords, configs, _IdentityScaler(),
        num_samples=2, device="cpu")  # source_vec defaults to None

    _seis, _coords, _mask, source_vec = unpack_variable_context(
        posterior.contexts[0], n_components=C, trace_length=T, n_cond=0)
    assert source_vec is None
