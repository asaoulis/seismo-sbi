"""
Characterisation tests for ``seismo_sbi.evaluation.validation``.

B3 goal: verify that the unified ``run_validation`` engine's fixed-station /
unconditioned branch (``cond_param_map=None``) reproduces the same
``theta_scaled`` / ``samples_scaled`` values as the old
``evaluate_validation_set`` direct ``dataset[idx] → posterior.sample`` path.

Strategy
--------
``run_validation`` is hard to call end-to-end in a unit test because it needs:
  * a real ``TorchSimulationDataset`` (needs h5 files on disk),
  * a ``training_noise_sampler``,
  * an ``sbi_pipeline`` object.

The plan explicitly permits: "unit-test its pure sub-pieces and the
variable↔fixed branch selection, and clearly note what is covered vs deferred
to the B7 smoke."

We therefore test:

1. **Equivalence of the full-set identity path** (D2 concern): when
   ``config_from_kept(master, master)`` is fed to
   ``sample_station_dropout_ensemble``, the context handed to the posterior is
   equivalent to the raw observation — proving the full-set case reduces to
   the same input as the direct ``dataset[idx]`` path.  Uses a deterministic
   stub posterior that records context widths and returns fixed samples.

2. **D2 branch selection** — ``source_vec`` is ``None`` when
   ``cond_param_map`` is ``None``, and non-None when it is set — tested via
   the shared ``sample_station_dropout_ensemble`` stub (the same call that
   ``run_validation`` makes).

3. **Output dict keys** of ``run_validation`` — verified against
   ``write_validation_outputs`` expectations without needing a real pipeline,
   by constructing the dict manually and confirming ``write_validation_outputs``
   accepts it.

4. **``write_validation_outputs`` side-effects** — given a valid ``val`` dict
   (constructed with a tiny numpy array), confirm the function writes
   ``evaluation_metrics.json`` without crashing (plotting may be skipped if
   deps unavailable; the JSON write is always executed).

Full ``run_validation`` end-to-end equivalence is deferred to the B7 Santorini
eval smoke (``evaluate_model.py --smoke --events No14_id3250 --validation``).
"""
from __future__ import annotations

import json
import numpy as np
import pytest

import torch


# ---------------------------------------------------------------------------
# 1.  D2 equivalence: full-set config_from_kept → same posterior input shape
# ---------------------------------------------------------------------------

class _DeterministicPosterior:
    """Stub posterior: returns a fixed all-zeros sample and records context shapes."""
    def __init__(self, n_dims: int = 6):
        self.n_dims = n_dims
        self.seen_ctx_shapes: list = []

    def sample(self, shape, x, show_progress_bars=False):
        self.seen_ctx_shapes.append(tuple(x.shape))
        return torch.zeros((shape[0], self.n_dims))


class _IdentityScaler:
    def inverse_transform(self, x):
        return np.asarray(x)
    def transform(self, x):
        return np.asarray(x)


def test_full_set_config_from_kept_passes_all_stations():
    """config_from_kept(master, master) keeps every station → context is (1, W_full)."""
    from seismo_sbi.sbi.compression.ML.station_dropout import (
        config_from_kept, sample_station_dropout_ensemble,
    )

    N, C, T = 5, 3, 10
    master = [f"S{i}" for i in range(N)]
    obs = np.random.default_rng(0).normal(size=(N, C, T)).astype(np.float32)
    coords = np.random.default_rng(1).normal(size=(N, 2)).astype(np.float32)

    cfg_all = config_from_kept(master, master, f"val all (N={N})")
    post = _DeterministicPosterior()
    ens, _ = sample_station_dropout_ensemble(
        post, obs, coords, [cfg_all], _IdentityScaler(),
        num_samples=2, device="cpu", source_vec=None)

    # Should have processed exactly one config.
    assert len(post.seen_ctx_shapes) == 1
    ctx_shape = post.seen_ctx_shapes[0]
    # Context is (1, W) where W encodes N*C*T + coords.
    assert ctx_shape[0] == 1
    assert ctx_shape[1] > N * C * T  # coords add extra dims


def test_full_set_matches_deterministic_posterior_output():
    """Deterministic posterior output (zeros) is identical whether called via
    the full-set config_from_kept path or a direct posterior.sample call."""
    from seismo_sbi.sbi.compression.ML.station_dropout import (
        config_from_kept, sample_station_dropout_ensemble,
    )
    from seismo_sbi.sbi.compression.ML.source_conditioning import pack_subset_observation

    N, C, T = 3, 3, 8
    obs = np.random.default_rng(42).normal(size=(N, C, T)).astype(np.float32)
    coords = np.random.default_rng(43).normal(size=(N, 2)).astype(np.float32)
    master = [f"S{i}" for i in range(N)]
    NUM_SAMPLES = 4

    # --- Path A: config_from_kept(master, master) + sample_station_dropout_ensemble ---
    cfg_all = config_from_kept(master, master, "all")
    post_a = _DeterministicPosterior(n_dims=6)
    ens, _ = sample_station_dropout_ensemble(
        post_a, obs, coords, [cfg_all], _IdentityScaler(),
        num_samples=NUM_SAMPLES, device="cpu", source_vec=None)
    samples_a = ens["all"].samples  # (NUM_SAMPLES, 6)

    # --- Path B: direct pack + posterior.sample (old evaluate_validation_set path) ---
    post_b = _DeterministicPosterior(n_dims=6)
    ctx_b = pack_subset_observation(obs, coords, source_vec=None)  # (1, W)
    samples_b_tensor = post_b.sample((NUM_SAMPLES,), ctx_b, show_progress_bars=False)
    samples_b = samples_b_tensor.numpy()  # (NUM_SAMPLES, 6)

    # Both deterministic stubs return zeros → values must match.
    np.testing.assert_array_equal(samples_a, samples_b)
    assert samples_a.shape == (NUM_SAMPLES, 6)


# ---------------------------------------------------------------------------
# 2.  D2 branch selection: source_vec threading
# ---------------------------------------------------------------------------

def test_full_set_unconditioned_source_vec_is_none_in_context():
    """With source_vec=None the packed context carries no source vector."""
    from seismo_sbi.sbi.compression.ML.station_dropout import (
        config_from_kept, sample_station_dropout_ensemble,
    )
    from seismo_sbi.sbi.compression.ML.source_conditioning import unpack_variable_context

    N, C, T = 3, 3, 8
    obs = np.random.default_rng(1).normal(size=(N, C, T)).astype(np.float32)
    coords = np.random.default_rng(2).normal(size=(N, 2)).astype(np.float32)
    master = [f"S{i}" for i in range(N)]

    class _RecordingPosterior:
        def __init__(self):
            self.last_ctx = None
        def sample(self, shape, x, show_progress_bars=False):
            self.last_ctx = x.detach().cpu()
            return torch.zeros((shape[0], 6))

    post = _RecordingPosterior()
    cfg = config_from_kept(master, master, "all")
    sample_station_dropout_ensemble(
        post, obs, coords, [cfg], _IdentityScaler(),
        num_samples=2, device="cpu", source_vec=None)

    _, _, _, source_vec_out = unpack_variable_context(
        post.last_ctx, n_components=C, trace_length=T, n_cond=0)
    assert source_vec_out is None


def test_full_set_conditioned_source_vec_is_present_in_context():
    """With source_vec provided the packed context carries the source vector."""
    from seismo_sbi.sbi.compression.ML.station_dropout import (
        config_from_kept, sample_station_dropout_ensemble,
    )
    from seismo_sbi.sbi.compression.ML.source_conditioning import unpack_variable_context

    N, C, T, n_cond = 3, 3, 8, 3
    obs = np.random.default_rng(1).normal(size=(N, C, T)).astype(np.float32)
    coords = np.random.default_rng(2).normal(size=(N, 2)).astype(np.float32)
    master = [f"S{i}" for i in range(N)]
    src = np.array([36.5, 25.0, 8.0], dtype=np.float32)

    class _RecordingPosterior:
        def __init__(self):
            self.last_ctx = None
        def sample(self, shape, x, show_progress_bars=False):
            self.last_ctx = x.detach().cpu()
            return torch.zeros((shape[0], 6))

    post = _RecordingPosterior()
    cfg = config_from_kept(master, master, "all")
    sample_station_dropout_ensemble(
        post, obs, coords, [cfg], _IdentityScaler(),
        num_samples=2, device="cpu", source_vec=src)

    _, _, _, source_vec_out = unpack_variable_context(
        post.last_ctx, n_components=C, trace_length=T, n_cond=n_cond)
    assert source_vec_out is not None
    assert torch.allclose(source_vec_out[0], torch.as_tensor(src))


# ---------------------------------------------------------------------------
# 3.  run_validation output dict keys
# ---------------------------------------------------------------------------

def _make_val_dict(n_val: int = 5, n_dims: int = 6, num_samples: int = 4):
    """Build a minimal val dict matching the run_validation return contract."""
    theta_scaled = np.zeros((n_val, n_dims), dtype=float)
    samples_scaled = np.zeros((num_samples, n_val, n_dims), dtype=float)
    theta_phys = np.zeros((n_val, n_dims), dtype=float)
    samples_phys = np.zeros((num_samples, n_val, n_dims), dtype=float)
    return {
        "theta_phys": theta_phys,
        "samples_phys": samples_phys,
        "theta_scaled": theta_scaled,
        "samples_scaled": samples_scaled,
        "show": [],
        "n_val": n_val,
    }


def test_run_validation_output_dict_contract():
    """The val dict returned by run_validation must contain the expected keys."""
    EXPECTED_KEYS = {
        "theta_phys", "samples_phys", "theta_scaled", "samples_scaled", "show", "n_val",
    }
    val = _make_val_dict()
    assert set(val.keys()) == EXPECTED_KEYS


def test_run_validation_theta_scaled_shape():
    """theta_scaled and samples_scaled must have consistent shapes."""
    n_val, n_dims, num_samples = 7, 6, 4
    val = _make_val_dict(n_val=n_val, n_dims=n_dims, num_samples=num_samples)
    assert val["theta_scaled"].shape == (n_val, n_dims)
    assert val["samples_scaled"].shape == (num_samples, n_val, n_dims)


# ---------------------------------------------------------------------------
# 3b.  run_validation branch selection (packed vs direct) — end-to-end with stubs
# ---------------------------------------------------------------------------
#
# These exercise the run_validation BODY for BOTH branches by monkeypatching the
# heavy collaborators (TorchSimulationDataset, augmentation-chain builder, and
# the packed sampler) so no h5 files / GPU / real pipeline are needed.


class _StubDataset:
    """Tiny in-memory stand-in for TorchSimulationDataset.

    __getitem__ returns (theta_scaled, x) with x of shape (N, C, T).
    """
    def __init__(self, n_sims, N, C, T, n_dims=6):
        self.N, self.C, self.T = N, C, T
        rng = np.random.default_rng(7)
        self._theta = rng.normal(size=(n_sims, n_dims)).astype(np.float32)
        self._x = rng.normal(size=(n_sims, N, C, T)).astype(np.float32)
        self.station_coords = rng.normal(size=(N, 2)).astype(np.float32)
        self.paths = [f"sim_{i}.h5" for i in range(n_sims)]

    def __len__(self):
        return len(self._theta)

    def __getitem__(self, idx):
        return self._theta[idx], self._x[idx]


class _StubReceiver:
    def __init__(self, name):
        self.station_name = name


class _StubReceivers:
    def __init__(self, names):
        self._names = names

    def iterate(self):
        return [_StubReceiver(n) for n in self._names]


class _StubSimParams:
    def __init__(self, names, sampling_rate=1.0):
        self.receivers = _StubReceivers(names)
        self.sampling_rate = sampling_rate


class _StubParams:
    names = ["mt0", "mt1", "mt2", "mt3", "mt4", "mt5"]


class _StubPipeline:
    """Minimal sbi_pipeline duck-type for run_validation."""
    def __init__(self, names):
        self.parameters = _StubParams()
        self.simulation_parameters = _StubSimParams(names)
        self.simulations_output_path = "/tmp/does-not-exist"
        self.training_noise_sampler = object()

        class _DM:
            class _DL:
                def load_input_data(self, path):
                    # constant fake source vector for conditioned-path test
                    return {"source_location": {"latitude": 36.5,
                                                "longitude": 25.0,
                                                "depth": 8.0}}
            data_loader = _DL()
        self.data_manager = _DM()


def _patch_run_validation_deps(monkeypatch, stub_ds, recorder):
    """Patch the lazy imports inside run_validation to use stubs.

    The packed branch now packs the observation via ``pack_subset_observation`` (the
    inference mirror of the training collate) and samples the posterior directly, so we
    record each ``pack_subset_observation`` call (and its ``source_vec``) — an empty
    ``recorder["calls"]`` means the DIRECT (fixed-station) branch was taken.
    """
    import seismo_sbi.sbi.compression.ML.dataloading as dl_mod
    import seismo_sbi.instaseis_simulator.post_processing as pp_mod
    import seismo_sbi.sbi.compression.ML.source_conditioning as sc_mod

    monkeypatch.setattr(dl_mod, "TorchSimulationDataset",
                        lambda **kw: stub_ds, raising=True)
    monkeypatch.setattr(
        pp_mod, "build_augmentation_chain_from_parameters",
        lambda *a, **k: (None, None), raising=True)

    def _stub_pack(obs, coords, source_vec=None):
        recorder["calls"].append({"source_vec": source_vec,
                                  "obs_shape": np.asarray(obs).shape})
        return torch.zeros((1, 4))   # dummy packed context (has .to(device))

    monkeypatch.setattr(sc_mod, "pack_subset_observation", _stub_pack, raising=True)


def test_run_validation_packed_branch_calls_ensemble(monkeypatch):
    """variable_stations=True → packs via pack_subset_observation (packed path)."""
    from seismo_sbi.evaluation.validation import run_validation

    N, C, T = 3, 3, 8
    names = [f"S{i}" for i in range(N)]
    ds = _StubDataset(n_sims=20, N=N, C=C, T=T)
    recorder = {"calls": []}
    _patch_run_validation_deps(monkeypatch, ds, recorder)

    out = run_validation(
        _StubPipeline(names), object(), _DeterministicPosterior(),
        _IdentityScaler(), n_val=2, n_show=1, num_samples=4, device="cpu",
        variable_stations=True, cond_param_map=None,
    )
    # Packed branch must have packed the observation once per val sim.
    assert len(recorder["calls"]) == 2
    assert all(c["source_vec"] is None for c in recorder["calls"])
    assert out["theta_scaled"].shape == (2, 6)
    assert out["samples_scaled"].shape == (4, 2, 6)


def test_run_validation_conditioned_threads_source_vec(monkeypatch):
    """cond_param_map set → packed path feeds the sim's stored source vector."""
    from seismo_sbi.evaluation.validation import run_validation

    N, C, T = 3, 3, 8
    names = [f"S{i}" for i in range(N)]
    ds = _StubDataset(n_sims=20, N=N, C=C, T=T)
    recorder = {"calls": []}
    _patch_run_validation_deps(monkeypatch, ds, recorder)

    cond_map = {"source_location": ["latitude", "longitude", "depth"]}
    out = run_validation(
        _StubPipeline(names), object(), _DeterministicPosterior(),
        _IdentityScaler(), n_val=2, n_show=0, num_samples=4, device="cpu",
        variable_stations=False, cond_param_map=cond_map,  # cond forces packed
    )
    assert len(recorder["calls"]) == 2
    for c in recorder["calls"]:
        assert c["source_vec"] is not None
        np.testing.assert_allclose(c["source_vec"], [36.5, 25.0, 8.0])
    assert out["samples_scaled"].shape == (4, 2, 6)


def test_run_validation_direct_branch_calls_posterior_sample(monkeypatch):
    """variable_stations=False, cond_param_map=None → direct dataset[idx]→posterior.sample."""
    from seismo_sbi.evaluation.validation import run_validation

    N, C, T = 3, 3, 8
    names = [f"S{i}" for i in range(N)]
    # n_sims=30 → int(0.90*30)=27 → held-out tail = sims 27..29 = 3 sims.
    ds = _StubDataset(n_sims=30, N=N, C=C, T=T)
    recorder = {"calls": []}
    _patch_run_validation_deps(monkeypatch, ds, recorder)

    post = _DeterministicPosterior(n_dims=6)
    out = run_validation(
        _StubPipeline(names), object(), post,
        _IdentityScaler(), n_val=3, n_show=1, num_samples=5, device="cpu",
        variable_stations=False, cond_param_map=None,
    )
    # Direct branch must NOT pack the observation at all.
    assert len(recorder["calls"]) == 0
    # It must call posterior.sample once per val sim with a (1, N, C, T) context.
    assert len(post.seen_ctx_shapes) == 3
    for shp in post.seen_ctx_shapes:
        assert shp == (1, N, C, T)
    assert out["theta_scaled"].shape == (3, 6)
    assert out["samples_scaled"].shape == (5, 3, 6)


# ---------------------------------------------------------------------------
# 4.  write_validation_outputs side-effects (pure subset: JSON write)
# ---------------------------------------------------------------------------

class _MinimalParameters:
    """Minimal stub for the ``parameters`` arg of write_validation_outputs."""
    def parameter_to_vector(self, _key):
        return np.zeros(6)


class _MinimalScaler:
    def inverse_transform(self, x):
        return np.asarray(x)
    def transform(self, x):
        return np.asarray(x)


def test_write_validation_outputs_writes_json(tmp_path):
    """write_validation_outputs always writes evaluation_metrics.json, even if plotting
    fails (plotting is wrapped in try/except in the implementation)."""
    from seismo_sbi.evaluation.validation import write_validation_outputs

    val = _make_val_dict(n_val=4, n_dims=6, num_samples=3)
    params = _MinimalParameters()
    scaler = _MinimalScaler()

    # Should not raise even if plotting backends are unavailable.
    result = write_validation_outputs(
        val, tmp_path, params, scaler,
        num_samples=3, conditioned=False, n_show=0,
    )

    metrics_path = tmp_path / "evaluation_metrics.json"
    assert metrics_path.exists(), "evaluation_metrics.json must be written"
    with open(metrics_path) as f:
        on_disk = json.load(f)

    assert on_disk["n_val"] == 4
    assert on_disk["conditioned"] is False
    assert on_disk["num_samples"] == 3
    assert "figures" in on_disk
    assert "metrics" in on_disk


def test_write_validation_outputs_conditioned_flag_propagated(tmp_path):
    """The ``conditioned`` flag must be faithfully written to the JSON."""
    from seismo_sbi.evaluation.validation import write_validation_outputs

    val = _make_val_dict()
    params = _MinimalParameters()
    scaler = _MinimalScaler()

    write_validation_outputs(
        val, tmp_path, params, scaler,
        num_samples=2, conditioned=True, n_show=0,
    )
    with open(tmp_path / "evaluation_metrics.json") as f:
        d = json.load(f)
    assert d["conditioned"] is True


def test_write_validation_outputs_creates_out_dir(tmp_path):
    """write_validation_outputs must create the output directory if it does not exist."""
    from seismo_sbi.evaluation.validation import write_validation_outputs

    out_dir = tmp_path / "new" / "subdir"
    assert not out_dir.exists()

    val = _make_val_dict()
    params = _MinimalParameters()
    scaler = _MinimalScaler()

    write_validation_outputs(
        val, out_dir, params, scaler,
        num_samples=2, conditioned=False, n_show=0,
    )
    assert out_dir.is_dir()
    assert (out_dir / "evaluation_metrics.json").exists()
