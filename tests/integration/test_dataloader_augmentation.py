"""Integration tests: training-time nuisance augmentation in the dataloader.

These verify that the Category-2 nuisance effects, when supplied as an
`augmentation_chain`, are folded into the CLEAN loaded data BEFORE noise is added
in `TorchSimulationDataset.__getitem__`, and that the absence of a chain is exact
back-compat (clean load + noise).  No h5 files are needed — `_load_sim` is stubbed.
"""

import os
import h5py
import numpy as np
import torch

from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
from seismo_sbi.instaseis_simulator.dataloader import SimulationDataLoader
from seismo_sbi.instaseis_simulator.post_processing import (
    PostProcessingChain,
    AmplitudeErrorEffect,
    TimeShiftErrorEffect,
    ComponentDropoutEffect,
)
from seismo_sbi.sbi.compression.ML.dataloading import (
    TorchSimulationDataset,
    make_torch_dataloader,
    StationSubsampler,
    _seed_worker,
)

TRACE_LEN = 32


def _receivers():
    return Receivers(receivers=[
        Receiver(0.0, 0.0, "XX", "STA1", ["Z"]),
        Receiver(1.0, 1.0, "XX", "STA2", ["Z"]),
    ])


def _make_dataset(augmentation_chain, augmentation_nuisance_params, D_clean):
    """Build a TorchSimulationDataset with _load_sim and noise stubbed out."""
    receivers = _receivers()
    loader = SimulationDataLoader(components=["Z"], receivers=receivers)

    ds = TorchSimulationDataset.__new__(TorchSimulationDataset)
    ds.data_loader = loader
    ds.parameter_name_map = {}
    ds.conditioning_param_map = {}
    ds.data_scaler = None
    ds.return_tensors = True
    ds.torch_dtype = torch.float32
    ds.augmentation_chain = augmentation_chain
    ds.augmentation_nuisance_params = augmentation_nuisance_params or {}
    ds.paths = ["dummy.h5"]
    # Zero noise → x == (possibly augmented) D, isolating the augmentation effect.
    ds.synthetic_noise_model_sampler = lambda: np.zeros((2, TRACE_LEN))
    ds._load_sim = lambda path: (np.array([]), D_clean.copy())
    return ds


def _clean_D():
    D = np.zeros((2, 1, TRACE_LEN), dtype=np.float64)
    D[0, 0] = np.arange(TRACE_LEN, dtype=float) + 1.0
    D[1, 0] = np.arange(TRACE_LEN, dtype=float) + 100.0
    return D


def test_no_chain_is_backward_compatible():
    """No augmentation chain ⇒ x == clean D (+ zero noise)."""
    D_clean = _clean_D()
    ds = _make_dataset(None, None, D_clean)
    _, x = ds[0]
    assert np.allclose(x.numpy(), D_clean)


def test_amplitude_augmentation_applied_before_noise():
    """Deterministic amplitude scale ⇒ x == 2 * clean D (noise is zero)."""
    D_clean = _clean_D()
    chain = PostProcessingChain([AmplitudeErrorEffect(scale_range=(2.0, 2.0))])
    ds = _make_dataset(chain, {"amplitude_error": 1.0}, D_clean)
    np.random.seed(0)
    _, x = ds[0]
    assert np.allclose(x.numpy(), 2.0 * D_clean)


def test_time_shift_augmentation_changes_data():
    """A non-zero time-shift augmentation must change the (non-constant) data."""
    D_clean = _clean_D()
    chain = PostProcessingChain([
        TimeShiftErrorEffect(sampling_rate=1.0, uniform_offset=0.0, gaussian_sigma=2.0)
    ])
    ds = _make_dataset(chain, {"time_shift_error": 1.0}, D_clean)
    np.random.seed(1)
    _, x = ds[0]
    assert not np.allclose(x.numpy(), D_clean)
    assert x.shape == (2, 1, TRACE_LEN)


def test_augmentation_is_reproducible_under_fixed_seed():
    """Same seed ⇒ identical augmented output (stochastic effect)."""
    D_clean = _clean_D()
    chain = PostProcessingChain([AmplitudeErrorEffect(scale_range=(0.3, 1.9))])
    ds = _make_dataset(chain, {"amplitude_error": 0.7}, D_clean)

    np.random.seed(123)
    _, x1 = ds[0]
    np.random.seed(123)
    _, x2 = ds[0]
    assert np.allclose(x1.numpy(), x2.numpy())


# ---------------------------------------------------------------------------
# T2: end-to-end factory wiring — make_torch_dataloader threads the chain through
# to a REAL TorchSimulationDataset reading h5 from disk (not a stub).
# ---------------------------------------------------------------------------

def _write_sim_h5(path, receivers, n=TRACE_LEN):
    """Write a minimal SBI-schema h5 (outputs/{station}/{component}) for the loader."""
    with h5py.File(path, "w") as f:
        out = f.create_group("outputs")
        for i, rec in enumerate(receivers.iterate()):
            g = out.create_group(rec.station_name)
            for comp in rec.components:
                g.create_dataset(comp, data=(np.arange(n, dtype=float) + 10.0 * i + 1.0))


def test_make_torch_dataloader_applies_augmentation_end_to_end(tmp_path):
    receivers = _receivers()
    loader = SimulationDataLoader(components=["Z"], receivers=receivers)
    for k in range(2):
        _write_sim_h5(os.path.join(str(tmp_path), f"sim_{k}.h5"), receivers)

    # Clean reference (no augmentation) straight from the loader.
    clean = loader.load_simulation_data_array(
        os.path.join(str(tmp_path), "sim_0.h5"), stacked=True, fill_unused=True
    )

    chain = PostProcessingChain([AmplitudeErrorEffect(scale_range=(2.0, 2.0))])
    dl = make_torch_dataloader(
        data_loader=loader,
        data_folder=str(tmp_path),
        parameter_name_map={},
        synthetic_noise_model_sampler=lambda: np.zeros((2, TRACE_LEN)),  # zero noise
        augmentation_chain=chain,
        augmentation_nuisance_params={"amplitude_error": 1.0},
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    np.random.seed(0)
    _, x = next(iter(dl))                 # (batch=2, n_stations=2, n_comp=1, T)
    # Deterministic scale_range=(2,2), prob=1 ⇒ every sim's data is exactly doubled.
    assert np.allclose(x[0].numpy(), 2.0 * clean)


def test_make_torch_dataloader_no_chain_is_clean(tmp_path):
    """No augmentation_chain ⇒ dataloader returns clean data (+ zero noise)."""
    receivers = _receivers()
    loader = SimulationDataLoader(components=["Z"], receivers=receivers)
    _write_sim_h5(os.path.join(str(tmp_path), "sim_0.h5"), receivers)
    clean = loader.load_simulation_data_array(
        os.path.join(str(tmp_path), "sim_0.h5"), stacked=True, fill_unused=True
    )
    dl = make_torch_dataloader(
        data_loader=loader, data_folder=str(tmp_path), parameter_name_map={},
        synthetic_noise_model_sampler=lambda: np.zeros((2, TRACE_LEN)),
        augmentation_chain=None, batch_size=1, shuffle=False, num_workers=0,
    )
    _, x = next(iter(dl))
    assert np.allclose(x[0].numpy(), clean)


# ---------------------------------------------------------------------------
# T4: _seed_worker is reproducible for a fixed (base seed, worker id) and distinct
# across worker ids.
# ---------------------------------------------------------------------------

def test_seed_worker_reproducible_and_distinct():
    torch.manual_seed(123)
    _seed_worker(0)
    a0 = np.random.uniform(size=5)
    torch.manual_seed(123)
    _seed_worker(0)
    a0_again = np.random.uniform(size=5)
    assert np.array_equal(a0, a0_again)          # same base seed + id ⇒ identical

    torch.manual_seed(123)
    _seed_worker(1)
    a1 = np.random.uniform(size=5)
    assert not np.array_equal(a0, a1)            # different worker id ⇒ distinct stream


# ---------------------------------------------------------------------------
# Post-noise component dropout — MUST be applied AFTER noise so a dropped
# channel is EXACTLY zero (matching a genuinely-absent component), unlike the
# pre-noise augmentation chain.  These are the critical ordering tests.
# ---------------------------------------------------------------------------

# STA1 present Z,N,E (3); STA2 present Z,N (E absent → zero-filled).  Global comps ZNE.
_GLOBAL_COMPS = ["Z", "N", "E"]
_NOISE_LEVEL = 0.5
_N_PRESENT = 5  # STA1:3 + STA2:2


def _multicomp_receivers():
    return Receivers(receivers=[
        Receiver(0.0, 0.0, "XX", "STA1", ["Z", "N", "E"]),
        Receiver(1.0, 1.0, "XX", "STA2", ["Z", "N"]),
    ])


def _multicomp_clean_D():
    """(2, 3, T) clean data: present channels are non-zero ramps, absent (STA2 E) is zero."""
    D = np.zeros((2, 3, TRACE_LEN), dtype=np.float64)
    D[0, 0] = np.arange(TRACE_LEN) + 1.0    # STA1 Z
    D[0, 1] = np.arange(TRACE_LEN) + 2.0    # STA1 N
    D[0, 2] = np.arange(TRACE_LEN) + 3.0    # STA1 E
    D[1, 0] = np.arange(TRACE_LEN) + 4.0    # STA2 Z
    D[1, 1] = np.arange(TRACE_LEN) + 5.0    # STA2 N (E stays zero)
    return D


def _make_post_noise_dataset(post_chain, post_params, D_clean, station_subsampler=None):
    receivers = _multicomp_receivers()
    loader = SimulationDataLoader(components=_GLOBAL_COMPS, receivers=receivers)
    ds = TorchSimulationDataset.__new__(TorchSimulationDataset)
    ds.data_loader = loader
    ds.parameter_name_map = {}
    ds.conditioning_param_map = {}
    ds.data_scaler = None
    ds.return_tensors = True
    ds.torch_dtype = torch.float32
    ds.augmentation_chain = None
    ds.augmentation_nuisance_params = {}
    ds.post_noise_augmentation_chain = post_chain
    ds.post_noise_nuisance_params = post_params or {}
    ds.station_subsampler = station_subsampler
    ds.station_coords = receivers.get_station_locations_array()
    ds.paths = ["dummy.h5"]
    # Constant NON-ZERO noise on present channels (zero-filled for absent ones by the loader).
    ds.synthetic_noise_model_sampler = lambda: np.full((_N_PRESENT, TRACE_LEN), _NOISE_LEVEL)
    ds._load_sim = lambda path: (np.array([]), D_clean.copy())
    return ds


def test_component_dropout_zeros_channel_exactly_after_noise():
    """The crux: with NON-ZERO noise, a dropped present channel is EXACTLY zero in x — proving
    it was zeroed AFTER noise (a pre-noise zeroing would leave 0 + noise)."""
    D = _multicomp_clean_D()
    chain = PostProcessingChain([ComponentDropoutEffect()])
    ds = _make_post_noise_dataset(chain, {"component_dropout": 1.0}, D)

    np.random.seed(0)
    _, x = ds[0]
    x = x.numpy()

    # STA1 (3 present): p=1 keeps exactly ONE present channel; the other two are EXACTLY 0.
    sta1_present = x[0, :3]  # Z, N, E all present
    nonzero = [j for j in range(3) if not np.allclose(sta1_present[j], 0.0)]
    zeroed = [j for j in range(3) if np.all(sta1_present[j] == 0.0)]
    assert len(nonzero) == 1 and len(zeroed) == 2
    # The surviving channel still carries signal + noise (= clean ramp + 0.5).
    j = nonzero[0]
    assert np.allclose(x[0, j], D[0, j] + _NOISE_LEVEL)
    # STA2 absent E channel stays exactly zero throughout.
    assert np.allclose(x[1, 2], 0.0)


def test_without_post_chain_channels_are_noisy_not_zero():
    """Contrast: without the post-noise chain, every present channel is signal+noise (non-zero),
    so it is the POST-noise dropout that produces the exact zeros above."""
    D = _multicomp_clean_D()
    ds = _make_post_noise_dataset(None, None, D)
    _, x = ds[0]
    x = x.numpy()
    # All present channels are non-zero (clean ramp + 0.5 noise); none accidentally zero.
    for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]:
        assert np.allclose(x[i, j], D[i, j] + _NOISE_LEVEL)
        assert not np.allclose(x[i, j], 0.0)
    # Absent STA2 E remains zero.
    assert np.allclose(x[1, 2], 0.0)


def test_component_dropout_keeps_at_least_one_per_station():
    D = _multicomp_clean_D()
    chain = PostProcessingChain([ComponentDropoutEffect()])
    ds = _make_post_noise_dataset(chain, {"component_dropout": 1.0}, D)
    for seed in range(20):
        np.random.seed(seed)
        _, x = ds[0]
        x = x.numpy()
        # Each station retains >=1 non-zero present channel (never an all-zero station).
        assert any(not np.allclose(x[0, j], 0.0) for j in range(3))
        assert any(not np.allclose(x[1, j], 0.0) for j in range(2))


def test_component_dropout_reproducible_under_fixed_seed():
    D = _multicomp_clean_D()
    chain = PostProcessingChain([ComponentDropoutEffect()])
    ds = _make_post_noise_dataset(chain, {"component_dropout": 0.5}, D)
    np.random.seed(7)
    _, x1 = ds[0]
    np.random.seed(7)
    _, x2 = ds[0]
    assert np.allclose(x1.numpy(), x2.numpy())


def test_component_dropout_with_variable_stations():
    """Post-noise dropout composes with variable-station subsampling: the returned (x_sub,...)
    tuple still has dropped channels exactly zero."""
    D = _multicomp_clean_D()
    chain = PostProcessingChain([ComponentDropoutEffect()])
    # keep_fraction=1.0 keeps both stations so we can assert on a known station set.
    sub = StationSubsampler(keep_fraction=1.0, min_stations=1)
    ds = _make_post_noise_dataset(chain, {"component_dropout": 1.0}, D, station_subsampler=sub)

    np.random.seed(0)
    theta, sample = ds[0]
    x_sub, coords_sub, source_vec = sample
    x_sub = x_sub.numpy()
    assert x_sub.shape == (2, 3, TRACE_LEN) and coords_sub.shape == (2, 2)
    # STA1 still keeps exactly one present channel after noise+dropout, then subsampling.
    nonzero = [j for j in range(3) if not np.allclose(x_sub[0, j], 0.0)]
    assert len(nonzero) == 1
