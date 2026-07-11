"""Benchmark harness for the training-time nuisance-augmentation dataloader.

Phase-2 diagnosis tool for `.claude/runs/ml-architectures/nuisance-sampling-route`.
NOT a test (kept out of the suite). Attributes the ~50% slowdown of the
`training_augmentation` route vs the baked route, and finds the `num_workers` knee.

Runs on the fabricated-kernel pipeline — NO Instaseis/CPS, no network. Builds a
realistic-shaped on-disk sim set (N stations x C components x T), then times:

  (a) `_load_sim` only                     -- disk read + theta marshalling
  (b) full `__getitem__`, chain=None       -- baked-equivalent floor (noise add only)
  (c) full `__getitem__`, aug chain        -- the augmentation route
  (d) `apply_chain_to_array` per effect    -- isolates adapter + each effect
  (e) end-to-end DataLoader throughput      -- sweep num_workers / pin_memory / prefetch

Also prints a fixed-seed checksum of one augmented sample so O1/O2 edits can be
proven behaviour-preserving (compare the number before/after).

Usage:
    conda run -n seismo-sbi python scripts/bench_aug_dataloader.py \
        --stations 30 --components ZNE --num-sims 128 --duration 300 \
        --getitem-iters 400 --loader-epochs 2
"""
import argparse
import os
import time

# Match the production thread-pinning (event_inversion.py / train_NPE.py) so the
# per-sample numbers reflect the real single-thread-per-worker regime.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

from seismo_sbi.instaseis_simulator.post_processing import (
    build_augmentation_chain,
    apply_chain_to_array,
)
from seismo_sbi.sbi.compression.ML.dataloading import (
    TorchSimulationDataset,
    make_torch_dataloaders,
)

from _bench_common import build_kernel_pipeline


def _time(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = time.perf_counter() - t0
    return dt / iters * 1e3   # ms / call


def _make_dataset(pipeline, data_vector_length, aug_chain, aug_params, components):
    data_loader = pipeline.data_manager.data_loader
    noise_sampler = lambda: np.random.normal(0.0, 1.0, data_vector_length)
    return TorchSimulationDataset(
        data_loader=data_loader,
        data_folder=pipeline.simulations_output_path + "/train",
        parameter_name_map=pipeline.parameters.names,
        synthetic_noise_model_sampler=noise_sampler,
        augmentation_chain=aug_chain,
        augmentation_nuisance_params=aug_params,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", type=int, default=30)
    ap.add_argument("--components", type=str, default="ZNE")
    ap.add_argument("--num-sims", type=int, default=128)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--sampling-rate", type=float, default=1.0)
    ap.add_argument("--getitem-iters", type=int, default=400)
    ap.add_argument("--loader-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=str, default="0,4,8,16",
                    help="comma-separated num_workers values for the sweep")
    ap.add_argument("--tmp-dir", type=str, default=None)
    args = ap.parse_args()

    import tempfile
    tmp_dir = args.tmp_dir or tempfile.mkdtemp(prefix="bench_aug_")
    print(f"=== bench: {args.stations} stations x {args.components} "
          f"x T(dur={args.duration},sr={args.sampling_rate}) | "
          f"{args.num_sims} sims | tmp={tmp_dir} ===")

    pipeline, dvl, T = build_kernel_pipeline(
        args.stations, args.components, args.num_sims,
        args.duration, args.sampling_rate, tmp_dir,
    )
    N, C = args.stations, len(args.components)
    print(f"shape: N={N} C={C} T={T}  data_vector_length={dvl}")

    # Full augmentation chain (amplitude + time-shift + coda all active).
    nuisance = {
        "amplitude_error": [1.0],
        "time_shift_error": [1.0],
        "scattering_coda": [1.0],
    }
    stage = {k: "training_augmentation" for k in nuisance}
    effect_cfg = {
        "time_shift_error": {"uniform_offset": 1.0, "gaussian_sigma": 1.0},
        "scattering_coda": {"alpha_range": (0.2, 0.6)},
    }
    full_chain, full_params = build_augmentation_chain(
        nuisance, stage, effect_cfg, sampling_rate=args.sampling_rate,
    )
    print(f"aug chain effects: {[type(e).__name__ for e in full_chain.effects]}")

    ds_aug = _make_dataset(pipeline, dvl, full_chain, full_params, args.components)
    ds_baked = _make_dataset(pipeline, dvl, None, {}, args.components)

    npaths = len(ds_aug.paths)

    # --- (a) _load_sim only ---
    counter = {"i": 0}
    def load_only():
        p = ds_aug.paths[counter["i"] % npaths]; counter["i"] += 1
        ds_aug._load_sim(p)
    a = _time(load_only, args.getitem_iters)

    # --- (b) full __getitem__, baked (chain=None) ---
    counter["i"] = 0
    def getitem_baked():
        ds_baked[counter["i"] % npaths]; counter["i"] += 1
    b = _time(getitem_baked, args.getitem_iters)

    # --- (c) full __getitem__, augmented ---
    counter["i"] = 0
    def getitem_aug():
        ds_aug[counter["i"] % npaths]; counter["i"] += 1
    c = _time(getitem_aug, args.getitem_iters)

    # --- (d) apply_chain_to_array per effect on a fixed D ---
    receivers = pipeline.data_manager.data_loader.receivers
    _, D0 = ds_aug._load_sim(ds_aug.paths[0])
    per_effect = {}
    # empty-chain round-trip floor
    from seismo_sbi.instaseis_simulator.post_processing import PostProcessingChain
    empty = PostProcessingChain([])
    per_effect["empty_roundtrip"] = _time(
        lambda: apply_chain_to_array(empty, D0, receivers, args.components, {}),
        args.getitem_iters)
    for key in ("amplitude_error", "time_shift_error", "scattering_coda"):
        ch, pr = build_augmentation_chain(
            {key: [1.0]}, {key: "training_augmentation"}, effect_cfg,
            sampling_rate=args.sampling_rate)
        per_effect[key] = _time(
            lambda ch=ch, pr=pr: apply_chain_to_array(ch, D0, receivers, args.components, pr),
            args.getitem_iters)

    print("\n--- per-sample timings (ms/call, single process) ---")
    print(f"  (a) _load_sim only              : {a:8.3f}")
    print(f"  (b) __getitem__ baked (no aug)  : {b:8.3f}")
    print(f"  (c) __getitem__ augmented       : {c:8.3f}")
    print(f"      -> aug overhead (c-b)       : {c - b:8.3f}  ({(c-b)/b*100:5.1f}% over baked)")
    print(f"      noise/marshalling (b-a)     : {b - a:8.3f}")
    print("  (d) apply_chain_to_array on fixed D:")
    for k, v in per_effect.items():
        print(f"        {k:22s}: {v:8.3f}")

    # --- fixed-seed equivalence checksum (prove O1/O2 preserve behaviour) ---
    np.random.seed(12345)
    theta_s, x_s = ds_aug[0]
    chk = float(np.asarray(x_s).astype(np.float64).sum())
    print(f"\n[equivalence checksum] fixed-seed ds_aug[0] x.sum() = {chk:.10e}")

    # --- (e) end-to-end DataLoader throughput sweep ---
    print("\n--- end-to-end loader throughput (samples/s, higher=better) ---")
    print(f"{'route':>10} {'workers':>8} {'pin':>4} {'prefetch':>9} {'samples/s':>12}")
    worker_list = [int(w) for w in args.workers.split(",")]
    train_max = int(0.9 * npaths)

    def sweep(label, aug_chain, aug_params):
        for nw in worker_list:
            for pin in ([False, True] if nw > 0 else [False]):
                train_loader, _ = make_torch_dataloaders(
                    data_loader=pipeline.data_manager.data_loader,
                    data_folder=pipeline.simulations_output_path + "/train",
                    parameter_name_map=pipeline.parameters.names,
                    synthetic_noise_model_sampler=lambda: np.random.normal(0.0, 1.0, dvl),
                    augmentation_chain=aug_chain,
                    augmentation_nuisance_params=aug_params,
                    train_max_index=train_max,
                    train_batch_size=args.batch_size,
                    num_workers=nw,
                    pin_memory=pin,
                )
                # warm up one epoch (spins persistent workers), then time
                for _ in range(1):
                    for _ in train_loader:
                        pass
                t0 = time.perf_counter()
                n_seen = 0
                for _ in range(args.loader_epochs):
                    for theta_b, x_b in train_loader:
                        n_seen += theta_b.shape[0]
                dt = time.perf_counter() - t0
                pf = 4 if nw > 0 else "-"
                print(f"{label:>10} {nw:>8} {str(pin):>4} {str(pf):>9} {n_seen/dt:>12.1f}")
                del train_loader

    sweep("baked", None, {})
    sweep("aug", full_chain, full_params)

    print("\n=== done ===")


if __name__ == "__main__":
    main()
