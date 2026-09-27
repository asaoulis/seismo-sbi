"""Benchmark NPE training throughput vs DataLoader ``num_workers``.

Companion to ``scripts/bench_aug_dataloader.py``. That one drained the loader with
NO model, so workers couldn't overlap anything; THIS one runs **real training**
(embedding net + flow, forward+backward on the GPU via Lightning) so per-sample CPU
augmentation can be hidden behind GPU compute — which is the whole point of workers.

Uses the fabricated-kernel pipeline (no Instaseis/CPS, no network), the same
substrate as ``tests/end_to_end/test_train_npe_one_epoch.py``, but with a realistic
station count, a larger sim set, and the full training-augmentation chain
(amplitude + time-shift + coda) staged on. For each ``num_workers`` it builds a FRESH
trainer and times ``epochs`` of ``CompressionTrainer.train`` headless, reporting
wall-clock and train-samples/s.

Usage:
    PYTHONPATH=. conda run -n seismo-sbi python scripts/bench_training_workers.py \
        --stations 30 --components ZNE --num-sims 1024 --duration 300 \
        --epochs 5 --batch-size 128 --model-dim 64 --workers 0,4,8,16,32
"""
import argparse
import os
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from seismo_sbi.sbi.compression.ML.train import CompressionTrainer
from seismo_sbi.sbi.scalers import FlexibleScaler

from _bench_common import build_kernel_pipeline, default_augmentation_chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", type=int, default=30)
    ap.add_argument("--components", type=str, default="ZNE")
    ap.add_argument("--num-sims", type=int, default=1024)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--sampling-rate", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--model-dim", type=int, default=64)
    ap.add_argument("--architecture", type=str, default="seismogram_transformer")
    ap.add_argument("--workers", type=str, default="0,4,8,16,32")
    ap.add_argument("--pin-memory", action="store_true", default=True)
    ap.add_argument("--prefetch-factor", type=int, default=6)
    args = ap.parse_args()

    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="bench_train_")
    print(f"=== training-worker bench: {args.stations}x{args.components} "
          f"| {args.num_sims} sims | {args.architecture} dim={args.model_dim} "
          f"| bs={args.batch_size} | {args.epochs} epochs/setting ===")

    pipeline, dvl, T = build_kernel_pipeline(args.stations, args.components, args.num_sims,
                                             args.duration, args.sampling_rate, tmp_dir)

    import torch
    print(f"shape N={args.stations} C={len(args.components)} T={T} dvl={dvl} | "
          f"cuda={torch.cuda.is_available()} cpus={os.cpu_count()}")

    aug_chain, aug_params, nuisance = default_augmentation_chain(args.sampling_rate, coda_prob=0.4)
    print(f"aug effects: {[type(e).__name__ for e in aug_chain.effects]} "
          f"(coda prob={nuisance['scattering_coda'][0]})")

    components = pipeline.data_manager.data_loader.components
    station_locations = pipeline.simulation_parameters.receivers.get_station_locations_array()
    data_scaler = FlexibleScaler(pipeline.parameters)
    num_sims = len(__import__("glob").glob(os.path.join(pipeline.simulations_output_path + "/train", "*.h5")))
    train_max = int(0.9 * num_sims)
    n_train = train_max

    def base_args(nw):
        return {
            "data_loader": pipeline.data_manager.data_loader,
            "data_folder": pipeline.simulations_output_path + "/train",
            "parameter_name_map": pipeline.parameters.names,
            "synthetic_noise_model_sampler": (lambda: np.random.normal(0.0, 1.0, dvl)),
            "augmentation_chain": aug_chain,
            "augmentation_nuisance_params": aug_params,
            "data_scaler": data_scaler,
            "train_max_index": train_max,
            "train_batch_size": args.batch_size,
            "val_batch_size": args.batch_size,
            "num_workers": nw,
            "pin_memory": args.pin_memory,
            "prefetch_factor": args.prefetch_factor,
        }

    print(f"\n{'workers':>8} {'wall(s)':>9} {'s/epoch':>9} {'train smp/s':>12} {'speedup':>8}")
    results = {}
    baseline = None
    for nw in [int(w) for w in args.workers.split(",")]:
        trainer = CompressionTrainer(
            components, station_locations, channels=args.model_dim,
            latent_dim=args.model_dim, architecture=args.architecture, trace_length=T,
        )
        t0 = time.perf_counter()
        trainer.train(
            f"bench_nw{nw}", epochs=args.epochs, output_path=tmp_dir,
            dataloader_args=base_args(nw),
            logger=None, enable_checkpointing=False, enable_progress_bar=False,
        )
        wall = time.perf_counter() - t0
        s_per_epoch = wall / args.epochs
        smp_s = args.epochs * n_train / wall
        if baseline is None:
            baseline = smp_s
        results[nw] = (wall, s_per_epoch, smp_s)
        print(f"{nw:>8} {wall:>9.1f} {s_per_epoch:>9.2f} {smp_s:>12.1f} {smp_s/baseline:>7.2f}x")

    best = max(results, key=lambda k: results[k][2])
    print(f"\nbest: num_workers={best} at {results[best][2]:.1f} train smp/s "
          f"({results[best][2]/baseline:.2f}x vs num_workers={args.workers.split(',')[0]})")
    print("=== done ===")


if __name__ == "__main__":
    main()
