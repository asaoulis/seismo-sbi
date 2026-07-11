"""Multi-epoch val-loss A/B — the QUALITY guard for the perf optimisations.

Speed benches prove the step is faster; this proves the optimisations don't degrade what the
model learns. Builds ONE learnable kernel dataset (data = Σ θ_i · kernel_i + noise — a genuine
θ→x map the NPE flow must learn), then trains several variants (eager / amp+sdpa / amp+sdpa+ds8 …)
through the FULL production path (variable stations + source conditioning + amplitude embedding +
RFF posenc + PMA-tokens + 8-transform flow + training augmentation) for N epochs, with identical
data + noise + shuffle seeds per variant, and compares the val-loss curves.

A perf change is ACCEPTED only if its final val-loss matches the eager baseline within tolerance
(or is better). bf16 (amp) and the ds8 token-halving are the numerics-touching ones this guards.

Usage:
    PYTHONPATH=.:scripts conda run -n seismo-sbi python scripts/bench_val_ab.py \
        --variants eager,amp+sdpa,amp+sdpa+ds8 --epochs 18 --num-sims 3000 --stations 16
"""
import argparse
import csv
import os
import tempfile

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

from seismo_sbi.sbi.compression.ML.train import CompressionTrainer
from seismo_sbi.sbi.compression.ML.dataloading import StationSubsampler
from seismo_sbi.sbi.scalers import FlexibleScaler
from seismo_sbi.instaseis_simulator.post_processing import build_augmentation_chain

from _bench_common import build_kernel_pipeline, production_model_config
from bench_ab import parse_variant


def aug_chain(sampling_rate):
    nuisance = {"amplitude_error": [1.0], "time_shift_error": [1.0], "scattering_coda": [0.4]}
    stage = {k: "training_augmentation" for k in nuisance}
    effect_cfg = {"time_shift_error": {"uniform_offset": 3.0, "gaussian_sigma": 3.0},
                  "scattering_coda": {"alpha_range": (0.2, 0.6)}}
    return build_augmentation_chain(nuisance, stage, effect_cfg, sampling_rate=sampling_rate)


def read_val_curve(csv_path):
    epochs = []
    if not os.path.exists(csv_path):
        return epochs
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            v = row.get("val_loss", "")
            if v not in ("", None):
                try:
                    epochs.append(float(v))
                except ValueError:
                    pass
    return epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=str, default="eager,amp+sdpa,amp+sdpa+ds8")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--num-sims", type=int, default=3000)
    ap.add_argument("--stations", type=int, default=16)
    ap.add_argument("--components", type=str, default="ZNE")
    ap.add_argument("--duration", type=float, default=200.0)
    ap.add_argument("--sampling-rate", type=float, default=1.0)
    ap.add_argument("--model-dim", type=int, default=256)
    ap.add_argument("--architecture", type=str, default="tcn")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", action="store_true", help="enable in-RAM sim cache")
    args = ap.parse_args()

    tmp_dir = tempfile.mkdtemp(prefix="bench_val_")
    print(f"=== val-loss A/B | variants={args.variants} | {args.num_sims} sims | "
          f"{args.epochs} epochs | B={args.batch_size} N={args.stations} ===")

    pipeline, dvl, T = build_kernel_pipeline(
        args.stations, args.components, args.num_sims, args.duration, args.sampling_rate, tmp_dir)
    components = pipeline.data_manager.data_loader.components
    station_locations = pipeline.simulation_parameters.receivers.get_station_locations_array()
    data_scaler = FlexibleScaler(pipeline.parameters)
    chain, params = aug_chain(args.sampling_rate)

    num_sims = len(__import__("glob").glob(pipeline.simulations_output_path + "/train/*.h5"))
    train_max = int(0.9 * num_sims)

    def dl_args(nw):
        return {
            "data_loader": pipeline.data_manager.data_loader,
            "data_folder": pipeline.simulations_output_path + "/train",
            "parameter_name_map": pipeline.parameters.names,
            "synthetic_noise_model_sampler": (lambda: np.random.normal(0.0, 1.0, dvl)),
            "augmentation_chain": chain,
            "augmentation_nuisance_params": params,
            "data_scaler": data_scaler,
            "train_max_index": train_max,
            "train_batch_size": args.batch_size,
            "val_batch_size": 256,
            "num_workers": nw,
            "pin_memory": True,
            "prefetch_factor": 6 if nw > 0 else None,
            "conditioning_param_map": {"source_location": ["latitude", "longitude", "depth"]},
            "station_subsampler": StationSubsampler(keep_fraction=(0.5, 1.0), min_stations=3),
            "cache_in_memory": args.cache,
            "cache_preload_workers": min(16, args.num_workers or 16),
        }

    from pytorch_lightning.loggers import CSVLogger

    import time as _time
    results = {}
    walltimes = {}
    epochs_map = {}
    for raw in [s.strip() for s in args.variants.split(",")]:
        # "spec:epochs" overrides the global --epochs for a wall-clock-matched comparison.
        if ":" in raw and not raw.startswith("ds"):
            spec, ep_str = raw.rsplit(":", 1)
            n_epochs = int(ep_str)
        else:
            spec, n_epochs = raw, args.epochs
        epochs_map[spec] = n_epochs
        perf = parse_variant(spec)
        nobn = perf.pop("_nobn", False)
        ds = perf.pop("_ds", None)
        dec = perf.pop("_dec", None)
        nt = perf.pop("_nt", None)
        model_config, flow_config, _ = production_model_config(
            model_dim=args.model_dim, station_encoder=args.architecture, downsample=ds)
        if dec is not None:
            model_config["input_decimate"] = {"factor": int(dec)}
        if nt is not None:
            flow_config = {**flow_config, "num_transforms": int(nt)}
        if perf:
            model_config["perf"] = dict(perf)
        if nobn:
            flow_config = {**flow_config, "use_batch_norm": False}

        # Identical data/noise/shuffle order across variants.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        trainer = CompressionTrainer(
            components, station_locations, channels=args.model_dim, latent_dim=args.model_dim,
            architecture="seismogram_transformer", trace_length=T,
            model_config=model_config, flow_config=flow_config, lr=1e-4, weight_decay=1e-4)
        tag = f"{spec.replace('+','_')}_e{n_epochs}"
        logger = CSVLogger(save_dir=tmp_dir, name=f"val_{tag}", version="")
        _t0 = _time.perf_counter()
        trainer.train(f"run_{tag}", epochs=n_epochs, output_path=tmp_dir,
                      dataloader_args=dl_args(args.num_workers), logger=logger,
                      enable_checkpointing=False, enable_progress_bar=False)
        wall = _time.perf_counter() - _t0
        curve = read_val_curve(os.path.join(tmp_dir, f"val_{tag}", "metrics.csv"))
        results[spec] = curve
        walltimes[spec] = wall
        last = curve[-1] if curve else float("nan")
        best = min(curve) if curve else float("nan")
        print(f"[{spec:16s}] epochs={n_epochs:3d} wall={wall:6.1f}s  final val_loss={last:8.4f}  "
              f"best={best:8.4f}  epochs_logged={len(curve)}")

    base = list(results.keys())[0]
    base_best = min(results[base]) if results.get(base) else float("nan")
    print(f"\n{'variant':16s} {'epochs':>6} {'wall(s)':>8} {'final':>9} {'best':>9} {'Δbest vs base':>14}")
    for spec, curve in results.items():
        if not curve:
            print(f"{spec:16s} {'NaN':>6}")
            continue
        b = min(curve)
        print(f"{spec:16s} {epochs_map[spec]:>6d} {walltimes[spec]:>8.1f} "
              f"{curve[-1]:>9.4f} {b:>9.4f} {b-base_best:>+14.4f}")
    print("\nlast-5-epoch val curves:")
    for spec, curve in results.items():
        tail = " ".join(f"{v:.3f}" for v in curve[-5:])
        print(f"  {spec:18s}: {tail}")
    print("=== done ===")


if __name__ == "__main__":
    main()
