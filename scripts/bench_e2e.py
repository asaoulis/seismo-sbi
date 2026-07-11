"""e2e per-epoch walltime bench — REAL headless training with a controllable load-time
augmentation chain, to measure when the dataloader stops being the limiter.

Each run = `variant|aug|cache|workers` where
  variant: perf spec for bench_ab.parse_variant (eager, amp+sdpa+ds8, ...)
  aug:     full  = amplitude + time_shift(3/3) + coda   (the production load-time chain)
           baked = amplitude only                        (time_shift + coda baked into sims)
           none  = no augmentation chain                 (dataloader floor)
  cache:   cache | nocache  (in-RAM sim cache)
  workers: dataloader worker count

Timing: per-epoch TRAIN time only (on_train_epoch_start -> last on_train_batch_end), so the
val loop, cache preload and Trainer setup are excluded; epoch 0 is warmup and dropped;
median over the remaining epochs. A sampler thread polls nvidia-smi during timed epochs.

Usage:
    PYTHONPATH=.:scripts conda run -n seismo-sbi python scripts/bench_e2e.py \
        --runs "eager|full|nocache|16,amp+sdpa+ds8|full|cache|16,amp+sdpa+ds8|baked|cache|16" \
        --num-sims 2500 --epochs 5
"""
import argparse
import os
import statistics
import subprocess
import tempfile
import threading
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
import pytorch_lightning as pl

from seismo_sbi.sbi.compression.ML.train import CompressionTrainer
from seismo_sbi.sbi.compression.ML.dataloading import StationSubsampler
from seismo_sbi.sbi.scalers import FlexibleScaler
from seismo_sbi.instaseis_simulator.post_processing import build_augmentation_chain

from _bench_common import build_kernel_pipeline, production_model_config
from bench_ab import parse_variant


def aug_chain(mode, sampling_rate):
    """Load-time augmentation chain by mode. 'full' matches the production
    brustle-lomax chain (uniform_offset/gaussian_sigma = 3.0, the bench_val_ab protocol);
    'baked' keeps only the cheap amplitude error (time_shift + coda moved to
    stage: simulation); 'none' disables augmentation entirely."""
    if mode == "none":
        return None, None
    nuisance = {"amplitude_error": [1.0]}
    effect_cfg = {}
    if mode == "full":
        nuisance.update({"time_shift_error": [1.0], "scattering_coda": [0.4]})
        effect_cfg = {"time_shift_error": {"uniform_offset": 3.0, "gaussian_sigma": 3.0},
                      "scattering_coda": {"alpha_range": (0.2, 0.6)}}
    elif mode != "baked":
        raise ValueError(f"unknown aug mode '{mode}'")
    stage = {k: "training_augmentation" for k in nuisance}
    return build_augmentation_chain(nuisance, stage, effect_cfg, sampling_rate=sampling_rate)


class GpuUtilSampler(threading.Thread):
    daemon = True

    def __init__(self, interval=0.5):
        super().__init__()
        self.interval = interval
        self.samples = []
        self.active = False
        self.stopped = False

    def run(self):
        while not self.stopped:
            if self.active:
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5)
                    self.samples.append(float(out.stdout.strip().splitlines()[0]))
                except Exception:
                    pass
            time.sleep(self.interval)


class EpochTimer(pl.Callback):
    """Times train batches per epoch; epoch 0 treated as warmup by the caller."""

    def __init__(self, gpu_sampler=None):
        self.epoch_times = []
        self._t0 = None
        self._t_last = None
        self.gpu = gpu_sampler

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = self._t_last = time.perf_counter()
        if self.gpu is not None and trainer.current_epoch >= 1:
            self.gpu.active = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._t_last = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.gpu is not None:
            self.gpu.active = False
        self.epoch_times.append(self._t_last - self._t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str,
                    default="eager|full|nocache|16,amp+sdpa+ds8|full|cache|16,"
                            "amp+sdpa+ds8|baked|cache|16,amp+sdpa+ds8|none|cache|16")
    ap.add_argument("--num-sims", type=int, default=2500)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--stations", type=int, default=16)
    ap.add_argument("--components", type=str, default="ZNE")
    ap.add_argument("--duration", type=float, default=200.0)
    ap.add_argument("--sampling-rate", type=float, default=1.0)
    ap.add_argument("--model-dim", type=int, default=256)
    ap.add_argument("--architecture", type=str, default="tcn")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tmp_dir = tempfile.mkdtemp(prefix="bench_e2e_")
    print(f"=== e2e per-epoch bench | {args.num_sims} sims | {args.epochs} epochs | "
          f"B={args.batch_size} N={args.stations} | runs={args.runs} ===", flush=True)

    pipeline, dvl, T = build_kernel_pipeline(
        args.stations, args.components, args.num_sims, args.duration, args.sampling_rate, tmp_dir)
    components = pipeline.data_manager.data_loader.components
    station_locations = pipeline.simulation_parameters.receivers.get_station_locations_array()
    data_scaler = FlexibleScaler(pipeline.parameters)

    import glob as _glob
    num_sims = len(_glob.glob(pipeline.simulations_output_path + "/train/*.h5"))
    train_max = int(0.9 * num_sims)

    results = []
    for raw in [s.strip() for s in args.runs.split(",") if s.strip()]:
        spec, aug_mode, cache_tok, nw_tok = [t.strip() for t in raw.split("|")]
        use_cache = cache_tok == "cache"
        nw = int(nw_tok)

        chain, params = aug_chain(aug_mode, args.sampling_rate)
        perf = parse_variant(spec)
        perf.pop("_nobn", False)
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

        dl_args = {
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
            "cache_in_memory": use_cache,
            "cache_preload_workers": 16,
        }

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        trainer = CompressionTrainer(
            components, station_locations, channels=args.model_dim, latent_dim=args.model_dim,
            architecture="seismogram_transformer", trace_length=T,
            model_config=model_config, flow_config=flow_config, lr=1e-4, weight_decay=1e-4)

        gpu = GpuUtilSampler()
        gpu.start()
        timer = EpochTimer(gpu_sampler=gpu)
        tag = raw.replace("|", "_").replace("+", "-")
        t0 = time.perf_counter()
        trainer.train(f"run_{tag}", epochs=args.epochs, output_path=tmp_dir,
                      dataloader_args=dl_args, logger=None,
                      enable_checkpointing=False, enable_progress_bar=False,
                      extra_callbacks=[timer])
        wall = time.perf_counter() - t0
        gpu.stopped = True

        timed = timer.epoch_times[1:] if len(timer.epoch_times) > 1 else timer.epoch_times
        sec_per_epoch = statistics.median(timed)
        smp_s = train_max / sec_per_epoch
        util = statistics.mean(gpu.samples) if gpu.samples else float("nan")
        results.append((raw, sec_per_epoch, smp_s, util, wall))
        print(f"[{raw:34s}] s/epoch={sec_per_epoch:7.2f}  samples/s={smp_s:7.1f}  "
              f"GPU-util~{util:5.1f}%  total-wall={wall:6.1f}s  "
              f"epochs={['%.2f' % t for t in timer.epoch_times]}", flush=True)

    base = results[0][1]
    print(f"\n{'run':36s} {'s/epoch':>8} {'smp/s':>8} {'GPU%':>6} {'x vs first':>10}")
    for raw, spe, smp, util, _ in results:
        print(f"{raw:36s} {spe:>8.2f} {smp:>8.1f} {util:>6.1f} {base / spe:>10.2f}")
    print("=== done ===")


if __name__ == "__main__":
    main()
