"""Model-only passthrough benchmark: fwd + bwd + optimizer-step ms/step.

The PRIMARY passthrough metric for `.claude/runs/ml-architectures/efficiency-optimization`.
Builds the REAL production architecture (the brustle-lomax YAML: tcn encoder + source
conditioning + variable stations + amplitude embedding + RFF posenc + PMA-tokens pooling +
8-transform NSF flow) via `CompressionTrainer`, synthesises ONE fixed variable-station batch
in the exact packed format `variable_station_collate` produces, moves it to the GPU once, and
times the training step `loss = -flow.log_prob(theta, context=x).mean(); loss.backward();
opt.step()` in a tight loop. NO dataloader, NO augmentation, NO disk — so the number isolates
GPU forward-pass + backprop compute, which is what the compute optimisations target.

NOT a test (kept out of the suite). Deterministic batch for a fixed --seed so before/after
numbers are comparable. Reports median/mean ms/step (warmup dropped) + peak GPU mem, and a
fixed-input log_prob checksum so numerics-touching opts can be checked for non-regression.

Usage:
    PYTHONPATH=. conda run -n seismo-sbi python scripts/bench_train_step.py \
        --batch-size 128 --stations 16 --duration 200 --steps 100 --warmup 25
"""
import argparse
import os
import statistics
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

from seismo_sbi.instaseis_simulator.utils import compute_data_vector_length
from seismo_sbi.sbi.compression.ML.train import CompressionTrainer

from _bench_common import production_model_config, make_variable_station_batch


def build_trainer(args, trace_length, station_locations, perf):
    """Construct the production CompressionTrainer. `perf` is an opt-toggle dict folded
    into model_config so the src perf hooks (added per optimisation) can read it."""
    components = list(args.components)
    model_config, flow_config, n_cond = production_model_config(
        model_dim=args.model_dim, station_encoder=args.architecture)
    if perf:
        model_config["perf"] = perf
    trainer = CompressionTrainer(
        components, station_locations, channels=args.model_dim, latent_dim=args.model_dim,
        architecture="seismogram_transformer", trace_length=trace_length,
        model_config=model_config, flow_config=flow_config,
        lr=1e-4, weight_decay=1e-4, lr_second_stage="cosine",
    )
    return trainer, n_cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--stations", type=int, default=16, help="master station count")
    ap.add_argument("--components", type=str, default="ZNE")
    ap.add_argument("--duration", type=float, default=200.0)
    ap.add_argument("--sampling-rate", type=float, default=1.0)
    ap.add_argument("--model-dim", type=int, default=256)
    ap.add_argument("--architecture", type=str, default="tcn", help="per-station encoder")
    ap.add_argument("--keep-low", type=float, default=0.5)
    ap.add_argument("--keep-high", type=float, default=1.0)
    ap.add_argument("--min-stations", type=int, default=3)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-n", action="store_true",
                    help="keep all stations every sample (keep_fraction=1.0) for a static-shape run")
    # --- opt toggles (wired into model_config['perf']; ignored by src until implemented) ---
    ap.add_argument("--amp", action="store_true", help="bf16 autocast on the embedding net")
    ap.add_argument("--compile", action="store_true", help="torch.compile the embedding net")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--sdpa", action="store_true", help="use fused scaled_dot_product_attention")
    ap.add_argument("--no-batchnorm", action="store_true", help="flow use_batch_norm=False")
    ap.add_argument("--tf32", action="store_true", help="allow TF32 matmul/cudnn")
    ap.add_argument("--label", type=str, default="run")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    trace_length = compute_data_vector_length(args.duration, args.sampling_rate) + 1
    rng = np.random.default_rng(args.seed)
    station_locations = np.stack([
        rng.uniform(36.05, 36.85, args.stations),
        rng.uniform(25.15, 25.95, args.stations),
    ], axis=1)

    perf = {}
    if args.amp:
        perf["amp"] = True
    if args.compile:
        perf["compile"] = True
    if args.channels_last:
        perf["channels_last"] = True
    if args.sdpa:
        perf["sdpa"] = True

    trainer, n_cond = build_trainer(args, trace_length, station_locations, perf)
    # Optional flow numerics toggle (O7a): rebuild flow without conditioner BatchNorm.
    if args.no_batchnorm:
        from seismo_sbi.sbi.compression.ML.maf import build_nsf
        trainer.flow = CompressionTrainer._assemble_flow(
            architecture="seismogram_transformer", num_seismic_components=len(args.components),
            model_config={**trainer._model_config}, flow_config={**trainer._flow_config, "use_batch_norm": False},
            feature_length=trainer._feature_length, latent_dim=trainer.latent_dim,
            num_dims=trainer.num_dims, station_locations=station_locations,
            trace_length=trace_length, device=device)
    flow = trainer.flow.to(device)
    flow.train()

    keep = 1.0 if args.fixed_n else (args.keep_low, args.keep_high)
    theta, context = make_variable_station_batch(
        args.batch_size, args.stations, list(args.components), trace_length, n_cond,
        keep_fraction=keep, min_stations=args.min_stations, seed=args.seed)
    theta = theta.to(device)
    context = context.to(device)

    n_params = sum(p.numel() for p in flow.parameters())
    print(f"=== bench_train_step [{args.label}] ===")
    print(f"arch={args.architecture} model_dim={args.model_dim} | B={args.batch_size} "
          f"N_master={args.stations} C={len(args.components)} T={trace_length} | "
          f"keep={keep} | n_cond={n_cond} | flow num_transforms={trainer._flow_config.get('num_transforms')}")
    print(f"context width W={context.shape[1]} | params={n_params/1e6:.2f}M | "
          f"perf={perf} no_bn={args.no_batchnorm} tf32={args.tf32}")
    print(f"device={device} cuda={torch.cuda.is_available()}")

    opt = torch.optim.AdamW(flow.parameters(), lr=1e-4, weight_decay=1e-4)

    def step():
        opt.zero_grad(set_to_none=True)
        log_prob = flow.log_prob(theta, context=context)
        loss = -log_prob.mean()
        loss.backward()
        opt.step()
        return loss

    # Checksum: a fixed-input eval log_prob mean (numerics non-regression probe).
    flow.eval()
    with torch.no_grad():
        lp = flow.log_prob(theta, context=context)
        checksum = float(lp.mean().item())
    flow.train()
    print(f"[checksum] eval log_prob.mean() = {checksum:.6f}  (finite={np.isfinite(checksum)})")

    # Warmup
    for _ in range(args.warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        loss = step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    times.sort()
    med = statistics.median(times)
    mean = statistics.mean(times)
    p10 = times[max(0, int(0.1 * len(times)) - 1)]
    peak_mb = (torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else 0.0
    print(f"\n--- {args.steps} timed steps (warmup {args.warmup} dropped) ---")
    print(f"ms/step  median={med:8.3f}  mean={mean:8.3f}  p10={p10:8.3f}  "
          f"min={times[0]:8.3f}  max={times[-1]:8.3f}")
    print(f"steps/s  median={1000.0/med:8.2f}  | samples/s={args.batch_size*1000.0/med:8.1f}")
    print(f"peak GPU mem (alloc) = {peak_mb:8.1f} MB | final loss={float(loss):.4f}")
    print("=== done ===")


if __name__ == "__main__":
    main()
