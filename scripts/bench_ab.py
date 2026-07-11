"""Interleaved A/B passthrough benchmark — reliable relative speedups under clock drift.

The A6000 on this box thermally drifts (1800-2100 MHz), so absolute ms/step wanders run to
run. This harness builds several model variants (by `perf` toggle) in ONE process, shares a
FIXED synthetic production batch, and times them ROUND-ROBIN (one step of each variant per
outer iteration). Because every variant sees the same GPU state in the same window, the
*ratios* are stable even while absolute numbers drift — exactly what we need to measure the
smaller stacked wins.

Also (optionally) proves equivalence: loads the eager model's weights into each variant
(`FusedMHA` mirrors `nn.MultiheadAttention`'s param names, so the state_dict transfers) and
reports the max abs difference of the embedding output — a behaviour-preserving check for the
fused-attention path.

Usage:
    PYTHONPATH=. conda run -n seismo-sbi python scripts/bench_ab.py \
        --variants eager,amp,sdpa,amp+sdpa --steps 80 --warmup 25
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

PERF_FLAGS = {"amp", "sdpa", "channels_last", "compile", "compile_flow"}


def parse_variant(spec):
    """'amp+sdpa+ds8' -> {'amp':True,'sdpa':True,'_ds':8}; 'eager' -> {}.

    Extended tokens: 'fused' (fused AdamW), 'decK' (model-entry input decimation
    factor K), 'ntK' (flow num_transforms K) — popped by builders like '_ds'.
    """
    perf = {}
    if spec in ("eager", "base", "baseline"):
        return perf
    for tok in spec.split("+"):
        tok = tok.strip()
        if tok in PERF_FLAGS:
            perf[tok] = True
        elif tok == "fused":
            perf["fused_adam"] = True
        elif tok == "nobn":
            perf["_nobn"] = True   # handled at build time (flow rebuild)
        elif tok.startswith("ds"):
            perf["_ds"] = int(tok[2:])   # tcn downsample (token-count) override
        elif tok.startswith("dec"):
            perf["_dec"] = int(tok[3:])  # model-entry input decimation factor
        elif tok.startswith("nt"):
            perf["_nt"] = int(tok[2:])   # flow num_transforms override
        else:
            raise ValueError(f"unknown perf token '{tok}' in variant '{spec}'")
    return perf


def build(perf, args, trace_length, station_locations, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    components = list(args.components)
    nobn = perf.pop("_nobn", False)
    ds = perf.pop("_ds", None)
    dec = perf.pop("_dec", None)
    nt = perf.pop("_nt", None)
    model_config, flow_config, n_cond = production_model_config(
        model_dim=args.model_dim, station_encoder=args.architecture, downsample=ds)
    if dec is not None:
        model_config["input_decimate"] = {"factor": int(dec)}
    if nt is not None:
        flow_config = {**flow_config, "num_transforms": int(nt)}
    if perf:
        model_config["perf"] = {k: v for k, v in perf.items()}
    if nobn:
        flow_config = {**flow_config, "use_batch_norm": False}
    tr = CompressionTrainer(
        components, station_locations, channels=args.model_dim, latent_dim=args.model_dim,
        architecture="seismogram_transformer", trace_length=trace_length,
        model_config=model_config, flow_config=flow_config,
        lr=1e-4, weight_decay=1e-4)
    return tr.flow.to("cuda"), n_cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=str, default="eager,amp,sdpa,amp+sdpa")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--stations", type=int, default=16)
    ap.add_argument("--components", type=str, default="ZNE")
    ap.add_argument("--duration", type=float, default=200.0)
    ap.add_argument("--sampling-rate", type=float, default=1.0)
    ap.add_argument("--model-dim", type=int, default=256)
    ap.add_argument("--architecture", type=str, default="tcn")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-n", action="store_true")
    ap.add_argument("--no-equiv", action="store_true", help="skip the equivalence check")
    ap.add_argument("--tf32", action="store_true")
    args = ap.parse_args()

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda")
    trace_length = compute_data_vector_length(args.duration, args.sampling_rate) + 1
    rng = np.random.default_rng(args.seed)
    station_locations = np.stack([rng.uniform(36.05, 36.85, args.stations),
                                  rng.uniform(25.15, 25.95, args.stations)], axis=1)

    specs = [s.strip() for s in args.variants.split(",")]
    keep = 1.0 if args.fixed_n else (0.5, 1.0)
    theta, context = make_variable_station_batch(
        args.batch_size, args.stations, list(args.components), trace_length,
        production_model_config(args.model_dim, args.architecture)[2],
        keep_fraction=keep, min_stations=3, seed=args.seed)
    theta, context = theta.to(device), context.to(device)

    print(f"=== bench_ab | variants={specs} | B={args.batch_size} N={args.stations} "
          f"T={trace_length} keep={keep} | steps={args.steps} warmup={args.warmup} ===")

    # Build all variants from the SAME seed so weights are identical where the modules match.
    flows = {}
    n_cond = None
    for spec in specs:
        perf = parse_variant(spec)
        flows[spec], n_cond = build(perf, args, trace_length, station_locations, args.seed)

    # --- equivalence: load eager weights into each variant, compare embedding output ---
    if not args.no_equiv and "eager" in flows:
        base = flows["eager"]
        base.eval()
        with torch.no_grad():
            e_base = base._embedding_net(context)
        for spec in specs:
            if spec == "eager":
                continue
            f = flows[spec]
            try:
                f._embedding_net.load_state_dict(base._embedding_net.state_dict())
                f.eval()
                with torch.no_grad():
                    e = f._embedding_net(context)
                d = float((e - e_base).abs().max())
                print(f"[equiv] {spec:12s}: max|Δ| embedding vs eager = {d:.3e}")
            except Exception as ex:
                print(f"[equiv] {spec:12s}: SKIP ({type(ex).__name__}: {str(ex)[:80]})")
            f.train()
        base.train()
        # rebuild eager+variants fresh so the equivalence weight-copy doesn't bias timing
        flows = {}
        for spec in specs:
            flows[spec], n_cond = build(parse_variant(spec), args, trace_length, station_locations, args.seed)

    opts = {s: torch.optim.AdamW(f.parameters(), lr=1e-4, weight_decay=1e-4)
            for s, f in flows.items()}
    for f in flows.values():
        f.train()

    def step(spec):
        f, opt = flows[spec], opts[spec]
        opt.zero_grad(set_to_none=True)
        loss = -f.log_prob(theta, context=context).mean()
        loss.backward()
        opt.step()
        return float(loss)

    # warmup all
    for _ in range(args.warmup):
        for spec in specs:
            step(spec)
    torch.cuda.synchronize()

    times = {s: [] for s in specs}
    last_loss = {s: None for s in specs}
    for _ in range(args.steps):
        for spec in specs:        # round-robin → identical GPU state per outer iter
            t0 = time.perf_counter()
            last_loss[spec] = step(spec)
            torch.cuda.synchronize()
            times[spec].append((time.perf_counter() - t0) * 1e3)

    base_med = statistics.median(times[specs[0]])
    print(f"\n{'variant':14s} {'median ms':>10} {'p20 ms':>9} {'speedup':>8} {'loss':>9}")
    for spec in specs:
        ts = sorted(times[spec])
        med = statistics.median(ts)
        p20 = ts[max(0, int(0.2 * len(ts)) - 1)]
        print(f"{spec:14s} {med:>10.2f} {p20:>9.2f} {base_med/med:>7.2f}x {last_loss[spec]:>9.3f}")
    print("=== done ===")


if __name__ == "__main__":
    main()
