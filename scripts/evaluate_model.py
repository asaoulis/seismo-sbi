#!/usr/bin/env python3
"""
evaluate_model.py
=================
ONE generic, stage-toggled evaluation CLI for a variable-station NPE.

**Standalone / public-clone safe.** This file is tracked at the public
``scripts/`` root and imports **nothing** from the gitignored
``santorini_pathbreaker/`` or ``continuity/`` areas at module load.  All reusable
machinery lives in the tested ``seismo_sbi.evaluation`` package (pipeline /
posterior build, the per-model ``OutputLayout``, the unified validation/TARP
engine, the station-usage writers, the moment-tensor primitives, and the
``EvalDomain`` protocol + ``load_domain`` loader).

The **domain specifics** (event discovery, station-set derivation, per-event
conditioning source vector, reference-solution overlay) are supplied by a
pluggable adapter loaded **on demand** via ``--domain
<module>:<Class>`` or ``--domain </abs/path/adapter.py>:<Class>`` — e.g. the
gitignored ``scripts/santorini_pathbreaker/eval_adapter.py:SantoriniDomain``.
With no ``--domain``, the per-event ``events``/``dropout`` stages are skipped and
only the **domain-agnostic** ``validation`` stage (held-out TARP + recovery
scatter) runs — so a fresh public clone runs ``--help``, ``--dry-run`` and the
validation stage with zero adapter code present.

For each domain event, using ONE trained checkpoint, this:

  1. runs ML inversion with the event's **all available** stations AND with the
     **filtered "good quality" subset** the traditional inversions used,
  2. overlays both ML posteriors on a single lune against the **Zahradnik
     reference** and the **traditional SBI + Gaussian-likelihood** solutions,
     plus an MT-component / nodal corner,
  3. runs a **station-dropout ensemble** on the all-available set, and
  4. writes everything into a **per-model -> per-event** tree
     ``<output_root>/<model_name>/<event>/`` (default output_root =
     ``scripts/santorini_pathbreaker/eval``; model_name = resolved checkpoint
     dir name).  At the model root: ``run_meta.json``,
     ``ml_posttrain_summary.json``, ``station_usage.json`` / ``.csv``, and a
     model-level held-out ``validation/`` (TARP + recovery scatter + examples +
     metrics).

Stages (the configurable on/off knobs):
    --stages events,dropout,validation       # default = all
      events     → per-event ML(all)+ML(filtered) recovery lune + corner +
                   summary + station breakdown
      dropout    → station-dropout ensemble per event (KDE lune + spread
                   summary); nests under the events stage
      validation → per-model held-out TARP + recovery scatter + examples + JSON

Back-compat aliases (edit the --stages set so muscle-memory / scripts keep
working): --no-dropout, --no-validation, --validation_only.

Usage (cwd anywhere):
    DOMAIN=$REPO/scripts/santorini_pathbreaker/eval_adapter.py:SantoriniDomain

    # validate registry + station sets only (no model):
    python scripts/evaluate_model.py --domain $DOMAIN --dry-run

    # cheap smoke once a checkpoint exists (few samples, 1 event):
    python scripts/evaluate_model.py --domain $DOMAIN --smoke --events No14_id3250

    # full post-train suite (all domain events):
    python scripts/evaluate_model.py --domain $DOMAIN \
        --ckpt_dir /data/alex/santorini/npe/santorini_first_ml/results

    # public clone, no adapter: held-out validation/TARP only (domain-agnostic):
    python scripts/evaluate_model.py --stages validation --ckpt_dir <run>
"""
import argparse
import datetime
import json
import os
import pickle
import traceback
from pathlib import Path

# OMP/MKL pinning (mirror the other inference scripts) BEFORE numpy/torch import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

HERE = Path(__file__).resolve().parent                       # scripts/

# Reusable, tested evaluation engine.  NOTHING gitignored is imported here — the
# domain adapter is loaded on demand via --domain (load_domain) inside main().
from seismo_sbi.evaluation import (                           # noqa: E402
    resolve_output_layout, git_rev, load_domain,
    build_eval_pipeline, build_ml_posterior, resolve_ckpt_dir,
    run_validation, write_validation_outputs,
    write_station_breakdown, write_station_usage,
)

DEFAULT_CONFIG = str(HERE / "configs" / "santorini" / "first_ml_npe.yaml")
DEFAULT_CKPT_DIR = "/data/alex/santorini/npe/santorini_first_ml/results"
# Per-model -> per-event tree.  Default keeps the Santorini eval tree location
# (overridable via --output_root); it is a plain path string, not an import.
DEFAULT_OUTPUT_ROOT = str(HERE / "santorini_pathbreaker" / "eval")

ALL_STAGES = ["events", "dropout", "validation"]


# --------------------------------------------------------------------------- #
# Station-dropout ensemble (domain-agnostic — pure src utilities).
# --------------------------------------------------------------------------- #
def _event_dropout(obs_all, coords_all, names_all, posterior, data_scaler, parameters,
                   out_dir, *, num_samples, n_subsets, keep_fraction, min_stations, seed,
                   device, event, source_vec=None):
    """Station-dropout ensemble on the event's all-available set, via the shared
    src utilities. ``source_vec`` (when the model is conditioned) is the event's
    source vector, shared across every dropout config."""
    from seismo_sbi.sbi.compression.ML.station_dropout import (
        make_dropout_configs, sample_station_dropout_ensemble,
    )
    from seismo_sbi.plotting.evaluation import (
        spread_stats, plot_ensemble_lune_kde, plot_ensemble_spread_summary,
    )
    from seismo_sbi.plotting.results_plotting import SBIPipelinePlotter

    configs = make_dropout_configs(
        names_all, keep_fraction=keep_fraction, n_subsets=n_subsets,
        min_stations=min_stations, seed=seed, include_full=True)
    ensemble, results = sample_station_dropout_ensemble(
        posterior, obs_all, coords_all, configs, data_scaler,
        num_samples=num_samples, device=device, event_name=event, source_vec=source_vec)
    stats = {c.label: spread_stats(ensemble[c.label].samples) for c in configs}

    with open(out_dir / f"ml_{event}_station_dropout.pkl", "wb") as f:
        pickle.dump((None, None, results), f)

    figures = {}
    plotter = SBIPipelinePlotter(str(out_dir), parameters)
    plotter.initialise_posterior_plotter(
        data_scaler, parameters.parameter_to_vector("information")[:6])
    try:
        p = out_dir / f"ml_{event}_dropout_lune_kde.svg"
        plot_ensemble_lune_kde(ensemble, plotter, figsave=p, legend=True)
        figures["dropout_lune_kde"] = str(p)
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] dropout KDE lune failed: {type(e).__name__}: {e}")
    try:
        p = out_dir / f"ml_{event}_dropout_spread_summary.png"
        plot_ensemble_spread_summary(configs, ensemble, figsave=p)
        figures["dropout_spread_summary"] = str(p)
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] dropout spread summary failed: {type(e).__name__}: {e}")
    return figures, stats, configs


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Generic stage-toggled NPE evaluation (pluggable --domain).")
    p.add_argument("--domain", default=None,
                   help="Evaluation-domain adapter as 'module:Class' or "
                        "'/abs/path/adapter.py:Class' (e.g. "
                        "scripts/santorini_pathbreaker/eval_adapter.py:SantoriniDomain). "
                        "Required for the events/dropout stages; omit it to run only the "
                        "domain-agnostic validation stage.")
    p.add_argument("--config", "-c", default=DEFAULT_CONFIG,
                   help="Master training YAML (defines the model station set). "
                        "Default: first_ml_npe.yaml")
    p.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR,
                   help="Run dir holding model_meta.json + checkpoints/ (auto-resolved).")
    p.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT,
                   help="Root for the per-model eval tree "
                        "(default scripts/santorini_pathbreaker/eval).")
    p.add_argument("--model_name", default=None,
                   help="Per-model subdir name (default: resolved checkpoint dir name).")
    p.add_argument("--num_samples", type=int, default=10000, help="Posterior samples per config.")
    p.add_argument("--n_subsets", type=int, default=4, help="Random subsets for the dropout ensemble.")
    p.add_argument("--keep_fraction", type=float, default=0.6, help="Dropout keep fraction.")
    p.add_argument("--min_stations", type=int, default=3, help="Dropout floor on station count.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible subsets.")
    p.add_argument("--events", nargs="*", default=None, help="Limit to these event names.")
    p.add_argument("--n_val", type=int, default=200,
                   help="Held-out validation sims for the TARP/recovery validation stage.")
    p.add_argument("--n_show", type=int, default=6,
                   help="Validation examples rendered as MT/nodal corner panels.")
    p.add_argument("--val_num_samples", type=int, default=2000,
                   help="Posterior samples per validation sim (TARP stage).")
    # Stage selection (the configurable on/off requirement).
    p.add_argument("--stages", default=",".join(ALL_STAGES),
                   help=f"Comma-separated subset of {ALL_STAGES}. Default = all.")
    # Back-compat aliases — edit the --stages set so old invocations keep working.
    p.add_argument("--no-dropout", dest="no_dropout", action="store_true",
                   help="[alias] Remove the dropout stage.")
    p.add_argument("--no-validation", dest="no_validation", action="store_true",
                   help="[alias] Remove the validation stage.")
    p.add_argument("--validation_only", action="store_true",
                   help="[alias] Run ONLY the validation stage.")
    p.add_argument("--smoke", action="store_true", help="num_samples=300, n_subsets=2, n_val=20.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print registry + station sets + path checks; no model load.")
    return p.parse_args()


def _resolve_stages(args):
    """Resolve the active stage set from --stages + the back-compat aliases."""
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        raise SystemExit(f"Unknown --stages entries {unknown}; valid: {ALL_STAGES}")
    stages = set(stages)
    if args.validation_only:
        stages = {"validation"}
    if args.no_dropout:
        stages.discard("dropout")
    if args.no_validation:
        stages.discard("validation")
    # The dropout stage nests under the events stage.
    if "dropout" in stages and "events" not in stages and not args.validation_only:
        stages.add("events")
    return stages


def _master_station_names(config_path):
    """Master station names from the training config (no DB needed)."""
    from seismo_sbi.sbi.configuration import SBI_Configuration
    cfg = SBI_Configuration(); cfg.parse_config_file(config_path)
    return [rec.station_name for rec in cfg.sim_parameters.receivers.iterate()], cfg


def _exists(p):
    try:
        return p is not None and Path(p).exists()
    except Exception:  # noqa: BLE001
        return False


def main():
    args = parse_args()
    stages = _resolve_stages(args)
    do_events = "events" in stages
    do_dropout = "dropout" in stages
    do_validation = "validation" in stages

    num_samples = 300 if args.smoke else args.num_samples
    n_subsets = 2 if args.smoke else args.n_subsets
    n_val = 20 if args.smoke else args.n_val
    n_show = 2 if args.smoke else args.n_show
    val_num_samples = 300 if args.smoke else args.val_num_samples

    # Load the pluggable domain adapter on demand — nothing gitignored is imported
    # at module load.  Without --domain the per-event events/dropout stages are
    # unavailable; only the domain-agnostic validation stage can run.
    domain = load_domain(args.domain) if args.domain else None
    if domain is None and (do_events or do_dropout):
        print("[note] no --domain adapter -> skipping the events/dropout stages "
              "(they need a domain). Pass --domain <module-or-path>:<Class> for "
              "per-event evaluation; continuing with the validation stage only.\n")
        do_events = do_dropout = False

    master_names, _cfg = _master_station_names(args.config)
    events = domain.discover_events(set(args.events) if args.events else None) if domain else []
    print(f"Master station set ({len(master_names)}): {master_names}")
    print(f"Domain: {domain.name if domain else '(none — validation only)'}")
    print(f"Discovered {len(events)} event(s).")
    print(f"Stages: {sorted(stages)}\n")

    # ----- dry run: registry + station-set derivation + path checks only ----- #
    if args.dry_run:
        if domain is None:
            print("DRY RUN: no --domain -> nothing to enumerate; only the "
                  "(domain-agnostic) validation stage is available.")
            return
        dry_model_root = Path(args.output_root) / (args.model_name or Path(args.ckpt_dir).name)
        print(f"Per-model output root: {dry_model_root}\n")
        for spec in events:
            alls, filt = domain.station_sets(spec, master_names)
            print(f"== {spec.event} (job={spec.job_name}) ==")
            print(f"   h5            : {spec.h5_path}  exists={_exists(spec.h5_path)}")
            for k, v in (spec.extra or {}).items():
                if isinstance(v, (str, Path)):
                    print(f"   {k:<13} : {v}  exists={_exists(v)}")
            print(f"   all-available ({len(alls)}): {alls}")
            print(f"   filtered      ({len(filt)}): {filt}")
            print(f"   dropped-by-qa : {sorted(set(alls) - set(filt))}")
            print(f"   -> event out  : {dry_model_root / spec.event}\n")
        print("DRY RUN complete (no model loaded).")
        return

    # ----- full run: build the master pipeline + ML posterior once ----- #
    from seismo_sbi.sbi.scalers import build_flexible_scaler
    from seismo_sbi.sbi.compression.ML.station_dropout import (
        config_from_kept, sample_station_dropout_ensemble)
    import torch

    ckpt_dir = resolve_ckpt_dir(args.ckpt_dir)
    print(f"Resolved checkpoint dir: {ckpt_dir}")
    model_name = args.model_name or ckpt_dir.name
    layout = resolve_output_layout(args.output_root, model_name)
    print(f"Per-model output tree: {layout.model_root}")
    config, sbi_pipeline, original_parameters = build_eval_pipeline(args.config)
    posterior = build_ml_posterior(ckpt_dir, sbi_pipeline)
    data_loader = sbi_pipeline.data_manager.data_loader
    data_scaler = build_flexible_scaler(original_parameters, config.raw_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model ready (device={device}). Evaluating {len(events)} events, "
          f"{num_samples} samples/config.\n")

    # Conditioned model? The training YAML's `ml_conditioning.param_map` is the single
    # source of truth for both training and inference, so the per-event source vector
    # is derived from it (None ⇒ unconditioned ⇒ source_vec stays None). Both this and
    # the variable-station flag come from the already-parsed `config.raw_config` (no
    # second file read).
    raw_config = config.raw_config or {}
    _cond_cfg = raw_config.get("ml_conditioning") or {}
    cond_param_map = _cond_cfg.get("param_map") or None
    variable_stations = bool((raw_config.get("ml_variable_stations") or {}).get("enabled", False))
    if cond_param_map:
        print(f"Conditioned model: feeding each event's catalogue source location as "
              f"conditioning (param_map={cond_param_map}).\n")

    # Provenance sidecar at the model root — makes each per-model eval self-describing.
    with open(layout.model_root / "run_meta.json", "w") as f:
        json.dump({
            "model_name": model_name,
            "ckpt_dir": str(ckpt_dir),
            "config": args.config,
            "stages": sorted(stages),
            "num_samples": num_samples,
            "n_subsets": n_subsets,
            "keep_fraction": args.keep_fraction,
            "min_stations": args.min_stations,
            "seed": args.seed,
            "dropout": do_dropout,
            "conditioned": cond_param_map is not None,
            "param_map": cond_param_map,
            "master_stations": list(master_names),
            "domain": (domain.name if domain else None),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_rev": git_rev(HERE),
        }, f, indent=2)

    summary = {}
    per_event_stations = {}
    for spec in (events if do_events else []):
        event = spec.event
        out_dir = layout.event_dir(event)
        print(f"==================== {event} ====================")
        try:
            alls, filt = domain.station_sets(spec, master_names)
            if not alls:
                print(f"  [skip] no master stations present in {event}.")
                continue
            if not filt:
                filt = alls  # nothing QA-dropped within master → both versions identical
            print(f"  all-available ({len(alls)}): {alls}")
            print(f"  filtered      ({len(filt)}): {filt}")

            # Per-event source-conditioning vector (None for an unconditioned model).
            # Shared across this event's station configs — same source, different subsets.
            source_vec = domain.source_vec(spec, cond_param_map)

            # Load the all-available observation ONCE, then sample ML(all) + ML(filtered)
            # as two station configs (filtered ⊆ all-available) via the shared src loop.
            obs_all, coords_all = data_loader.load_event_subset(
                spec.h5_path, alls, stacked=True)            # (N,C,T), (N,2)
            cmp_configs = [config_from_kept(alls, alls, f"ML all (N={len(alls)})"),
                           config_from_kept(alls, filt, f"ML filtered (N={len(filt)})")]
            cmp_ens, _ = sample_station_dropout_ensemble(
                posterior, obs_all, coords_all, cmp_configs, data_scaler,
                num_samples=num_samples, device=device, event_name=event, source_vec=source_vec)
            ml_all = cmp_ens[cmp_configs[0].label].samples[:, :6]
            ml_filt = cmp_ens[cmp_configs[1].label].samples[:, :6]

            # Reference overlay (recovery lune + corner) + scalar summary are the domain's.
            figs = domain.reference_overlay(
                spec, {"ml_all": ml_all, "ml_filt": ml_filt,
                       "n_all": len(alls), "n_filt": len(filt)},
                original_parameters, data_scaler, out_dir)

            ev_summary = {"all_available": alls, "filtered": filt, "figures": figs}
            ev_summary.update(domain.event_summary(spec, ml_all, ml_filt))

            dropout_configs = None
            if do_dropout:
                names_all = alls
                d_figs, d_stats, dropout_configs = _event_dropout(
                    obs_all, coords_all, names_all, posterior, data_scaler, original_parameters,
                    out_dir, num_samples=num_samples, n_subsets=n_subsets,
                    keep_fraction=args.keep_fraction, min_stations=args.min_stations,
                    seed=args.seed, device=device, event=event, source_vec=source_vec)
                ev_summary["dropout_figures"] = d_figs
                ev_summary["dropout_spread"] = d_stats

            # Station-usage breakdown for this event (which stations each posterior uses).
            stations = write_station_breakdown(
                out_dir, event, master_names, alls, filt, dropout_configs)
            per_event_stations[event] = stations
            ev_summary["dropped_by_qa"] = stations["dropped_by_qa"]
            ev_summary["stations_json"] = str(out_dir / f"ml_{event}_stations.json")

            with open(out_dir / f"ml_{event}_summary.json", "w") as f:
                json.dump(ev_summary, f, indent=2)
            summary[event] = ev_summary
            ka = ev_summary.get("kagan_all_vs_ref_deg")
            kf = ev_summary.get("kagan_filtered_vs_ref_deg")
            if ka is not None and kf is not None:
                print(f"  done. Kagan(all)={ka:.1f}deg Kagan(filtered)={kf:.1f}deg")
            else:
                print(f"  done ({event}).")
        except Exception as e:  # noqa: BLE001  — one event must not abort the rest
            print(f"  [ERROR] {event}: {type(e).__name__}: {e}")
            traceback.print_exc()
            summary[event] = {"error": f"{type(e).__name__}: {e}"}

    if do_events:
        # Cross-event station-usage matrix (all vs filtered) at the model root.
        if per_event_stations:
            write_station_usage(layout.model_root, master_names, per_event_stations)

        top = layout.model_root / "ml_posttrain_summary.json"
        with open(top, "w") as f:
            json.dump({"model_name": model_name, "ckpt_dir": str(ckpt_dir), "config": args.config,
                       "conditioned": cond_param_map is not None, "param_map": cond_param_map,
                       "num_samples": num_samples, "events": summary}, f, indent=2)
        ok = [e for e, s in summary.items() if "error" not in s]
        print(f"\nPer-event eval complete: {len(ok)}/{len(summary)} events OK.")
        print(f"  model root   : {layout.model_root}")
        print(f"  summary      : {top}")
        print(f"  station usage: {layout.model_root / 'station_usage.csv'}")

    # Model-level validation + TARP coverage (held-out sims). Non-fatal: never abort a run.
    if do_validation:
        print(f"\n==================== MODEL VALIDATION / TARP ({model_name}) ====================")
        try:
            val = run_validation(
                sbi_pipeline, original_parameters, posterior, data_scaler,
                n_val=n_val, n_show=n_show, num_samples=val_num_samples, device=device,
                variable_stations=variable_stations, cond_param_map=cond_param_map)
            result = write_validation_outputs(
                val, layout.validation_dir(), original_parameters, data_scaler,
                num_samples=val_num_samples, conditioned=cond_param_map is not None,
                n_show=n_show)
            tarp_err = (result.get("metrics", {}).get("coverage", {}) or {}).get(
                "tarp_calibration_error")
            print(f"  validation done: n_val={val['n_val']} TARP_calib_err={tarp_err} "
                  f"-> {layout.validation_dir()}")
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] validation/TARP: {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
