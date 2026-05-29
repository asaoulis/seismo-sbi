#!/usr/bin/env python3
"""
stage_npe_checkpoint.py
=======================
Copy (or symlink) the best checkpoint produced by train_NPE.py into the
directory structure expected by MachineLearningCompressor / get_best_model()
in src/seismo_sbi/sbi/compression/ML/utils.py, i.e.:

    <examples_dir>/ml_models/<model_name>/ckpts/<checkpoint>.ckpt

train_NPE.py writes checkpoints at:
    <output_dir>/<run_name>/<job_name>/<run_name>/checkpoints/best_model-*.ckpt

This script bridges the gap so that run_ml_inversion.py can locate the
freshly-trained model with:
    --ckpt_dir examples/ml_models  --model_name <model_name>

Usage:
    python scripts/continuity/stage_npe_checkpoint.py \\
        --output_dir examples/data/pipeline_outputs/continuity_train \\
        --run_name   continuity_20261201 \\
        --job_name   results \\
        --model_name continuity_20261201 \\
        --examples_dir examples/

Arguments:
    --output_dir   output_directory from the YAML config (relative or absolute).
    --run_name     --run_name passed to train_NPE.py (same value used twice in path).
    --job_name     job_name from the YAML config (default: results).
    --model_name   target name under ml_models/<model_name>/ckpts/ (usually same
                   as run_name, but can differ).
    --examples_dir examples/ directory (default: auto-detected from repo root).
    --symlink      create a symlink instead of copying (default: copy).
    --dry_run      print what would be done without doing it.

Exit codes:
    0  success
    1  error (checkpoint not found, bad paths, etc.)
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Checkpoint discovery  (mirrors find_best_checkpoint_path in train.py)
# ---------------------------------------------------------------------------

def find_best_checkpoint(ckpt_dir: Path) -> Path:
    """
    Return the checkpoint with the lowest val_loss encoded in its filename.
    Pattern: best_model-val_loss=<float>[-v<N>].ckpt  or  best_model-<float>.ckpt
    """
    candidates = list(ckpt_dir.glob("best_model-*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No best_model-*.ckpt checkpoints found in: {ckpt_dir}")

    def _score(p: Path) -> float:
        # New-style: best_model-val_loss=0.12.ckpt
        m = re.search(r"best_model-val_loss=(-?\d+(?:\.\d+)?)(?:-v\d+)?\.ckpt$", p.name)
        if m:
            return float(m.group(1))
        # Old-style: best_model-0.12.ckpt
        m2 = re.search(r"best_model-(-?\d+(?:\.\d+)?)(?:-v\d+)?\.ckpt$", p.name)
        if m2:
            return float(m2.group(1))
        return float("inf")

    best = min(candidates, key=_score)
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage the best train_NPE.py checkpoint into ml_models/<model_name>/ckpts/."
    )
    p.add_argument("--output_dir", required=True,
                   help="output_directory from the training YAML config.")
    p.add_argument("--run_name", required=True,
                   help="--run_name passed to train_NPE.py.")
    p.add_argument("--job_name", default="results",
                   help="job_name in the YAML config (default: results).")
    p.add_argument("--model_name", default=None,
                   help="Target model_name under ml_models/. Defaults to --run_name.")
    p.add_argument("--examples_dir", default=None,
                   help="Path to examples/ directory. Defaults to <repo_root>/examples.")
    p.add_argument("--symlink", action="store_true",
                   help="Create a symlink instead of copying the checkpoint file.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would be done without actually doing it.")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Resolve examples_dir ------------------------------------------------
    if args.examples_dir:
        examples_dir = Path(args.examples_dir).resolve()
    else:
        # Auto-detect: this script lives at scripts/continuity/stage_npe_checkpoint.py
        repo_root = Path(__file__).resolve().parent.parent.parent
        examples_dir = repo_root / "examples"
    if not examples_dir.exists():
        print(f"ERROR: examples_dir not found: {examples_dir}", file=sys.stderr)
        sys.exit(1)

    # ---- Resolve source checkpoint directory ---------------------------------
    # train_NPE.py writes to:
    #   <output_dir>/<config_run_name>/<job_name>/<train_run_name>/checkpoints/
    # The config run_name is baked in the YAML; the --run_name arg is the *train*
    # run_name which appears at the inner level.  For our continuity configs the
    # YAML run_name == config.pipeline_parameters.run_name, but the script is
    # given --output_dir which already includes that prefix (matches
    # output_directory in the YAML).  The path built by train_NPE.py is:
    #
    #   data_path = Path(output_directory) / run_name_from_yaml / job_name
    #   trainer.train(run_name_from_arg, ..., output_path=data_path)
    #   -> checkpoints land at data_path / <run_name_from_arg> / checkpoints/
    #
    # The user supplies --output_dir=<output_directory> from the YAML (e.g.
    # ./data/pipeline_outputs/continuity_train).
    # For LV2_continuity_train.yaml: run_name=continuity_train, job_name=results.
    # So the full path is:
    #   <output_dir>/continuity_train/results/<run_name>/checkpoints/
    # But we don't have the YAML's run_name here; the user can supply it via
    # --job_name path override.  To keep things flexible we accept both forms:
    #   1. --output_dir already is the full parent of <run_name>/checkpoints/
    #      (i.e. includes the YAML run_name + job_name segments already)
    #   2. --output_dir is just output_directory and we auto-append
    #      <yaml_run_name>/<job_name>  — but we don't know yaml_run_name here.
    #
    # Simplest safe approach: try the full chain, then fall back to shortened.

    output_dir = Path(args.output_dir)
    run_name = args.run_name
    job_name = args.job_name
    model_name = args.model_name or run_name

    # Primary: output_dir / <run_name> / checkpoints/
    candidate1 = output_dir / run_name / "checkpoints"
    # Secondary: output_dir / <job_name> / <run_name> / checkpoints/
    candidate2 = output_dir / job_name / run_name / "checkpoints"
    # Tertiary: output_dir / checkpoints/ (if caller already resolved path)
    candidate3 = output_dir / "checkpoints"

    ckpt_dir = None
    for candidate in [candidate1, candidate2, candidate3]:
        if candidate.exists() and any(candidate.glob("best_model-*.ckpt")):
            ckpt_dir = candidate
            break

    if ckpt_dir is None:
        print(f"ERROR: could not locate checkpoints directory. Tried:", file=sys.stderr)
        for c in [candidate1, candidate2, candidate3]:
            print(f"  {c}", file=sys.stderr)
        print("Make sure train_NPE.py has completed and checkpoints exist.",
              file=sys.stderr)
        sys.exit(1)

    # ---- Find best checkpoint -----------------------------------------------
    try:
        best_ckpt = find_best_checkpoint(ckpt_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # ---- Destination ---------------------------------------------------------
    dest_dir = examples_dir / "ml_models" / model_name / "ckpts"

    print(f"Source checkpoint : {best_ckpt}")
    print(f"Destination dir   : {dest_dir}")
    print(f"Action            : {'symlink' if args.symlink else 'copy'}")

    if args.dry_run:
        print("\nDRY RUN — no files written.")
        sys.exit(0)

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / best_ckpt.name

    if dest_file.exists() or dest_file.is_symlink():
        dest_file.unlink()

    if args.symlink:
        dest_file.symlink_to(best_ckpt.resolve())
        print(f"Symlink created: {dest_file} -> {best_ckpt.resolve()}")
    else:
        shutil.copy2(best_ckpt, dest_file)
        print(f"Copied: {best_ckpt} -> {dest_file}")

    print("\nCheckpoint staged successfully.")
    print(f"Use with run_ml_inversion.py:")
    print(f"    --ckpt_dir {examples_dir / 'ml_models'}  --model_name {model_name}")
    sys.exit(0)


if __name__ == "__main__":
    main()
