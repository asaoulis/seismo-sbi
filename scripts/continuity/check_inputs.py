#!/usr/bin/env python3
"""
check_inputs.py — Read-only existence guard for all precomputed LV2 artifacts.

Fails loudly (nonzero exit + clear error messages) if any required artifact is
missing. Must never trigger download, preprocessing, or simulation.

Usage:
    python scripts/continuity/check_inputs.py
    # or from examples/ cwd:
    python ../scripts/continuity/check_inputs.py
"""
import sys
from pathlib import Path

# Resolve repo root: this script lives at scripts/continuity/check_inputs.py
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES = REPO_ROOT / "examples"


def check(path: Path, label: str) -> bool:
    """Return True if path exists, else print error and return False."""
    if path.exists():
        print(f"  OK  {label}")
        return True
    else:
        print(f"  MISSING  {label}")
        print(f"           -> {path}")
        return False


def main():
    print("=" * 70)
    print("LV2 Continuity — Input Artifact Guard")
    print("=" * 70)
    print(f"REPO_ROOT  : {REPO_ROOT}")
    print(f"examples/  : {EXAMPLES}")
    print()

    failures = []

    # ------------------------------------------------------------------ #
    # 1. Preprocessed event HDF5
    # ------------------------------------------------------------------ #
    print("[1] Preprocessed event data")
    items = [
        (
            EXAMPLES / "data/preprocessed/LV2/events/LV2_noise_filt_20_50_1hz.h5",
            "data/preprocessed/LV2/events/LV2_noise_filt_20_50_1hz.h5",
        ),
    ]
    for path, label in items:
        if not check(path, label):
            failures.append(label)

    print()

    # ------------------------------------------------------------------ #
    # 2. CPS Green's-function caches (precomputed model perturbations)
    # ------------------------------------------------------------------ #
    print("[2] CPS Green's-function caches")
    gf_items = [
        (
            EXAMPLES / "data/models/LV2_perturbations/kappa_5.0",
            "data/models/LV2_perturbations/kappa_5.0/",
        ),
        (
            EXAMPLES / "data/models/LV2_perturbations/kappa_5.0_fiducial",
            "data/models/LV2_perturbations/kappa_5.0_fiducial/",
        ),
    ]
    for path, label in gf_items:
        if not check(path, label):
            failures.append(label)
        else:
            # Also verify they are non-empty directories
            contents = list(path.iterdir())
            if not contents:
                msg = f"{label} exists but is EMPTY"
                print(f"  EMPTY  {label}")
                failures.append(msg)

    print()

    # ------------------------------------------------------------------ #
    # 3. Config support files
    # ------------------------------------------------------------------ #
    print("[3] Config support files")
    config_items = [
        (EXAMPLES / "configs/stations.txt",                "configs/stations.txt"),
        (EXAMPLES / "configs/components.json",             "configs/components.json"),
        (EXAMPLES / "configs/LV2_station_shifts.json",     "configs/LV2_station_shifts.json"),
        (EXAMPLES / "configs/SoCal.plain.txt",             "configs/SoCal.plain.txt"),
        (EXAMPLES / "configs/LV2.yaml",                    "configs/LV2.yaml"),
    ]
    for path, label in config_items:
        if not check(path, label):
            failures.append(label)

    print()

    # ------------------------------------------------------------------ #
    # 4. Pre-trained ML checkpoint
    # ------------------------------------------------------------------ #
    print("[4] Pre-trained ML checkpoint")
    ckpt_items = [
        (
            EXAMPLES / "ml-checkpoints/checkpoints/best_model-LV2.ckpt",
            "ml-checkpoints/checkpoints/best_model-LV2.ckpt",
        ),
    ]
    for path, label in ckpt_items:
        if not check(path, label):
            failures.append(label)

    print()
    print("=" * 70)

    if failures:
        print(f"FAIL — {len(failures)} artifact(s) missing:")
        for f in failures:
            print(f"  - {f}")
        print()
        print("Do NOT run this workflow without all inputs precomputed.")
        print("Never re-download or re-preprocess LV2 data from within this script.")
        sys.exit(1)
    else:
        print("PASS — all required LV2 artifacts are present.")
        print("Safe to proceed with the continuity inversions.")
        sys.exit(0)


if __name__ == "__main__":
    main()
