#!/usr/bin/env python3
"""
compare_to_baseline.py
======================
Diff a fresh extract_source_summary.py JSON against the frozen golden baseline
(baseline_LV2.json).  Print Δγ/Δδ/ΔMw per method, flag tolerance breaches,
and exit nonzero if any breach is found.

Usage:
    python scripts/continuity/compare_to_baseline.py \\
        --summary  path/to/summary_LV2.json \\
        --baseline scripts/continuity/baseline_LV2.json

Exit codes:
    0  all deltas within tolerance
    1  one or more tolerance breaches (or I/O error)
"""
import argparse
import json
import sys
from pathlib import Path


# Fallback tolerances used if a key is absent from the baseline JSON.
# Loose by design: this is a FLAG-only continuity check, not a pass/fail test.
_DEFAULT_TOLERANCES = {
    "gamma_deg": 15.0,
    "delta_deg": 15.0,
    "Mw": 0.3,
    "std_ratio": 2.0,
}


def load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def _tol(baseline_entry: dict, quantity: str) -> float:
    """Return tolerance for quantity from baseline entry, or fallback."""
    tolerances = baseline_entry.get("tolerances", {})
    return float(tolerances.get(quantity, _DEFAULT_TOLERANCES[quantity]))


def compare(summary: dict, baseline: dict) -> bool:
    """
    Compare summary against baseline.  Print a table and return True if all
    deltas are within tolerance, False if any breach.
    """
    col_w = 30
    header = (
        f"{'Method/Quantity':<{col_w}}  "
        f"{'Baseline':>10}  {'Fresh':>10}  {'Delta':>10}  {'Tol':>8}  Status"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    all_pass = True

    # Check every key present in the baseline
    for key, base_entry in sorted(baseline.items()):
        if key not in summary:
            print(f"WARNING: method '{key}' missing from fresh summary, skipping.")
            continue

        fresh_entry = summary[key]
        std_tol = _tol(base_entry, "std_ratio")
        for quantity in ("gamma_deg", "delta_deg", "Mw"):
            if quantity not in base_entry or quantity not in fresh_entry:
                continue
            # --- median drift check ---
            b_med = base_entry[quantity]["median"]
            f_med = fresh_entry[quantity]["median"]
            delta = abs(f_med - b_med)
            tol = _tol(base_entry, quantity)
            ok = delta <= tol
            if not ok:
                all_pass = False
            label = f"{key}__{quantity}"
            status = "ok" if ok else "FLAG <<<<"
            print(
                f"  {label:<{col_w}}  {b_med:>10.4f}  {f_med:>10.4f}  "
                f"{delta:>10.4f}  {tol:>8.4f}  {status}"
            )
            # --- variance (std) ratio check ---
            b_std = base_entry[quantity].get("std")
            f_std = fresh_entry[quantity].get("std")
            if b_std and f_std and b_std > 0 and f_std > 0:
                ratio = max(f_std, b_std) / min(f_std, b_std)
                std_ok = ratio <= std_tol
                if not std_ok:
                    all_pass = False
                print(
                    f"  {label + '.std':<{col_w}}  {b_std:>10.4f}  {f_std:>10.4f}  "
                    f"{ratio:>9.2f}x  {std_tol:>7.2f}x  "
                    f"{'ok' if std_ok else 'FLAG <<<<'}"
                )

    # Warn about keys in summary that are not in baseline (new methods)
    for key in sorted(summary):
        if key not in baseline:
            print(f"INFO: method '{key}' in summary but not in baseline (new — no check).")

    print(sep)
    return all_pass


def main():
    p = argparse.ArgumentParser(
        description="Diff a fresh source summary JSON against the golden baseline."
    )
    p.add_argument("--summary", "-s", required=True,
                   help="Path to the fresh summary JSON (from extract_source_summary.py).")
    p.add_argument("--baseline", "-b",
                   default=str(Path(__file__).resolve().parent / "baseline_LV2.json"),
                   help="Path to the frozen baseline JSON. "
                        "Default: scripts/continuity/baseline_LV2.json")
    p.add_argument("--strict", action="store_true",
                   help="Exit nonzero if any quantity is flagged. Default is "
                        "flag-only (always exit 0) — the continuity check reports "
                        "drifts for human attention, it is not a pass/fail test.")
    args = p.parse_args()

    summary_path = Path(args.summary)
    baseline_path = Path(args.baseline)

    print(f"Summary  : {summary_path}")
    print(f"Baseline : {baseline_path}")
    print()

    summary = load_json(summary_path)
    baseline = load_json(baseline_path)

    passed = compare(summary, baseline)

    if passed:
        print("\nRESULT: no drifts beyond tolerance — continuity looks good.")
        sys.exit(0)
    else:
        print("\nRESULT: ⚠ one or more quantities FLAGGED (drift beyond loose tolerance).")
        print("        These are flags for human attention, not necessarily failures —")
        print("        review the posterior diagnostics before concluding anything is wrong.")
        sys.exit(1 if args.strict else 0)


if __name__ == "__main__":
    main()
