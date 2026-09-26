"""Posterior summaries of the LV2 inversions: gamma, delta and Mw per method.

``summarise_pkl`` reads an ``inversion_results.pkl`` (``(job_data, job_results, results)``, as
event_inversion.py and run_ml_inversion.py write it) and adds, per ``<event>__<method>``, the
median, 68 and 95 per cent intervals and standard deviation of each quantity. Run as a script
it writes them to a JSON file; ``--add_tolerances`` embeds the loose drift tolerances a golden
baseline carries.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

GAMMA_TOLERANCE_DEG = 15.0
DELTA_TOLERANCE_DEG = 15.0
MW_TOLERANCE = 0.3
#: A posterior more than this many times wider or narrower than the baseline is flagged.
STD_RATIO_TOLERANCE = 2.0


def load_pkl(pkl_path: Path):
    """Load (job_data, job_results, inversion_results) from a pkl file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not (isinstance(data, (tuple, list)) and len(data) == 3):
        raise ValueError(
            f"{pkl_path}: expected (job_data, job_results, inversion_results), "
            f"got {type(data)} of length {len(data) if hasattr(data, '__len__') else '?'}"
        )
    _job_data, _job_results, inversion_results = data
    return inversion_results


def percentile_ci(arr, lo=16.0, hi=84.0):
    """Return (lo_pct, median, hi_pct) for arr."""
    return (
        float(np.percentile(arr, lo)),
        float(np.percentile(arr, 50.0)),
        float(np.percentile(arr, hi)),
    )


def compute_summary(samples: np.ndarray) -> dict:
    """
    Compute gamma/delta/Mw statistics from moment tensor samples.

    Parameters
    ----------
    samples : np.ndarray, shape (N, 6)
        Rows are moment tensor samples [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
        in the parametrisation used by the pipeline (up-south-east ordering
        matching ``seismo_sbi.utils.mt_conventions.create_matrix``).

    Returns
    -------
    dict with keys 'gamma_deg', 'delta_deg', 'Mw'.
    """
    from seismo_sbi.plotting.lune import mts6_to_gamma_delta
    from seismo_sbi.plotting.distributions import get_MW_and_epsilon

    if samples.ndim != 2 or samples.shape[1] != 6:
        raise ValueError(
            f"Expected samples shape (N, 6), got {samples.shape}. "
            "Check that the pkl contains moment-tensor-only samples."
        )

    gamma, delta = mts6_to_gamma_delta(samples)
    mw_arr = np.array([get_MW_and_epsilon(s)[0] for s in samples])

    def _ci_dict(arr):
        lo68, med, hi68 = percentile_ci(arr, 16.0, 84.0)
        lo95, _, hi95 = percentile_ci(arr, 2.5, 97.5)
        return {
            "median": float(med),
            "ci68_lo": float(lo68),
            "ci68_hi": float(hi68),
            "ci95_lo": float(lo95),
            "ci95_hi": float(hi95),
            "std": float(np.std(arr)),
        }

    return {
        "gamma_deg": _ci_dict(gamma),
        "delta_deg": _ci_dict(delta),
        "Mw": _ci_dict(mw_arr),
        "num_samples": int(samples.shape[0]),
    }


def summarise_pkl(pkl_path: Path, output: dict, add_tolerances: bool):
    """Extract summaries from one pkl and merge into output dict."""
    inversion_results = load_pkl(pkl_path)
    if not inversion_results:
        print(f"  WARNING: {pkl_path} contains no InversionResults, skipping.")
        return

    for result in inversion_results:
        event_name = result.event_name
        method = result.inversion_config.inversion_method
        key = f"{event_name}__{method}"

        samples = result.inversion_data.samples
        if samples is None or len(samples) == 0:
            print(f"  WARNING: {key} has no samples, skipping.")
            continue

        print(f"  Processing {key} ({len(samples)} samples)...")
        try:
            summary = compute_summary(np.asarray(samples))
        except Exception as e:
            print(f"  ERROR computing summary for {key}: {e}")
            continue

        summary["note"] = str(pkl_path)

        if add_tolerances:
            summary["tolerances"] = {
                "gamma_deg": GAMMA_TOLERANCE_DEG,
                "delta_deg": DELTA_TOLERANCE_DEG,
                "Mw": MW_TOLERANCE,
                "std_ratio": STD_RATIO_TOLERANCE,
            }

        output[key] = summary
        print(f"    gamma={summary['gamma_deg']['median']:.2f}°  "
              f"delta={summary['delta_deg']['median']:.2f}°  "
              f"Mw={summary['Mw']['median']:.3f}")


def main():
    p = argparse.ArgumentParser(
        description="Compute gamma/delta/Mw summaries from inversion pkl(s)."
    )
    p.add_argument("--pkl", action="append", required=True, metavar="PKL",
                   help="Path to an inversion_results.pkl (repeat for multiple files).")
    p.add_argument("--output", "-o", required=True,
                   help="Path to write the summary JSON.")
    p.add_argument("--add_tolerances", action="store_true",
                   help="Embed tolerance values in the JSON for use as a golden baseline.")
    args = p.parse_args()

    output = {}

    for pkl_path_str in args.pkl:
        pkl_path = Path(pkl_path_str)
        if not pkl_path.exists():
            print(f"ERROR: pkl file not found: {pkl_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading {pkl_path} ...")
        summarise_pkl(pkl_path, output, args.add_tolerances)

    if not output:
        print("ERROR: no summaries were produced (all pkls empty?)", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(output)} method-summaries to: {out_path}")
    if args.add_tolerances:
        print(f"Tolerances embedded: gamma={GAMMA_TOLERANCE_DEG}°, "
              f"delta={DELTA_TOLERANCE_DEG}°, Mw={MW_TOLERANCE}.")


if __name__ == "__main__":
    main()
