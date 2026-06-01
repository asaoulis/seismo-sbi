"""
Regression tests for scripts/continuity/compare_to_baseline.py.

Locks shut the silent no-op bug: when a fresh summary shares no method keys
with the baseline, the comparison must NOT report a green "looks good" result.
Also covers the normal within-tolerance (green) and out-of-tolerance (flag)
paths.
"""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
COMPARE_PY = REPO / "scripts" / "continuity" / "compare_to_baseline.py"


def _load_compare_module():
    spec = importlib.util.spec_from_file_location("compare_to_baseline", COMPARE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


compare_mod = _load_compare_module()


def _entry(gamma=0.0, delta=40.0, mw=4.9, std=2.0, tol=15.0, mw_tol=0.3):
    def q(median):
        return {
            "median": median, "ci68_lo": median - std, "ci68_hi": median + std,
            "ci95_lo": median - 2 * std, "ci95_hi": median + 2 * std, "std": std,
        }
    return {
        "gamma_deg": q(gamma),
        "delta_deg": q(delta),
        "Mw": q(mw),
        "num_samples": 1000,
        "tolerances": {"gamma_deg": tol, "delta_deg": tol, "Mw": mw_tol, "std_ratio": 2.0},
    }


def test_disjoint_keys_report_no_checks_ran():
    """The core bug: zero overlapping methods must be reported, not passed."""
    baseline = {"LV2__theory_optimal_score": _entry()}
    summary = {"LV2__ml_compressor": _entry()}
    all_pass, n_compared = compare_mod.compare(summary, baseline)
    assert n_compared == 0


def test_disjoint_keys_exit_nonzero(tmp_path, capsys):
    """main() must exit nonzero (even without --strict) on a no-op comparison."""
    baseline = {"LV2__theory_optimal_score": _entry()}
    summary = {"LV2__ml_compressor": _entry()}
    bpath = tmp_path / "baseline.json"
    spath = tmp_path / "summary.json"
    bpath.write_text(json.dumps(baseline))
    spath.write_text(json.dumps(summary))

    import sys
    argv = sys.argv
    sys.argv = ["compare_to_baseline.py", "--summary", str(spath), "--baseline", str(bpath)]
    try:
        with pytest.raises(SystemExit) as exc:
            compare_mod.main()
    finally:
        sys.argv = argv
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "NO CHECKS RAN" in out


def test_matching_within_tolerance_passes():
    baseline = {"LV2__ml_compressor": _entry(gamma=-13.0, delta=38.0, mw=4.93)}
    summary = {"LV2__ml_compressor": _entry(gamma=-12.0, delta=39.0, mw=4.95)}
    all_pass, n_compared = compare_mod.compare(summary, baseline)
    assert n_compared == 1
    assert all_pass is True


def test_matching_outside_tolerance_flags():
    baseline = {"LV2__ml_compressor": _entry(gamma=-13.0, delta=38.0, mw=4.93)}
    # delta drifts > 15 deg
    summary = {"LV2__ml_compressor": _entry(gamma=-12.0, delta=60.0, mw=4.95)}
    all_pass, n_compared = compare_mod.compare(summary, baseline)
    assert n_compared == 1
    assert all_pass is False
