"""
Unit tests for seismo_sbi.plotting.evaluation — the dependency-light data-assembly
helpers and the TARP coverage wrapper. The heavy lune/chainconsumer rendering is
exercised in the e2e smoke (it needs basemap + a real ModelParameters), not here.
"""
import os
import pickle
from collections import namedtuple

import numpy as np
import pytest

from seismo_sbi.plotting import evaluation as ev


# Lightweight stand-ins matching the InversionResult / InversionData NamedTuple API.
FakeData = namedtuple("FakeData", ["theta0", "samples", "data_scaler", "compression_data"])
FakeCfg = namedtuple("FakeCfg", ["train_noise", "test_noise", "inversion_method"])
FakeResult = namedtuple("FakeResult", ["event_name", "inversion_data", "inversion_config"])


def _result(method, n=100, seed=0):
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(n, 6)) * 1e15
    return FakeResult("LV2", FakeData(rng.normal(size=6) * 1e15, samples, None, None),
                      FakeCfg("", "gaussian_filtered", method))


def test_load_inversion_pkl_both_shapes(tmp_path):
    ml = (None, None, [_result("ml_compressor")])
    gold = ([1], [2, 3], [_result("theory_optimal_score"),
                          _result("gaussian_likelihood_theory_optimal_score")])
    mlp = tmp_path / "ml.pkl"
    gp = tmp_path / "gold.pkl"
    mlp.write_bytes(pickle.dumps(ml))
    gp.write_bytes(pickle.dumps(gold))

    assert len(ev.load_inversion_pkl(mlp)) == 1
    assert len(ev.load_inversion_pkl(gp)) == 2


def test_load_inversion_pkl_bad_shape(tmp_path):
    p = tmp_path / "bad.pkl"
    p.write_bytes(pickle.dumps({"not": "a tuple"}))
    with pytest.raises(ValueError):
        ev.load_inversion_pkl(p)


def test_discover_ml_runs(tmp_path):
    for label in ("train_cnn_nonuisance", "train_cnn_nuisance"):
        d = tmp_path / f"continuity_ml_{label}"
        d.mkdir()
        (d / "inversion_results_ml.pkl").write_bytes(pickle.dumps((None, None, [])))
    # a non-matching dir should be ignored
    (tmp_path / "continuity_cnn_nonuisance").mkdir()
    runs = ev.discover_ml_runs(tmp_path)
    assert set(runs) == {"train_cnn_nonuisance", "train_cnn_nuisance"}


def test_build_recovery_dict_orders_gold_first():
    gold = [_result("theory_optimal_score"),
            _result("gaussian_likelihood_theory_optimal_score")]
    ml = {"train_cnn_nonuisance": [_result("ml_compressor")]}
    rec = ev.build_recovery_dict(gold, ml)
    keys = list(rec)
    assert keys[0] == "Optimal Score"        # gold score first (truth line source)
    assert keys[1] == "Gaussian"
    assert keys[2] == "NPE ML"                # single ML run -> bare label


def test_build_recovery_dict_multiple_ml_labelled():
    gold = [_result("theory_optimal_score")]
    ml = {"a": [_result("ml_compressor")], "b": [_result("ml_compressor")]}
    rec = ev.build_recovery_dict(gold, ml)
    assert "NPE ML (a)" in rec and "NPE ML (b)" in rec


def test_tarp_coverage_shapes():
    tarp = pytest.importorskip("tarp")  # noqa: F841
    rng = np.random.default_rng(1)
    n_samples, n_sims, n_dims = 200, 40, 3
    theta = rng.uniform(size=(n_sims, n_dims))
    # well-calibrated: samples centred on truth
    samples = theta[None, :, :] + rng.normal(scale=0.1, size=(n_samples, n_sims, n_dims))
    ecp, alpha = ev.tarp_coverage(samples, theta, num_bootstrap=20, seed=0)
    assert ecp.shape[0] == 20
    assert ecp.shape[1] == alpha.shape[0]


def test_tarp_coverage_rejects_bad_shape():
    pytest.importorskip("tarp")
    with pytest.raises(ValueError):
        ev.tarp_coverage(np.zeros((10, 3)), np.zeros((10, 3)))


# --------------------------------------------------------------------------- #
# Quantitative metrics
# --------------------------------------------------------------------------- #
def _synthetic_val(n_sims=400, n_samples=300, n_dims=3, scale=0.2, seed=0):
    """
    Well-calibrated Gaussian posteriors (no pyrocko needed). The posterior centre is
    offset from the truth by the posterior width (centre = truth + N(0, scale)), so the
    truth is distributed like a posterior draw — giving nominal credible-interval
    coverage and ~zero mean bias.
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(n_sims, n_dims))
    centre = theta + rng.normal(scale=scale, size=(n_sims, n_dims))
    samples = centre[None, :, :] + rng.normal(scale=scale, size=(n_samples, n_sims, n_dims))
    return {"theta_phys": theta, "samples_phys": samples}


def test_compute_metrics_structure_and_calibration():
    val = _synthetic_val(n_dims=3)
    m = ev.compute_evaluation_metrics(val)
    # structure
    assert set(m) >= {"meta", "mt_space", "figure_of_merit", "coverage"}
    assert m["meta"]["n_val"] == 400 and m["meta"]["n_dims"] == 3
    # per-param keys present for each of the 3 dims
    p0 = next(iter(m["mt_space"].values()))
    assert set(p0) == {"std", "bias", "mae", "rmse", "ci68", "ci90", "iqr"}
    # well-calibrated unbiased posterior: small bias, coverage near nominal
    biases = [v["bias"] for v in m["mt_space"].values()]
    assert all(abs(b) < 0.1 for b in biases)
    assert 0.55 < m["coverage"]["emp_68"] < 0.8
    assert 0.8 < m["coverage"]["emp_90"] < 0.99
    # non-6-dim => no derived space / δγ FoM, but full_mt FoM still computed
    assert m["derived"] == {}
    assert "full_mt" in m["figure_of_merit"]
    assert "delta_gamma" not in m["figure_of_merit"]


def test_figure_of_merit_matches_sqrt_det_cov():
    # one example, known covariance => sqrt(det) is exact (mean over a single sim)
    rng = np.random.default_rng(3)
    cov = np.array([[1.0, 0.3], [0.3, 2.0]])
    draws = rng.multivariate_normal([0, 0], cov, size=20000)
    val = {"theta_phys": np.zeros((1, 2)),
           "samples_phys": draws[:, None, :]}     # (n_samples, 1, 2)
    m = ev.compute_evaluation_metrics(val)
    expected = np.sqrt(np.linalg.det(cov))
    assert abs(m["figure_of_merit"]["full_mt"] - expected) < 0.05


def test_tarp_calibration_error_zero_for_perfect():
    val = _synthetic_val(n_dims=2)
    alpha = np.linspace(0, 1, 11)
    ecp = np.tile(alpha, (5, 1))               # perfectly calibrated (ecp == alpha)
    m = ev.compute_evaluation_metrics(val, ecp=ecp, alpha=alpha)
    assert m["coverage"]["tarp_calibration_error"] == pytest.approx(0.0, abs=1e-9)


def test_tarp_curve_persisted():
    val = _synthetic_val(n_dims=2)
    alpha = np.linspace(0, 1, 11)
    ecp = np.tile(alpha, (5, 1)) + np.linspace(-0.01, 0.01, 5)[:, None]
    m = ev.compute_evaluation_metrics(val, ecp=ecp, alpha=alpha)
    curve = m["coverage"]["tarp_curve"]
    assert len(curve["alpha"]) == 11
    assert len(curve["ecp_mean"]) == 11 and len(curve["ecp_std"]) == 11
    assert curve["ecp_mean"] == pytest.approx(alpha.tolist(), abs=1e-9)


def test_plot_cross_run_tarp_writes(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    alpha = list(np.linspace(0, 1, 11))
    all_metrics = {}
    for label, off in (("runA", 0.0), ("runB", 0.05)):
        all_metrics[label] = {"metrics": {"coverage": {"tarp_curve": {
            "alpha": alpha,
            "ecp_mean": [a + off for a in alpha],
            "ecp_std": [0.01] * 11}}}}
    # a run without a stored curve must be skipped gracefully
    all_metrics["runC"] = {"metrics": {"coverage": {}}}
    out = tmp_path / "out"
    p = ev.plot_cross_run_tarp("runA", all_metrics, out)
    assert p and os.path.exists(p)


def test_plot_cross_run_tarp_none_when_no_curves(tmp_path):
    assert ev.plot_cross_run_tarp("x", {"r": {"metrics": {"coverage": {}}}}, tmp_path) is None


def test_compute_metrics_derived_6mt():
    pytest.importorskip("pyrocko")
    rng = np.random.default_rng(7)
    theta = rng.normal(size=(6, 6)) * 1e15
    samples = theta[None] + rng.normal(scale=1e14, size=(200, 6, 6))
    m = ev.compute_evaluation_metrics(
        {"theta_phys": theta, "samples_phys": samples}, derived_max_samples=50)
    assert set(m["derived"]) == {"gamma", "delta", "Mw", "strike", "dip", "rake"}
    for key in ("delta_gamma", "sdr", "full_mt", "Mw"):
        assert key in m["figure_of_merit"]


def _write_metrics(path, label, ts, fom=1.0):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"schema_version": 1, "run_label": label, "config": "c.yaml",
               "timestamp": ts,
               "metrics": {"figure_of_merit": {"delta_gamma": fom, "full_mt": fom},
                           "coverage": {"emp_68": 0.68}}},
              open(path, "w"))


def test_scan_run_metrics_dedup_and_cap(tmp_path):
    # two runs, one with an older + newer copy (newer should win); + an unreadable file
    _write_metrics(tmp_path / "runA" / "artifacts" / ev.METRICS_FILENAME, "runA", 100, fom=5)
    _write_metrics(tmp_path / "runA_v2" / "artifacts" / ev.METRICS_FILENAME, "runA", 200, fom=9)
    _write_metrics(tmp_path / "runB" / "artifacts" / ev.METRICS_FILENAME, "runB", 150, fom=3)
    bad = tmp_path / "runC" / "artifacts" / ev.METRICS_FILENAME
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json")

    scanned = ev.scan_run_metrics(tmp_path)
    assert set(scanned) == {"runA", "runB"}                       # runC dropped (unreadable)
    assert scanned["runA"]["metrics"]["figure_of_merit"]["delta_gamma"] == 9  # newest wins
    # cap keeps the newest N
    capped = ev.scan_run_metrics(tmp_path, max_runs=1)
    assert set(capped) == {"runA"}                                # ts=200 newest


def test_plot_cross_run_comparison_writes(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    _write_metrics(tmp_path / "runA" / "artifacts" / ev.METRICS_FILENAME, "runA", 100, fom=5)
    _write_metrics(tmp_path / "runB" / "artifacts" / ev.METRICS_FILENAME, "runB", 150, fom=3)
    scanned = ev.scan_run_metrics(tmp_path)
    out = tmp_path / "out"
    bars = ev.plot_cross_run_comparison("runA", scanned, out)
    assert bars and all(os.path.exists(p) for p in bars.values())
