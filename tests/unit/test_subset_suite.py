"""Unit tests for the station-subset disagreement suite engine (subset_suite.py).

Covers subset construction (LOO / mode split / half splits, the min_stations floor),
the suite runner's record schema, and the fake generator's planted misspecification
(deterministic, recoverable — the smoke-gate contract).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2]
           / "scripts" / "santorini_pathbreaker" / "lomax_catalogue"))

import subset_suite as ss   # noqa: E402


KEPT = ["CMBO", "SANT", "SAP3", "SNT1", "THERA", "AMGA", "APE", "IOSI", "MHLO"]


def test_suite_subsets_structure():
    subs = ss.suite_subsets(KEPT, event_key="ev1", n_half_splits=2, seed=0)
    assert subs["full"] == KEPT
    loo = [k for k in subs if k.startswith("loo_")]
    assert len(loo) == len(KEPT)
    for lbl in loo:
        sta = lbl.split("loo_", 1)[1]
        assert sta not in subs[lbl] and len(subs[lbl]) == len(KEPT) - 1
    # mode split present: 5 on-island + 4 off-island here
    assert set(subs["modeA"]) == {"CMBO", "SANT", "SAP3", "SNT1", "THERA"}
    assert set(subs["modeB"]) == {"AMGA", "APE", "IOSI", "MHLO"}
    # half splits partition the kept set
    assert sorted(subs["half0a"] + subs["half0b"]) == sorted(KEPT)


def test_suite_subsets_thin_mode_b_omitted():
    kept = ["CMBO", "SANT", "SAP3", "SNT1", "AMGA", "APE"]   # only 2 mode-B
    subs = ss.suite_subsets(kept, event_key="ev2")
    assert "modeA" not in subs and "modeB" not in subs


def test_suite_subsets_min_station_floor_blocks_loo():
    kept = ["CMBO", "SANT", "SAP3"]                          # LOO would leave 2 < 3
    subs = ss.suite_subsets(kept, event_key="ev3")
    assert not any(k.startswith("loo_") for k in subs)
    assert "half0a" not in subs                              # 3 < 2*min_stations


def test_suite_subsets_reproducible_half_splits():
    a = ss.suite_subsets(KEPT, event_key="evX", seed=3)
    b = ss.suite_subsets(KEPT, event_key="evX", seed=3)
    c = ss.suite_subsets(KEPT, event_key="evY", seed=3)
    assert a["half0a"] == b["half0a"]
    assert a["half0a"] != c["half0a"] or a["half1a"] != c["half1a"]


def _flat_sampler(n=200, seed=0):
    """Subset-independent sampler: same posterior whatever the subset."""
    rng = np.random.default_rng(seed)
    base = np.array([0.0, 1e16, -1e16, 0.0, 0.0, 0.0])

    def fn(names, label):
        return base + 3e14 * rng.standard_normal((n, 6))
    return fn


def test_run_suite_schema_and_null_behaviour():
    out = ss.run_suite(_flat_sampler(), KEPT, event_key="ev1", n_half_splits=1)
    assert set(out) == {"subsets", "stats"}
    st = out["stats"]
    assert st["n_loo"] == len(KEPT)
    # subset-independent posteriors => small whitened influences
    assert st["D_max_w"] < 1.0
    assert np.isfinite(st["mode_split"]["lune_w"])
    assert np.isfinite(st["H_med_w"])
    for k in ("jack_var_gamma", "jack_var_delta", "jack_var_Mw"):
        assert k in st
    full = out["subsets"]["full"]
    assert len(full["point6"]) == 6 and full["n_samples"] == 200


def test_fake_sampler_deterministic_and_plants_recoverable():
    meta = {"event_id": "20250210T000000", "ml": 4.0, "depth_km": 8.0,
            "lat": 36.5, "lon": 25.6, "t_days": 5.0,
            "stations_all": KEPT, "stations_used": KEPT}
    fn1 = ss.make_fake_subset_sampler(dict(meta), seed=1)
    fn2 = ss.make_fake_subset_sampler(dict(meta), seed=1)
    s1, s2 = fn1(KEPT, "full"), fn2(KEPT, "full")
    assert np.allclose(s1, s2)

    # plant assignment is deterministic per event_id
    a = ss.plant_assignment("ev-abc", plant_frac=0.5, seed=0)
    b = ss.plant_assignment("ev-abc", plant_frac=0.5, seed=0)
    assert a == b


def test_planted_station_raises_loo_influence():
    meta = {"event_id": "20250211T111111", "ml": 4.0, "depth_km": 8.0,
            "lat": 36.5, "lon": 25.6, "t_days": 5.0,
            "stations_all": KEPT, "stations_used": KEPT}
    # find an event key that plants a *station* under plant_frac=1.0
    m_clean = dict(meta)
    fn_clean = ss.make_fake_subset_sampler(m_clean, seed=2, plant_frac=0.0)
    out_clean = ss.run_suite(fn_clean, KEPT, event_key=meta["event_id"], seed=2)

    m_hot = dict(meta)
    hot_fn = None
    for bump in range(40):
        m_try = dict(meta, event_id=f"{meta['event_id']}_{bump}")
        fn = ss.make_fake_subset_sampler(m_try, seed=2, plant_frac=1.0,
                                         plant_scale=40.0)
        if m_try["_plant"]["planted"] == "station":
            m_hot, hot_fn = m_try, fn
            break
    assert hot_fn is not None
    out_hot = ss.run_suite(hot_fn, KEPT, event_key=m_hot["event_id"], seed=2)
    assert out_hot["stats"]["D_max_w"] > 3 * out_clean["stats"]["D_max_w"]
    assert out_hot["stats"]["D_max_station"] == m_hot["_plant"]["station"]


def test_write_and_load_record_roundtrip(tmp_path):
    rec = {"event_id": "e1", "stats": {"D_max_w": np.float64(1.5),
                                       "bad": float("nan")},
           "_samples": {"full": np.zeros((2, 6))}}
    ss.write_record(tmp_path / "e1.json", rec)
    loaded = ss.load_records(tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["stats"]["D_max_w"] == 1.5
    assert loaded[0]["stats"]["bad"] is None          # NaN -> null
    assert "_samples" not in loaded[0]                # private keys stripped


def test_covariates_block_types():
    c = ss.covariates_block(mw_hat=3.7, n_kept=12, n_comps=34, depth_km=9.0,
                            rms=1e-6, ml=3.9, flags={"neighbour_in_window": 1.0})
    assert c["neighbour_in_window"] is True
    assert c["log10_rms"] == pytest.approx(-6.0)


def test_obs_rms_ignores_zero_traces():
    obs = np.zeros((2, 3, 10))
    obs[0, 0, :] = 2.0
    assert ss.obs_rms(obs) == pytest.approx(2.0)


# ---- deep-ensemble channel (the PRIMARY) ------------------------------------

def _cloud6(rng, center, frac, n=250):
    import numpy as _np
    return center + frac * _np.linalg.norm(center) * rng.standard_normal((n, 6))


def test_ensemble_stats_agreement_and_divergence():
    rng = np.random.default_rng(20)
    dc = np.array([0.0, 1e16, -1e16, 0.0, 0.0, 0.0])
    iso = np.array([1e16, 1e16, 1e16, 0.0, 0.0, 0.0])
    agree = ss.ensemble_stats({"tcn": _cloud6(rng, dc, 0.03),
                               "cnn": _cloud6(rng, dc, 0.03),
                               "pno": _cloud6(rng, dc, 0.03)})
    assert agree["n_models"] == 3 and len(agree["pairs"]) == 3
    assert agree["E_med_w"] < 0.7
    diverge = ss.ensemble_stats({"tcn": _cloud6(rng, dc, 0.03),
                                 "cnn": _cloud6(rng, dc, 0.03),
                                 "pno": _cloud6(rng, 0.5 * dc + 0.5 * iso, 0.03)})
    assert diverge["E_max_w"] > 2.0
    assert diverge["E_max_w"] > agree["E_max_w"]
    assert np.isfinite(diverge["ens_spread_ratio"])


def test_ensemble_blind_to_coherent_data_shift():
    """A data-level error moves ALL members the same way => E stays small.

    This is the documented blind spot that keeps the station channel in the battery."""
    rng = np.random.default_rng(21)
    dc = np.array([0.0, 1e16, -1e16, 0.0, 0.0, 0.0])
    iso = np.array([1e16, 1e16, 1e16, 0.0, 0.0, 0.0])
    shifted = 0.5 * dc + 0.5 * iso           # very wrong, but identically wrong
    ens = ss.ensemble_stats({m: _cloud6(rng, shifted, 0.03)
                             for m in ("tcn", "cnn", "pno")})
    assert ens["E_med_w"] < 0.7


def test_symmetric_kl_matrix_gaussians():
    rng = np.random.default_rng(22)

    def gauss_lp(mu):
        return lambda x: -0.5 * np.sum((np.asarray(x) - mu) ** 2, axis=1)

    a = rng.standard_normal((400, 3))
    b = rng.standard_normal((400, 3)) + 3.0
    same = ss.symmetric_kl_matrix({"a": a, "b": a.copy()},
                                  {"a": gauss_lp(0.0), "b": gauss_lp(0.0)})
    apart = ss.symmetric_kl_matrix({"a": a, "b": b},
                                   {"a": gauss_lp(0.0), "b": gauss_lp(3.0)})
    assert abs(same["skl_med"]) < 0.2
    assert apart["skl_med"] > 5.0            # ~ d*mu^2 = 27, trimmed
    missing = ss.symmetric_kl_matrix({"a": a, "b": b}, {"a": gauss_lp(0.0), "b": None})
    assert missing["pairs"] is None


def test_fake_model_sampler_ood_plant_diverges_models():
    meta = {"event_id": "20250212T000000", "ml": 4.0, "depth_km": 8.0,
            "lat": 36.5, "lon": 25.6, "t_days": 5.0,
            "stations_all": KEPT, "stations_used": KEPT}
    # find an event id planted as "ood" under plant_frac=1.0
    m_ood = None
    for bump in range(60):
        m_try = dict(meta, event_id=f"{meta['event_id']}_{bump}")
        ss.make_fake_model_sampler(m_try, "m0", seed=3, plant_frac=1.0,
                                   plant_scale=30.0)
        if m_try["_plant"]["planted"] == "ood":
            m_ood = m_try
            break
    assert m_ood is not None
    full = {}
    for m in ("m0", "m1", "m2"):
        fn = ss.make_fake_model_sampler(dict(m_ood), m, seed=3, plant_frac=1.0,
                                        plant_scale=30.0)
        full[m] = fn(KEPT, "full")
    hot = ss.ensemble_stats(full)
    clean_meta = dict(meta, event_id="clean_ev")
    clean = ss.ensemble_stats({m: ss.make_fake_model_sampler(
        dict(clean_meta), m, seed=3, plant_frac=0.0)(KEPT, "full")
        for m in ("m0", "m1", "m2")})
    assert hot["E_med_w"] > 3 * clean["E_med_w"]
