#!/usr/bin/env python3
"""Theta-scaler provenance: a checkpoint must carry the scaling it was trained under.

The failure this guards is silent. The inverse transform is rebuilt at inference from
whatever YAML is passed; edit `ml_scaler`/`bounds` after training and every recovered
moment is wrong by a constant factor, with no exception anywhere — it surfaces only as a
systematic magnitude offset in the finished catalogue.
"""
import numpy as np
import pytest

from seismo_sbi.sbi.scalers import (
    FlexibleScaler, MomentTensorScaler, build_flexible_scaler,
    check_scaler_provenance, scaler_provenance,
)
from seismo_sbi.sbi.types.parameters import ModelParameters

BOUNDS = [[-5e17] * 6, [5e17] * 6]


class _Holder:
    """Stands in for a FlexibleScaler carrying a MomentTensorScaler."""

    def __init__(self, mt=None):
        if mt is not None:
            self.mt = mt


def _mw_to_u(sc, mw):
    return ((1.5 * mw + 9.1) - sc.log10_m0_min) / (sc.log10_m0_max - sc.log10_m0_min)


def test_provenance_captures_the_resolved_window():
    sc = MomentTensorScaler(bounds=BOUNDS, n_decades=4.0)
    p = scaler_provenance(_Holder(sc))
    assert p["moment_tensor"] == "scale_shape"
    assert p["log10_m0_min"] == pytest.approx(sc.log10_m0_min)
    assert p["log10_m0_max"] == pytest.approx(sc.log10_m0_max)


def test_a_linear_scaler_is_recorded_as_such():
    assert scaler_provenance(_Holder())["moment_tensor"] == "linear"


def test_matching_scalers_pass():
    sc = MomentTensorScaler(bounds=BOUNDS, n_decades=4.0)
    h = _Holder(sc)
    assert check_scaler_provenance({"theta_scaler": scaler_provenance(h)}, h) is True


def test_provenance_may_live_under_model_config():
    """train_NPE records it inside model_config, which is what lands in model_meta.json."""
    sc = MomentTensorScaler(bounds=BOUNDS, n_decades=4.0)
    h = _Holder(sc)
    meta = {"model_config": {"theta_scaler": scaler_provenance(h)}}
    assert check_scaler_provenance(meta, h) is True


def test_a_changed_decade_count_is_caught(capsys):
    """The exact regression: retrain-free config edit that silently rescales every moment."""
    trained = scaler_provenance(_Holder(MomentTensorScaler(bounds=BOUNDS, n_decades=4.0)))
    now = _Holder(MomentTensorScaler(bounds=BOUNDS, n_decades=9.0))
    assert check_scaler_provenance({"theta_scaler": trained}, now) is False
    assert "MISMATCH" in capsys.readouterr().out


def test_a_changed_bounds_is_caught():
    trained = scaler_provenance(_Holder(MomentTensorScaler(bounds=BOUNDS, n_decades=4.0)))
    now = _Holder(MomentTensorScaler(bounds=[[-1e18] * 6, [1e18] * 6], n_decades=4.0))
    assert check_scaler_provenance({"theta_scaler": trained}, now) is False


def test_switching_scale_shape_to_linear_is_caught():
    trained = scaler_provenance(_Holder(MomentTensorScaler(bounds=BOUNDS, n_decades=4.0)))
    assert check_scaler_provenance({"theta_scaler": trained}, _Holder()) is False


def test_strict_mode_raises_instead_of_warning():
    trained = scaler_provenance(_Holder(MomentTensorScaler(bounds=BOUNDS, n_decades=4.0)))
    now = _Holder(MomentTensorScaler(bounds=BOUNDS, n_decades=9.0))
    with pytest.raises(ValueError, match="MISMATCH"):
        check_scaler_provenance({"theta_scaler": trained}, now, strict=True)


def test_a_legacy_checkpoint_without_provenance_warns_but_passes(capsys):
    """Checkpoints predating the record must keep loading — just not silently."""
    h = _Holder(MomentTensorScaler(bounds=BOUNDS, n_decades=4.0))
    assert check_scaler_provenance({}, h) is True
    assert "no theta_scaler provenance" in capsys.readouterr().out


def test_equivalent_windows_written_differently_compare_equal():
    """Provenance records the resolved NUMBERS, so `auto` == the same explicit window."""
    lo, hi = 1.5 * 3.5 + 9.1, 1.5 * 5.5 + 9.1
    a = _Holder(MomentTensorScaler(log10_m0_range=(lo, hi)))
    b = _Holder(MomentTensorScaler(bounds=[[-(10 ** hi) * np.sqrt(2)] * 6,
                                           [(10 ** hi) * np.sqrt(2)] * 6],
                                   n_decades=hi - lo))
    assert check_scaler_provenance({"theta_scaler": scaler_provenance(a)}, b) is True


# ---- the auto window itself ---------------------------------------------------

def test_auto_window_puts_the_prior_snugly_in_unit_interval():
    """What `mt_log_decades: auto` buys: no arbitrary cutoff, no wasted range."""
    sc = MomentTensorScaler(log10_m0_range=(1.5 * 3.5 + 9.1, 1.5 * 5.5 + 9.1))
    assert _mw_to_u(sc, 3.5) == pytest.approx(0.0, abs=1e-12)
    assert _mw_to_u(sc, 5.5) == pytest.approx(1.0, abs=1e-12)


def test_the_fixed_decade_window_wastes_range():
    """Documents what we are moving away from: the prior occupies only [0.20, 0.95]."""
    sc = MomentTensorScaler(bounds=BOUNDS, n_decades=4.0)
    assert _mw_to_u(sc, 3.5) == pytest.approx(0.200, abs=5e-3)
    assert _mw_to_u(sc, 5.5) == pytest.approx(0.950, abs=5e-3)


def test_auto_clips_below_the_prior_floor_where_the_fixed_window_did_not():
    """The coupling flagged in the config: `auto` removes the sub-prior headroom.

    ~30% of delivered events have a median Mw below the 3.5 floor, so this is not
    hypothetical — switching to `auto` without widening `mw_min` clamps them.
    """
    auto = MomentTensorScaler(log10_m0_range=(1.5 * 3.5 + 9.1, 1.5 * 5.5 + 9.1))
    fixed = MomentTensorScaler(bounds=BOUNDS, n_decades=4.0)
    assert _mw_to_u(auto, 3.24) < 0.0, "auto cannot represent Mw below the prior floor"
    assert 0.0 < _mw_to_u(fixed, 3.24) < 1.0, "the 4.0-decade window could"
    # and the clip in transform is what turns that into a silent clamp
    m6 = np.zeros((1, 6))
    m6[0, 0] = np.sqrt(2) * 10 ** (1.5 * 3.24 + 9.1)
    u = 2.0 * auto.transform(m6) - 1.0
    assert np.linalg.norm(u) == pytest.approx(0.0, abs=1e-9)


def test_round_trip_is_exact_for_both_windows():
    rng = np.random.default_rng(0)
    for sc in (MomentTensorScaler(bounds=BOUNDS, n_decades=4.0),
               MomentTensorScaler(log10_m0_range=(1.5 * 3.5 + 9.1, 1.5 * 5.5 + 9.1))):
        mw = rng.uniform(3.6, 5.4, 200)
        m6 = rng.normal(size=(200, 6))
        m6 *= (np.sqrt(2) * 10 ** (1.5 * mw + 9.1) / np.linalg.norm(m6, axis=1))[:, None]
        back = sc.inverse_transform(sc.transform(m6))
        assert np.abs(back / m6 - 1.0).max() < 1e-9


# ---------------------------------------------------------------------------
# Regression: provenance against a REAL FlexibleScaler, not the _Holder stub.
#
# Every test above hands `scaler_provenance` an object with the MomentTensorScaler as a
# direct attribute. A real FlexibleScaler does not look like that — it keeps its per-block
# sub-scalers in the LIST `self.scalers` — so the stub tests all passed while the function
# returned "linear" for every genuine scale_shape scaler in production. That is precisely
# the silent mis-recording this module exists to prevent, so it is pinned here.
# ---------------------------------------------------------------------------

def _mt_and_location_params():
    """Same two-block parameter set used by tests/unit/test_moment_tensor_scaler.py."""
    p = ModelParameters()
    p.names = {
        "source_location": ["latitude", "longitude", "depth", "time_shift"],
        "moment_tensor": ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"],
    }
    p.theta_fiducial = {
        "source_location": [36.5, 25.5, 10.0, 0.0],
        "moment_tensor": [1e15] * 6,
    }
    p.bounds = {
        "source_location": [[36.0, 25.0, 0.0, -2.0], [37.0, 26.0, 55.0, 2.0]],
        "moment_tensor": BOUNDS,
    }
    return p


def test_a_real_flexible_scaler_is_recorded_as_scale_shape():
    """The bug: sub-scalers live in a list, so a vars()-only scan found nothing."""
    scaler = FlexibleScaler(_mt_and_location_params(),
                            moment_tensor_scaling="scale_shape", mt_log_decades=4.0)
    mt = scaler.scalers[1]
    assert isinstance(mt, MomentTensorScaler)          # guards the block ordering assumption

    p = scaler_provenance(scaler)
    assert p["moment_tensor"] == "scale_shape", \
        "a real FlexibleScaler must not be recorded as 'linear'"
    assert p["log10_m0_min"] == pytest.approx(mt.log10_m0_min)
    assert p["log10_m0_max"] == pytest.approx(mt.log10_m0_max)


def test_a_real_linear_flexible_scaler_is_recorded_as_linear():
    scaler = FlexibleScaler(_mt_and_location_params())   # default: linear
    assert scaler_provenance(scaler)["moment_tensor"] == "linear"


def test_a_real_flexible_scaler_round_trips_through_check():
    scaler = FlexibleScaler(_mt_and_location_params(),
                            moment_tensor_scaling="scale_shape", mt_log_decades=4.0)
    meta = {"model_config": {"theta_scaler": scaler_provenance(scaler)}}
    assert check_scaler_provenance(meta, scaler) is True


def test_a_real_scale_shape_scaler_is_caught_against_a_linear_checkpoint(capsys):
    """Without the list scan this silently PASSED — both sides read 'linear'."""
    linear = FlexibleScaler(_mt_and_location_params())
    scale_shape = FlexibleScaler(_mt_and_location_params(),
                                  moment_tensor_scaling="scale_shape", mt_log_decades=4.0)
    ok = check_scaler_provenance({"theta_scaler": scaler_provenance(linear)}, scale_shape)
    assert ok is False
    assert "MISMATCH" in capsys.readouterr().out


# ---- the mw32 campaign's window, pinned end-to-end through build_flexible_scaler ----

def test_auto_window_from_a_mw_3p2_gutenberg_richter_prior():
    """`mt_log_decades: auto` + mw_min 3.2 / mw_max 5.5 => log10 M0 in [13.900, 17.350].

    This is the exact scaling the mw32 training campaign runs under, and the number that
    must appear in every one of its checkpoints' model_meta.json.
    """
    raw_config = {
        "ml_scaler": {"moment_tensor": "scale_shape", "mt_log_decades": "auto"},
        "simulations": {
            "sampling_method": {
                "moment_tensor": {
                    "type": "gutenberg_richter",
                    "b_value": 1.1, "mc": 3.2, "mw_min": 3.2, "mw_max": 5.5,
                    "magnitude_conversion": "identity",
                }
            }
        },
    }
    scaler = build_flexible_scaler(_mt_and_location_params(), raw_config)
    p = scaler_provenance(scaler)
    assert p["moment_tensor"] == "scale_shape"
    assert p["log10_m0_min"] == pytest.approx(13.900)     # 1.5 * 3.2 + 9.1
    assert p["log10_m0_max"] == pytest.approx(17.350)     # 1.5 * 5.5 + 9.1
    assert p["log10_m0_max"] - p["log10_m0_min"] == pytest.approx(3.45)


def test_the_old_and_new_prior_floors_produce_different_provenance():
    """mw_min 3.5 -> 3.2 must be visible in the recorded provenance, i.e. it is a
    retrain-requiring change and check_scaler_provenance will say so."""
    def _prov(mw_min):
        raw = {
            "ml_scaler": {"moment_tensor": "scale_shape", "mt_log_decades": "auto"},
            "simulations": {"sampling_method": {"moment_tensor": {
                "type": "gutenberg_richter", "b_value": 1.1, "mc": mw_min,
                "mw_min": mw_min, "mw_max": 5.5, "magnitude_conversion": "identity"}}},
        }
        return build_flexible_scaler(_mt_and_location_params(), raw)

    old, new = _prov(3.5), _prov(3.2)
    assert scaler_provenance(old) != scaler_provenance(new)
    assert check_scaler_provenance({"theta_scaler": scaler_provenance(old)}, new) is False
