"""Tests for the anisotropy injection effects (anisotropy-robustness task).

Contract (mirrors the module's other effects):
- key absent or 0.0 ⇒ strict identity (same arrays pass through);
- no input mutation;
- AzimuthalAnisotropyEffect: coherent cos 2φ delay pattern — fast-azimuth
  stations advanced, slow-azimuth stations delayed, magnitude ∝ distance and
  ∝ the nuisance multiplier; loud error when active without a source location;
- ShearSplittingEffect: Silver & Chan operator — identity for δt=0; a pulse
  polarised along the fast axis is unchanged; along the slow axis it is
  delayed by δt; Z untouched; energy preserved by the rotations.
"""
import numpy as np
import pytest

from seismo_sbi.instaseis_simulator.post_processing import (
    AzimuthalAnisotropyEffect, EFFECT_REGISTRY, ShearSplittingEffect,
    build_post_processing_chain)
from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers

SR = 1.0                          # samples per second, like the Santorini config
T = 256


def _gauss_pulse(center, width=6.0):
    t = np.arange(T, dtype=float)
    return np.exp(-0.5 * ((t - center) / width) ** 2)


def _receivers_at(*lat_lon_name):
    recs = [Receiver(lat, lon, "XX", name, ["Z", "E", "N"])
            for lat, lon, name in lat_lon_name]
    return Receivers(receivers=recs)


def _pulse_map(receivers):
    return {r.station_name: {c: _gauss_pulse(100.0) for c in ("Z", "E", "N")}
            for r in receivers.iterate()}


def _peak(trace):
    """Sub-sample peak position via quadratic interpolation."""
    i = int(np.argmax(trace))
    if 0 < i < len(trace) - 1:
        y0, y1, y2 = trace[i - 1], trace[i], trace[i + 1]
        return i + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return float(i)


# source at origin-ish; stations ~60 km north (az 0) and ~60 km east (az 90)
SRC = (36.0, 25.0)
RECS = _receivers_at((36.54, 25.0, "NORTH"), (36.0, 25.67, "EAST"))


class TestAzimuthalAnisotropyEffect:
    def _effect(self, **kw):
        kw.setdefault("fast_azimuth_deg", 0.0)      # fast axis = north
        kw.setdefault("aniso_fraction", 0.05)       # exaggerated for resolvable shifts
        kw.setdefault("ref_velocity_kms", 3.5)
        return AzimuthalAnisotropyEffect(SR, **kw)

    def test_identity_when_key_absent_or_zero(self):
        eff = self._effect()
        m = _pulse_map(RECS)
        assert eff(m, RECS) is m
        assert eff(m, RECS, azimuthal_anisotropy=0.0) is m

    def test_cos2phi_pattern_fast_advanced_slow_delayed(self):
        eff = self._effect(source_latitude=SRC[0], source_longitude=SRC[1])
        m = _pulse_map(RECS)
        out = eff(m, RECS, azimuthal_anisotropy=1.0)
        # NORTH station: azimuth 0 = fast axis -> advance (peak earlier)
        assert _peak(out["NORTH"]["Z"]) < 100.0 - 0.3
        # EAST station: azimuth 90 = slow axis -> delay (peak later)
        assert _peak(out["EAST"]["Z"]) > 100.0 + 0.3
        # magnitude ~ (D/V)*A: D~60km, V=3.5, A=0.05 -> ~0.86 s
        expected = 60.0 / 3.5 * 0.05
        assert abs((100.0 - _peak(out["NORTH"]["Z"])) - expected) < 0.35
        # all components of a station share the shift
        assert _peak(out["NORTH"]["E"]) == pytest.approx(_peak(out["NORTH"]["Z"]), abs=1e-6)

    def test_multiplier_scales_delay(self):
        eff = self._effect(source_latitude=SRC[0], source_longitude=SRC[1])
        m = _pulse_map(RECS)
        d1 = 100.0 - _peak(eff(m, RECS, azimuthal_anisotropy=1.0)["NORTH"]["Z"])
        d3 = 100.0 - _peak(eff(m, RECS, azimuthal_anisotropy=3.0)["NORTH"]["Z"])
        assert d3 == pytest.approx(3.0 * d1, rel=0.05)

    def test_source_location_from_nuisance_params_wins(self):
        eff = self._effect()                        # no constructor source
        m = _pulse_map(RECS)
        out = eff(m, RECS, azimuthal_anisotropy=1.0, source_location=SRC)
        assert _peak(out["NORTH"]["Z"]) < 100.0 - 0.3

    def test_active_without_source_location_raises(self):
        eff = self._effect()
        with pytest.raises(ValueError, match="source location"):
            eff(_pulse_map(RECS), RECS, azimuthal_anisotropy=1.0)

    def test_no_input_mutation(self):
        eff = self._effect(source_latitude=SRC[0], source_longitude=SRC[1])
        m = _pulse_map(RECS)
        before = {s: {c: v.copy() for c, v in comps.items()} for s, comps in m.items()}
        eff(m, RECS, azimuthal_anisotropy=1.0)
        for s in m:
            for c in m[s]:
                np.testing.assert_array_equal(m[s][c], before[s][c])

    def test_station_delays_cos2phi_zero_at_45deg(self):
        eff = self._effect(source_latitude=SRC[0], source_longitude=SRC[1])
        recs = _receivers_at((36.38, 25.47, "AZ45"))     # ~NE of source
        delays = eff.station_delays(recs, SRC)
        assert abs(delays["AZ45"]) < 0.1                 # node of cos 2phi


class TestShearSplittingEffect:
    def _effect(self, **kw):
        kw.setdefault("fast_azimuth_deg", 30.0)
        kw.setdefault("delay_s", 4.0)               # exaggerated, resolvable at 1 Hz
        return ShearSplittingEffect(SR, **kw)

    def test_identity_when_key_absent_or_zero(self):
        eff = self._effect()
        m = _pulse_map(RECS)
        assert eff(m, RECS) is m
        assert eff(m, RECS, shear_wave_splitting=0.0) is m

    def test_zero_delay_roundtrip_is_identity(self):
        eff = ShearSplittingEffect(SR, fast_azimuth_deg=30.0, delay_s=0.0)
        m = _pulse_map(RECS)
        out = eff(m, RECS, shear_wave_splitting=1.0)
        for s in out:
            np.testing.assert_allclose(out[s]["N"], m[s]["N"], atol=1e-10)
            np.testing.assert_allclose(out[s]["E"], m[s]["E"], atol=1e-10)

    def test_fast_polarised_pulse_unchanged_slow_delayed(self):
        phi = np.deg2rad(30.0)
        pulse = _gauss_pulse(100.0)
        m = {"STA": {"Z": pulse.copy(),
                     "N": np.cos(phi) * pulse, "E": np.sin(phi) * pulse}}
        out = self._effect()(m, RECS, shear_wave_splitting=1.0)
        # fast-polarised: unchanged
        np.testing.assert_allclose(out["STA"]["N"], m["STA"]["N"], atol=1e-8)
        np.testing.assert_allclose(out["STA"]["E"], m["STA"]["E"], atol=1e-8)
        # slow-polarised: delayed by delay_s * multiplier
        m2 = {"STA": {"Z": pulse.copy(),
                      "N": -np.sin(phi) * pulse, "E": np.cos(phi) * pulse}}
        out2 = self._effect()(m2, RECS, shear_wave_splitting=1.0)
        slow = -np.sin(phi) * out2["STA"]["N"] + np.cos(phi) * out2["STA"]["E"]
        assert _peak(slow) == pytest.approx(104.0, abs=0.05)
        # Z untouched
        np.testing.assert_array_equal(out2["STA"]["Z"], pulse)

    def test_station_missing_horizontal_passthrough(self):
        m = {"STA": {"Z": _gauss_pulse(100.0)}}
        out = self._effect()(m, RECS, shear_wave_splitting=1.0)
        np.testing.assert_array_equal(out["STA"]["Z"], m["STA"]["Z"])

    def test_no_input_mutation(self):
        m = _pulse_map(RECS)
        before = {s: {c: v.copy() for c, v in comps.items()} for s, comps in m.items()}
        self._effect()(m, RECS, shear_wave_splitting=1.0)
        for s in m:
            for c in m[s]:
                np.testing.assert_array_equal(m[s][c], before[s][c])


class TestRegistryAndChain:
    def test_keys_registered(self):
        assert EFFECT_REGISTRY["azimuthal_anisotropy"] is AzimuthalAnisotropyEffect
        assert EFFECT_REGISTRY["shear_wave_splitting"] is ShearSplittingEffect

    def test_chain_construction_and_identity_at_zero(self):
        chain = build_post_processing_chain(
            ["azimuthal_anisotropy", "shear_wave_splitting"],
            effect_configs={
                "azimuthal_anisotropy": {"sampling_rate": SR,
                                         "source_latitude": SRC[0],
                                         "source_longitude": SRC[1]},
                "shear_wave_splitting": {"sampling_rate": SR, "delay_s": 2.0},
            })
        assert len(chain.effects) == 2
        m = _pulse_map(RECS)
        out = chain(m, RECS, {"azimuthal_anisotropy": 0.0, "shear_wave_splitting": 0.0})
        assert out is m
