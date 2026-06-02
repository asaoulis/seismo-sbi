"""Unit tests for the AxiSEM background-model I/O and perturbation."""

import numpy as np
import pytest

from seismo_sbi.instaseis_simulator.axisem import (
    read_bm,
    write_bm,
    perturb_background_model,
    generate_ensemble,
)
from seismo_sbi.instaseis_simulator.axisem.model_io import BackgroundModel

FIDUCIAL_BM = (
    "/home/alex/work/seismo-sbi/scripts/axisem/madeira/background_model.bm"
)


@pytest.fixture
def fiducial():
    return read_bm(FIDUCIAL_BM)


def test_read_parses_header_and_data(fiducial):
    assert fiducial.columns == ["radius", "rho", "vpv", "vsv", "qka", "qmu"]
    assert fiducial.meta["NAME"] == "prem_iso"
    assert fiducial.meta["ANELASTIC"] == "T"
    assert fiducial.meta["ANISOTROPIC"] == "F"
    assert fiducial.meta["UNITS"] == "m"
    # Descending radius, surface at 6371 km.
    r = fiducial.radius
    assert r[0] == 6371000.0
    assert r[-1] == 0.0
    assert np.all(np.diff(r) <= 0)


def test_discontinuities_detected(fiducial):
    # The madeira PREM-ish model has duplicated-radius discontinuity rows.
    discs = fiducial.discontinuity_rows()
    assert len(discs) >= 8
    for i in discs:
        assert fiducial.radius[i] == fiducial.radius[i + 1]


def test_roundtrip_equality(fiducial, tmp_path):
    out = tmp_path / "rt.bm"
    write_bm(fiducial, out)
    reloaded = read_bm(out)
    assert reloaded.columns == fiducial.columns
    assert reloaded.meta == fiducial.meta
    np.testing.assert_allclose(reloaded.data, fiducial.data, rtol=0, atol=1e-6)
    # Discontinuity structure preserved exactly.
    assert reloaded.discontinuity_rows() == fiducial.discontinuity_rows()


def test_perturb_preserves_structure(fiducial):
    p = perturb_background_model(
        fiducial, vp_sigma=0.03, vs_sigma=0.03, width_sigma=0.02, seed=1
    )
    # Same number of rows + discontinuities.
    assert p.n_rows == fiducial.n_rows
    assert p.discontinuity_rows() == fiducial.discontinuity_rows()
    # Radius still strictly descending (non-increasing) and endpoints pinned.
    assert np.all(np.diff(p.radius) <= 0)
    assert p.radius[0] == fiducial.radius[0]
    assert p.radius[-1] == pytest.approx(fiducial.radius[-1], abs=1e-3)


def test_perturb_velocity_physical_guards(fiducial):
    p = perturb_background_model(
        fiducial, vp_sigma=0.1, vs_sigma=0.1, width_sigma=0.0, seed=2
    )
    vp = p.column("vpv")
    vs = p.column("vsv")
    # Fluid layers (outer core) stay fluid.
    fluid_fid = fiducial.column("vsv") == 0.0
    assert np.all(vs[fluid_fid] == 0.0)
    # Vs < Vp / sqrt(2) everywhere.
    assert np.all(vs <= vp / np.sqrt(2.0) + 1e-9)
    # Velocities actually changed for solid layers.
    assert not np.allclose(vp, fiducial.column("vpv"))


def test_perturb_is_seed_reproducible(fiducial):
    a = perturb_background_model(fiducial, vp_sigma=0.03, width_sigma=0.02, seed=7)
    b = perturb_background_model(fiducial, vp_sigma=0.03, width_sigma=0.02, seed=7)
    c = perturb_background_model(fiducial, vp_sigma=0.03, width_sigma=0.02, seed=8)
    np.testing.assert_array_equal(a.data, b.data)
    assert not np.array_equal(a.data, c.data)


def test_max_depth_perturbs_only_crust(fiducial):
    # Only nodes shallower than max_depth move; deeper nodes are identical.
    R = fiducial.radius.max()
    depth = (R - fiducial.radius) / 1000.0
    md = 50.0
    p = perturb_background_model(
        fiducial, vp_sigma=0.05, vs_sigma=0.05, width_sigma=0.05,
        rho_mode="brocher", max_depth_km=md, seed=5,
    )
    deep = depth >= md
    crust = ~deep
    np.testing.assert_array_equal(p.data[deep], fiducial.data[deep])   # deep fixed
    assert not np.allclose(p.column("vpv")[crust], fiducial.column("vpv")[crust])
    assert p.discontinuity_rows() == fiducial.discontinuity_rows()
    assert np.all(np.diff(p.radius) <= 0)


def test_perturbation_preserves_monotonicity(fiducial):
    # The fiducial crust is non-decreasing in Vp/Vs with depth; the increment-
    # based perturbation must never introduce a reversal (backward bending),
    # across many seeds and strong perturbations.
    R = fiducial.radius.max()
    depth = (R - fiducial.radius) / 1000.0
    md = 35.0
    k = int((depth < md).sum())
    for seed in range(50):
        p = perturb_background_model(
            fiducial, vp_sigma=0.08, vs_sigma=0.08, width_sigma=0.05,
            rho_mode="brocher", max_depth_km=md, seed=seed,
        )
        # crust + splice region (indices 0..k) must stay non-decreasing
        assert np.all(np.diff(p.column("vpv")[:k + 1]) >= -1e-6), f"Vp reversal seed {seed}"
        assert np.all(np.diff(p.column("vsv")[:k + 1]) >= -1e-6), f"Vs reversal seed {seed}"


def test_width_perturbation_disabled_keeps_radii(fiducial):
    p = perturb_background_model(fiducial, vp_sigma=0.0, vs_sigma=0.0,
                                 width_sigma=0.0, seed=3)
    np.testing.assert_array_equal(p.radius, fiducial.radius)


def test_generate_ensemble_layout(fiducial, tmp_path):
    manifest = generate_ensemble(
        FIDUCIAL_BM, tmp_path, n_members=3,
        vp_sigma=0.02, vs_sigma=0.02, width_sigma=0.01, base_seed=10,
    )
    assert (tmp_path / "fiducial" / "background_model.bm").exists()
    assert manifest["n_members"] == 3
    seeds = set()
    for m in manifest["members"]:
        mbm = tmp_path / m["id"] / "background_model.bm"
        assert mbm.exists()
        seeds.add(m["seed"])
        # Each member is a valid, loadable model.
        rm = read_bm(mbm)
        assert rm.n_rows == fiducial.n_rows
    assert seeds == {10, 11, 12}
    # Fiducial is unperturbed.
    fid = read_bm(tmp_path / "fiducial" / "background_model.bm")
    np.testing.assert_allclose(fid.data, fiducial.data, atol=1e-6)


def test_anisotropic_columns_generic(tmp_path):
    # A minimal anisotropic-style model round-trips on column names alone.
    cols = ["radius", "rho", "vpv", "vsv", "vph", "vsh", "eta", "qka", "qmu"]
    data = np.array([
        [6371000., 2600., 5800., 3200., 5800., 3200., 1.0, 57827., 600.],
        [6360000., 2600., 5800., 3200., 5800., 3200., 1.0, 57827., 600.],
        [6360000., 3300., 8000., 4500., 8100., 4550., 0.9, 57827., 200.],
        [6000000., 3400., 8100., 4600., 8200., 4650., 0.9, 57827., 200.],
    ])
    m = BackgroundModel(columns=cols, data=data,
                        meta={"NAME": "ani", "UNITS": "m",
                              "ANELASTIC": "T", "ANISOTROPIC": "T"})
    out = tmp_path / "ani.bm"
    write_bm(m, out)
    rm = read_bm(out)
    assert rm.columns == cols
    np.testing.assert_allclose(rm.data, data, atol=1e-6)
    assert rm.discontinuity_rows() == [1]
