"""Staging an AxiSEM ensemble from the checked-in example configuration."""

import json
from pathlib import Path

import pytest
import yaml

from seismo_sbi.simulators.axisem.build_ensemble import (
    DEFAULT_TEMPLATES_DIR, _set_inparam_key, build_ensemble,
)

EXAMPLE_CONFIG = Path(__file__).resolve().parents[2] / "examples/configs/axisem_ensemble.yaml"


@pytest.fixture
def staged(tmp_path):
    config = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    config["ensemble"]["out_dir"] = str(tmp_path / "ensemble")
    config["ensemble"]["fiducial_bm"] = str(EXAMPLE_CONFIG.parent
                                            / config["ensemble"]["fiducial_bm"])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return build_ensemble(config_path, dry_run=True)


def test_example_config_names_paths_that_exist():
    config = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    assert (EXAMPLE_CONFIG.parent / config["ensemble"]["fiducial_bm"]).exists()


def test_the_library_ships_the_solver_input_templates():
    assert (DEFAULT_TEMPLATES_DIR / "inparam_basic").exists()
    assert (DEFAULT_TEMPLATES_DIR / "inparam_advanced").exists()


def test_dry_run_stages_the_reference_and_one_member(staged):
    members = json.loads(staged.read_text())["members"]
    assert [entry["id"] for entry in members] == ["fiducial", "member_000"]
    for entry in members:
        member_dir = staged.parent / entry["id"]
        assert (member_dir / "background_model.bm").exists()
        assert (member_dir / "inparam_basic").exists()
        assert (member_dir / "inparam_advanced").exists()


def test_meshname_carries_the_ensemble_name_and_period(staged):
    members = json.loads(staged.read_text())["members"]
    assert members[0]["meshname"] == "example5s_fiducial_5s"


def test_submit_args_use_the_full_mesh_decomposition(staged):
    submit_args = json.loads(staged.read_text())["submit_args"]
    assert submit_args["ncpu"] == submit_args["ntheta"] * submit_args["nrad"]


def test_rendered_inparam_carries_the_configured_seismogram_length(staged):
    basic = (staged.parent / "fiducial" / "inparam_basic").read_text()
    assert "SEISMOGRAM_LENGTH   300.0" in basic


def test_setting_a_key_absent_from_the_template_raises():
    with pytest.raises(KeyError):
        _set_inparam_key("SIMULATION_TYPE   force\n", "NO_SUCH_KEY", "1")
