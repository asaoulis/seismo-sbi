"""Stage an AxiSEM ensemble: one directory per member, ready for the mesher and the solver.

Reads the ensemble configuration, writes ``background_model.bm`` for the reference model and
every perturbed member (or ingests a set of ``.bm`` files somebody else produced), renders the
``inparam_basic`` / ``inparam_advanced`` the solver reads from the templates the configuration
names, and writes ``members.json`` with the submission arguments each member needs. Meshing and
solving happen elsewhere; this only prepares the inputs.
"""

import json
import re
from pathlib import Path

import yaml

from .model_io import read_bm
from .perturbed_models import BM_FILENAME, generate_ensemble

#: Solver input templates a member's inparam files are rendered from, overridable per ensemble
#: with the ``axisem.templates_dir`` configuration key.
DEFAULT_TEMPLATES_DIR = Path(__file__).resolve().parent / "inparams"


def _set_inparam_key(text: str, key: str, value) -> str:
    """Replace the value on the line ``^<key> ...`` (preserving the key)."""
    pattern = re.compile(rf"^({re.escape(key)})\s+\S.*$", re.MULTILINE)
    if not pattern.search(text):
        raise KeyError(f"key {key!r} not found in inparam template")
    return pattern.sub(rf"\1   {value}", text)


def _render_inparams(member_dir: Path, meshname: str, axisem_cfg: dict, ncpu: int,
                     templates_dir: Path):
    """Render inparam_basic + inparam_advanced into ``member_dir``."""
    basic = (templates_dir / "inparam_basic").read_text()
    basic = _set_inparam_key(basic, "SEISMOGRAM_LENGTH",
                             f"{float(axisem_cfg['seismogram_length'])}")
    basic = _set_inparam_key(basic, "MESHNAME", meshname)
    basic = _set_inparam_key(basic, "SIMULATION_TYPE", axisem_cfg["simulation_type"])
    basic = _set_inparam_key(basic, "ATTENUATION",
                             "true" if axisem_cfg.get("attenuation", True) else "false")
    (member_dir / "inparam_basic").write_text(basic)

    # The buffer must exceed the number of processors or the solver aborts, and a buffer above
    # 256 on a mesh below 10 s stalls the dump processors for hours; 128 is the safe default.
    advanced = (templates_dir / "inparam_advanced").read_text()
    override = axisem_cfg.get("netcdf_dump_buffer")
    if override:
        dump_buffer = int(override)
    else:
        dump_buffer = max(ncpu + 16, 128)
    advanced = _set_inparam_key(advanced, "NETCDF_DUMP_BUFFER", str(dump_buffer))
    (member_dir / "inparam_advanced").write_text(advanced)


def _ingest_bm_members(from_bm_dir, dry_run: bool):
    """``(out_dir, member entries)`` for a ``.bm`` ensemble produced outside this builder.

    Seeds are carried over from the set's own ``ensemble_manifest.json`` when it has one.
    """
    out_dir = Path(from_bm_dir).resolve()
    if not (out_dir / "fiducial" / BM_FILENAME).exists():
        raise SystemExit(f"{out_dir} has no fiducial/{BM_FILENAME}; not a .bm ensemble dir")
    member_ids = sorted(d.name for d in out_dir.iterdir()
                        if d.is_dir() and d.name.startswith("member_")
                        and (d / BM_FILENAME).exists())
    if not member_ids:
        raise SystemExit(f"{out_dir} contains no member_*/ {BM_FILENAME}")
    if dry_run:
        member_ids = member_ids[:1]
    seeds = {}
    set_manifest = out_dir / "ensemble_manifest.json"
    if set_manifest.exists():
        for entry in json.loads(set_manifest.read_text()).get("members", []):
            seeds[entry["id"]] = entry.get("seed")
    return out_dir, [{"id": mid, "seed": seeds.get(mid)} for mid in member_ids]


def _generate_bm_members(ensemble_cfg: dict, config_dir: Path, dry_run: bool):
    """``(out_dir, member entries)`` for a fresh perturbation ensemble around the reference."""
    fiducial_bm = Path(ensemble_cfg["fiducial_bm"])
    if not fiducial_bm.is_absolute():
        fiducial_bm = config_dir / fiducial_bm
    out_dir = Path(ensemble_cfg["out_dir"])
    n_members = 1 if dry_run else int(ensemble_cfg["n_members"])
    pert = ensemble_cfg["perturbation"]
    manifest = generate_ensemble(
        fiducial_bm, out_dir, n_members,
        vp_sigma=float(pert["vp_sigma"]),
        vs_sigma=float(pert["vs_sigma"]),
        width_sigma=float(pert["width_sigma"]),
        rho_mode=pert.get("rho_mode", "fixed"),
        max_depth_km=(float(pert["max_depth_km"])
                      if pert.get("max_depth_km") is not None else None),
        base_seed=int(pert.get("base_seed", 0)),
    )
    return out_dir, manifest["members"]


def build_ensemble(config_path, *, dry_run: bool = False, from_bm_dir=None, name=None) -> Path:
    """Stage every member under the configured ``out_dir``; returns the ``members.json`` path.

    ``dry_run`` stages the reference plus one member only. ``from_bm_dir`` ingests background
    models written elsewhere instead of perturbing the reference. ``name`` overrides the
    ensemble name the mesh names are built from.
    """
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    ens, ax, cl = cfg["ensemble"], cfg["axisem"], cfg["cluster"]

    ncpu = int(ax["ntheta_slices"]) * int(ax["nradial_slices"])
    if ncpu > int(cl["cpus_per_node"]):
        raise SystemExit(
            f"ntheta*nrad = {ncpu} exceeds cpus_per_node = {cl['cpus_per_node']}; "
            "the solver would not fit on one node."
        )
    templates_dir = Path(ax.get("templates_dir") or DEFAULT_TEMPLATES_DIR)
    if not templates_dir.is_absolute():
        templates_dir = config_path.parent / templates_dir

    if from_bm_dir:
        out_dir, manifest_members = _ingest_bm_members(from_bm_dir, dry_run)
        ens_name = name or ens.get("name") or out_dir.name
    else:
        out_dir, manifest_members = _generate_bm_members(ens, config_path.parent, dry_run)
        ens_name = name or ens["name"]

    submit_args = {
        "mesh_period": float(ax["dominant_period"]),
        "ntheta": int(ax["ntheta_slices"]),
        "nrad": int(ax["nradial_slices"]),
        "ncl": int(ax["coarsening_layers"]),
        "wall_time_hours": float(cl["walltime_hours"]),
        "partition": cl["partition"],
        "ncpu": ncpu,
    }

    members = []
    period_tag = f"{int(round(float(ax['dominant_period'])))}s"
    for member in [{"id": "fiducial", "seed": None}] + manifest_members:
        member_dir = out_dir / member["id"]
        meshname = f"{ens_name}_{member['id']}_{period_tag}"
        _render_inparams(member_dir, meshname, ax, ncpu, templates_dir)
        read_bm(member_dir / BM_FILENAME)
        members.append({"id": member["id"], "meshname": meshname,
                        "seed": member["seed"], "bm": BM_FILENAME})

    members_doc = {
        "ensemble_name": ens_name,
        "submit_args": submit_args,
        "cluster": {
            "host": cl["host"],
            "axisem_dir": cl["axisem_dir"],
            "ensemble_remote_dir": cl["ensemble_remote_dir"],
            "conda_env": cl["conda_env"],
            "max_in_flight": int(cl.get("max_in_flight", 1)),
        },
        "members": members,
    }
    members_path = out_dir / "members.json"
    members_path.write_text(json.dumps(members_doc, indent=2))
    return members_path
