"""Generate an ensemble of perturbed AxiSEM background models on disk.

Layout produced (consumed downstream by the cluster pipeline; the eventual
repacked Instaseis DBs mirror this member layout for
``InstaseisEnsembleSimulator``)::

    out_dir/
        fiducial/
            background_model.bm        # unperturbed reference
        member_000/
            background_model.bm        # perturbed
        member_001/
            ...
        ensemble_manifest.json
"""

from __future__ import annotations

import json
from pathlib import Path

from .model_io import read_bm, write_bm
from .perturb import perturb_background_model

BM_FILENAME = "background_model.bm"


def member_id(index: int) -> str:
    return f"member_{index:03d}"


def generate_ensemble(
    fiducial_bm,
    out_dir,
    n_members: int,
    *,
    vp_sigma: float = 0.02,
    vs_sigma: float = 0.02,
    width_sigma: float = 0.0,
    rho_mode: str = "fixed",
    max_depth_km: float = None,
    base_seed: int = 0,
    seeds=None,
) -> dict:
    """Write ``fiducial/`` plus ``n_members`` perturbed members under ``out_dir``.

    Returns the manifest dict (also written to ``ensemble_manifest.json``).
    Each member gets a distinct seed: either the supplied ``seeds`` list or
    ``base_seed + index``.
    """
    fiducial = read_bm(fiducial_bm)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if seeds is None:
        seeds = [base_seed + i for i in range(n_members)]
    elif len(seeds) != n_members:
        raise ValueError("len(seeds) must equal n_members")

    # Fiducial (unperturbed reference).
    fid_dir = out_dir / "fiducial"
    fid_dir.mkdir(exist_ok=True)
    write_bm(fiducial, fid_dir / BM_FILENAME)

    perturb_kwargs = dict(
        vp_sigma=vp_sigma, vs_sigma=vs_sigma,
        width_sigma=width_sigma, rho_mode=rho_mode, max_depth_km=max_depth_km,
    )

    members = []
    for i in range(n_members):
        mid = member_id(i)
        mdir = out_dir / mid
        mdir.mkdir(exist_ok=True)
        perturbed = perturb_background_model(fiducial, seed=seeds[i], **perturb_kwargs)
        write_bm(perturbed, mdir / BM_FILENAME)
        members.append({"id": mid, "seed": int(seeds[i])})

    manifest = {
        "fiducial_source": str(fiducial_bm),
        "n_members": n_members,
        "perturbation": perturb_kwargs,
        "members": members,
    }
    with open(out_dir / "ensemble_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest
