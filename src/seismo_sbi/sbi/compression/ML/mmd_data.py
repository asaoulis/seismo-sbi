"""Data assembly for the misspecification-robust MMD auxiliary loss.

Builds the two summary-space sample sets the loss compares (see ``mmd.py`` and the
``ml_mmd`` block in ``scripts/train_NPE.py``):

* the REAL side — QA-cleaned real events, packed exactly as at inference (per-event
  post-QA station/component masks, conditioning = catalogue location), one fixed
  context tensor built once from the QA run's ``mmd_manifest.json``;
* the PSIM side — a DataLoader over the posterior-matched simulation suite that
  reproduces the training augmentation path (fresh noise + amplitude draws per epoch)
  while pinning each sample's station/component availability to its PARENT real event
  (``TorchSimulationDataset(fixed_item_masks=...)``).

Conditioning-noise is deliberately OFF for the psim loader: the suite's true source
locations are already scattered around the catalogue values (by the generator), and the
conditioning vector is the exact catalogue location — reproducing ``cond - truth ~
coordinate_std`` exactly as the real events have it physically. Adding the training-time
conditioning perturbation on top would double-count the location error.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataloading import TorchSimulationDataset, variable_station_collate
from .source_conditioning import pack_variable_context


def load_mmd_manifest(manifest_path, *, clean_only=True, events_h5_dir=None,
                      solutions_root=None):
    """Read the QA run's ``mmd_manifest.json`` -> list of event entries.

    ``clean_only`` keeps only events with ``mmd_clean`` (no neighbour-in-window /
    contamination flag): an interloper wavetrain is contamination, not model
    misspecification, and would pollute the MMD target.

    The manifest stores ABSOLUTE paths from the machine that built it; on another
    host (e.g. the cluster) pass ``events_h5_dir`` / ``solutions_root`` to rebuild
    the per-event ``h5`` / ``samples`` paths as ``<events_h5_dir>/<event_id>.h5`` and
    ``<solutions_root>/<event_id>/samples.npy`` (these two keys are plain YAML leaves
    in the ``ml_mmd`` config block, so the cluster orchestrator's path remap covers
    them where it cannot rewrite paths embedded inside a JSON file).
    """
    doc = json.loads(Path(manifest_path).read_text())
    events = doc["events"] if isinstance(doc, dict) else doc
    if clean_only:
        events = [e for e in events if e.get("mmd_clean", True)]
    if events_h5_dir is not None:
        for e in events:
            e["h5"] = str(Path(events_h5_dir) / f"{e['event_id']}.h5")
    if solutions_root is not None:
        for e in events:
            e["samples"] = str(Path(solutions_root) / e["event_id"] / "samples.npy")
    if not events:
        raise ValueError(f"no usable events in {manifest_path} (clean_only={clean_only})")
    return events


def build_real_context(manifest_path, data_loader, *, clean_only=True,
                       events_h5_dir=None, torch_dtype=torch.float32):
    """Pack the QA-cleaned real events into one fixed context tensor (N_real, W).

    Each event is loaded through ``load_event_subset_with_components`` (per-station
    component subsets in receiver master order, dropped channels zero-filled), padded to
    the set-wide max station count, and packed with ``pack_variable_context`` — the same
    packing the variable-station collate applies to training batches, so the embedding
    net unpacks both identically. The conditioning vector is the event's catalogue
    location (NO perturbation — it IS the value inference conditions on).
    """
    events = load_mmd_manifest(manifest_path, clean_only=clean_only,
                               events_h5_dir=events_h5_dir)
    loaded = []
    for e in events:
        comp_map = {s: c for s, c in (e["components_used"] or {}).items() if c}
        if not comp_map:
            continue
        obs, coords, kept = data_loader.load_event_subset_with_components(
            str(e["h5"]), comp_map, stacked=True)
        sv = (torch.as_tensor(np.asarray(e["conditioning_vec"], float),
                              dtype=torch_dtype)
              if e.get("conditioning_vec") is not None else None)
        loaded.append((torch.as_tensor(obs, dtype=torch_dtype),
                       torch.as_tensor(coords, dtype=torch_dtype), sv))
    max_n = max(x.shape[0] for x, _, _ in loaded)
    packed = []
    for x, crd, sv in loaded:
        n, C, T = x.shape
        x_pad = x.new_zeros((max_n, C, T)); x_pad[:n] = x
        crd_pad = crd.new_zeros((max_n, 2)); crd_pad[:n] = crd
        mask = torch.zeros(max_n, dtype=torch.bool); mask[:n] = True
        packed.append(pack_variable_context(x_pad, crd_pad, mask, sv))
    return torch.stack(packed, dim=0)


def _psim_masks(paths, events_by_id, master_names, components):
    """Per-psim-file ``(keep_indices, zero_channels)`` from the parent event's QA map.

    psim files are named ``<parent_event_id>__<k>.h5``; the parent's
    ``components_used`` gives kept stations (-> master-order keep indices) and, for
    partially-kept stations, the dropped components (-> (master_station_idx,
    component_idx) zero-fill channels, exactly the QA zero-fill the real side gets).
    """
    name_to_idx = {n: i for i, n in enumerate(master_names)}
    comp_to_idx = {c: i for i, c in enumerate(components)}
    masks = []
    for p in paths:
        stem = Path(p).stem
        parent = stem.split("__")[0]
        e = events_by_id.get(parent)
        if e is None:
            raise KeyError(f"psim file {p} has no parent event {parent!r} in the manifest")
        comp_map = {s: c for s, c in (e["components_used"] or {}).items() if c}
        kept_names = [n for n in master_names if n in comp_map]   # master order
        keep = np.asarray([name_to_idx[n] for n in kept_names], dtype=int)
        zero = [(name_to_idx[s], comp_to_idx[c])
                for s in kept_names for c in components if c not in comp_map[s]]
        masks.append((keep, zero))
    return masks


def build_psim_loader(psim_folder, manifest_path, *, data_loader,
                      synthetic_noise_model_sampler,
                      augmentation_chain=None, augmentation_nuisance_params=None,
                      conditioning_param_map=None, batch_size=64, clean_only=True,
                      cache_in_memory=True, torch_dtype=torch.float32, seed=0):
    """DataLoader over the posterior-matched sim suite for the MMD's psim side.

    Reuses ``TorchSimulationDataset`` so noise + amplitude augmentation are EXACTLY the
    training path, with three deliberate differences (see module docstring + the task's
    ``training_implementation.md``): per-parent-event fixed masks instead of random
    station/component dropout, conditioning-noise OFF, and no theta (the MMD only needs
    contexts). ``num_workers=0`` — the suite is small and cached in RAM.
    """
    events = load_mmd_manifest(manifest_path, clean_only=clean_only)
    events_by_id = {e["event_id"]: e for e in events}
    master_names = [r.station_name for r in data_loader.receivers.iterate()]
    components = list(data_loader.components)

    import glob as _glob
    import os as _os
    paths = sorted(_glob.glob(_os.path.join(str(psim_folder), "*.h5")))
    # keep only sims whose parent survived the clean_only filter (same sorted order the
    # dataset will glob, so fixed_item_masks stays aligned with dataset indices)
    keep_paths = [p for p in paths if Path(p).stem.split("__")[0] in events_by_id]
    if len(keep_paths) < 2:
        raise ValueError(f"psim suite under {psim_folder} has {len(keep_paths)} usable "
                         f"sims (need >=2 for the unbiased MMD)")
    if len(keep_paths) != len(paths):
        # the dataset globs the folder itself — a mismatch means flagged parents' sims
        # are present; hide them via a glob that can't match, i.e. fail loudly instead.
        raise ValueError(
            f"{len(paths) - len(keep_paths)} psim sims under {psim_folder} belong to "
            "flagged/absent parent events — regenerate the suite from the same manifest "
            "filter (clean_only) so the folder and manifest agree")
    masks = _psim_masks(keep_paths, events_by_id, master_names, components)

    dataset = TorchSimulationDataset(
        data_loader=data_loader,
        data_folder=str(psim_folder),
        parameter_name_map={},                       # theta unused by the MMD
        synthetic_noise_model_sampler=synthetic_noise_model_sampler,
        data_scaler=None,
        augmentation_chain=augmentation_chain,
        augmentation_nuisance_params=augmentation_nuisance_params,
        conditioning_param_map=conditioning_param_map,
        conditioning_noise_std=None,                 # OFF by design (no double-count)
        station_subsampler=None,
        post_noise_augmentation_chain=None,          # masks replace random dropout
        cache_in_memory=cache_in_memory,
        torch_dtype=torch_dtype,
        fixed_item_masks=masks,
    )
    # Condition each psim sample on its PARENT's catalogue location, not the sim's own
    # (scattered) stored source_location — see the module docstring.
    if conditioning_param_map:
        dataset.fixed_conditioning = [
            np.asarray(events_by_id[Path(p).stem.split("__")[0]]["conditioning_vec"],
                       float)
            for p in keep_paths]
    gen = torch.Generator(); gen.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=True,
                      num_workers=0, collate_fn=variable_station_collate,
                      generator=gen, drop_last=False)
