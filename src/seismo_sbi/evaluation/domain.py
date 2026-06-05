"""
domain.py
=========
Pluggable *evaluation domain* contract for the generic evaluation CLI.

A "domain" supplies the four pieces of behaviour that differ between
deployments (Santorini, LV2/continuity, …):

  1. **event discovery** — what events exist and where their data / reference
     moment tensor / source location live;
  2. **station-set derivation** — for each event, the *all-available* and the
     QA-*filtered* station subsets (each a subset of the model's master set);
  3. **conditioning source vector** — the per-event raw source vector fed to a
     *conditioned* model at inference (``None`` for an unconditioned model);
  4. **reference overlay + scalar summary** — the recovery-lune / corner overlay
     of the ML posterior(s) against the domain's reference solution(s), and a
     scalar summary (e.g. Kagan angle vs the reference).

The public CLI (``scripts/evaluate_model.py``) is tracked, but each concrete
adapter (e.g. the Santorini one) may live in a *gitignored* area.  The CLI
therefore imports **no** adapter at module load: it calls :func:`load_domain`
with a ``--domain`` spec only when an event-level stage is requested, so a fresh
public clone runs ``--help``, ``--dry-run`` and the (domain-agnostic) validation
stage with zero adapter code present.

``EvalDomain`` is a :class:`typing.Protocol` (structural) — adapters need only
provide methods of the right shape; no inheritance required.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class EventSpec:
    """A single evaluation event, as a typed record.

    Attributes
    ----------
    event:
        Event identifier (used as the per-event output subdir name).
    job_name:
        ``jobs.real_events`` key (the observation handle the pipeline loads).
    h5_path:
        Path to the event observation h5 (``None`` if not yet processed).
    ref_mt:
        ``(6,)`` physical-unit reference moment tensor ``[Mrr,Mtt,Mpp,Mrt,Mrp,Mtp]``.
    ref_lat, ref_lon, ref_depth_km:
        Reference / catalogue source location (depth in km).  Fed as conditioning
        to a conditioned model via :meth:`EvalDomain.source_vec`.
    extra:
        Adapter-private payload (e.g. QA-verdict paths, components.json path,
        traditional-inversion pickle path).  The CLI never reads ``extra``.
    """

    event: str
    job_name: str
    h5_path: "str | None"
    ref_mt: np.ndarray
    ref_lat: float
    ref_lon: float
    ref_depth_km: float
    extra: dict = field(default_factory=dict)


@runtime_checkable
class EvalDomain(Protocol):
    """Structural contract an evaluation-domain adapter must satisfy."""

    name: str

    def discover_events(self, only: "set[str] | None") -> "list[EventSpec]":
        """Enumerate the domain's events (optionally restricted to ``only``)."""
        ...

    def station_sets(self, spec: EventSpec,
                     master_names: "list[str]") -> "tuple[list[str], list[str]]":
        """``(all_available, filtered)`` — each a subset of ``master_names`` in
        master order.  ``all_available`` = present in the event data; ``filtered``
        = the QA-kept subset the traditional inversions used."""
        ...

    def source_vec(self, spec: EventSpec,
                   cond_param_map: "dict | None") -> "np.ndarray | None":
        """Per-event conditioning vector in ``cond_param_map`` order, or ``None``
        when the model is unconditioned (``cond_param_map is None``)."""
        ...

    def reference_overlay(self, spec: EventSpec, ml_ensembles: dict,
                          parameters, data_scaler, out_dir: Path) -> dict:
        """Assemble ``{label: InversionData}`` (ML ensembles + the domain
        reference + any traditional solutions) and render the recovery lune +
        MT/nodal corner via the src plotting helpers.  Returns a figure-path
        dict.  ``ml_ensembles`` carries the already-sampled ML posteriors so the
        adapter does no sampling (keys ``ml_all``, ``ml_filt``, ``n_all``,
        ``n_filt``)."""
        ...

    def event_summary(self, spec: EventSpec, ml_all: np.ndarray,
                      ml_filt: np.ndarray) -> dict:
        """Domain scalar summary for one event (e.g. Kagan(all/filtered) vs the
        reference).  Free to return ``{}``."""
        ...


def load_domain(spec: str) -> EvalDomain:
    """Load an :class:`EvalDomain` adapter from a ``--domain`` spec string.

    ``spec`` is ``"<module>:<ClassName>"`` where ``<module>`` is either

      * a dotted import path on ``sys.path`` (``pkg.mod:Class``), or
      * a filesystem path to a ``.py`` file (``/abs/path/adapter.py:Class``) —
        detected by a trailing ``.py`` or a path separator.

    The adapter class is instantiated with **no arguments** (adapters read their
    own paths).  A file-path adapter's directory is prepended to ``sys.path`` so
    it can import its gitignored siblings.

    Raises a clear, actionable error if the module/class cannot be imported, so a
    public clone missing the adapter gets a useful message instead of a bare
    ``ModuleNotFoundError``.
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(
            f"--domain spec must be 'module:ClassName' or '/path/to/adapter.py:ClassName'; "
            f"got {spec!r}.")
    mod_part, _, cls_name = spec.rpartition(":")
    if not mod_part or not cls_name:
        raise ValueError(
            f"--domain spec must be 'module:ClassName' or '/path/to/adapter.py:ClassName'; "
            f"got {spec!r}.")

    is_path = mod_part.endswith(".py") or ("/" in mod_part) or ("\\" in mod_part)
    try:
        if is_path:
            path = Path(mod_part).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"adapter file not found: {path}")
            # Let the adapter import its gitignored siblings (diag_utils, …).
            parent = str(path.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            mod_spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(mod_spec)
            mod_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(mod_part)
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            f"Could not import evaluation domain adapter from --domain {spec!r}: "
            f"{type(exc).__name__}: {exc}. Provide a valid 'module:ClassName' on "
            f"sys.path or an absolute '/path/to/adapter.py:ClassName'."
        ) from exc

    try:
        cls = getattr(module, cls_name)
    except AttributeError as exc:
        raise ImportError(
            f"Domain adapter module {mod_part!r} has no attribute {cls_name!r} "
            f"(from --domain {spec!r})."
        ) from exc
    return cls()
