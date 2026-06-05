"""
seismo_sbi.evaluation
=====================
Evaluation harness helpers: pipeline build, output layout, validation/TARP engine,
and station-usage writers.

All imports are lazy inside functions (heavy deps: torch, seismo_sbi pipeline
classes) so importing this package during the fast unit-test gate is cheap.

Public API
----------
From ``inference``:
    build_eval_pipeline, build_ml_posterior, resolve_ckpt_dir, load_real_observation

From ``layout``:
    OutputLayout, resolve_output_layout, git_rev

From ``validation``:
    run_validation, write_validation_outputs

From ``station_usage``:
    write_station_breakdown, write_station_usage

From ``moment_tensor``:
    pyrocko_mt, kagan

From ``domain``:
    EventSpec, EvalDomain, load_domain
"""
from seismo_sbi.evaluation.inference import (
    build_eval_pipeline,
    build_ml_posterior,
    resolve_ckpt_dir,
    load_real_observation,
    recovered_mt_samples,
)
from seismo_sbi.evaluation.layout import (
    OutputLayout,
    resolve_output_layout,
    git_rev,
)
from seismo_sbi.evaluation.validation import (
    run_validation,
    write_validation_outputs,
)
from seismo_sbi.evaluation.station_usage import (
    write_station_breakdown,
    write_station_usage,
)
from seismo_sbi.evaluation.moment_tensor import (
    pyrocko_mt,
    kagan,
)
from seismo_sbi.evaluation.domain import (
    EventSpec,
    EvalDomain,
    load_domain,
)

__all__ = [
    # inference
    "build_eval_pipeline",
    "build_ml_posterior",
    "resolve_ckpt_dir",
    "load_real_observation",
    "recovered_mt_samples",
    # layout
    "OutputLayout",
    "resolve_output_layout",
    "git_rev",
    # validation
    "run_validation",
    "write_validation_outputs",
    # station_usage
    "write_station_breakdown",
    "write_station_usage",
    # moment_tensor
    "pyrocko_mt",
    "kagan",
    # domain
    "EventSpec",
    "EvalDomain",
    "load_domain",
]
