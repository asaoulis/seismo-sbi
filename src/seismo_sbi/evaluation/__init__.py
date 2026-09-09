"""Evaluation harness: pipeline build, output layout, validation and station usage.

``inference`` builds the evaluation pipeline and posterior and loads a real observation;
``layout`` resolves the output tree; ``validation`` runs the held-out validation and TARP
coverage; ``station_usage`` writes the per-station breakdowns; ``moment_tensor`` holds the
tensor-comparison primitives; ``domain`` is the adapter contract for a study. Imports are lazy
inside functions, so importing this package pulls in neither torch nor the pipeline classes.
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
