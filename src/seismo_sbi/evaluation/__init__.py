"""Evaluation harness: pipeline build, held-out validation and tensor comparison.

``inference`` builds the evaluation pipeline and posterior and loads a real observation;
``validation`` runs the held-out validation and TARP coverage; ``moment_tensor`` holds the
tensor-comparison primitives. Imports are lazy inside functions, so importing this package
pulls in neither torch nor the pipeline classes.
"""
from seismo_sbi.evaluation.inference import (
    build_eval_pipeline,
    build_ml_posterior,
    resolve_ckpt_dir,
    load_real_observation,
    recovered_mt_samples,
)
from seismo_sbi.evaluation.validation import (
    run_validation,
    write_validation_outputs,
)
from seismo_sbi.evaluation.moment_tensor import (
    pyrocko_mt,
    kagan,
)

__all__ = [
    # inference
    "build_eval_pipeline",
    "build_ml_posterior",
    "resolve_ckpt_dir",
    "load_real_observation",
    "recovered_mt_samples",
    # validation
    "run_validation",
    "write_validation_outputs",
    # moment_tensor
    "pyrocko_mt",
    "kagan",
]
