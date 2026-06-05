"""
inference.py
============
Pipeline / posterior / observation build helpers for the evaluation harness.

Lifted verbatim (behaviour-preserving) from
``scripts/continuity/_eval_inference.py``.  The only change is the rename
``build_continuity_pipeline`` → ``build_eval_pipeline`` (the function is not
continuity-specific; both Santorini and LV2 domains use it).

Not a CLI — imported by evaluation drivers and the new
``src/seismo_sbi/evaluation/`` package.

Heavy deps (seismo_sbi pipeline, torch) are imported lazily inside each
function so importing this module during the fast unit-test gate is cheap.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np


def build_eval_pipeline(config_path, *, setup_training_noise=False,
                        regenerate_dataset=False):
    """
    Parse a YAML config and build a fully-loaded SingleEventPipeline, mirroring
    exactly what the evaluation drivers (LV2 and Santorini) do.

    Returns ``(config, sbi_pipeline, original_parameters)``.

    If ``setup_training_noise`` is True, also primes the training noise sampler's
    adaptive covariance from the first jobs.real_events entry (needed to draw
    validation examples with the same on-the-fly noise the model trained on).

    By default (``regenerate_dataset=False``) the existing simulation dataset on
    disk is REUSED rather than regenerated.  ``simulate_test_jobs`` always rewrites
    the full ``random_events`` set from scratch (run_and_save_simulations does not
    skip existing files), which for a continuity run means redrawing 40k–75k sims
    on every evaluation — minutes of pure waste, and it clobbers the very dataset
    the model trained on.  For evaluation we only need the existing sims (for the
    held-out validation tail) plus the compressors, so we glob what's there.
    """
    from pathlib import Path as _Path
    from seismo_sbi.sbi.configuration import SBI_Configuration
    from seismo_sbi.sbi.pipeline import SingleEventPipeline

    config = SBI_Configuration()
    config.parse_config_file(config_path)

    sbi_pipeline = SingleEventPipeline(config.pipeline_parameters, config_path)
    sbi_pipeline.compression_methods = config.compression_methods
    sbi_pipeline.load_seismo_parameters(
        config.sim_parameters, config.model_parameters, config.dataset_parameters
    )
    original_parameters = deepcopy(sbi_pipeline.parameters)

    # Snapshot the config-default receiver time shifts NOW, before any compressor
    # mutates them.  set_time_shifts() *replaces* the shared receivers map, and the
    # theory-covariance estimator (built in load_compressors below) zeroes it
    # (EnsembleTheoryCovarianceEstimationSimulator.__init__).  So reading
    # receiver_time_shifts_map after the build can return {station: 0} instead of
    # the config defaults — which would silently turn the NPE time-shift undo into
    # a no-op.  Stash the true defaults so load_real_observation can invert them.
    sbi_pipeline._default_receiver_time_shifts = dict(
        sbi_pipeline.simulation_parameters.receivers.receiver_time_shifts_map)

    existing_sims = sorted(
        _Path(sbi_pipeline.simulations_output_path).glob("random_event_*.h5"))
    if regenerate_dataset or not existing_sims:
        if not existing_sims:
            print("No existing sims found — generating the dataset.")
        test_jobs_paths = sbi_pipeline.simulate_test_jobs(
            config.dataset_parameters, config.test_job_simulations
        )
    else:
        print(f"Reusing {len(existing_sims)} existing sims at "
              f"{sbi_pipeline.simulations_output_path} (no regeneration).")
        test_jobs_paths = existing_sims
    sbi_pipeline.compute_data_vector_properties(test_jobs_paths, config.real_event_jobs)
    score_compression_data, extra_gradients = sbi_pipeline.compute_required_compression_data(
        config.compression_methods,
        config.model_parameters,
        rerun_if_stencil_exists=config.pipeline_parameters.generate_dataset,
    )
    sbi_pipeline.load_compressors(
        config.compression_methods, score_compression_data,
        extra_gradients=extra_gradients, freeze=True,
    )
    sbi_pipeline.load_test_noises(config.sbi_noise_model, config.test_noise_models)

    if setup_training_noise:
        if not config.real_event_jobs:
            raise ValueError("setup_training_noise requires a jobs.real_events entry.")
        real_noise_path = next(iter(config.real_event_jobs.values()))
        cov_data = sbi_pipeline.data_manager.load_noise_parametrisation_data(real_noise_path)
        sbi_pipeline.training_noise_sampler.set_adaptive_covariance_with_misc_data(cov_data)

    return config, sbi_pipeline, original_parameters


def build_ml_posterior(ckpt_dir, sbi_pipeline, dim=256):
    """
    Rebuild the ML compressor + neural posterior from a checkpoint directory,
    reading the architecture from the checkpoint's model_meta.json sidecar so any
    encoder (cnn / pno / tcn) reloads correctly.  Returns the sbi DirectPosterior.
    """
    import json as _json
    from seismo_sbi.sbi.compression.ML.train import CompressionTrainer

    ckpt_dir = Path(ckpt_dir)
    components = sbi_pipeline.data_manager.data_loader.components
    station_locations = sbi_pipeline.simulation_parameters.receivers.get_station_locations_array()

    meta_path = ckpt_dir / "model_meta.json"
    meta = _json.load(open(meta_path)) if meta_path.exists() else {}
    trainer = CompressionTrainer(
        components, station_locations, dim, dim,
        trace_length=meta.get("trace_length", sbi_pipeline.trace_length),
        architecture=meta.get("architecture", "seismogram_transformer"),
        model_config=meta.get("model_config"),
        flow_config=meta.get("flow_config"),
    )
    trainer.load_best(str(ckpt_dir))
    return trainer.build_posterior()


def resolve_ckpt_dir(ckpt_dir) -> Path:
    """Return the directory that actually holds model_meta.json (+ checkpoints/).

    Accepts either that directory directly or any ancestor of it (the training
    output_directory nests the run dir a few levels deep).  Shared by the
    continuity and santorini eval drivers.
    """
    ckpt_dir = Path(ckpt_dir)
    if (ckpt_dir / "model_meta.json").exists():
        return ckpt_dir
    matches = sorted(ckpt_dir.glob("**/model_meta.json"))
    if not matches:
        # Fall back to a checkpoint glob (staged runs may lack a meta sidecar).
        ckpts = sorted(ckpt_dir.glob("**/checkpoints/best_model-*.ckpt"))
        if ckpts:
            return ckpts[0].parent.parent
        raise FileNotFoundError(
            f"No model_meta.json or checkpoints/best_model-*.ckpt under {ckpt_dir}.")
    if len(matches) > 1:
        print(f"WARNING: multiple model_meta.json under {ckpt_dir}; using {matches[0]}")
    return matches[0].parent


def load_real_observation(config, sbi_pipeline, job_name="LV2"):
    """
    Load the real event observation as an (n_stations, n_components, T) array,
    inverting the receiver time shifts (the ML model trained on unshifted data).
    """
    real_event_path = config.real_event_jobs.get(job_name)
    if real_event_path is None:
        raise KeyError(
            f"Job '{job_name}' not in jobs.real_events. "
            f"Available: {list(config.real_event_jobs.keys())}")

    components = sbi_pipeline.data_manager.data_loader.components
    # Use the config-default shifts snapshotted at pipeline-build time, NOT the live
    # receivers map (which the theory-covariance estimator zeroes during the build).
    # Falls back to the live map only if the snapshot is unavailable (e.g. a pipeline
    # not built via build_eval_pipeline).
    forward_shifts = getattr(sbi_pipeline, "_default_receiver_time_shifts", None)
    if forward_shifts is None:
        forward_shifts = sbi_pipeline.simulation_parameters.receivers.receiver_time_shifts_map
    inverted_shifts = {k: -v for k, v in forward_shifts.items()}
    print(f"NPE time-shift undo: inverting config-default shifts {dict(forward_shifts)}")
    shifted = sbi_pipeline.data_manager.data_loader.load_simulation_data_array_with_shifts(
        real_event_path, inverted_shifts)
    n_stations = len(sbi_pipeline.simulation_parameters.receivers.receivers)
    return shifted.reshape(n_stations, len(components), -1)


def recovered_mt_samples(inv_data) -> np.ndarray:
    """Physical-unit moment-tensor samples ``(n, 6)`` from an ``InversionData``.

    ``inv_data.samples`` are ALREADY in physical units — the pipeline applies the
    ``FlexibleScaler`` inverse before saving (median ~1e16 N m, matching the MLE).
    Applying ``data_scaler.inverse_transform`` here again double-scales to ~1e33
    (Mw 16, float overflow in the lune), so do NOT transform — just slice the first
    six MT columns.  Lifted from ``compare_to_reference.recovered_samples``."""
    return np.asarray(inv_data.samples)[:, :6]
