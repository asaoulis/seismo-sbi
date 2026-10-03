"""Build the pipeline, posterior and observation an evaluation run needs.

:func:`load_trained_posterior` gives the posterior, theta scaler and pipeline of a trained run;
:func:`build_eval_pipeline` constructs a pipeline that also simulates the test jobs,
:func:`build_ml_posterior` loads a trained model into a posterior, :func:`resolve_ckpt_dir`
finds the checkpoint directory to load from, and :func:`load_real_observation` reads one named
real event (:func:`load_observation` reads one event file).
"""
from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from seismo_sbi.simulators.simulation_io import SimulationDataLoader

logger = logging.getLogger(__name__)


class TrainedPosterior(NamedTuple):
    """A trained NPE ready for inference.

    ``posterior`` samples the scaled parameters, in [0, 1]; ``data_scaler.inverse_transform`` maps
    them to physical units. ``pipeline`` holds the receivers and the data loader an observation is
    read with, and ``config`` is the parsed configuration of the run.
    """

    posterior: Any
    data_scaler: Any
    pipeline: Any
    config: Any
    run_directory: Path


def load_trained_posterior(config_path, run_directory, *, strict=True) -> TrainedPosterior:
    """The posterior, theta scaler and pipeline of a trained run; no simulation is read.

    ``config_path`` is the configuration the run was trained with and ``run_directory`` the run,
    or a directory holding exactly one run (see :func:`resolve_ckpt_dir`). The flow is rebuilt from
    the run's ``model_meta.json``; the scaler is built from the configuration with the M0
    convention the sidecar records and checked against the sidecar's scaling record, raising on a
    mismatch unless ``strict`` is False. The pipeline's forward model places the source time on
    the moment-rate function as the run was trained (:func:`recorded_stf_alignment`).
    """
    from seismo_sbi.sbi.configuration import SBI_Configuration
    from seismo_sbi.sbi.datasets.training_data import build_pipeline
    from seismo_sbi.sbi.npe.training.train import CompressionTrainer, recorded_stf_alignment
    from seismo_sbi.sbi.pipeline import SingleEventPipeline
    from seismo_sbi.sbi.scalers import build_flexible_scaler, check_scaler_provenance

    run_directory = resolve_ckpt_dir(run_directory)
    posterior = CompressionTrainer.from_run_directory(run_directory).build_posterior()
    model_meta = json.loads((run_directory / "model_meta.json").read_text())
    config = SBI_Configuration.from_file(config_path)
    config.sim_parameters = config.sim_parameters._replace(stf_alignment=recorded_stf_alignment(model_meta))
    pipeline = build_pipeline(config, config_path, pipeline_class=SingleEventPipeline)
    data_scaler = build_flexible_scaler(deepcopy(pipeline.parameters), config.raw_config,
                                        model_meta=model_meta)
    check_scaler_provenance(model_meta, data_scaler, strict=strict)
    return TrainedPosterior(posterior, data_scaler, pipeline, config, run_directory)


def build_eval_pipeline(config_path, *, setup_training_noise=False,
                        regenerate_dataset=False, skip_compression=None):
    """
    Parse a YAML config and build a fully-loaded SingleEventPipeline, exactly as the
    evaluation drivers do.

    Returns ``(config, sbi_pipeline, original_parameters)``.

    If ``setup_training_noise`` is True, also primes the training noise sampler's
    adaptive covariance from the first jobs.real_events entry (needed to draw
    validation examples with the same on-the-fly noise the model trained on).

    By default (``regenerate_dataset=False``) the simulation dataset on disk is reused:
    ``simulate_test_jobs`` would redraw every ``random_events`` simulation and overwrite
    the dataset the model trained on. The dataset is simulated only when none exists.
    """
    from pathlib import Path as _Path
    from seismo_sbi.sbi.configuration import SBI_Configuration
    from seismo_sbi.sbi.pipeline import SingleEventPipeline
    from seismo_sbi.sbi.datasets.training_data import build_pipeline

    config = SBI_Configuration()
    config.parse_config_file(config_path)

    sbi_pipeline = build_pipeline(config, config_path, pipeline_class=SingleEventPipeline)
    original_parameters = deepcopy(sbi_pipeline.parameters)

    existing_sims = sorted(
        _Path(sbi_pipeline.simulations_output_path).glob("random_event_*.h5"))
    if regenerate_dataset or not existing_sims:
        if not existing_sims:
            logger.info("No existing sims found — generating the dataset.")
        test_jobs_paths = sbi_pipeline.simulate_test_jobs(
            config.dataset_parameters, config.test_job_simulations
        )
    else:
        logger.info(f"Reusing {len(existing_sims)} existing sims at "
                    f"{sbi_pipeline.simulations_output_path} (no regeneration).")
        test_jobs_paths = existing_sims
    sbi_pipeline.compute_data_vector_properties(test_jobs_paths, config.real_event_jobs)
    # An NPE-only evaluation never uses the score compressors, and the stencil cannot run on the
    # multi-ensemble simulator, so ``skip_compression_data`` bypasses both.
    if skip_compression is None:
        skip_compression = config.training.skip_compression_stencil
    if not skip_compression:
        score_compression_data, extra_gradients = sbi_pipeline.compute_required_compression_data(
            config.compression_methods,
            config.model_parameters,
            rerun_if_stencil_exists=config.pipeline_parameters.generate_dataset,
        )
        sbi_pipeline.load_compressors(
            config.compression_methods, score_compression_data,
            extra_gradients=extra_gradients, freeze=True,
        )
    else:
        logger.info("skip_compression_data set — skipping score/Fisher stencil + compressor "
                    "load (ML-NPE eval needs no compressors).")
    sbi_pipeline.load_test_noises(config.sbi_noise_model, config.test_noise_models)

    if setup_training_noise:
        if not config.real_event_jobs:
            raise ValueError("setup_training_noise requires a jobs.real_events entry.")
        real_noise_path = next(iter(config.real_event_jobs.values()))
        cov_data = sbi_pipeline.data_manager.load_noise_parametrisation_data(real_noise_path)
        sbi_pipeline.rescale_training_noise(cov_data)

    return config, sbi_pipeline, original_parameters


def build_ml_posterior(ckpt_dir, sbi_pipeline, dim=256):
    """
    Rebuild the ML compressor + neural posterior from a checkpoint directory,
    reading the architecture from the checkpoint's model_meta.json sidecar so any
    encoder (cnn / pno / tcn) reloads correctly.  Returns the sbi DirectPosterior.
    """
    import json as _json
    from seismo_sbi.sbi.npe.training.train import CompressionTrainer, recorded_stf_alignment

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
    """The directory holding ``model_meta.json`` and ``checkpoints/``: ``ckpt_dir`` itself or the
    one run directory found beneath it.
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
        logger.warning(f"multiple model_meta.json under {ckpt_dir}; using {matches[0]}")
    return matches[0].parent


def load_real_observation(config, sbi_pipeline, job_name):
    """Load one real event as an ``(n_stations, n_components, n_samples)`` array.

    The receiver time shifts are inverted, since the model is trained on unshifted data.

    :param job_name: key of the event under ``jobs.real_events`` in the configuration.
    """
    real_event_path = config.real_event_jobs.get(job_name)
    if real_event_path is None:
        raise KeyError(
            f"Job '{job_name}' not in jobs.real_events. "
            f"Available: {list(config.real_event_jobs.keys())}")

    components = sbi_pipeline.data_manager.data_loader.components
    forward_shifts = sbi_pipeline.default_receiver_time_shifts
    logger.info(f"NPE time-shift undo: inverting config-default shifts {dict(forward_shifts)}")
    return load_observation(real_event_path, sbi_pipeline.simulation_parameters.receivers, components,
                            forward_shifts)


def load_observation(event_path, receivers, components, time_shifts=None):
    """One event file as an ``(n_stations, n_components, n_samples)`` array, with the receiver
    time shifts ``time_shifts`` (``{station: samples}``) undone.

    ``receivers`` is left carrying the inverted shifts, and a simulator holding the same object
    applies them to what it simulates next; ``components`` is the component layout, such as ``"ZEN"``.
    """
    inverted_shifts = {k: -v for k, v in (time_shifts or {}).items()}
    shifted = SimulationDataLoader(components, receivers).load_simulation_data_array_with_shifts(
        event_path, inverted_shifts)
    n_stations = len(receivers.receivers)
    return shifted.reshape(n_stations, len(components), -1)


def recovered_mt_samples(inv_data) -> np.ndarray:
    """Physical-unit moment-tensor samples ``(n, 6)`` from an ``InversionData``.

    ``inv_data.samples`` are ALREADY in physical units — the pipeline applies the
    ``FlexibleScaler`` inverse before saving (median ~1e16 N m, matching the MLE).
    Applying ``data_scaler.inverse_transform`` here again double-scales to ~1e33
    (Mw 16, float overflow in the lune), so do NOT transform — just slice the first
    six MT columns.  Lifted from ``compare_to_reference.recovered_samples``."""
    return np.asarray(inv_data.samples)[:, :6]
