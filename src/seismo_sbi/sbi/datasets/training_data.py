"""Assemble the dataset an NPE training run consumes.

Builds the pipeline named by a configuration, simulates or finds the training set on disk, then
loads the compressors, the noise model and the parameter scaler and builds the augmentation
chains. The result is a :class:`TrainingData` holding everything the trainer and the dataloaders
need; the training script calls these in order.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..pipeline_variants import PIPELINE_CLASSES
from ..scalers import FlexibleScaler, build_flexible_scaler, check_scaler_provenance, recorded_theta_scaler
from seismo_sbi.nuisance_effects.post_processing import build_augmentation_chain_from_parameters
from ...utils.errors import InvalidConfiguration


@dataclass
class TrainingData:
    """The simulated dataset and the per-run objects an NPE training run consumes.

    ``station_locations`` has shape (n_stations, n_coordinates); ``trace_length`` is the
    per-trace sample count; the two augmentation chains are applied to a training sample before
    and after sensor noise is added. ``stf_alignment`` is the forward model's source-time
    convention (:attr:`~seismo_sbi.simulators.base.Simulator.stf_alignment`).
    """

    simulation_paths: list
    components: str
    station_locations: np.ndarray
    trace_length: int
    data_scaler: FlexibleScaler
    augmentation_chain: object
    augmentation_nuisance_params: dict
    post_noise_chain: object
    post_noise_nuisance_params: dict
    stf_alignment: str = "peak"


def build_pipeline(config, config_path, num_simulations=None, pipeline_class=None):
    """Build the pipeline named by ``config.pipeline_type`` and load its seismic parameters.

    ``num_simulations`` overrides the configured size of the training dataset;
    ``pipeline_class`` overrides the class the configuration names.
    """
    if num_simulations is not None:
        config.dataset_parameters = config.dataset_parameters._replace(
            num_simulations=num_simulations)
        print(f"Overriding num_simulations -> {num_simulations}")

    if pipeline_class is None:
        pipeline_class = PIPELINE_CLASSES[config.pipeline_type]
    pipeline = pipeline_class(config.pipeline_parameters, config_path)
    pipeline.compression_methods = config.compression_methods
    pipeline.load_seismo_parameters(config.sim_parameters, config.model_parameters,
                                    config.dataset_parameters)
    return pipeline


def generate_training_dataset(pipeline, config, skip_compression_stencil=False):
    """Simulate, or find on disk, the training simulations and the score/Fisher stencil.

    Returns the simulation paths. Only the classical compressed inversion reads the stencil, so
    ``skip_compression_stencil`` leaves it unbuilt for an ML training run.
    """
    if config.pipeline_parameters.generate_dataset:
        simulation_paths = pipeline.simulate_test_jobs(config.dataset_parameters,
                                                       config.test_job_simulations)
    else:
        simulation_paths = list(Path(pipeline.simulations_output_path).glob('*.h5'))

    pipeline.compute_data_vector_properties(simulation_paths, config.real_event_jobs)
    if skip_compression_stencil:
        pipeline.score_compression_data, pipeline.extra_gradients = None, None
    else:
        pipeline.score_compression_data, pipeline.extra_gradients = (
            pipeline.compute_required_compression_data(
                config.compression_methods, config.model_parameters,
                rerun_if_stencil_exists=config.pipeline_parameters.generate_dataset))
    print(f"Training dataset ready: {len(simulation_paths)} simulations at "
          f"{pipeline.simulations_output_path}")
    return simulation_paths


def training_scaler(parameters, raw_config, training, models_output_path) -> FlexibleScaler:
    """The parameter scaler a run trains with.

    A run that warm-starts from ``training.warm_start_run_name`` keeps the scalar-moment convention
    recorded in that run's ``model_meta.json`` under ``models_output_path`` (six-component when it
    records none or has no sidecar), and raises if its scaling differs from a recorded one.
    A cold start gets the full-tensor scaler.
    """
    if not training.warm_start_run_name:
        return build_flexible_scaler(parameters, raw_config)
    meta_path = Path(models_output_path) / training.warm_start_run_name / "model_meta.json"
    source_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    scaler = build_flexible_scaler(parameters, raw_config, model_meta=source_meta)
    check_scaler_provenance(source_meta, scaler, strict=bool(recorded_theta_scaler(source_meta)))
    return scaler


def prepare_training_data(pipeline, config, simulation_paths, training):
    """Load the compressors, the noise model and the parameter scaler, and build the
    augmentation chains, returning the :class:`TrainingData` a trainer consumes."""
    if not training.skip_compression_stencil:
        pipeline.load_compressors(config.compression_methods, pipeline.score_compression_data,
                                  extra_gradients=pipeline.extra_gradients)
    pipeline.load_test_noises(config.sbi_noise_model, config.test_noise_models)
    rescale_training_noise_to_event(pipeline, config)

    augmentation_chain, augmentation_nuisance_params = build_augmentation_chain_from_parameters(
        pipeline.parameters, sampling_rate=pipeline.simulation_parameters.sampling_rate)
    post_noise_chain, post_noise_nuisance_params = build_augmentation_chain_from_parameters(
        pipeline.parameters, stage="training_augmentation_post_noise")
    print(f"Training-time nuisance augmentation: {list(augmentation_nuisance_params) or 'none'}")
    print(f"Post-noise augmentation: {list(post_noise_nuisance_params) or 'none'}")

    data_scaler = training_scaler(pipeline.parameters, config.raw_config, training,
                                  pipeline.models_output_path)
    print(f"Moment-tensor scaling: {data_scaler.moment_tensor_scaling}")
    return TrainingData(
        simulation_paths=simulation_paths,
        components=pipeline.data_manager.data_loader.components,
        station_locations=pipeline.simulation_parameters.receivers.get_station_locations_array(),
        trace_length=pipeline.trace_length,
        data_scaler=data_scaler,
        augmentation_chain=augmentation_chain,
        augmentation_nuisance_params=augmentation_nuisance_params,
        post_noise_chain=post_noise_chain,
        post_noise_nuisance_params=post_noise_nuisance_params,
        stf_alignment=pipeline.simulator_wrapper.stf_alignment,
    )


def rescale_training_noise_to_event(pipeline, config):
    """Scale the training noise covariance to one real event's pre-event variance.

    Skipped for white ``gaussian`` noise, whose level is fixed by ``noise_level``, and in
    generic-event mode (``real_noise`` with ``rescale: false``), where the sampler draws noise
    windows verbatim and the event file need not hold every station.
    """
    if not config.sbi_noise_model.follows_event:
        return
    if not config.real_event_jobs:
        raise InvalidConfiguration(
            "Training the noise covariance needs at least one jobs.real_events entry to "
            "parametrise it from.")
    real_noise_path = next(iter(config.real_event_jobs.values()))
    covariance_data = pipeline.data_manager.load_noise_parametrisation_data(real_noise_path)
    pipeline.rescale_training_noise(covariance_data)


def preload_noise_cache(pipeline, cache):
    """Read the real-noise window pool into RAM when ``ml_cache.noise`` asks for it."""
    if cache.noise and hasattr(pipeline.training_noise_sampler, "preload_cache"):
        pipeline.training_noise_sampler.preload_cache(max_workers=cache.preload_workers)
