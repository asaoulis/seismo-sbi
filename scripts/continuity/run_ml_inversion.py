#!/usr/bin/env python3
"""
run_ml_inversion.py
====================
Reproduce the ML-NPE inversion from examples/theory_errors_LV2.ipynb,
loading a pre-trained (or freshly-trained + staged) ML checkpoint
and sampling the posterior.

Why a wrapper instead of a YAML config?
    MachineLearningCompressor takes Python objects (seismogram_preprocessor,
    scaler, model_type) that cannot be described as plain YAML strings.
    This script replicates exactly what the notebook does, saving results
    in the same (job_data, job_results, inversion_results) pkl format as
    event_inversion.py so extract_source_summary.py can parse it uniformly.

Usage (cwd = examples/):
    # Use the pre-trained checkpoint from ml-checkpoints/ (smoke mode):
    python ../scripts/continuity/run_ml_inversion.py \\
        --config configs/LV2_continuity.yaml \\
        --ckpt_dir ./ml-checkpoints \\
        --model_name LV2 \\
        --output_dir ./data/pipeline_outputs/continuity_ml

    # After train + stage (full mode):
    python ../scripts/continuity/run_ml_inversion.py \\
        --config configs/LV2_continuity.yaml \\
        --ckpt_dir ./ml_models \\
        --model_name continuity_<YYYYMMDD> \\
        --output_dir ./data/pipeline_outputs/continuity_ml

Output pkl:
    <output_dir>/inversion_results_ml.pkl
    Contains: (job_data, job_results, inversion_results) where inversion_results
    is a list with one InversionResult (inversion_method='ml_compressor').
    job_data and job_results are None (no SBI dataset training here, only sampling).
"""
import argparse
import os
import pickle
from copy import deepcopy
from pathlib import Path

# Prevent OMP/MKL contention with multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import torch
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="ML-NPE inversion for the LV2 event.")
    p.add_argument("--config", "-c", required=True,
                   help="Path to a continuity YAML config (relative to cwd, e.g. configs/LV2_continuity.yaml).")
    p.add_argument("--ckpt_dir", default="./ml-checkpoints",
                   help="Parent directory containing checkpoints/best_model-*.ckpt (smoke) "
                        "or <model_name>/ckpts/ (staged). Default: ./ml-checkpoints")
    p.add_argument("--model_name", default="LV2",
                   help="Model name passed to CompressionTrainer.load_best(). "
                        "For the pre-trained smoke checkpoint use 'LV2'. "
                        "For a freshly staged checkpoint use the run_name.")
    p.add_argument("--num_samples", type=int, default=10000,
                   help="Number of posterior samples to draw. Default: 10000.")
    p.add_argument("--output_dir", default="./data/pipeline_outputs/continuity_ml",
                   help="Directory to write inversion_results_ml.pkl.")
    p.add_argument("--job_name", default="LV2",
                   help="Key in jobs.real_events to use as observation. Default: LV2.")
    return p.parse_args()


def main():
    args = parse_args()
    config_path = args.config
    ckpt_dir = Path(args.ckpt_dir)
    model_name = args.model_name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Build the SBI pipeline (same as notebook cells 7 + 11)
    # ------------------------------------------------------------------ #
    from seismo_sbi.sbi.configuration import SBI_Configuration
    from seismo_sbi.sbi.pipeline import SingleEventPipeline
    from seismo_sbi.sbi.scalers import FlexibleScaler
    from seismo_sbi.sbi.compression.ML.train import CompressionTrainer
    from seismo_sbi.sbi.types.results import InversionData, InversionResult, InversionConfig

    print("Parsing config file...")
    config = SBI_Configuration()
    config.parse_config_file(config_path)
    print("Successfully parsed config file.")

    sbi_pipeline = SingleEventPipeline(config.pipeline_parameters, config_path)
    sbi_pipeline.compression_methods = config.compression_methods
    sbi_pipeline.load_seismo_parameters(
        config.sim_parameters, config.model_parameters, config.dataset_parameters
    )
    original_parameters = deepcopy(sbi_pipeline.parameters)

    # Run test-job simulations and set up compressors (stencil cache is reused)
    test_jobs_paths = sbi_pipeline.simulate_test_jobs(
        config.dataset_parameters, config.test_job_simulations
    )
    sbi_pipeline.compute_data_vector_properties(test_jobs_paths, config.real_event_jobs)
    score_compression_data, extra_gradients = sbi_pipeline.compute_required_compression_data(
        config.compression_methods,
        config.model_parameters,
        rerun_if_stencil_exists=config.pipeline_parameters.generate_dataset,
    )
    sbi_pipeline.load_compressors(
        config.compression_methods, score_compression_data,
        extra_gradients=extra_gradients, freeze=True
    )
    sbi_pipeline.load_test_noises(config.sbi_noise_model, config.test_noise_models)

    # ------------------------------------------------------------------ #
    # 2. Build the ML compressor (notebook cell 26)
    # ------------------------------------------------------------------ #
    components = sbi_pipeline.data_manager.data_loader.components
    station_locations = sbi_pipeline.simulation_parameters.receivers.get_station_locations_array()
    dim = 256
    trainer = CompressionTrainer(
        components, station_locations, dim, dim,
        trace_length=sbi_pipeline.trace_length
    )

    print(f"Loading ML checkpoint from: {ckpt_dir}")
    trainer.load_best(str(ckpt_dir))
    posterior = trainer.build_posterior()

    # ------------------------------------------------------------------ #
    # 3. Load the real observation (notebook cell 27)
    # Note: The ML model was trained on unshifted data, so we invert the
    # time shifts that were applied for the other two methods.
    # ------------------------------------------------------------------ #
    job_name = args.job_name
    real_event_path = config.real_event_jobs.get(job_name)
    if real_event_path is None:
        raise KeyError(
            f"Job '{job_name}' not found in config jobs.real_events. "
            f"Available: {list(config.real_event_jobs.keys())}"
        )

    forward_time_shifts = sbi_pipeline.simulation_parameters.receivers.receiver_time_shifts_map
    inverted_time_shifts = {k: -v for k, v in forward_time_shifts.items()}

    shifted_real_event = sbi_pipeline.data_manager.data_loader.load_simulation_data_array_with_shifts(
        real_event_path, inverted_time_shifts
    )
    num_stations = len(sbi_pipeline.simulation_parameters.receivers.receivers)
    num_components = len(components)
    real_data_vector = shifted_real_event.reshape(num_stations, num_components, -1)

    # ------------------------------------------------------------------ #
    # 4. Sample the posterior (notebook cell 27, continued)
    # ------------------------------------------------------------------ #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor_obs = torch.as_tensor(real_data_vector, dtype=torch.float32).to(device).unsqueeze(0)

    print(f"Drawing {args.num_samples} posterior samples...")
    samples = posterior.sample((args.num_samples,), tensor_obs, show_progress_bars=True)

    data_scaler = FlexibleScaler(original_parameters)
    ml_inversion_data = InversionData(
        theta0=None,
        samples=data_scaler.inverse_transform(samples.cpu().numpy()),
        data_scaler=data_scaler,
    )

    # ------------------------------------------------------------------ #
    # 5. Build InversionResult and serialize to pkl
    #
    # InversionConfig(train_noise, test_noise, inversion_method)
    # InversionResult(event_name, inversion_data, inversion_config)
    # ------------------------------------------------------------------ #
    inversion_config = InversionConfig(
        train_noise="",
        test_noise="gaussian_filtered",
        inversion_method="ml_compressor",
    )
    inversion_result = InversionResult(
        event_name=job_name,
        inversion_data=ml_inversion_data,
        inversion_config=inversion_config,
    )

    # job_data and job_results are None: no SBI training here, only sampling.
    # Shape matches (job_data, job_results, inversion_results) from event_inversion.py.
    out_pkl = output_dir / "inversion_results_ml.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump((None, None, [inversion_result]), f)

    print(f"Saved ML inversion results to: {out_pkl}")
    print(f"Samples shape: {ml_inversion_data.samples.shape}")


if __name__ == "__main__":
    main()
