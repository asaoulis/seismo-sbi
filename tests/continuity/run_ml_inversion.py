"""The ML-NPE leg of the LV2 continuity check: sample a trained NPE on the real LV2 event.

Builds the evaluation pipeline from the continuity config, loads the checkpoint under
``--ckpt_dir``, draws ``--num_samples`` posterior samples of the observation (whose receiver
time shifts are undone, the network having been trained on unshifted data) and writes
``<output_dir>/inversion_results_ml.pkl`` as ``(None, None, [InversionResult])``, the layout
event_inversion.py uses. Run from ``examples/``.
"""
import argparse
import os
import pickle
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import torch  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="ML-NPE inversion for the LV2 event.")
    p.add_argument("--config", "-c", required=True,
                   help="Path to a continuity YAML config (relative to cwd, e.g. configs/LV2_continuity.yaml).")
    p.add_argument("--ckpt_dir", default="./ml-checkpoints",
                   help="Parent directory containing checkpoints/best_model-*.ckpt (smoke) "
                        "or <model_name>/ckpts/ (staged). Default: ./ml-checkpoints")
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from seismo_sbi.evaluation.inference import (
        build_eval_pipeline, build_ml_posterior, load_real_observation,
    )
    from seismo_sbi.sbi.scalers import build_flexible_scaler
    from seismo_sbi.sbi.types.results import InversionData, InversionResult, InversionConfig

    print("Parsing config file...")
    config, sbi_pipeline, original_parameters = build_eval_pipeline(config_path)
    print("Successfully parsed config file.")

    print(f"Loading ML checkpoint from: {ckpt_dir}")
    posterior = build_ml_posterior(ckpt_dir, sbi_pipeline)

    job_name = args.job_name
    real_data_vector = load_real_observation(config, sbi_pipeline, job_name=job_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor_obs = torch.as_tensor(real_data_vector, dtype=torch.float32).to(device).unsqueeze(0)

    print(f"Drawing {args.num_samples} posterior samples...")
    samples = posterior.sample((args.num_samples,), tensor_obs, show_progress_bars=True)

    data_scaler = build_flexible_scaler(original_parameters, config.raw_config)
    ml_inversion_data = InversionData(
        theta0=None,
        samples=data_scaler.inverse_transform(samples.cpu().numpy()),
        data_scaler=data_scaler,
    )

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

    out_pkl = output_dir / "inversion_results_ml.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump((None, None, [inversion_result]), f)

    print(f"Saved ML inversion results to: {out_pkl}")
    print(f"Samples shape: {ml_inversion_data.samples.shape}")


if __name__ == "__main__":
    main()
