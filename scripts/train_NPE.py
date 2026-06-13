import os
import argparse
import pickle
# prevent processes from using multiple threads
# this is necessary because otherwise the multiprocessing
# in emcee may use more threads than requested
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from pathlib import Path
from seismo_sbi.sbi.configuration import SBI_Configuration
from seismo_sbi.sbi.pipeline import SingleEventPipeline, MultiEventPipeline, VaryDatasetSizeEventPipeline
from seismo_sbi.sbi import utils as utils
from seismo_sbi.sbi.compression.ML.train import CompressionTrainer
from seismo_sbi.sbi.scalers import ZeroOneScaler, FlexibleScaler, build_flexible_scaler
from seismo_sbi.instaseis_simulator.post_processing import build_augmentation_chain_from_parameters

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for running a complete SBI pipeline. Requires a pre-specified configuration file. ')
    parser.add_argument('--config', '-c', type=str, help='Filepath of sbi_pipeline configuration file.', required = True)
    # run name argument
    parser.add_argument('--run_name', '-n', type=str, help='Name of the run. Used to create a subfolder in the output directory.', default = 'default_run')
    parser.add_argument('--epochs', '-e', type=int, help='Number of training epochs.', default = 300)
    parser.add_argument('--architecture', '-a', type=str, default=None,
                        help="Per-station encoder (station_encoder): 'cnn' (default), 'pno', 'tcn'. "
                             "Overrides the optional 'ml_architecture' key in the YAML config.")
    parser.add_argument('--generate_only', action='store_true',
                        help="Stop after generating the training dataset + compression stencil "
                             "(no NDE training). Used for the CPU dataset-generation stage of the "
                             "remote two-stage workflow; the GPU train stage re-loads these sims.")
    parser.add_argument('--num_simulations', type=int, default=None,
                        help="Override simulations.num_simulations from the config (caps the "
                             "training dataset size). Used by the remote gen-submit stage.")
    parser.add_argument('--csv_logger', action='store_true',
                        help="ALSO log per-epoch metrics to a Lightning CSVLogger (metrics.csv "
                             "beside the checkpoints), ALONGSIDE W&B. Gives a deterministic on-disk "
                             "val_loss for remote monitoring (train-monitor) while the run still "
                             "lands in W&B.")
    parser.add_argument('--no_wandb', action='store_true',
                        help="Disable W&B logging (e.g. CI). Combine with --csv_logger for "
                             "on-disk-only metrics, or leave both off to disable logging entirely.")
    parser.add_argument('--skip_compression_data', action='store_true',
                        help="Skip the score/Fisher compression stencil + the Gaussian "
                             "(optimal_score) compressor. ML/NPE training (CompressionTrainer) "
                             "trains its own embedding net on raw waveforms and never consumes "
                             "score_compression_data, so the stencil is pure overhead for ML runs "
                             "— and forking instaseis's numba-JIT forward model across the "
                             "stencil's loky workers can abort with 'ReferenceError: underlying "
                             "object has vanished'. Use for the ML gen + train stages.")
    args = parser.parse_args()
    return args

def main():

    # Parse arguments and prepare configuration data

    args = parse_arguments()
    config_path = args.config

    print("Parsing config file...")
    config = SBI_Configuration()
    config.parse_config_file(config_path)
    print("Successfully parsed config file.")

    # Optional dataset-size override (remote gen-submit stage). DatasetGenerationParameters
    # is a NamedTuple, so _replace gives a clean immutable override.
    if args.num_simulations is not None:
        config.dataset_parameters = config.dataset_parameters._replace(
            num_simulations=args.num_simulations)
        print(f"Overriding num_simulations -> {args.num_simulations}")

    ### Start SBI Pipeline

    Pipeline = SingleEventPipeline if config.pipeline_type == 'single_event' else MultiEventPipeline
    Pipeline = VaryDatasetSizeEventPipeline if config.pipeline_type == 'vary_dataset_size' else Pipeline
    sbi_pipeline = Pipeline(config.pipeline_parameters, config_path)
    sbi_pipeline.compression_methods = config.compression_methods
    sbi_pipeline.load_seismo_parameters(config.sim_parameters, config.model_parameters, config.dataset_parameters)

    if config.pipeline_parameters.generate_dataset:
        test_jobs_paths = sbi_pipeline.simulate_test_jobs(config.dataset_parameters, config.test_job_simulations)
    else:
        path = sbi_pipeline.simulations_output_path
        test_jobs_paths = list(Path(path).glob('*.h5'))

    sbi_pipeline.compute_data_vector_properties(test_jobs_paths, config.real_event_jobs)
    # The score/Fisher stencil + Gaussian (optimal_score) compressor are only needed for the
    # classical compressed-inversion path; ML/NPE training builds + trains its own embedding net
    # on raw waveforms and never reads score_compression_data. Skipping the stencil bypasses it
    # entirely for ML runs (and dodges the instaseis numba/loky stencil crash). Enable via the
    # --skip_compression_data CLI flag OR a top-level `skip_compression_data: true` config key
    # (the latter rides the normal config sync so the remote ML gen/train stages pick it up).
    skip_compression_data = args.skip_compression_data or bool(
        config.raw_config.get("skip_compression_data", False))
    if skip_compression_data:
        print("[skip_compression_data] Skipping score/Fisher stencil + Gaussian compressor "
              "(unused by ML/NPE training).")
        score_compression_data, extra_gradients = None, None
    else:
        score_compression_data, extra_gradients = sbi_pipeline.compute_required_compression_data(config.compression_methods,
                                                                                config.model_parameters,
                                                                                rerun_if_stencil_exists = config.pipeline_parameters.generate_dataset)

    # CPU dataset-generation stage of the remote two-stage workflow: the expensive
    # forward simulations (training dataset via simulate_test_jobs, and the score/Fisher
    # stencil via compute_required_compression_data) are now on disk. Stop here so the
    # GPU train stage (generate_dataset: false) re-loads them without re-simulating.
    if args.generate_only:
        print(f"[generate_only] dataset ready: "
              f"{len(test_jobs_paths)} simulations at {sbi_pipeline.simulations_output_path}")
        return

    if not skip_compression_data:
        sbi_pipeline.load_compressors(config.compression_methods, score_compression_data, extra_gradients=extra_gradients)

    sbi_pipeline.load_test_noises(config.sbi_noise_model, config.test_noise_models)

    # Rescale the training noise covariance to a specific real event's pre-event variance,
    # parametrised from that event's h5 misc group (jobs.real_events). SKIPPED in generic-event
    # mode (real_noise with `rescale: false`): the sampler then draws windows verbatim, and the
    # event h5 need not even contain every master-list station (it usually won't).
    _nm = config.sbi_noise_model
    rescale_to_event = (_nm.get('type') != 'real_noise') or _nm.get('rescale', True)
    if rescale_to_event:
        if not config.real_event_jobs:
            raise ValueError("train_NPE.py requires at least one jobs.real_events entry in the "
                             "config to parametrise the training noise covariance.")
        real_noise_path = next(iter(config.real_event_jobs.values()))
        covariance_data = sbi_pipeline.data_manager.load_noise_parametrisation_data(real_noise_path)
        sbi_pipeline.training_noise_sampler.set_adaptive_covariance_with_misc_data(covariance_data)

    components = sbi_pipeline.data_manager.data_loader.components
    station_locations = sbi_pipeline.simulation_parameters.receivers.get_station_locations_array()
    # Scaler parametrisation is config-driven (top-level `ml_scaler` block) so it matches
    # at inference time; default 'linear' reproduces the legacy FlexibleScaler behaviour.
    data_scaler = build_flexible_scaler(sbi_pipeline.parameters, config.raw_config)
    print(f"Moment-tensor scaling: {data_scaler.moment_tensor_scaling}")
    model_dim = 256

    # Per-station encoder selection (station_encoder): CLI overrides the optional
    # top-level 'ml_architecture' key in the YAML; default 'cnn'.
    import yaml as _yaml
    _raw_cfg = (_yaml.safe_load(open(config_path)) or {})
    _cfg_arch = _raw_cfg.get("ml_architecture")
    architecture = args.architecture or _cfg_arch or "cnn"
    print(f"Using per-station encoder (station_encoder): {architecture}")

    model_config = {"station_encoder": architecture}

    # Optional per-station encoder hyperparameters via a top-level 'ml_encoder' YAML block,
    # forwarded verbatim to the encoder __init__ (see station_encoders.py). Example:
    #   ml_encoder:
    #     downsample: 8       # tcn/pno temporal stride — bigger ⇒ fewer time tokens L ⇒ less
    #                         #   transformer/PMA work (a measured ~1.25x train-step speedup at 8 vs 4)
    #     channels: 32        # tcn intermediate channel count
    #     n_blocks: 4
    # Absent ⇒ encoder defaults (tcn downsample=4). A capacity-affecting knob — change only with
    # a val-loss check (proven non-regressing for downsample 8 on the kernel + brustle-lomax tasks).
    _enc_cfg = _raw_cfg.get("ml_encoder")
    if _enc_cfg:
        enc_cfg = {k: v for k, v in _enc_cfg.items() if k != "enabled"}
        # `input_decimate` is a MODEL-entry knob (Nyquist-aware decimation before the
        # encoder), not an encoder kwarg — pop it out so the encoder __init__ never sees
        # it. Accepts `input_decimate: 3` or `{factor: 3, antialias: true}`.
        _dec = enc_cfg.pop("input_decimate", None)
        if _dec:
            model_config["input_decimate"] = (
                dict(_dec) if isinstance(_dec, dict) else {"factor": int(_dec)}
            )
            print(f"Model-entry input decimation: {model_config['input_decimate']}")
        if enc_cfg:
            model_config["encoder_config"] = enc_cfg
            print(f"Per-station encoder config: {model_config['encoder_config']}")

    # Optional source-location conditioning via a top-level 'ml_conditioning' YAML block:
    #   ml_conditioning:
    #     param_map: {source_location: [latitude, longitude, depth]}
    #     d_cond: 16
    #     coord_mode: geographic
    #     inject: [token_add, film]      # subset of relative_posemb/token_add/film/concat_context
    # n_cond defaults to the number of params in param_map.
    conditioning_param_map = None
    _cond_cfg = _raw_cfg.get("ml_conditioning")
    if _cond_cfg:
        conditioning_param_map = _cond_cfg["param_map"]
        n_cond = _cond_cfg.get("n_cond", sum(len(v) for v in conditioning_param_map.values()))
        model_config["conditioning"] = {
            "n_cond": n_cond,
            "d_cond": _cond_cfg.get("d_cond", model_dim),
            "coord_mode": _cond_cfg.get("coord_mode", "geographic"),
            "inject": _cond_cfg.get("inject", []),
            "n_fourier": _cond_cfg.get("n_fourier", 0),
        }
        print(f"Source-location conditioning enabled: {model_config['conditioning']}")

    # Optional source-location UNCERTAINTY augmentation (v3): a `source_location_error` nuisance
    # with stage 'training_augmentation' supplies a per-coordinate Gaussian std (`coordinate_std`,
    # in the conditioning param_map order) that perturbs the conditioning vector in the dataloader
    # (fresh per sample). Only meaningful when conditioning is active.
    conditioning_noise_std = None
    if conditioning_param_map:
        _sle_cfg = ((_raw_cfg.get("parameters", {}) or {}).get("nuisance", {}) or {}).get(
            "source_location_error")
        if _sle_cfg and _sle_cfg.get("stage") == "training_augmentation":
            conditioning_noise_std = _sle_cfg.get("coordinate_std")
            print(f"Source-location conditioning noise (training augmentation): "
                  f"coordinate_std={conditioning_noise_std}")

    # Optional variable-station training via a top-level 'ml_variable_stations' YAML block:
    #   ml_variable_stations:
    #     enabled: true
    #     keep_fraction: [0.5, 1.0]          # (low, high) uniform range, or a fixed float
    #     min_stations: 1
    #     station_coords_mode: absolute      # or 'relative' (needs ml_conditioning)
    # When enabled, each training sample uses a random subset of the master station set and
    # the model is built in variable-station mode (pad+mask batching, per-sample coords).
    from seismo_sbi.sbi.compression.ML.dataloading import StationSubsampler
    station_subsampler = None
    _var_cfg = _raw_cfg.get("ml_variable_stations")
    if _var_cfg and _var_cfg.get("enabled", False):
        station_subsampler = StationSubsampler(
            keep_fraction=_var_cfg.get("keep_fraction"),
            min_stations=_var_cfg.get("min_stations", 1),
        )
        model_config["variable_stations"] = True
        model_config["station_coords_mode"] = _var_cfg.get("station_coords_mode", "absolute")
        print(f"Variable-station training enabled: coords_mode="
              f"{model_config['station_coords_mode']}, keep_fraction={station_subsampler.keep_fraction}")

    # Optional per-station amplitude embedding via a top-level 'ml_amplitude_embedding' block:
    #   ml_amplitude_embedding:
    #     enabled: true
    #     mode: array_relative      # array_relative | absolute
    #     per_component: false
    #     reference: mean           # mean | median  (array_relative only)
    #     num_freqs: 16
    #     sigma: 1.0
    #     learnable_freqs: false
    #     scale: 1.0                # fixed std for the array-relative token feature
    #     distance_correction: true # de-bias the reference for geometric spreading (needs conditioning)
    #     snr_weighting: true       # down-weight noise-dominated stations in the reference
    #     snr_floor_quantile: 0.2   # low-percentile |x| used as the rough per-station noise floor
    # Lifts per-station amplitude out of the waveform channel into a full-width Random-Fourier
    # token so cross-station relative amplitude reaches the transformer attention; the per-event
    # reference (≈log M0) is added to the pooled embedding (see amplitude_embedding.py).
    _amp_cfg = _raw_cfg.get("ml_amplitude_embedding")
    if _amp_cfg and _amp_cfg.get("enabled", False):
        model_config["amplitude_embedding"] = {
            k: v for k, v in _amp_cfg.items() if k != "enabled"
        }
        print(f"Per-station amplitude embedding enabled: {model_config['amplitude_embedding']}")

    # Optional RFF station positional encoding via a top-level 'ml_positional_encoding' block:
    #   ml_positional_encoding:
    #     enabled: true
    #     mode: fourier            # fourier (sinusoidal / absence => legacy sinusoid)
    #     num_freqs: 16
    #     sigma: 1.0
    #     learnable_freqs: false
    #     include_depth: true      # also RFF-encode source depth (needs ml_conditioning, n_cond >= 3)
    #     inject_every_layer: true # re-inject the geometry before every transformer block (§3.2.c)
    #     standardize: running     # per-feature input standardisation ('running' | 'none')
    # Replaces the mis-scaled Vaswani station sinusoid with a well-scaled Random-Fourier-Feature
    # map of source-relative geometry (distance, periodic azimuth) or absolute (lat, lon); azimuth
    # enters as (cos, sin) so 11° ≈ 350°. coords_kind is auto-derived (relative iff relative_posemb
    # / station_coords_mode='relative'). Absent ⇒ unchanged legacy sinusoid (see positional_encoding.py).
    _pe_cfg = _raw_cfg.get("ml_positional_encoding")
    if _pe_cfg and _pe_cfg.get("enabled", False):
        model_config["positional_encoding"] = {
            k: v for k, v in _pe_cfg.items() if k != "enabled"
        }
        print(f"RFF station positional encoding enabled: {model_config['positional_encoding']}")

    # Optional Set-Transformer PMA pooling head via a top-level 'ml_pooling' block:
    #   ml_pooling:
    #     enabled: true
    #     pool_over: tokens        # tokens (pool the N·L final tokens) | stations (time-collapse,
    #                              #   one token/station, then pool the N station tokens — §3.4b)
    #     num_seeds: 4             # k learnable seeds (k=1 => single-vector summary)
    #     num_heads: null          # default = transformer nheads
    #     seed_self_attention: false  # SAB among the k seed outputs (opt-in; speculative)
    #     combine: linear          # linear (learned concat→Linear) | mean | first
    #     ffn: true                # MAB position-wise FFN inside the pool (textbook PMA)
    #     dim_feedforward: null    # MAB / TimePool FFN width (default 2*channels)
    #     dropout: 0.0
    #     time_pool_heads: null    # pool_over=stations only: TimePool attention heads (default num_heads)
    # Replaces the legacy "query tokens then unweighted mean" read-out with a proper PMA head whose
    # k seeds are combined by a LEARNED linear map (the §3.4a fix). Enabling it disables the in-block
    # query cross-attention (the axial blocks become a pure set-encoder; the encoder is unchanged).
    # Absent ⇒ no head ⇒ byte-identical legacy pooling (see pma_pooling.py).
    _pool_cfg = _raw_cfg.get("ml_pooling")
    if _pool_cfg and _pool_cfg.get("enabled", False):
        model_config["pma_pooling"] = {
            k: v for k, v in _pool_cfg.items() if k != "enabled"
        }
        print(f"Set-Transformer PMA pooling head enabled: {model_config['pma_pooling']}")

    # Optional NDE-head (normalising-flow) overrides via a top-level 'ml_flow' block:
    #   ml_flow:
    #     num_transforms: 8          # flow coupling-transform depth (default 5). 8->5 is a
    #                                #   measured ~1.17x model-step speedup; CAPACITY change —
    #                                #   quality-validate (TARP) before lowering in production.
    #     num_blocks: 2              # residual blocks per spline conditioner
    #     dropout_probability: 0.0
    #     use_batch_norm: true
    # The flow's hidden width stays tied to the embedding channel width (model_dim); it is
    # intentionally NOT exposed here. Absent => DEFAULT_FLOW_CONFIG (legacy num_transforms=5).
    flow_config = None
    _flow_cfg = _raw_cfg.get("ml_flow")
    if _flow_cfg:
        flow_config = {k: v for k, v in _flow_cfg.items() if k != "enabled"}
        # `hidden_features` here is OPTIONAL and decouples the flow's hidden width from the
        # embedding channel width (model_dim); when absent the flow width stays tied to
        # channels (the legacy behaviour, preserved for run comparability).
        print(f"NDE-head (flow) overrides: {flow_config}")

    # Optional performance toggles via a top-level 'ml_perf' block (speed-only, opt-in):
    #   ml_perf:
    #     amp: true            # bf16 autocast scoped to the EMBEDDING net (flow head stays fp32)
    #     amp_dtype: bfloat16  # bfloat16 (default) | float16
    #     sdpa: true           # fused scaled_dot_product_attention (numerically-equivalent attn)
    #     fused_adam: true     # fused-kernel AdamW (same update rule; falls back on CPU/old torch)
    #     compile: true        # torch.compile the full log-prob (embedding + flow). Needs a
    #                          #   WORKING torch.compile (>= 2.2; broken on torch 2.0) — gate-test first.
    #     compile_flow: true   # compile only the flow transform stack (fallback if `compile`
    #                          #   recompile-thrashes on the embedding's mask branches)
    # The flow's precision-brittle transforms (LULinear log-det + BatchNorm conditioner) are NEVER
    # autocast — the autocast is wrapped around SeismogramTransformer.forward and casts the context
    # back to fp32 before the flow. Absent ⇒ all off ⇒ byte-identical legacy compute.
    _perf_cfg = _raw_cfg.get("ml_perf") or {}
    if _perf_cfg:
        model_config["perf"] = {k: v for k, v in _perf_cfg.items() if k != "enabled"}
        print(f"ML perf toggles: {model_config['perf']}")

    # Optional optimizer / LR-schedule overrides via a top-level 'ml_optimizer' block:
    #   ml_optimizer:
    #     lr: 1.0e-4
    #     weight_decay: 1.0e-4
    #     lr_schedule: constant      # cosine (default; warmup->decay to lr*0.1) | constant | cyclic
    # Absent => lr=1e-4, weight_decay=1e-4, cosine (the legacy schedule).
    _opt_cfg = _raw_cfg.get("ml_optimizer") or {}
    lr = float(_opt_cfg.get("lr", 1e-4))
    weight_decay = float(_opt_cfg.get("weight_decay", 1e-4))
    lr_second_stage = _opt_cfg.get("lr_schedule", "cosine")
    if _opt_cfg:
        print(f"Optimizer overrides: lr={lr}, weight_decay={weight_decay}, "
              f"lr_schedule={lr_second_stage}")

    trainer = CompressionTrainer(components, station_locations, channels=model_dim, latent_dim=model_dim,
                                 trace_length=sbi_pipeline.trace_length,
                                 model_config=model_config,
                                 flow_config=flow_config,
                                 lr=lr, weight_decay=weight_decay,
                                 lr_second_stage=lr_second_stage)
    # Build the training-time nuisance augmentation chain from the config: only
    # nuisances staged `training_augmentation` (Category-2 post-processing effects)
    # are folded in per-batch on the clean simulations. Empty chain ⇒ no augmentation.
    augmentation_chain, augmentation_nuisance_params = build_augmentation_chain_from_parameters(
        sbi_pipeline.parameters,
        sampling_rate=sbi_pipeline.simulation_parameters.sampling_rate,
    )
    print(f"Training-time nuisance augmentation: {list(augmentation_nuisance_params.keys()) or 'none'}")

    # Post-noise augmentation (e.g. component_dropout): applied to x = D + noise so dropped
    # channels are exactly zero (matching genuinely-absent components). Empty chain ⇒ no-op.
    post_noise_chain, post_noise_nuisance_params = build_augmentation_chain_from_parameters(
        sbi_pipeline.parameters,
        stage="training_augmentation_post_noise",
    )
    print(f"Post-noise augmentation: {list(post_noise_nuisance_params.keys()) or 'none'}")

    # Optional in-RAM caches via a top-level 'ml_cache' block (opt-in; speed-only). The per-sample
    # HDF5 reads (clean sim load + a 2nd open for conditioning + the RealNoiseSampler noise read)
    # are ~25-40 ms/sample and DON'T scale with workers (shared-file reads plateau). Once the model
    # step is fast (ml_perf), they dominate and starve the GPU. Preloading into contiguous RAM
    # buffers (fork-shared copy-on-write across workers) removes them.
    #   ml_cache:
    #     sims: true            # cache clean sim arrays (sim_count * N*C*T * dtype bytes)
    #     noise: true           # cache the RealNoiseSampler window pool
    #     dtype: float32        # sim-cache storage dtype (float32 == model precision; float64 byte-exact)
    #     preload_workers: 16
    _cache_cfg = _raw_cfg.get("ml_cache") or {}
    cache_sims = bool(_cache_cfg.get("sims", False))
    cache_noise = bool(_cache_cfg.get("noise", False))
    cache_dtype = str(_cache_cfg.get("dtype", "float32"))
    cache_workers = int(_cache_cfg.get("preload_workers", 16))
    if cache_noise and hasattr(sbi_pipeline.training_noise_sampler, "preload_cache"):
        sbi_pipeline.training_noise_sampler.preload_cache(max_workers=cache_workers)
    if _cache_cfg:
        print(f"ML in-RAM cache: sims={cache_sims} noise={cache_noise} dtype={cache_dtype}")

    num_sims = len(test_jobs_paths)
    train_max_index = int(0.90 * num_sims)
    dataloader_args = {
        'data_loader': sbi_pipeline.data_manager.data_loader,
        'data_folder': sbi_pipeline.simulations_output_path,
        'parameter_name_map': sbi_pipeline.parameters.names,
        'synthetic_noise_model_sampler': sbi_pipeline.training_noise_sampler,
        'augmentation_chain': augmentation_chain,
        'augmentation_nuisance_params': augmentation_nuisance_params,
        'data_scaler': data_scaler,
        'train_max_index': train_max_index,
        'train_batch_size': 128,
        'val_batch_size': 256,
        'train_shuffle': True,
        'val_shuffle': False,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 6,
        'conditioning_param_map': conditioning_param_map,
        'conditioning_noise_std': conditioning_noise_std,
        'station_subsampler': station_subsampler,
        'post_noise_augmentation_chain': post_noise_chain,
        'post_noise_nuisance_params': post_noise_nuisance_params,
        'cache_in_memory': cache_sims,
        'cache_dtype': cache_dtype,
        'cache_preload_workers': cache_workers,
    }
    run_name = args.run_name
    data_path = Path(config.pipeline_parameters.output_directory)/ config.pipeline_parameters.run_name / config.pipeline_parameters.job_name
    # Default logging is W&B (cloud; also writes a readable wandb-summary.json locally).
    # --csv_logger ADDS a deterministic on-disk metrics.csv (beside the checkpoints at
    # data_path/run_name/) ALONGSIDE W&B, so the remote `train-monitor` verb can parse
    # val_loss without network access AND the run still lands in W&B. --no_wandb drops W&B.
    loggers = []
    if not args.no_wandb:
        loggers.append("wandb")
    if args.csv_logger:
        from pytorch_lightning.loggers import CSVLogger
        loggers.append(CSVLogger(save_dir=str(data_path), name=run_name, version=""))
        print(f"CSV metrics logging to {Path(data_path)/run_name/'metrics.csv'}")
    logger = loggers if len(loggers) > 1 else (loggers[0] if loggers else False)
    trainer.train(run_name, epochs=args.epochs, output_path=data_path,
                  dataloader_args=dataloader_args, logger=logger)

if __name__ == '__main__':
    main()
