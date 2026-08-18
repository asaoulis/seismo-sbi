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

# Give numba an explicitly writable on-disk cache directory.
#
# instaseis JITs with @njit(cache=True) (finite_elem_mapping), so on the FIRST instaseis.open_db
# numba must resolve a cache "locator". Its fallback chain is: NUMBA_CACHE_DIR -> the source
# directory (site-packages/instaseis/) -> the user-wide cache (~/.cache). On a cluster the conda
# env AND ~/.cache both live in $HOME, so the moment HOME is full, read-only, or over quota EVERY
# locator fails and numba raises, killing the job at simulator-construction time:
#     RuntimeError: cannot cache function 'compute_theta_r': no locator available for file ...
# That took down two 500k dataset-generation jobs before they ran a single simulation.
#
# A compute job must not depend on a writable HOME for a JIT cache, so default it to node-local
# temp (TMPDIR-aware; per-user to avoid collisions on shared nodes). setdefault means an explicit
# NUMBA_CACHE_DIR from the submit script or the operator still wins. This MUST run before numba is
# first imported — numba reads the cache dir into numba.core.config at import — hence its position
# above the seismo_sbi imports, alongside the thread-count block.
if not os.environ.get("NUMBA_CACHE_DIR"):
    import tempfile as _tempfile
    _numba_cache = os.path.join(_tempfile.gettempdir(),
                                f"numba_cache_{os.environ.get('USER', 'seismo')}")
    os.makedirs(_numba_cache, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = _numba_cache

from pathlib import Path
from seismo_sbi.sbi.configuration import SBI_Configuration
from seismo_sbi.sbi.pipeline import SingleEventPipeline, MultiEventPipeline, VaryDatasetSizeEventPipeline
from seismo_sbi.sbi import utils as utils
from seismo_sbi.sbi.compression.ML.train import CompressionTrainer, load_warm_start_weights
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
    parser.add_argument('--devices', type=int, default=1,
                        help="Number of GPUs to train on. 1 (default) is the single-device path. "
                             ">1 engages multi-GPU DistributedDataParallel (one rank per GPU); each "
                             "rank gets its OWN batch of train_batch_size, so size the batch for a "
                             "single GPU. Set to $NGPU by the srun DDP launcher (submit_train.sh).")
    parser.add_argument('--train-batch-size', dest='train_batch_size', type=int, default=None,
                        help="Per-GPU training batch size. Overrides the config's ml_batch.train "
                             "(and the legacy default of 128). For DDP the GLOBAL batch is "
                             "train_batch_size * devices.")
    parser.add_argument('--val-batch-size', dest='val_batch_size', type=int, default=None,
                        help="Per-GPU validation batch size. Overrides the config's ml_batch.val "
                             "(default 2 * train_batch_size).")
    parser.add_argument('--write_meta_only', action='store_true',
                        help="Build the model exactly as a training run would, write the "
                             "model_meta.json sidecar next to the checkpoints, and exit WITHOUT "
                             "training. Recovers the sidecar for a run whose trainer.fit never "
                             "returned (e.g. a SLURM wall-clock kill), whose .ckpt files are fine "
                             "but which load_best would otherwise reload against default "
                             "architecture settings. MUST be given the same config (and "
                             "--architecture / --run_name) the run used; nothing is trained and no "
                             "checkpoint is touched, so it is safe to re-run.")
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

    # Optional per-worker cap on the open-Instaseis-DB LRU cache, from
    # `seismic_context.querier_cache_maxsize`. MUST be exported into the ENVIRONMENT (not just set
    # as a module global) and BEFORE any simulator is constructed: joblib/loky starts its workers
    # via spawn, so a runtime module global set in the parent would NOT reach them, whereas the
    # environment is inherited.
    #
    # WHY IT EXISTS: each cached handle costs ~55 MB resident and the cache is per worker process,
    # so an unbounded cap costs n_workers * n_members * 55 MB — ~206 GB for a 62-member Mode-A/B
    # ensemble at 60 workers, which OOM-killed a 500k dataset gen at 47% (SIGKILL'd loky worker).
    # See the memory-budget note in seismo_sbi/instaseis_simulator/ensemble.py. Purely a
    # memory/wall-clock trade: a miss costs one instaseis.open_db, never a different result.
    import yaml as _yaml_cap
    _qcap = (_yaml_cap.safe_load(Path(config_path).read_text())
             .get("seismic_context", {}) or {}).get("querier_cache_maxsize")
    if _qcap is not None:
        os.environ["SEISMO_QUERIER_CACHE_MAXSIZE"] = str(int(_qcap))
        print(f"Instaseis querier cache capped at {int(_qcap)} open handles/worker "
              f"(~{int(_qcap) * 55 / 1024:.1f} GB per worker process).")

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
    # Pin the theta scaling INTO the checkpoint. `model_config` is held by reference by
    # CompressionTrainer and dumped to model_meta.json, so the checkpoint is self-describing.
    # Without it the inverse transform is rebuilt at inference from whatever YAML is passed,
    # and an `ml_scaler`/`bounds` edit after training silently biases every recovered moment
    # by a constant factor instead of raising.
    from seismo_sbi.sbi.scalers import scaler_provenance
    model_config["theta_scaler"] = scaler_provenance(data_scaler)
    print(f"theta scaler provenance recorded: {model_config['theta_scaler']}")

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

    # Optional summary BOTTLENECK via a top-level 'ml_summary_bottleneck' block:
    #   ml_summary_bottleneck:
    #     dim: 32                  # width of the summary the MMD loss compares
    # Narrows ONLY the summary: the encoder keeps its channel width and the flow keeps its
    # `latent_dim`-wide context, so neither capacity changes (do NOT lower `channels` for this
    # — model_dim ties the flow's hidden width too, so that would shrink the flow as well and
    # confound a latent-dim result with a capacity result). Absent => unchanged behaviour.
    _bneck_cfg = _raw_cfg.get("ml_summary_bottleneck")
    if _bneck_cfg and _bneck_cfg.get("dim"):
        model_config["summary_bottleneck"] = {"dim": int(_bneck_cfg["dim"])}
        print(f"Summary bottleneck enabled: {model_config['summary_bottleneck']} "
              f"(flow context stays {model_dim}-wide)")

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
    #     lr_schedule: constant      # cosine (default; warmup->decay to lr*lr_min_factor) | constant | cyclic
    #     lr_min_factor: 0.2         # cosine floor as a fraction of lr: eta_min = lr*factor.
    #                                #   0.1 = legacy (lr/10); 0.2 = lr/5. Cosine branch only.
    # Absent => lr=1e-4, weight_decay=1e-4, cosine to lr/10 (the legacy schedule).
    _opt_cfg = _raw_cfg.get("ml_optimizer") or {}
    lr = float(_opt_cfg.get("lr", 1e-4))
    weight_decay = float(_opt_cfg.get("weight_decay", 1e-4))
    lr_second_stage = _opt_cfg.get("lr_schedule", "cosine")
    lr_min_factor = float(_opt_cfg.get("lr_min_factor", 0.1))
    if _opt_cfg:
        print(f"Optimizer overrides: lr={lr}, weight_decay={weight_decay}, "
              f"lr_schedule={lr_second_stage}, lr_min_factor={lr_min_factor} "
              f"(cosine eta_min={lr * lr_min_factor:.3e})")

    trainer = CompressionTrainer(components, station_locations, channels=model_dim, latent_dim=model_dim,
                                 trace_length=sbi_pipeline.trace_length,
                                 model_config=model_config,
                                 flow_config=flow_config,
                                 lr=lr, weight_decay=weight_decay,
                                 lr_second_stage=lr_second_stage,
                                 lr_min_factor=lr_min_factor)
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

    # Per-GPU batch sizing: CLI overrides the optional top-level `ml_batch` YAML block,
    # which overrides the legacy 128/256/8 defaults. Under DDP each rank/GPU consumes a
    # full train_batch_size, so this is the PER-GPU batch (global batch = train_bs * devices).
    #   ml_batch:
    #     train: 32           # per-GPU training batch (e.g. 32 * 4 GPUs = 128 global)
    #     val: 64             # per-GPU validation batch (default: 2 * train)
    #     num_workers: 8      # DataLoader workers per rank
    _batch_cfg = _raw_cfg.get("ml_batch") or {}
    train_bs = int(args.train_batch_size or _batch_cfg.get("train", 128))
    val_bs = int(args.val_batch_size or _batch_cfg.get("val", 2 * train_bs))
    num_workers = int(_batch_cfg.get("num_workers", 8))
    if args.devices > 1:
        print(f"DDP enabled: devices={args.devices}, per-GPU train_batch_size={train_bs} "
              f"(global batch {train_bs * args.devices}), val_batch_size={val_bs}")
    # The noise pool is training data, not metadata: --write_meta_only exits before any
    # dataloader is built, so preloading it would burn ~30 min (4.2 GB for this config) to
    # produce a file that does not contain a single noise-derived field.
    if cache_noise and args.write_meta_only:
        print("[write_meta_only] skipping the RealNoiseSampler cache preload (unused by the sidecar)")
    elif cache_noise and hasattr(sbi_pipeline.training_noise_sampler, "preload_cache"):
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
        'train_batch_size': train_bs,
        'val_batch_size': val_bs,
        'train_shuffle': True,
        'val_shuffle': False,
        'num_workers': num_workers,
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
    # ---- Misspecification-robust MMD auxiliary loss (top-level 'ml_mmd' block; opt-in).
    # Aligns summaries of QA-cleaned REAL events with a POSTERIOR-MATCHED sim suite in
    # embedding space (unbiased mixture-RBF MMD; see sbi/compression/ML/mmd.py and the
    # ml-architectures/model-misspec-mmd task). Absent/disabled => byte-identical legacy loss.
    #   ml_mmd:
    #     enabled: true
    #     real_events_manifest: /data/.../mmd_manifest.json   # from the QA'd catalogue run
    #     psim_data_folder: /data/.../psim_suite              # generate_psim_suite.py output
    #     lambda_mmd: 0.05       # weight after ramp (paper range 0.01-0.1)
    #     warmup_epochs: 5       # NLL-only epochs before the ramp
    #     ramp_epochs: 5         # linear ramp 0 -> lambda_mmd
    #     every_n_steps: 1       # compute the MMD term every N optimizer steps
    #     batch_size: 64         # sub-batch per side per MMD evaluation
    #     clean_only: true       # exclude neighbour/contamination-flagged events
    _mmd_cfg = _raw_cfg.get("ml_mmd") or {}
    if _mmd_cfg.get("enabled", False):
        from seismo_sbi.sbi.compression.ML.mmd_data import (
            build_real_context, build_psim_loader)
        real_ctx = build_real_context(
            _mmd_cfg["real_events_manifest"],
            sbi_pipeline.data_manager.data_loader,
            clean_only=bool(_mmd_cfg.get("clean_only", True)),
            # optional host-portable override: the manifest stores absolute paths from
            # the machine that built it; this key is a plain YAML leaf, so the cluster
            # orchestrator's path remap covers it.
            events_h5_dir=_mmd_cfg.get("events_h5_dir"))
        psim_loader = build_psim_loader(
            _mmd_cfg["psim_data_folder"], _mmd_cfg["real_events_manifest"],
            data_loader=sbi_pipeline.data_manager.data_loader,
            synthetic_noise_model_sampler=sbi_pipeline.training_noise_sampler,
            augmentation_chain=augmentation_chain,
            augmentation_nuisance_params=augmentation_nuisance_params,
            conditioning_param_map=conditioning_param_map,
            batch_size=int(_mmd_cfg.get("batch_size", 64)),
            clean_only=bool(_mmd_cfg.get("clean_only", True)))
        trainer.model.enable_mmd(_mmd_cfg, real_ctx, psim_loader)
        # Record the MMD block in the model_meta.json sidecar so a checkpoint knows how it
        # was trained. MUST go through record_model_config: CompressionTrainer.__init__
        # MERGES model_config into a NEW dict, so mutating our own copy here (as this used
        # to do) never reached the sidecar — every MMD checkpoint was metadata-identical to
        # a non-MMD one, and a lambda sweep would be unattributable after the fact.
        model_config["mmd"] = {k: v for k, v in _mmd_cfg.items() if k != "enabled"}
        trainer.record_model_config(mmd=model_config["mmd"])
        print(f"MMD auxiliary loss enabled: N_real={real_ctx.shape[0]}, "
              f"N_psim={len(psim_loader.dataset)}, lambda={_mmd_cfg.get('lambda_mmd', 0.05)}, "
              f"warmup={_mmd_cfg.get('warmup_epochs', 5)}+ramp={_mmd_cfg.get('ramp_epochs', 5)} epochs")

    run_name = args.run_name
    data_path = Path(config.pipeline_parameters.output_directory)/ config.pipeline_parameters.run_name / config.pipeline_parameters.job_name

    # Sidecar-only recovery path: everything above has built the model exactly as a real run
    # would (architecture, model_config incl. the theta-scaler provenance and any MMD block,
    # flow_config, trace_length, station locations), so writing the sidecar here yields the
    # SAME file trainer.train() would have written. Placed after the MMD branch so an
    # MMD-trained run's block is included. No dataloaders are built and no .ckpt is touched.
    if args.write_meta_only:
        meta_path = trainer.write_model_meta(Path(data_path) / run_name)
        print(f"[write_meta_only] wrote {meta_path} (no training performed)")
        return

    # Optional WARM START via a top-level 'ml_warm_start' block:
    #   ml_warm_start:
    #     from_run_name: japan_stffix_v1_tcn   # a training run under THIS data_path
    # Loads that run's best checkpoint into the freshly-built flow and trains on from there.
    # Deliberately keyed by run NAME, not an absolute path, so the same config works locally
    # and on the cluster (data_path is already per-machine via output_directory).
    # NOT a Lightning resume: optimizer moments / LR position / epoch counter start fresh, so
    # --epochs is a brand-new schedule budget (warmup = 5% of it, cosine T_max = the rest).
    # Missing checkpoint or any key/shape mismatch RAISES: a continuation that silently fell
    # back to random init would look like a continuation in every log but the weights.
    _ws_cfg = _raw_cfg.get("ml_warm_start") or {}
    _ws_from = _ws_cfg.get("from_run_name")
    if _ws_from:
        _ws_ckpt = load_warm_start_weights(trainer.model, Path(data_path) / _ws_from)
        print(f"Warm start: loaded {_ws_ckpt} into the flow (strict=True); "
              f"optimizer + LR schedule start FRESH at lr={lr} over {args.epochs} epochs")
        # Provenance into model_meta.json — without this the continuation checkpoint is
        # indistinguishable from a from-scratch run at the same lr (see write_model_meta).
        trainer.record_model_config({"warm_start_checkpoint": str(_ws_ckpt)})

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
                  dataloader_args=dataloader_args, logger=logger, devices=args.devices)

if __name__ == '__main__':
    main()
