"""Training of the neural compressor: an embedding net plus a conditional normalising flow.

:class:`CompressionTrainer` builds the pair, trains it with Lightning, and writes a
``model_meta.json`` sidecar so a checkpoint can be rebuilt without re-supplying its
architecture. The free functions below wire in the optional extras a configuration may ask
for — the auxiliary MMD loss, a warm start from an earlier run, and the metric loggers.
"""

import json
import re
from pathlib import Path

import numpy as np
import torch

from seismo_sbi.sbi.npe.networks.seismogram_transformer import SeismogramTransformer
from seismo_sbi.sbi.npe.training.lightning_module import NPELightningModule
from seismo_sbi.sbi.npe.maf import build_nsf
from seismo_sbi.sbi.npe.data.dataloading import make_torch_dataloaders
from seismo_sbi.sbi.npe.training.checkpoint_loading import unpickling_torch_load

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

def _build_seismogram_transformer(*, num_seismic_components, model_config,
                                  feature_length, latent_dim, station_locations, device,
                                  trace_length, **_unused):
    """Default embedding net (context extractor). Output width must equal latent_dim.

    ``trace_length`` is the per-trace sample count of the data; it sizes the CNN's
    conv stack so the model matches the data instead of assuming a fixed length.
    """
    return SeismogramTransformer(
        num_seismic_components,
        model_config,
        feature_length,
        num_outputs=latent_dim,          # not used for embedding, but required by ctor
        noise_model=None,                # not used in NPE; pass None
        seismogram_locations=station_locations,
        device=device,
        input_length=trace_length,
    )


#: Embedding-net builders, selectable by name from a configuration. Each returns an nn.Module
#: emitting a context of width ``latent_dim``; take ``**_unused`` so the kwargs bundle can grow.
EMBEDDING_NET_REGISTRY = {
    "seismogram_transformer": _build_seismogram_transformer,
}

# Default hyperparameters, grouped so callers can override piecemeal instead of editing code.
DEFAULT_MODEL_CONFIG = {"layers": 4, "nheads": 4, "timeemb": 64, "posemb": 64}
DEFAULT_FLOW_CONFIG = {
    "num_transforms": 5,
    "num_blocks": 2,
    "dropout_probability": 0.0,
    "use_batch_norm": True,
}


class CompressionTrainer:

    def __init__(self, components, station_locations, channels=128, latent_dim=128,
                 architecture="seismogram_transformer", trace_length=200,
                 num_dims=6, feature_length=128, lr=1e-4, weight_decay=1e-4,
                 model_config=None, flow_config=None, lr_second_stage="cosine",
                 lr_min_factor=0.1):
        """Build the embedding net and the conditional normalising flow.

        ``components`` and ``station_locations`` describe the recording geometry; ``trace_length`` is
        the per-trace sample count of the data. ``model_config`` and ``flow_config`` override entries of
        ``DEFAULT_MODEL_CONFIG`` and ``DEFAULT_FLOW_CONFIG``. ``lr_second_stage`` is the learning-rate
        schedule after warm-up: ``"cosine"`` decays to ``lr * lr_min_factor``, ``"constant"`` holds
        ``lr``, ``"cyclic"`` cycles.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_seismic_components = len(components)

        model_config = {**DEFAULT_MODEL_CONFIG, "channels": channels, **(model_config or {})}
        flow_config = {**DEFAULT_FLOW_CONFIG, **(flow_config or {})}

        self.num_dims = num_dims
        self.latent_dim = latent_dim
        self.trace_length = trace_length
        self.architecture = architecture
        self.num_seismic_components = num_seismic_components
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_second_stage = lr_second_stage
        self.lr_min_factor = float(lr_min_factor)
        # The MERGED dict built above, a NEW object: entries the caller adds to its own dict
        # afterwards never reach the sidecar — use :meth:`record_model_config` for those.
        self._model_config = model_config
        self._flow_config = flow_config
        self._feature_length = feature_length
        self._station_locations = station_locations
        self._station_locations_shape = (
            list(station_locations.shape)
            if hasattr(station_locations, "shape")
            else None
        )

        # Conditional MAF over theta | x with embedding integrated in the flow.
        self.flow = self._assemble_flow(
            architecture=architecture,
            num_seismic_components=num_seismic_components,
            model_config=model_config,
            flow_config=flow_config,
            feature_length=feature_length,
            latent_dim=latent_dim,
            num_dims=num_dims,
            station_locations=station_locations,
            trace_length=trace_length,
            device=self.device,
        )

        # ``model_config['perf']`` also holds the optimizer and compile options; absent means all off.
        _perf = model_config.get("perf", {}) or {}
        self.model = NPELightningModule(
            flow=self.flow,
            lr=lr,
            weight_decay=weight_decay,
            lr_second_stage=lr_second_stage,
            lr_min_factor=lr_min_factor,
            fused_adam=bool(_perf.get("fused_adam", False)),
            compile_forward=bool(_perf.get("compile", False)),
            compile_flow=bool(_perf.get("compile_flow", False)),
        )

    @staticmethod
    def _assemble_flow(*, architecture, num_seismic_components, model_config, flow_config,
                       feature_length, latent_dim, num_dims, station_locations, trace_length,
                       device):
        """The embedding net named by ``architecture`` wrapped in a conditional NSF flow."""
        if architecture not in EMBEDDING_NET_REGISTRY:
            raise KeyError(
                f"unknown architecture '{architecture}'; "
                f"registered: {sorted(EMBEDDING_NET_REGISTRY)}"
            )
        embedding_net = EMBEDDING_NET_REGISTRY[architecture](
            num_seismic_components=num_seismic_components,
            model_config=model_config,
            feature_length=feature_length,
            latent_dim=latent_dim,
            station_locations=station_locations,
            device=device,
            trace_length=trace_length,
        )
        # The flow's hidden width follows the embedding channels unless flow_config overrides
        # it; popped so it is not also passed below as a duplicate keyword.
        flow_kwargs = dict(flow_config)
        flow_hidden_features = flow_kwargs.pop("hidden_features", None) or model_config["channels"]
        return build_nsf(
            dim=num_dims,
            conditional_dim=latent_dim,
            hidden_features=flow_hidden_features,
            embedding_net=embedding_net,
            **flow_kwargs,
        )

    @classmethod
    def from_configuration(cls, training, components, station_locations, trace_length,
                           theta_scaler_provenance):
        """Build the trainer described by a :class:`TrainingConfiguration`.

        ``theta_scaler_provenance`` is recorded in the checkpoint's sidecar so the parameter
        scaling used at inference cannot silently differ from the one trained under.
        """
        return cls(
            components, station_locations,
            channels=training.model_dim, latent_dim=training.model_dim,
            trace_length=trace_length,
            model_config=training.to_model_config(theta_scaler_provenance),
            flow_config=training.flow,
            lr=training.optimizer.lr,
            weight_decay=training.optimizer.weight_decay,
            lr_second_stage=training.optimizer.lr_schedule,
            lr_min_factor=training.optimizer.lr_min_factor,
        )

    def record_model_config(self, **entries):
        """Merge extra entries into the ``model_config`` recorded in ``model_meta.json``.

        Settings resolved only once the trainer exists, such as the MMD auxiliary-loss block, are
        registered here so the checkpoint's metadata carries them.
        """
        self._model_config.update(entries)
        return self._model_config

    def train(self, run_name, epochs=10, output_path=Path("model_ckpts"), dataloader_args: dict = None,
              logger="wandb", enable_checkpointing=True, enable_progress_bar=True,
              extra_callbacks=None, devices=1, strategy=None):
        """Train the flow; returns the trained model.

        ``logger=None`` disables logging, ``enable_checkpointing=False`` skips writing ``.ckpt`` files
        and ``enable_progress_bar=False`` silences the bar. ``extra_callbacks`` are appended to the
        Trainer's. ``devices`` is the number of accelerators; above one, each rank trains on its own
        batch of ``train_batch_size`` samples, so size that for a single device, and ``strategy``
        overrides the distributed strategy.
        """
        if dataloader_args is None or "train_max_index" not in dataloader_args:
            raise ValueError("dataloader_args must include train_max_index and either a dataset or "
                             "data_loader, data_folder, parameter_name_map and synthetic_noise_model_sampler.")

        # Build train/val dataloaders from a single split index
        train_dataloader, val_dataloader = make_torch_dataloaders(**dataloader_args)

        output_path = Path(output_path) / run_name

        # "wandb" builds a WandbLogger, None/False disables logging, a list logs to all of its
        # elements, and anything else is taken to be a constructed Lightning logger.
        def _resolve_logger(spec):
            if spec == "wandb":
                return WandbLogger(project="seismo-sbi", name=output_path.parent.name + '/' + run_name)
            if spec in (None, False):
                return None
            return spec

        if isinstance(logger, (list, tuple)):
            resolved = [r for r in (_resolve_logger(x) for x in logger) if r is not None]
            pl_logger = resolved if resolved else False
        else:
            pl_logger = _resolve_logger(logger)
            if pl_logger is None:
                pl_logger = False

        callbacks = []
        if enable_checkpointing:
            callbacks.append(create_best_checkpoint_callback(output_path))
        # LearningRateMonitor requires a logger to write to; only add it when logging is on.
        if pl_logger is not False:
            callbacks.append(LearningRateMonitor(logging_interval='epoch'))
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        # find_unused_parameters is the safe default for this many-branch model: variable
        # stations, conditioning, amplitude and positional embeddings and the pooling head.
        if strategy is None:
            strategy = (DDPStrategy(find_unused_parameters=True) if devices and devices > 1 else "auto")

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices=devices,
            num_nodes=1,
            strategy=strategy,
            callbacks=callbacks,
            precision=32,
            logger=pl_logger,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
        )

        # Written before fit too, so a run killed mid-fit still records how it was trained.
        if enable_checkpointing and trainer.is_global_zero:
            self.write_model_meta(output_path)
        trainer.fit(self.model, train_dataloader, val_dataloader)

        # Rank 0 only, so multi-GPU ranks do not race-write the same sidecar; on the
        # single-device path is_global_zero is always true.
        if enable_checkpointing and trainer.is_global_zero:
            self.write_model_meta(output_path)

        return self.model

    def write_model_meta(self, output_path: Path) -> Path:
        """Write the ``model_meta.json`` sidecar describing this trainer's architecture.

        :meth:`train` calls it before and after fitting; ``train_NPE.py --stage meta`` calls it for a
        run that was killed before its sidecar was written, from the same configuration the run used.
        """
        output_path = Path(output_path)
        meta = {
            "architecture": self.architecture,
            "model_config": self._model_config,
            "flow_config": self._flow_config,
            "trace_length": self.trace_length,
            "num_seismic_components": self.num_seismic_components,
            "num_dims": self.num_dims,
            "latent_dim": self.latent_dim,
            "feature_length": self._feature_length,
            "station_locations_shape": self._station_locations_shape,
            # Store the actual coordinates so the sidecar is self-describing and
            # load_best can rebuild the embedding net without re-supplying them.
            "station_locations": np.asarray(self._station_locations).tolist(),
        }
        meta_path = output_path / "model_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            # default=str guards against a non-JSON value sneaking into a config dict
            # aborting the dump after a (possibly long) successful training run.
            json.dump(meta, f, indent=2, default=str)
        return meta_path

    def load_best(self, output_path: Path) -> Path:
        """Load the best checkpoint under ``output_path`` into ``self.model``; returns its path.

        The ``model_meta.json`` sidecar, when present, gives the ``architecture``, ``model_config`` and
        ``flow_config`` the flow is rebuilt with before the weights are loaded; without it the flow of
        this object is used.
        """
        output_path = Path(output_path)
        ckpt_path = find_best_checkpoint_path(output_path)
        print(ckpt_path)

        # Rebuild from the sidecar so a checkpoint loads into a structurally matching flow
        # whatever this trainer was constructed with; without one, keep the flow from __init__.
        meta_path = output_path / "model_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            station_locations = (
                np.asarray(meta["station_locations"])
                if meta.get("station_locations") is not None
                else self._station_locations
            )
            self.architecture = meta.get("architecture", self.architecture)
            self.num_dims = meta.get("num_dims", self.num_dims)
            self.latent_dim = meta.get("latent_dim", self.latent_dim)
            self.trace_length = meta.get("trace_length", self.trace_length)
            self.num_seismic_components = meta.get("num_seismic_components", self.num_seismic_components)
            self._model_config = meta.get("model_config", self._model_config)
            self._flow_config = meta.get("flow_config", self._flow_config)
            self._feature_length = meta.get("feature_length", self._feature_length)
            self.flow = self._assemble_flow(
                architecture=self.architecture,
                num_seismic_components=self.num_seismic_components,
                model_config=self._model_config,
                flow_config=self._flow_config,
                feature_length=self._feature_length,
                latent_dim=self.latent_dim,
                num_dims=self.num_dims,
                station_locations=station_locations,
                trace_length=self.trace_length,
                device=self.device,
            )

        self._load_checkpoint(ckpt_path)
        return ckpt_path

    def _load_checkpoint(self, ckpt_path):
        """Load the weights at ``ckpt_path`` into the flow and freeze it for inference."""
        with unpickling_torch_load():
            self.model = NPELightningModule.load_from_checkpoint(
                ckpt_path,
                flow=self.flow,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        self.model.eval()
        self.model.freeze()

    def build_posterior(self):
        """An ``sbi`` ``DirectPosterior`` over the trained flow, on the flow's device."""
        # use sbi to build a direct posterior from the trained flow
        from sbi.inference.posteriors import DirectPosterior
        from sbi import utils as utils

        device = str(self.device)
        prior = utils.BoxUniform(low=np.zeros((self.num_dims)), high=np.ones((self.num_dims)), device=device)

        posterior = DirectPosterior(
                    posterior_estimator=self.model.flow.to(device),
                    prior=prior,
                    device=device,
                )
        return posterior


def create_best_checkpoint_callback(output_path):
    best_checkpoint_callback = ModelCheckpoint(
            dirpath=f'{output_path}/checkpoints',
            filename='best_model-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        )
    return best_checkpoint_callback

def find_best_checkpoint_path(output_path: Path) -> Path:
    """
    Find the best .ckpt under <output_path>/checkpoints by parsing the val_loss
    encoded in the filename 'best_model-{val_loss:.2f}.ckpt'.
    """
    ckpt_dir = Path(output_path) / "checkpoints"
    candidates = list(ckpt_dir.glob("best_model-*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    def score_from_name(p: Path) -> float:
        m = re.search(r"best_model-val_loss=(-?\d+(?:\.\d+)?)(?:-v\d+)?\.ckpt$", p.name)
        return float(m.group(1)) if m else float("inf")
    best = min(candidates, key=score_from_name)
    return best

def load_warm_start_weights(model, source_path: Path) -> Path:
    """Load the best checkpoint under ``<source_path>/checkpoints`` into ``model``'s weights;
    returns its path.

    Only the weights are transferred: the optimiser state, the learning-rate schedule and the epoch
    counter start afresh, so the new run declares its own ``--epochs``. Every architecture block
    (``ml_architecture``, ``ml_conditioning``, ``ml_variable_stations``, ``ml_amplitude_embedding``,
    ``ml_positional_encoding``, ``ml_pooling``, ``ml_flow``, ``ml_encoder``) must match the source
    run; a mismatch raises.
    """
    ckpt_path = find_best_checkpoint_path(source_path)
    # weights_only=False: a Lightning .ckpt carries non-tensor entries (hyper_parameters,
    # callback state) beside the weights, which the weights_only unpickler rejects.
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in state:
        raise KeyError(f"{ckpt_path} has no 'state_dict' — not a Lightning checkpoint")
    model.load_state_dict(state["state_dict"], strict=True)
    return ckpt_path


def apply_warm_start(trainer, training, models_output_path):
    """Load an earlier run's best weights into the fresh flow, if ``ml_warm_start`` asks for it.

    The source run is named relative to ``models_output_path`` so the same configuration works
    on any machine. The checkpoint that was loaded is recorded in the sidecar.
    """
    if not training.warm_start_run_name:
        return
    checkpoint_path = load_warm_start_weights(
        trainer.model, Path(models_output_path) / training.warm_start_run_name)
    trainer.record_model_config(warm_start_checkpoint=str(checkpoint_path))
    print(f"Warm start from {checkpoint_path}; the optimizer and LR schedule start fresh at "
          f"lr={training.optimizer.lr} over {training.epochs} epochs")


def enable_mmd_loss(trainer, training, pipeline, data):
    """Switch on the misspecification-robust MMD auxiliary loss, if ``ml_mmd`` asks for it.

    It aligns the summaries of QA-cleaned real events with those of a posterior-matched
    simulation suite in embedding space. Absent or disabled leaves the plain likelihood loss.
    """
    mmd = training.mmd
    if not mmd.get("enabled", False):
        return
    from seismo_sbi.sbi.npe.data.mmd_data import build_real_context, build_psim_loader

    clean_only = bool(mmd.get("clean_only", True))
    real_context = build_real_context(
        mmd["real_events_manifest"], pipeline.data_manager.data_loader,
        clean_only=clean_only,
        # The manifest holds absolute paths from the machine that wrote it; this relocates the
        # event files without rewriting it.
        events_h5_dir=mmd.get("events_h5_dir"))
    psim_loader = build_psim_loader(
        mmd["psim_data_folder"], mmd["real_events_manifest"],
        data_loader=pipeline.data_manager.data_loader,
        synthetic_noise_model_sampler=pipeline.training_noise_sampler,
        augmentation_chain=data.augmentation_chain,
        augmentation_nuisance_params=data.augmentation_nuisance_params,
        conditioning_param_map=training.conditioning.param_map,
        batch_size=int(mmd.get("batch_size", 64)),
        clean_only=clean_only)
    trainer.model.enable_mmd(mmd, real_context, psim_loader)
    # Through the recorder, not the caller's dict: __init__ merged model_config into a new
    # object, so a checkpoint would otherwise be indistinguishable from a non-MMD one.
    trainer.record_model_config(mmd={key: value for key, value in mmd.items()
                                     if key != "enabled"})
    print(f"MMD auxiliary loss enabled: N_real={real_context.shape[0]}, "
          f"N_psim={len(psim_loader.dataset)}, lambda={mmd.get('lambda_mmd', 0.05)}, "
          f"warmup={mmd.get('warmup_epochs', 5)}+ramp={mmd.get('ramp_epochs', 5)} epochs")


def attach_loggers(logging, run_directory):
    """The Lightning logger specification for the configured metric sinks.

    ``csv`` writes a ``metrics.csv`` beside the checkpoints, readable without network access;
    ``wandb`` streams the run to Weights & Biases as well.
    """
    loggers = []
    if logging.wandb:
        loggers.append("wandb")
    if logging.csv:
        run_directory = Path(run_directory)
        loggers.append(CSVLogger(save_dir=str(run_directory.parent),
                                 name=run_directory.name, version=""))
        print(f"CSV metrics logging to {run_directory / 'metrics.csv'}")
    if len(loggers) > 1:
        return loggers
    return loggers[0] if loggers else False
