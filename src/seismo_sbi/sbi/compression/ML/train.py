import torch
from torch import nn
import os
import json
from pathlib import Path
import re
from glob import glob
import numpy as np

from .seismogram_transformer import SeismogramTransformer, NPELightningModule
from .maf import build_nsf, build_maf
from .dataloading import make_torch_dataloader, make_torch_dataloaders

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
  # added
# from lightning.pytorch.profiler import AdvancedProfiler, SimpleProfiler, PyTorchProfiler

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


# Registry of embedding-net builders. Add new ML compression architectures here and they
# become selectable by name from train_NPE.py and the e2e test. Each builder receives the
# uniform kwargs bundle assembled in CompressionTrainer.__init__ (num_seismic_components,
# model_config, feature_length, latent_dim, station_locations, device, trace_length) and
# must return an nn.Module emitting a context of width `latent_dim` (the flow's conditional
# dimension). Accept **_unused so the bundle can grow without breaking existing builders.
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
                 model_config=None, flow_config=None):
        """Build the embedding net + conditional normalising flow.

        trace_length: per-trace sample count of the data (CNN input length). Defaults to
            200 for backward compatibility; pass the pipeline's real ``trace_length``.
        model_config / flow_config: optional overrides merged over DEFAULT_MODEL_CONFIG /
            DEFAULT_FLOW_CONFIG.
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
        # Store resolved configs + station locations for checkpoint metadata and rebuild.
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

        # Lightning module that maximizes log p_phi(theta | x)
        self.model = NPELightningModule(
            flow=self.flow,
            lr=lr,
            weight_decay=weight_decay,
        )

    @staticmethod
    def _assemble_flow(*, architecture, num_seismic_components, model_config, flow_config,
                       feature_length, latent_dim, num_dims, station_locations, trace_length,
                       device):
        """Build the embedding net (by registry name) + conditional NSF flow.

        Shared by ``__init__`` and ``load_best`` so a checkpoint can be rebuilt from its
        sidecar metadata (architecture / model_config / flow_config) rather than whatever
        configuration the loading trainer happened to be constructed with.
        """
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
        # hidden_features matches the resolved model channels (model_config carries it).
        return build_nsf(
            dim=num_dims,
            conditional_dim=latent_dim,
            hidden_features=model_config["channels"],
            embedding_net=embedding_net,
            **flow_config,
        )

    def train(self, run_name, epochs=10, output_path=Path("model_ckpts"), dataloader_args: dict = None,
              logger="wandb", enable_checkpointing=True, enable_progress_bar=True):
        """Train the flow.

        Defaults preserve production behaviour (W&B logging + checkpointing). For headless
        runs (tests/CI) pass ``logger=None`` (or ``False``) to disable logging entirely,
        ``enable_checkpointing=False`` to skip writing .ckpt files, and
        ``enable_progress_bar=False`` for clean output. Returns the trained model so callers
        that disabled checkpointing can use it without reading a checkpoint from disk.
        """
        if dataloader_args is None or "train_max_index" not in dataloader_args:
            raise ValueError("dataloader_args must include: data_loader, data_folder, parameter_name_map, synthetic_noise_model_sampler, and train_max_index.")

        # Build train/val dataloaders from a single split index
        train_dataloader, val_dataloader = make_torch_dataloaders(**dataloader_args)

        output_path = Path(output_path) / run_name

        # Resolve the logger: "wandb" -> WandbLogger (production); None/False -> no logging;
        # a list/tuple -> resolve each element and log to ALL of them (e.g.
        # ["wandb", CSVLogger(...)] writes the W&B run AND a deterministic metrics.csv);
        # anything else is treated as an already-constructed Lightning logger.
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

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices=1,
            callbacks=callbacks,
            precision=32,
            logger=pl_logger,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
        )

        trainer.fit(self.model, train_dataloader, val_dataloader)

        # Write sidecar metadata so new architectures can be reloaded without
        # hard-coding defaults.  Old checkpoints that lack this file fall back
        # to the current defaults in load_best() for backward compatibility.
        if enable_checkpointing:
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

        return self.model

    def load_best(self, output_path: Path) -> Path:
        """
        Locate and load the best-performing checkpoint into self.model.
        Returns the path to the checkpoint that was loaded.

        If a ``model_meta.json`` sidecar exists alongside the checkpoint directory,
        it is loaded and its ``architecture`` / ``model_config`` / ``flow_config``
        values are used to rebuild the flow before loading weights.  Old checkpoints
        that lack the sidecar fall back silently to the current object's flow.
        """
        output_path = Path(output_path)
        ckpt_path = find_best_checkpoint_path(output_path)
        print(ckpt_path)

        # Rebuild the flow from sidecar metadata so a checkpoint trained with a different
        # architecture / model_config / flow_config (e.g. a PNO or conditioned model) loads
        # into a structurally matching flow even when this trainer was constructed with
        # different/default settings. Old checkpoints lacking the sidecar fall back to the
        # flow built in __init__.
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

        self.model = NPELightningModule.load_from_checkpoint(
            ckpt_path,
            flow=self.flow,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.model.eval()
        self.model.freeze()
        return ckpt_path
        
    
    def build_posterior(self):
        # use sbi to build a direct posterior from the trained flow
        from sbi.inference.posteriors import DirectPosterior
        from sbi import utils as utils

        device = str(self.device)
        prior = utils.BoxUniform(low=np.zeros((self.num_dims)), high=np.ones((self.num_dims)), device=device)

        posterior = DirectPosterior(
                    posterior_estimator=self.model.flow.to(device),
                    prior=prior,
                    # x_shape=self._x_shape,
                    device=device,
                )
        return posterior
from pytorch_lightning.callbacks import ModelCheckpoint

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