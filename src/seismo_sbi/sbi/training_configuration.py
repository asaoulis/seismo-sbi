"""Typed view of the ``ml_*`` blocks of an SBI configuration file.

:meth:`TrainingConfiguration.from_yaml_block` parses the whole training section once. The
launcher folds in the command-line overrides a scheduler sets, :meth:`to_model_config` builds
the architecture dictionary stored in a checkpoint's ``model_meta.json`` sidecar, and
:meth:`dataloader_args` the keyword arguments of the training and validation dataloaders.
Unknown ``ml_*`` keys raise, so a mistyped block is not silently ignored.
"""

from dataclasses import dataclass, field

from ..utils.errors import InvalidConfiguration

#: Top-level configuration blocks that configure ML training.
TRAINING_BLOCKS = frozenset({
    "ml_architecture", "ml_encoder", "ml_conditioning", "ml_variable_stations",
    "ml_amplitude_embedding", "ml_positional_encoding", "ml_pooling", "ml_summary_bottleneck",
    "ml_flow", "ml_perf", "ml_optimizer", "ml_cache", "ml_batch", "ml_mmd", "ml_warm_start",
    "ml_logging", "ml_scaler",
})

#: Keys of the blocks read by name here; every other block is forwarded verbatim to a
#: constructor, so its keys are only validated there.
NAMED_BLOCK_KEYS = {
    "ml_conditioning": {"param_map", "n_cond", "d_cond", "coord_mode", "inject", "n_fourier"},
    "ml_variable_stations": {"enabled", "keep_fraction", "min_stations", "station_coords_mode"},
    "ml_summary_bottleneck": {"dim"},
    "ml_optimizer": {"lr", "weight_decay", "lr_schedule", "lr_min_factor"},
    "ml_batch": {"train", "val", "num_workers", "prefetch_factor", "train_fraction"},
    "ml_cache": {"sims", "noise", "dtype", "preload_workers"},
    "ml_warm_start": {"from_run_name"},
    "ml_logging": {"wandb", "csv"},
}


@dataclass
class EncoderConfig:
    """Per-station encoder choice (``ml_architecture``) and its keyword arguments (``ml_encoder``).

    ``input_decimate`` is a Nyquist-aware decimation applied at the model entry, before the
    encoder, so it is not one of the encoder's own arguments.
    """

    station_encoder: str = "cnn"
    input_decimate: dict = None
    options: dict = field(default_factory=dict)

    @classmethod
    def from_yaml_block(cls, config):
        options = _block_without_switch(config.get("ml_encoder"))
        decimate = options.pop("input_decimate", None) if options else None
        if decimate:
            decimate = dict(decimate) if isinstance(decimate, dict) else {"factor": int(decimate)}
        return cls(station_encoder=config.get("ml_architecture") or "cnn",
                   input_decimate=decimate or None,
                   options=options or {})


@dataclass
class ConditioningConfig:
    """Source-location conditioning of the encoder (``ml_conditioning``).

    ``coordinate_noise_std`` holds the per-coordinate Gaussian widths, in the ``param_map``
    order, of the training-time perturbation of the conditioning vector; it comes from a
    ``source_location_error`` nuisance staged as a training augmentation.
    """

    param_map: dict = None
    n_cond: int = None
    d_cond: int = None
    coord_mode: str = "geographic"
    inject: list = field(default_factory=list)
    n_fourier: int = 0
    coordinate_noise_std: list = None

    @classmethod
    def from_yaml_block(cls, config):
        block = config.get("ml_conditioning")
        if not block:
            return cls()
        if "param_map" not in block:
            raise InvalidConfiguration(
                "ml_conditioning needs a param_map, e.g. "
                "{source_location: [latitude, longitude, depth]}"
            )
        param_map = block["param_map"]
        return cls(
            param_map=param_map,
            n_cond=block.get("n_cond", sum(len(names) for names in param_map.values())),
            d_cond=block.get("d_cond"),
            coord_mode=block.get("coord_mode", "geographic"),
            inject=block.get("inject", []),
            n_fourier=block.get("n_fourier", 0),
            coordinate_noise_std=_source_location_error_std(config),
        )

    def to_model_entry(self, default_d_cond):
        """The ``conditioning`` entry of the model configuration."""
        return {
            "n_cond": self.n_cond,
            "d_cond": default_d_cond if self.d_cond is None else self.d_cond,
            "coord_mode": self.coord_mode,
            "inject": self.inject,
            "n_fourier": self.n_fourier,
        }


@dataclass
class VariableStationsConfig:
    """Training on a random subset of the master station set (``ml_variable_stations``)."""

    enabled: bool = False
    keep_fraction: object = None
    min_stations: int = 1
    station_coords_mode: str = "absolute"

    @classmethod
    def from_yaml_block(cls, config):
        block = config.get("ml_variable_stations") or {}
        if not block.get("enabled", False):
            return cls()
        return cls(
            enabled=True,
            keep_fraction=block.get("keep_fraction"),
            min_stations=block.get("min_stations", 1),
            station_coords_mode=block.get("station_coords_mode", "absolute"),
        )

    def build_subsampler(self):
        """The dataloader's station subsampler, or ``None`` when every station is kept."""
        if not self.enabled:
            return None
        from seismo_sbi.sbi.npe.data.dataloading import StationSubsampler
        return StationSubsampler(keep_fraction=self.keep_fraction,
                                 min_stations=self.min_stations)


@dataclass
class EmbeddingConfig:
    """Optional token features and pooling head of the set encoder.

    Each entry is its YAML block minus the ``enabled`` switch, forwarded verbatim to the
    embedding net; ``None`` means the feature is off and the model is built without it.
    """

    amplitude: dict = None
    positional_encoding: dict = None
    pma_pooling: dict = None
    summary_bottleneck_dim: int = None

    @classmethod
    def from_yaml_block(cls, config):
        bottleneck = config.get("ml_summary_bottleneck") or {}
        return cls(
            amplitude=_enabled_block(config, "ml_amplitude_embedding"),
            positional_encoding=_enabled_block(config, "ml_positional_encoding"),
            pma_pooling=_enabled_block(config, "ml_pooling"),
            summary_bottleneck_dim=int(bottleneck["dim"]) if bottleneck.get("dim") else None,
        )


@dataclass
class OptimizerConfig:
    """AdamW settings and the post-warmup learning-rate schedule (``ml_optimizer``).

    ``lr_schedule`` is "cosine" (decaying to ``lr * lr_min_factor``), "constant" or "cyclic".
    """

    lr: float = 1e-4
    weight_decay: float = 1e-4
    lr_schedule: str = "cosine"
    lr_min_factor: float = 0.1

    @classmethod
    def from_yaml_block(cls, config):
        block = config.get("ml_optimizer") or {}
        return cls(
            lr=float(block.get("lr", 1e-4)),
            weight_decay=float(block.get("weight_decay", 1e-4)),
            lr_schedule=block.get("lr_schedule", "cosine"),
            lr_min_factor=float(block.get("lr_min_factor", 0.1)),
        )


@dataclass
class BatchConfig:
    """Per-GPU batch sizes and dataloader settings (``ml_batch``).

    Each rank of a multi-GPU run consumes a full ``train`` batch, so the global batch is
    ``train * devices``. ``val`` defaults to twice ``train``; read it through :attr:`val_size`.
    ``train_fraction`` is the share of the simulations used for training rather than validation.
    """

    train: int = 128
    val: int = None
    num_workers: int = 8
    prefetch_factor: int = 6
    train_fraction: float = 0.90

    @classmethod
    def from_yaml_block(cls, config):
        block = config.get("ml_batch") or {}
        return cls(
            train=int(block.get("train", 128)),
            val=int(block["val"]) if block.get("val") is not None else None,
            num_workers=int(block.get("num_workers", 8)),
            prefetch_factor=int(block.get("prefetch_factor", 6)),
            train_fraction=float(block.get("train_fraction", 0.90)),
        )

    @property
    def val_size(self):
        """Validation batch size: twice the training batch unless the configuration sets one."""
        return self.val if self.val is not None else 2 * self.train


@dataclass
class CacheConfig:
    """In-RAM caches that remove the per-sample HDF5 reads (``ml_cache``).

    ``dtype`` is the storage precision of the cached simulations; float32 matches the model.
    """

    sims: bool = False
    noise: bool = False
    dtype: str = "float32"
    preload_workers: int = 16

    @classmethod
    def from_yaml_block(cls, config):
        block = config.get("ml_cache") or {}
        return cls(
            sims=bool(block.get("sims", False)),
            noise=bool(block.get("noise", False)),
            dtype=str(block.get("dtype", "float32")),
            preload_workers=int(block.get("preload_workers", 16)),
        )


@dataclass
class LoggingConfig:
    """Where per-epoch metrics go (``ml_logging``): a local ``metrics.csv``, Weights & Biases, or both."""

    wandb: bool = False
    csv: bool = True

    @classmethod
    def from_yaml_block(cls, config):
        block = config.get("ml_logging") or {}
        return cls(wandb=bool(block.get("wandb", False)), csv=bool(block.get("csv", True)))


@dataclass
class TrainingConfiguration:
    """Everything an NPE training run reads from its configuration file.

    ``flow``, ``perf`` and ``mmd`` are forwarded verbatim to the flow head, the Lightning module
    and the auxiliary MMD loss respectively.
    """

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    variable_stations: VariableStationsConfig = field(default_factory=VariableStationsConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    flow: dict = None
    perf: dict = None
    mmd: dict = field(default_factory=dict)
    warm_start_run_name: str = None
    model_dim: int = 256
    skip_compression_stencil: bool = False
    querier_cache_maxsize: int = None
    epochs: int = 300
    devices: int = 1

    @classmethod
    def from_yaml_block(cls, config):
        """Build from a parsed configuration file; every key it reads is optional."""
        reject_unknown_training_keys(config)
        return cls(
            encoder=EncoderConfig.from_yaml_block(config),
            conditioning=ConditioningConfig.from_yaml_block(config),
            variable_stations=VariableStationsConfig.from_yaml_block(config),
            embeddings=EmbeddingConfig.from_yaml_block(config),
            optimizer=OptimizerConfig.from_yaml_block(config),
            batch=BatchConfig.from_yaml_block(config),
            cache=CacheConfig.from_yaml_block(config),
            logging=LoggingConfig.from_yaml_block(config),
            flow=_block_without_switch(config.get("ml_flow")),
            perf=_block_without_switch(config.get("ml_perf")),
            mmd=config.get("ml_mmd") or {},
            warm_start_run_name=(config.get("ml_warm_start") or {}).get("from_run_name"),
            skip_compression_stencil=bool(config.get("skip_compression_data", False)),
            querier_cache_maxsize=(config.get("seismic_context") or {}).get(
                "querier_cache_maxsize"),
        )

    def apply_overrides(self, station_encoder=None, epochs=None, devices=None,
                        train_batch_size=None):
        """Fold in the command-line overrides a scheduler sets; ``None`` keeps the config value."""
        if station_encoder is not None:
            self.encoder.station_encoder = station_encoder
        if epochs is not None:
            self.epochs = epochs
        if devices is not None:
            self.devices = devices
        if train_batch_size is not None:
            self.batch.train = train_batch_size
        return self

    def to_model_config(self, theta_scaler_provenance):
        """The architecture dictionary handed to the trainer and stored in ``model_meta.json``.

        ``theta_scaler_provenance`` pins the parameter scaling into the checkpoint, so the
        inverse transform used at inference cannot silently differ from the trained one.
        """
        model_config = {"station_encoder": self.encoder.station_encoder,
                        "theta_scaler": theta_scaler_provenance}
        if self.encoder.input_decimate:
            model_config["input_decimate"] = self.encoder.input_decimate
        if self.encoder.options:
            model_config["encoder_config"] = self.encoder.options
        if self.conditioning.param_map is not None:
            model_config["conditioning"] = self.conditioning.to_model_entry(self.model_dim)
        if self.variable_stations.enabled:
            model_config["variable_stations"] = True
            model_config["station_coords_mode"] = self.variable_stations.station_coords_mode
        if self.embeddings.amplitude is not None:
            model_config["amplitude_embedding"] = self.embeddings.amplitude
        if self.embeddings.positional_encoding is not None:
            model_config["positional_encoding"] = self.embeddings.positional_encoding
        if self.embeddings.pma_pooling is not None:
            model_config["pma_pooling"] = self.embeddings.pma_pooling
        if self.embeddings.summary_bottleneck_dim:
            model_config["summary_bottleneck"] = {"dim": self.embeddings.summary_bottleneck_dim}
        if self.perf is not None:
            model_config["perf"] = self.perf
        return model_config

    def dataloader_args(self, pipeline, data):
        """Keyword arguments of the training/validation dataloaders for this dataset."""
        return {
            "data_loader": pipeline.data_manager.data_loader,
            "data_folder": pipeline.simulations_output_path,
            "parameter_name_map": pipeline.parameters.names,
            "synthetic_noise_model_sampler": pipeline.training_noise_sampler,
            "data_scaler": data.data_scaler,
            "augmentation_chain": data.augmentation_chain,
            "augmentation_nuisance_params": data.augmentation_nuisance_params,
            "post_noise_augmentation_chain": data.post_noise_chain,
            "post_noise_nuisance_params": data.post_noise_nuisance_params,
            "conditioning_param_map": self.conditioning.param_map,
            "conditioning_noise_std": self.conditioning.coordinate_noise_std,
            "station_subsampler": self.variable_stations.build_subsampler(),
            "cache_in_memory": self.cache.sims,
            "cache_dtype": self.cache.dtype,
            "cache_preload_workers": self.cache.preload_workers,
            **self.loader_args(len(data.simulation_paths)),
        }

    def loader_args(self, num_simulations):
        """The split and batching of ``num_simulations`` samples into training and validation.

        These are the keyword arguments of
        :func:`~seismo_sbi.sbi.npe.data.dataloading.make_torch_dataloaders` that do not
        build the dataset; with ``dataset=`` added they are the ``dataloader_args`` of a training
        run on an :class:`~seismo_sbi.sbi.npe.data.array_dataset.ArraySimulationDataset`.
        """
        return {
            "train_max_index": int(self.batch.train_fraction * num_simulations),
            "train_batch_size": self.batch.train,
            "val_batch_size": self.batch.val_size,
            "train_shuffle": True,
            "val_shuffle": False,
            "num_workers": self.batch.num_workers,
            "pin_memory": True,
            "prefetch_factor": self.batch.prefetch_factor,
        }


def reject_unknown_training_keys(config):
    """Raise :class:`InvalidConfiguration` on a mistyped ``ml_*`` block or block key."""
    unknown = sorted(key for key in config
                     if key.startswith("ml_") and key not in TRAINING_BLOCKS)
    if unknown:
        raise InvalidConfiguration(
            f"Unknown training blocks {unknown}. Allowed: {sorted(TRAINING_BLOCKS)}")
    for block_name, allowed in NAMED_BLOCK_KEYS.items():
        extra = sorted(set(config.get(block_name) or {}) - allowed)
        if extra:
            raise InvalidConfiguration(
                f"Unknown keys {extra} in {block_name}. Allowed: {sorted(allowed)}")


def _block_without_switch(block):
    """A YAML block minus its ``enabled`` switch, or ``None`` when the block is absent or empty."""
    if not block:
        return None
    return {name: value for name, value in block.items() if name != "enabled"}


def _enabled_block(config, block_name):
    """As :func:`_block_without_switch`, but ``None`` unless the block sets ``enabled: true``."""
    block = config.get(block_name) or {}
    if not block.get("enabled", False):
        return None
    return {name: value for name, value in block.items() if name != "enabled"}


def _source_location_error_std(config):
    """Per-coordinate conditioning-noise widths from a training-augmentation location nuisance."""
    nuisance = (config.get("parameters") or {}).get("nuisance") or {}
    block = nuisance.get("source_location_error") or {}
    if block.get("stage") == "training_augmentation":
        return block.get("coordinate_std")
    return None
