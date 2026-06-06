import torch
import math

from torch import nn

from .cnn_feature_extractor import ConvolutionalFeatureExtractor
from .csdi_transformer import ConditionalTransformer
from .axial_transformer import SeismogramAxialTransformer
from .station_encoders import build_station_encoder
from .amplitude_embedding import AmplitudeTokenEmbedding
from .source_conditioning import (
    SourceConditioner,
    FiLM,
    relative_station_geometry,
    unpack_context,
    unpack_variable_context,
)

import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, ExponentialLR, StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR, CyclicLR

import torch
import torch.nn as nn
import torch.nn.functional as F

class SeismogramTransformer(nn.Module):

    def __init__(self, num_seismic_components, transformer_config,
                        feature_length, num_outputs, noise_model,
                        seismogram_locations : torch.Tensor, device,
                        aggregation: str = "mean", input_length: int = 200) -> None:
        super().__init__()

        self.feature_length = feature_length
        self.noise_model = noise_model
        # Per-trace sample count the encoder will receive. Used to size the encoder's
        # output length; must match the actual trace length of the data at train time.
        self.input_length = input_length

        # Validate aggregation choice
        if aggregation not in ("mean", "query"):
            raise ValueError(f"Invalid aggregation '{aggregation}'. Choose 'mean' or 'query'.")
        self.aggregation = aggregation
        d_model = transformer_config['channels']

        # --- Pluggable per-station encoder ---
        encoder_name = transformer_config.get("station_encoder", "cnn")
        encoder_cfg = transformer_config.get("encoder_config", {})
        self.station_encoder = build_station_encoder(
            encoder_name,
            num_seismic_components=num_seismic_components,
            input_length=input_length,
            d_model=d_model,
            **encoder_cfg,
        )
        self.L = self.station_encoder.output_length   # temporal token count
        enc_D = self.station_encoder.output_dim       # encoder output width

        # Projection from encoder width to transformer width (Identity when equal).
        self.encoder_proj = (
            nn.Linear(enc_D, d_model) if enc_D != d_model else nn.Identity()
        )

        mode = 'axial'
        # New: allow configuring pooling and CLS from transformer_config
        pool_method = transformer_config.get("pooling", "mean")  # supports: mean, first, attn, max, gem
        use_cls = transformer_config.get("use_cls_token", False)
        num_q = transformer_config.get("num_query_tokens", 8)
        self.all_station_transformer = SeismogramAxialTransformer(
            seismogram_locations,
            d_model=d_model,
            nheads=transformer_config["nheads"],
            num_layers=transformer_config["layers"],
            time_steps=self.L,
            conv_length=d_model,
            # Size the sinusoidal time-embedding buffer to the encoder's token count.
            # Encoders (e.g. PNO/TCN with light downsampling) can emit L > the old default
            # of 60; without this the time_embed[:L] add would fail to broadcast.
            max_time_steps=max(self.L, 60),
            mode=mode,
            pool_queries=pool_method,
            use_cls_token=use_cls,
            num_query_tokens=num_q,
            temporal_pool_tokens=transformer_config.get("temporal_pool_tokens", 0),   # New
        )

        self.source_param_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_outputs)
        )

        # --- Optional source-location conditioning (opt-in; see source_conditioning.py) ---
        # Absent ⇒ self._n_cond == 0 and embed() takes the unchanged 4-D context path.
        self.d_model = d_model
        self._n_components = num_seismic_components
        self._n_stations = int(seismogram_locations.shape[0])
        self._configure_conditioning(transformer_config.get("conditioning", None), d_model)

        # --- Optional per-station amplitude embedding (opt-in; see amplitude_embedding.py) ---
        # Absent ⇒ self.amplitude_embedding is None and embed() is byte-identical to before.
        amp_cfg = transformer_config.get("amplitude_embedding", None)
        self.amplitude_embedding = (
            AmplitudeTokenEmbedding.from_config(d_model, num_seismic_components, amp_cfg)
            if amp_cfg else None
        )
        if (self.amplitude_embedding is not None
                and self.amplitude_embedding.uses_distance and self._n_cond == 0):
            raise ValueError(
                "amplitude_embedding.distance_correction needs a source location (n_cond > 0): "
                "configure a `conditioning` block so per-station epicentral distance is available."
            )

        # --- Variable-station support (opt-in) ---
        # When enabled, embed() expects the packed variable-station context (padded
        # seismograms + per-sample coords + validity mask + optional source vec) produced by
        # variable_station_collate, builds a key_padding_mask, and feeds per-sample station
        # coordinates to the transformer instead of the fixed station_coords buffer.
        self._variable_stations = bool(transformer_config.get("variable_stations", False))
        # How station position is encoded for variable configs: "absolute" feeds the raw
        # (lat, lon) coords; "relative" feeds source-relative (distance, azimuth) and requires
        # source conditioning (n_cond > 0) so a source vector is present to measure against.
        self._station_coords_mode = transformer_config.get("station_coords_mode", "absolute")
        if self._variable_stations and self._station_coords_mode not in ("absolute", "relative"):
            raise ValueError(
                f"station_coords_mode must be 'absolute' or 'relative', "
                f"got '{self._station_coords_mode}'."
            )
        if self._variable_stations and self._station_coords_mode == "relative" and self._n_cond == 0:
            raise ValueError(
                "station_coords_mode='relative' needs source conditioning (n_cond > 0) so a "
                "source location is available; configure `conditioning` or use 'absolute'."
            )

    def _configure_conditioning(self, cond_cfg, d_model):
        self._n_cond = 0
        self._inject = ()
        self._coord_mode = "geographic"
        self.source_conditioner = None
        self.film = None
        self.token_proj = None
        self.concat_proj = None
        if not cond_cfg:
            return
        valid = {"relative_posemb", "token_add", "film", "concat_context"}
        inject = tuple(cond_cfg.get("inject", []))
        unknown = set(inject) - valid
        if unknown:
            raise ValueError(f"Unknown conditioning.inject options {sorted(unknown)}; valid: {sorted(valid)}")
        n_cond = int(cond_cfg["n_cond"])
        d_cond = int(cond_cfg.get("d_cond", d_model))
        self._n_cond = n_cond
        self._inject = inject
        self._coord_mode = cond_cfg.get("coord_mode", "geographic")
        self.source_conditioner = SourceConditioner(
            n_cond=n_cond, d_cond=d_cond, coord_mode=self._coord_mode,
            n_fourier=int(cond_cfg.get("n_fourier", 0)),
        )
        if "film" in inject:
            self.film = FiLM(d_cond=d_cond, d_model=d_model)
        if "token_add" in inject:
            self.token_proj = nn.Linear(d_cond, d_model)
        if "concat_context" in inject:
            # Project back to d_model so the flow's conditional_dim is unchanged.
            self.concat_proj = nn.Linear(d_model + d_cond, d_model)

    def sample_noise_model(self, batch_size):
        return torch.stack([self.noise_model() for _ in range(batch_size)], dim =0)
    
    def forward(self, x : torch.Tensor):
        aggregated_station_info = self.embed(x)
        outputs = self.source_param_predictor(aggregated_station_info)
        return outputs

    def embed(self, x: torch.Tensor):
        # Unpack source conditioning if the context is packed (2-D). With no conditioning
        # configured, x stays 4-D and source_vec is None → original behaviour.
        source_vec = None
        # Per-sample station coords + validity mask, only set on the variable-station path.
        var_coords = None
        var_mask = None
        if self._variable_stations and x.dim() == 2:
            # Variable-station packed context: recover padded seismograms, per-sample coords,
            # the validity mask, and (optionally) the source vector. max_N is inferred inside.
            x, var_coords, var_mask, source_vec = unpack_variable_context(
                x, self._n_components, self.input_length, self._n_cond
            )
        elif self._variable_stations and x.dim() == 4:
            # Inference convenience: a 4-D (B, N, C, T) tensor for the full master station set.
            # Treat every station as valid and take coordinates from the fixed buffer, so
            # callers can run the full configuration without explicit packing. (Subsets or
            # relative-coord inference must supply a packed 2-D context with per-sample coords.)
            if self._station_coords_mode == "relative":
                raise ValueError(
                    "Variable-station model with station_coords_mode='relative' needs a packed "
                    "2-D context carrying the source vector and per-sample coords; a bare 4-D "
                    "tensor has no source location to measure against."
                )
            B_, N_ = x.shape[0], x.shape[1]
            var_mask = torch.ones(B_, N_, dtype=torch.bool, device=x.device)
            var_coords = self.all_station_transformer.station_coords.to(x.dtype)
            if var_coords.dim() == 2:
                var_coords = var_coords.unsqueeze(0).expand(B_, N_, 2)
        elif self._n_cond > 0 and x.dim() == 2:
            x, source_vec = unpack_context(
                x, self._n_stations, self._n_components, self.input_length, self._n_cond
            )
        elif self._n_cond == 0 and x.dim() == 2:
            # A packed context reached a model with no conditioning configured — almost
            # certainly a source_location set on an unconditioned checkpoint. Fail clearly
            # instead of the cryptic "not enough values to unpack" from x.shape below.
            raise ValueError(
                "Received a packed 2-D context but this model has no conditioning configured "
                "(n_cond == 0). Did you set source_location on an unconditioned model, or load "
                "a conditioned checkpoint into a default-constructed trainer?"
            )

        (batch_size, num_stations, num_seismic_components, trace_length) = x.shape
        B, N = batch_size, num_stations

        # Source embedding (shared across stations/time) — only when conditioning is active.
        source_emb = self.source_conditioner(source_vec) if (source_vec is not None) else None

        # Flatten to feed per-station encoder: (B*N, C, T)
        x_flat = x.reshape(B * N, num_seismic_components, trace_length)
        # Encoder → (B*N, L, D)
        feats = self.station_encoder(x_flat)
        # Optional projection from encoder width to transformer d_model → (B*N, L, d_model)
        feats = self.encoder_proj(feats)
        # Reshape to (B, N, L, d_model) for the axial transformer
        feature_sequences = feats.view(B, N, self.L, -1)

        # --- Conditioning injections on the per-station activations (B, N, L, d_model) ---
        if source_emb is not None and self.film is not None:
            feature_sequences = self.film(feature_sequences, source_emb)
        if source_emb is not None and self.token_proj is not None:
            # Add a projected source embedding to every token (broadcast over N and L).
            feature_sequences = feature_sequences + self.token_proj(source_emb)[:, None, None, :]

        # Source-relative station positional embedding (distance, azimuth) override.
        station_override = None
        if source_vec is not None and ("relative_posemb" in self._inject):
            station_override = relative_station_geometry(
                source_vec, self.all_station_transformer.station_coords, self._coord_mode
            )

        # --- Variable-station coords + key-padding mask ---
        key_padding_mask = None
        if self._variable_stations:
            # Per-sample station position: absolute (lat, lon) coords fed directly, or
            # source-relative (distance, azimuth) computed from this sample's coords.
            if self._station_coords_mode == "relative":
                station_override = relative_station_geometry(
                    source_vec, var_coords, self._coord_mode
                )
            else:
                station_override = var_coords
            # Transformer mask is (B, N, L) with True = pad: invert the station validity
            # mask and broadcast over the encoder token axis L. Keep it as an expanded view —
            # the transformer's reshape/permute consumers materialise only where needed.
            key_padding_mask = (~var_mask).unsqueeze(-1).expand(B, N, self.L)

        # --- Optional per-station amplitude embedding ---
        # Inject the array-relative radiation-pattern token BEFORE the transformer (so
        # cross-station attention sees relative amplitude), and stash the per-event global
        # (≈log M0) vector to add to the pooled embedding AFTER the transformer. When distance
        # correction is enabled and a source location is available, supply per-station epicentral
        # distance so the geometric-spreading trend is removed before forming the reference.
        amp_global = None
        if self.amplitude_embedding is not None:
            station_distance = None
            if self.amplitude_embedding.uses_distance and source_vec is not None:
                amp_coords = (
                    var_coords if var_coords is not None
                    else self.all_station_transformer.station_coords
                )
                station_distance = relative_station_geometry(
                    source_vec, amp_coords, self._coord_mode
                )[..., 0:1]   # (B, N, 1) epicentral distance
            amp_token, amp_global = self.amplitude_embedding(
                x, mask=var_mask, distance=station_distance
            )
            feature_sequences = feature_sequences + amp_token[:, :, None, :]

        # Contextualize with transformer
        transformer_output = self.all_station_transformer(
            feature_sequences,
            key_padding_mask=key_padding_mask,
            station_coords_override=station_override,
        )  # (x, q, pooled)
        pooled = transformer_output[2]  # pooled embedding (B, d_model)

        # Global concat-conditioning, projected back to d_model.
        if source_emb is not None and self.concat_proj is not None:
            pooled = self.concat_proj(torch.cat([pooled, source_emb], dim=-1))

        # Absolute-moment (M0) path: add the per-event global amplitude vector to the pooled
        # embedding so the flow sees absolute scale even though the per-station tokens carried
        # only the array-relative pattern.
        if amp_global is not None:
            pooled = pooled + amp_global
        return pooled


class LightningModel(pl.LightningModule):

    def __init__(self, loss_function = nn.MSELoss() , lr=0.001, **kwargs):

        super().__init__()

        self.model = SeismogramTransformer(**kwargs)
        self.loss_func = loss_function

        self.lr = lr
        self.learning_rate_sched = None

        self.weight_decay = 0

    def forward(self, x):
        return self.model.forward(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat , _= self.shared_step(batch)

        return y_hat, y
    
    def shared_step(self, batch, eval_type=""):

        x, y = batch
        noise = self.model.sample_noise_model(batch_size=x.shape[0])

        y_hat = self(x + noise)
        loss = self.loss_func(y_hat, y)
        loss_dict = {f'{eval_type}loss' : loss}

        return y_hat, loss_dict

    def training_step(self, batch, batch_idx):
        

        _, loss = self.shared_step(batch)


        self._log_loss(loss)
   
        return loss

    def validation_step(self, batch, batch_idx):

        _, loss = self.shared_step(batch, "val_")

        self._log_loss(loss)

        return loss

    def _log_loss(self, loss):
        
        for l in loss.keys():
            self.log(l, loss[l])

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        """

        
        opt = torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate), weight_decay=self.weight_decay)
        if not self.learning_rate_sched:
            return opt
        if self.learning_rate_sched == 'one_cycle':
            print("Using one cycle LR: ", self.learning_rate_sched_opt)
            sch = OneCycleLR(opt, **self.learning_rate_sched_opt)
        elif self.learning_rate_sched == 'reduce_on_plateau':
            sch = ReduceLROnPlateau(opt, **self.learning_rate_sched_opt)
        elif self.learning_rate_sched == 'exponential':
            sch = ExponentialLR(opt, **self.learning_rate_sched_opt)
        elif self.learning_rate_sched == 'step':
            sch = StepLR(opt, **self.learning_rate_sched_opt)
        sch = {"scheduler": sch, "interval": "epoch", "monitor": "val_loss"}
        return [opt], [sch]

class NPELightningModule(pl.LightningModule):
    def __init__(self, flow, lr=1e-3, weight_decay=0.0, **kwargs):
        super().__init__()
        self.flow = flow
        self.lr = lr
        self.weight_decay = weight_decay
        # New: allow selecting second-stage scheduler ("cosine" or "cyclic")
        self.lr_second_stage = "cosine"
        self.cyclic_period_steps = 8000

    def forward(self, x, theta):
        # The flow contains the embedding_net; pass x as context to be embedded internally.
        return self.flow.log_prob(theta, context=x)

    def training_step(self, batch, batch_idx):
        theta, x = batch
        log_prob = self.forward(x, theta)
        loss = -log_prob.mean()
        self.log("loss", loss, prog_bar=True)
        self.log("log_prob", log_prob.mean())
        return loss

    def validation_step(self, batch, batch_idx):
        theta, x = batch
        log_prob = self.forward(x, theta)
        val_loss = -log_prob.mean()
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_log_prob", log_prob.mean())
        return val_loss

    def configure_optimizers(self):
        # Optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Default: 2-phase schedule → warmup → cosine or cyclic
        max_epochs = getattr(self.trainer, "max_epochs", None) or 500

        warmup_epochs = max(1, int(0.05 * max_epochs))
        # Guard against tiny max_epochs (e.g. 1 in tests) where warmup consumes all epochs:
        # CosineAnnealingLR(T_max=0) divides by zero.
        cosine_epochs = max(1, max_epochs - warmup_epochs)  # remaining epochs

        # If using cyclic LR, switch to step-based scheduling and compute warmup in steps
        use_cyclic = (str(self.lr_second_stage).lower() == "cyclic")

        if use_cyclic:
            # Try to derive steps per epoch
            total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
            if total_steps is not None and max_epochs > 0:
                steps_per_epoch = max(1, total_steps // max_epochs)
            else:
                steps_per_epoch = getattr(self.trainer, "num_training_batches", None) or 1

            warmup_steps = max(1, warmup_epochs * steps_per_epoch)

            # Linear warmup to base LR over warmup_steps
            def lr_lambda_warmup(step_idx):
                return min(float(step_idx + 1) / float(max(1, warmup_steps)), 1.0)
            warmup = LambdaLR(optimizer, lr_lambda=lr_lambda_warmup)

            # Cyclic LR with given period (in steps)
            half_period = max(1, self.cyclic_period_steps // 2)
            cyclic = CyclicLR(
                optimizer,
                base_lr=self.lr * 0.05,
                max_lr=self.lr,
                step_size_up=half_period,
                step_size_down=self.cyclic_period_steps - half_period,
                mode="triangular",
                cycle_momentum=False
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cyclic],
                milestones=[warmup_steps],
            )

            return [optimizer], [{
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }]

        # Linear warmup to base LR (epoch-based)
        def lr_lambda_warmup(epoch):
            return float(epoch + 1) / float(max(1, warmup_epochs))
        warmup = LambdaLR(optimizer, lr_lambda=lr_lambda_warmup)

        # Cosine annealing down to 10% of base LR
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=self.lr * 0.1)

        # Combine: warmup → cosine
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }]
