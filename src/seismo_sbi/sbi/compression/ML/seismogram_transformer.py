import torch
import math

from torch import nn

from .cnn_feature_extractor import ConvolutionalFeatureExtractor
from .csdi_transformer import ConditionalTransformer
from .axial_transformer import SeismogramAxialTransformer
from .station_encoders import build_station_encoder, InputDecimator
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

        # --- Optional performance toggles (opt-in via model_config['perf']; see train.py) ---
        # Absent ⇒ all default to off and the model is byte-identical to before. These are pure
        # speed knobs for the EMBEDDING net only; the NSF flow head (precision-brittle: LULinear
        # log-det + BatchNorm conditioner) is left in fp32 by construction — autocast here wraps
        # SeismogramTransformer.forward and casts its context output back to fp32 before the flow.
        perf = transformer_config.get("perf", {}) or {}
        self._amp = bool(perf.get("amp", False))
        _amp_dtype = str(perf.get("amp_dtype", "bfloat16")).lower()
        self._amp_dtype = torch.bfloat16 if _amp_dtype in ("bfloat16", "bf16") else torch.float16
        # SDPA (fused scaled_dot_product_attention) for the axial / PMA attentions.
        self._use_sdpa = bool(perf.get("sdpa", False))

        # --- Optional Nyquist-aware model-entry input decimation (opt-in) ---
        # model_config['input_decimate'] = {"factor": k, "antialias": bool}. Band-limited
        # data sampled above its Nyquist rate decimates losslessly (min period 6 s @ 1 Hz
        # sampling ⇒ factor 3). Packed-context UNPACKING keeps the original trace length
        # (self.input_length); only the encoder and everything downstream see T/k.
        dec_cfg = transformer_config.get("input_decimate", None) or {}
        dec_factor = int(dec_cfg.get("factor", 1) or 1)
        self.input_decimator = (
            InputDecimator(dec_factor, num_seismic_components,
                           antialias=bool(dec_cfg.get("antialias", True)))
            if dec_factor > 1 else None
        )
        encoder_input_length = (
            self.input_decimator.output_length(input_length)
            if self.input_decimator is not None else input_length
        )

        # --- Pluggable per-station encoder ---
        encoder_name = transformer_config.get("station_encoder", "cnn")
        encoder_cfg = transformer_config.get("encoder_config", {})
        self.station_encoder = build_station_encoder(
            encoder_name,
            num_seismic_components=num_seismic_components,
            input_length=encoder_input_length,
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
        # NOTE: the axial transformer itself is constructed at the END of __init__ (below the
        # conditioning / variable-station setup) so the opt-in §3.2 RFF positional encoder can be
        # told whether station coordinates are source-relative or absolute (posemb_coords_kind).

        # --- Optional summary BOTTLENECK (model_config["summary_bottleneck"]) --------------
        # Narrows the summary the MMD auxiliary loss lives in WITHOUT touching either the
        # encoder or the flow.  MMD's sample complexity grows with dimension, and the training
        # term compares only `batch_size` (64) samples per side, so a 256-d summary is a
        # weakly-powered, high-variance regime for the estimator.
        #
        # The head becomes  d_model -> d_model -> bottleneck -> num_outputs:
        #   * the encoder is untouched (d_model unchanged),
        #   * the flow still receives a `num_outputs`-wide context, so its conditioner widths
        #     and parameter count are UNCHANGED (train_NPE ties the flow's hidden width to the
        #     embedding channel width, so shrinking `channels` instead would silently shrink
        #     the flow too — see train_NPE.py:369),
        #   * everything downstream is a deterministic function of `bottleneck` numbers, so the
        #     summary really is that many dimensions.
        # Absent => byte-identical to the previous two-layer head.
        _bneck_cfg = (transformer_config or {}).get("summary_bottleneck") or {}
        bneck = _bneck_cfg.get("dim") if isinstance(_bneck_cfg, dict) else _bneck_cfg
        self.summary_bottleneck_dim = int(bneck) if bneck else None
        _head_out = self.summary_bottleneck_dim or num_outputs
        self.source_param_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, _head_out)
        )
        # LayerNorm on the bottleneck: the pooling head already normalises its output "for a
        # consistent scale into the flow", and the same argument applies with more force here —
        # the MMD kernel bandwidth is a median heuristic over these vectors, so a drifting scale
        # is exactly what destabilises it.
        self.summary_expand = (
            nn.Sequential(nn.LayerNorm(self.summary_bottleneck_dim),
                          nn.Linear(self.summary_bottleneck_dim, num_outputs))
            if self.summary_bottleneck_dim else None
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

        # --- Build the all-station axial transformer (last, so the §3.2 RFF positional encoder
        # knows whether station coords are source-relative). Optional 'positional_encoding' block:
        #   ml_positional_encoding:
        #     enabled: true
        #     mode: fourier            # fourier (sinusoidal/absence => legacy sinusoid)
        #     num_freqs: 16
        #     sigma: 1.0
        #     learnable_freqs: false
        #     include_depth: true      # RFF-encode source depth (needs n_cond >= 3)
        #     inject_every_layer: true # re-inject the geometry before every block (§3.2.c)
        #     standardize: running
        posemb_config = transformer_config.get("positional_encoding", None)
        # coords_kind is static per model: source-relative iff the source-relative station
        # embedding is in use (relative_posemb injection, or variable-station relative mode).
        posemb_coords_kind = (
            "relative"
            if (("relative_posemb" in self._inject) or (self._station_coords_mode == "relative"))
            else "absolute"
        )
        inject_every_layer = (
            posemb_config.get("inject_every_layer", True) if posemb_config else True
        )
        # --- Optional §3.4 PMA pooling head. Absent ⇒ pma_cfg is None ⇒ pma_pooling_config=None ⇒
        # the axial transformer keeps the legacy query-mean / CLS pooling (byte-identical). When
        # enabled, the head owns the learned seeds and pools the final encoded token set, so the
        # in-block query cross-attention is turned off (num_query_tokens=0).
        #   ml_pooling:
        #     enabled: true
        #     pool_over: tokens        # tokens (pool N·L tokens) | stations (time-collapse, pool N)
        #     num_seeds: 4             # k learnable seeds
        #     seed_self_attention: false
        #     combine: linear          # linear (learned) | mean | first
        pma_cfg = transformer_config.get("pma_pooling", None)
        if pma_cfg and use_cls:
            raise ValueError(
                "pma_pooling is incompatible with use_cls_token=True (CLS has its own pooling); "
                "disable one of them."
            )
        self.all_station_transformer = SeismogramAxialTransformer(
            seismogram_locations,
            d_model=d_model,
            nheads=transformer_config["nheads"],
            num_layers=transformer_config["layers"],
            time_steps=self.L,
            conv_length=d_model,
            # Size the sinusoidal time-embedding buffer to the encoder's token count (PNO/TCN
            # can emit L > the old default of 60; without this the time_embed[:L] add fails).
            max_time_steps=max(self.L, 60),
            mode=mode,
            pool_queries=pool_method,
            use_cls_token=use_cls,
            num_query_tokens=(0 if pma_cfg else num_q),
            temporal_pool_tokens=transformer_config.get("temporal_pool_tokens", 0),
            posemb_config=posemb_config,
            posemb_coords_kind=posemb_coords_kind,
            inject_every_layer=inject_every_layer,
            pma_pooling_config=pma_cfg,
            use_sdpa=self._use_sdpa,
        )
        # include_depth needs a source depth in the conditioning vector (lat, lon, depth, ...).
        _posenc = self.all_station_transformer.station_posenc
        if _posenc is not None and _posenc.include_depth and self._n_cond < 3:
            raise ValueError(
                "positional_encoding.include_depth needs a source depth (n_cond >= 3): "
                "configure an `ml_conditioning` block with at least (latitude, longitude, depth)."
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
        # Optional bf16 autocast scoped to the EMBEDDING net only. The flow (which calls this
        # forward as its embedding_net) then receives an fp32 context, so the precision-brittle
        # coupling/LU/BatchNorm transforms keep running in fp32. autocast keeps LayerNorm/softmax
        # in fp32 automatically (its op allowlist) and routes matmuls/convs to bf16 tensor cores.
        if self._amp and x.is_cuda:
            with torch.autocast("cuda", dtype=self._amp_dtype):
                aggregated_station_info = self.embed(x)
                outputs = self.source_param_predictor(aggregated_station_info)
                if self.summary_expand is not None:
                    outputs = self.summary_expand(outputs)
            return outputs.float()
        aggregated_station_info = self.embed(x)
        outputs = self.source_param_predictor(aggregated_station_info)
        if self.summary_expand is not None:
            outputs = self.summary_expand(outputs)
        return outputs

    def summary_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        """The vector the MMD auxiliary loss should compare.

        With ``summary_bottleneck`` configured this is the NARROW pre-expansion summary
        (``summary_bottleneck_dim`` wide); without it, it is the ordinary ``forward`` output,
        so callers need no branch and behaviour is unchanged for existing configs.
        """
        if self.summary_expand is None:
            return self.forward(x)
        if self._amp and x.is_cuda:
            with torch.autocast("cuda", dtype=self._amp_dtype):
                z = self.source_param_predictor(self.embed(x))
            return z.float()
        return self.source_param_predictor(self.embed(x))

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
            _posenc = self.all_station_transformer.station_posenc
            if _posenc is not None and _posenc.include_depth:
                raise ValueError(
                    "positional_encoding.include_depth needs a packed 2-D context carrying the "
                    "source vector; a bare 4-D tensor has no source depth to encode."
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

        # Model-entry decimation: after unpacking (which needs the original trace length),
        # before the encoder/amplitude paths — everything downstream sees T/k samples.
        if self.input_decimator is not None:
            x = self.input_decimator(x)

        (batch_size, num_stations, num_seismic_components, trace_length) = x.shape
        B, N = batch_size, num_stations

        # Source embedding (shared across stations/time) — only when conditioning is active.
        source_emb = self.source_conditioner(source_vec) if (source_vec is not None) else None

        # Source depth (km), RFF-encoded by the §3.2 positional encoder when include_depth is set.
        # Conditioning param_map order is (latitude, longitude, depth, ...) ⇒ index 2.
        source_depth = (
            source_vec[:, 2:3] if (source_vec is not None and source_vec.shape[1] >= 3) else None
        )

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
            source_depth=source_depth,
            station_mask=var_mask,
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

def fused_adam_supported(params):
    """True if every parameter satisfies torch's fused-AdamW preconditions.

    Checked eagerly because torch validates them LAZILY, inside the first
    ``optimizer.step()`` — so wrapping the ``AdamW(..., fused=True)`` constructor in a
    try/except catches neither failure. Both have bitten real runs:

    * **device** — some torch versions accept fused CPU params at construction and only
      reject them at step time.
    * **dtype** — torch >= 2.5 (``_device_dtype_check_for_fused``) rejects COMPLEX params.
      The pno encoder's ``SpectralConv1d`` spectral weights are ``complex64``.

    Falling back to unfused AdamW is mathematically identical (same update rule); fused
    only saves per-parameter kernel launches. So this returns a plain bool and the caller
    silently takes the slower, always-correct path.
    """
    params = list(params)
    if not params:
        return False
    return (all(p.is_cuda for p in params)
            and not any(p.is_complex() for p in params))


class NPELightningModule(pl.LightningModule):
    def __init__(self, flow, lr=1e-3, weight_decay=0.0, lr_second_stage="cosine",
                 lr_min_factor=0.1,
                 fused_adam=False, compile_forward=False, compile_flow=False, **kwargs):
        super().__init__()
        self.flow = flow
        self.lr = lr
        self.weight_decay = weight_decay
        # Second-stage LR schedule applied AFTER the linear warmup:
        #   "cosine"   (default) — anneal down to lr*lr_min_factor over the remaining epochs;
        #   "constant"           — hold flat at the base lr (no decay);
        #   "cyclic"             — triangular CyclicLR (step-based).
        self.lr_second_stage = lr_second_stage
        # Cosine floor as a fraction of the base LR: eta_min = lr * lr_min_factor.
        # 0.1 is the legacy value (every run before 2026-08-11 annealed to lr/10); the
        # santorini mw-fix campaign moved to 0.2 (lr/5) so the tail of a 100-epoch run keeps
        # a usefully large step. Only read by the "cosine" branch.
        self.lr_min_factor = float(lr_min_factor)
        self.cyclic_period_steps = 8000
        # --- Optional perf toggles (opt-in via model_config['perf']; default-off = legacy) ---
        # fused_adam: one fused CUDA optimizer kernel instead of a per-parameter launch storm
        #   (~8M params in many small tensors — real saving on this launch-bound workload).
        # compile_forward: torch.compile the WHOLE log-prob (embedding + flow) — the flow's
        #   small per-transform kernels dominate launch overhead. Wrapped as a closure so
        #   state_dict keys are unchanged (no _orig_mod. prefix in checkpoints).
        # compile_flow: compile ONLY the transform stack + base density (embedding eager) —
        #   fallback if the embedding's data-dependent mask branches thrash recompilation.
        # Requires a working torch.compile (torch >= 2.2; broken on 2.0) — only set these on
        # an env where the one-epoch e2e gate passes with them on.
        self._fused_adam = bool(fused_adam)
        self._log_prob_fn = None
        self._flow_tail_fn = None
        if compile_forward and hasattr(torch, "compile"):
            self._log_prob_fn = torch.compile(self._log_prob)
        elif compile_flow and hasattr(torch, "compile"):
            self._flow_tail_fn = torch.compile(self._flow_log_prob_from_embedded)
        # Misspecification-robust MMD auxiliary loss (opt-in via enable_mmd; None = legacy).
        self._mmd_cfg = None
        self._mmd_psim_loader = None
        self._mmd_psim_iter = None
        self._mmd_bandwidth_ema = None
        self._mmd_last_beta = float("nan")
        self._mmd_last_beta_ema = float("nan")
        self._mmd_last_z_scale = float("nan")

    def enable_mmd(self, mmd_config: dict, real_context, psim_loader):
        """Arm the summary-space MMD auxiliary loss (Huang et al. 2023-style, two-sample).

        ``real_context``: pre-packed context tensor (N_real, W) of the QA-cleaned real
        events — registered as a NON-persistent buffer (moves with the module to GPU;
        checkpoints stay lean, the caller re-supplies it on reload). ``psim_loader``: a
        DataLoader over the posterior-matched simulation suite that reproduces the
        training-time augmentation path (noise + amplitude + per-parent-event masks) and
        yields ``(theta, context)`` batches — iterated cyclically, one batch per MMD step.

        The total loss becomes ``nll + lambda(t) * MMD^2_u(embed(real), embed(psim))``
        with lambda ramped 0 -> lambda_mmd after ``warmup_epochs`` over ``ramp_epochs``.
        Checkpoint selection stays on the NLL-only ``val_loss``; the MMD is logged as a
        diagnostic (``train_mmd2`` / ``val_mmd2``). Designed for single-device training
        (each DDP rank would draw independent sub-batches — fine, but the logged MMD is
        then per-rank).
        """
        from .mmd import DEFAULT_BANDWIDTH_SCALES
        cfg = dict(mmd_config or {})
        self._mmd_cfg = {
            "lambda_mmd": float(cfg.get("lambda_mmd", 0.05)),
            "warmup_epochs": int(cfg.get("warmup_epochs", 5)),
            "ramp_epochs": int(cfg.get("ramp_epochs", 5)),
            "every_n_steps": max(1, int(cfg.get("every_n_steps", 1))),
            "batch_size": int(cfg.get("batch_size", 64)),
            "bandwidth_scales": tuple(cfg.get("bandwidth_scales",
                                              DEFAULT_BANDWIDTH_SCALES)),
            "bandwidth_ema": float(cfg.get("bandwidth_ema", 0.9)),
        }
        self.register_buffer("mmd_real_context",
                             torch.as_tensor(real_context), persistent=False)
        self._mmd_psim_loader = psim_loader
        self._mmd_psim_iter = None
        self._mmd_bandwidth_ema = None

    def _next_psim_context(self):
        if self._mmd_psim_iter is None:
            self._mmd_psim_iter = iter(self._mmd_psim_loader)
        try:
            _, ctx = next(self._mmd_psim_iter)
        except StopIteration:
            self._mmd_psim_iter = iter(self._mmd_psim_loader)
            _, ctx = next(self._mmd_psim_iter)
        return ctx.to(device=self.device, dtype=self.mmd_real_context.dtype)

    def _mmd_lambda(self):
        cfg = self._mmd_cfg
        epoch = int(self.current_epoch)
        if epoch < cfg["warmup_epochs"]:
            return 0.0
        ramp = max(1, cfg["ramp_epochs"])
        frac = min(1.0, (epoch - cfg["warmup_epochs"] + 1) / ramp)
        return cfg["lambda_mmd"] * frac

    def _mmd_term(self):
        """One MMD^2_u evaluation between fresh real/psim summary sub-batches.

        Summaries are cast to float32 before the kernel (the embedding may run under
        bf16 autocast via the perf toggles; the O(B^2) kernel is cheap in fp32 and the
        estimator is noise-sensitive). Bandwidth = median heuristic on the pooled
        sub-batches, EMA-smoothed across steps, detached from the graph.
        """
        from .mmd import median_bandwidth, rbf_mixture_mmd2_unbiased
        cfg = self._mmd_cfg
        n_real = self.mmd_real_context.shape[0]
        b = min(cfg["batch_size"], n_real)
        idx = torch.randperm(n_real, device=self.mmd_real_context.device)[:b]
        # Read the summary BOTTLENECK when the embedding net has one, so the kernel lives in
        # the narrow space rather than the flow-facing expansion (which is a rank-limited
        # linear image of it, and so would reintroduce the wide-space geometry the bottleneck
        # exists to avoid). Falls back to the plain forward otherwise -> unchanged behaviour.
        _emb = self.flow._embedding_net
        _summarise = getattr(_emb, "summary_bottleneck", _emb)
        z_real = _summarise(self.mmd_real_context[idx]).float()
        z_psim = _summarise(self._next_psim_context()).float()
        beta = median_bandwidth(z_real, z_psim)
        ema = cfg["bandwidth_ema"]
        self._mmd_bandwidth_ema = (beta if self._mmd_bandwidth_ema is None
                                   else ema * self._mmd_bandwidth_ema + (1 - ema) * beta)
        # Degeneracy diagnostics (logged by the caller, never fed back into the loss).
        # MMD^2 is exactly invariant to a GLOBAL rescale of the summaries, because beta
        # is a median heuristic on those same summaries — so "shrink everything" is not
        # a way to cheat the penalty in steady state. It IS a way to cheat *transiently*:
        # the EMA lags by ~1/(1-ema) steps, so an embedding contracting faster than that
        # leaves beta stale-and-too-large, every d^2/beta^2 -> 0, every kernel -> 1 and
        # MMD^2_u -> 0 with no alignment whatsoever. Record the instantaneous beta (which
        # tracks the true scale) alongside the lagged EMA actually used: a widening gap
        # between them, or a collapsing z-scale, is the signature of that failure.
        self._mmd_last_beta = beta
        self._mmd_last_beta_ema = self._mmd_bandwidth_ema
        with torch.no_grad():
            self._mmd_last_z_scale = float(
                torch.cat([z_real, z_psim], dim=0).pow(2).mean().sqrt())
        bandwidths = [self._mmd_bandwidth_ema * s for s in cfg["bandwidth_scales"]]
        return rbf_mixture_mmd2_unbiased(z_real, z_psim, bandwidths)

    def _log_prob(self, theta, x):
        return self.flow.log_prob(theta, context=x)

    def _flow_log_prob_from_embedded(self, theta, embedded):
        # nflows.Flow.log_prob with the embedding hoisted out, so only the launch-bound
        # transform stack + base density are compiled.
        noise, logabsdet = self.flow._transform(theta, context=embedded)
        return self.flow._distribution.log_prob(noise, context=embedded) + logabsdet

    def forward(self, x, theta):
        # The flow contains the embedding_net; pass x as context to be embedded internally.
        if self._log_prob_fn is not None:
            return self._log_prob_fn(theta, x)
        if self._flow_tail_fn is not None:
            embedded = self.flow._embedding_net(x)
            return self._flow_tail_fn(theta, embedded)
        return self.flow.log_prob(theta, context=x)

    def training_step(self, batch, batch_idx):
        theta, x = batch
        log_prob = self.forward(x, theta)
        loss = -log_prob.mean()
        # MMD auxiliary loss (armed via enable_mmd; absent => byte-identical legacy loss).
        if self._mmd_cfg is not None and self.global_step % self._mmd_cfg["every_n_steps"] == 0:
            lam = self._mmd_lambda()
            mmd2 = self._mmd_term()
            self.log("train_mmd2", mmd2, prog_bar=True)
            self.log("mmd_lambda", lam)
            self.log("mmd_beta", self._mmd_last_beta)
            self.log("mmd_beta_ema", self._mmd_last_beta_ema)
            self.log("mmd_z_scale", self._mmd_last_z_scale)
            if lam > 0:
                loss = loss + lam * mmd2
        self.log("loss", loss, prog_bar=True)
        self.log("log_prob", log_prob.mean())
        return loss

    def validation_step(self, batch, batch_idx):
        theta, x = batch
        log_prob = self.forward(x, theta)
        val_loss = -log_prob.mean()
        # sync_dist=True averages across DDP ranks so ModelCheckpoint's monitored
        # val_loss is the true mean over the whole val split (not just rank 0's shard).
        # No-op on a single device ⇒ single-GPU/CPU behaviour is unchanged.
        # NOTE: val_loss stays NLL-ONLY even with MMD armed — checkpoint selection must
        # keep tracking posterior quality; val_mmd2 (below) is a diagnostic.
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_log_prob", log_prob.mean(), sync_dist=True)
        if self._mmd_cfg is not None and batch_idx == 0:
            with torch.no_grad():
                self.log("val_mmd2", self._mmd_term(), sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        # Optimizer. fused=True is numerically equivalent (same update rule, fused kernel).
        # Guard the fused PRECONDITIONS up front — torch checks them lazily, inside the first
        # optimizer.step(), so the try/except around the constructor below never sees them.
        # There are two, and both are real failures we have hit:
        #   * device: some torch versions only reject fused CPU params at step() time.
        #   * dtype: torch >= 2.5's _device_dtype_check_for_fused rejects COMPLEX params
        #     ("`fused=True` requires all the params to be floating point Tensors ... but
        #     torch.complex64 and cuda"). The pno encoder's SpectralConv1d spectral weights
        #     are complex64, so `--arch pno` with ml_perf.fused_adam died on its first step
        #     after ~70 min of cache preload (job 1342786). Falling back to the unfused
        #     AdamW is exactly equivalent mathematically — it costs kernel launches, not
        #     correctness — so a complex-parameter model should quietly take that path
        #     rather than force every pno config to special-case ml_perf.
        optimizer = None
        if self._fused_adam and fused_adam_supported(self.parameters()):
            try:
                optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                              weight_decay=self.weight_decay, fused=True)
            except (TypeError, RuntimeError, ValueError):
                optimizer = None
        if optimizer is None:
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

        # Constant second stage: hold the LR flat at the base lr after warmup (no decay).
        if str(self.lr_second_stage).lower() == "constant":
            # LambdaLR returning 1.0 keeps the optimiser at its base lr for every epoch.
            constant = LambdaLR(optimizer, lr_lambda=lambda *_: 1.0)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, constant],
                milestones=[warmup_epochs],
            )
            return [optimizer], [{
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }]

        # Cosine annealing down to lr_min_factor * base LR (0.1 legacy, 0.2 = lr/5)
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs,
                                   eta_min=self.lr * self.lr_min_factor)

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
