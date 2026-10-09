"""Embedding network of the NPE.

:class:`SeismogramTransformer` encodes each station's traces with a pluggable station encoder,
mixes stations with :class:`~.axial_transformer.SeismogramAxialTransformer`, and maps the result
and any source conditioning to a fixed-length summary, which conditions the flow.
"""

import torch

from torch import nn

from seismo_sbi.sbi.npe.networks.axial_transformer import SeismogramAxialTransformer
from seismo_sbi.sbi.npe.networks.station_encoders import build_station_encoder, BandLimit, InputDecimator
from seismo_sbi.sbi.npe.networks.amplitude_embedding import AmplitudeTokenEmbedding
from seismo_sbi.sbi.npe.source_conditioning import (
    SourceConditioner,
    FiLM,
    relative_station_geometry,
    unpack_context,
    unpack_variable_context,
)


class SeismogramTransformer(nn.Module):
    """Embedding network: a per-station encoder, an axial transformer over stations and time,
    and a head mapping the pooled result to a fixed-length summary.

    Input is ``(batch, n_stations, n_components, n_samples)``, or a packed 2-D context when
    source conditioning or variable stations are configured; output is ``(batch, num_outputs)``.
    """

    def __init__(self, num_seismic_components, transformer_config,
                        feature_length, num_outputs, noise_model,
                        seismogram_locations : torch.Tensor, device,
                        aggregation: str = "mean", input_length: int = 200) -> None:
        """``transformer_config`` holds ``channels`` and the optional encoder, conditioning, pooling,
        amplitude, bottleneck and performance entries; ``seismogram_locations`` is
        ``(n_stations, 2)`` latitude and longitude in degrees; ``input_length`` is the per-trace
        sample count.
        """
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

        # ``perf.amp`` runs the embedding net in reduced precision; the flow head stays in fp32.
        perf = transformer_config.get("perf", {}) or {}
        self._amp = bool(perf.get("amp", False))
        _amp_dtype = str(perf.get("amp_dtype", "bfloat16")).lower()
        self._amp_dtype = torch.bfloat16 if _amp_dtype in ("bfloat16", "bf16") else torch.float16
        # SDPA (fused scaled_dot_product_attention) for the axial / PMA attentions.
        self._use_sdpa = bool(perf.get("sdpa", False))

        # The same spectral limit on every input, before anything else sees the traces.
        limit_cfg = transformer_config.get("band_limit", None) or {}
        self.band_limit = BandLimit(**limit_cfg) if limit_cfg else None

        # Band-limited data sampled above its Nyquist rate decimates losslessly. Unpacking keeps
        # the original trace length; only the encoder and everything after it sees T/k.
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
        pool_method = transformer_config.get("pooling", "mean")  # supports: mean, first, attn, max, gem
        use_cls = transformer_config.get("use_cls_token", False)
        num_q = transformer_config.get("num_query_tokens", 8)
        # The axial transformer is built at the end of __init__ so the positional encoder can be
        # told whether station coordinates are source-relative or absolute.

        # Head d_model -> d_model -> bottleneck -> num_outputs: a narrower space for the MMD kernel,
        # with the encoder and flow context widths unchanged.
        _bneck_cfg = (transformer_config or {}).get("summary_bottleneck") or {}
        bneck = _bneck_cfg.get("dim") if isinstance(_bneck_cfg, dict) else _bneck_cfg
        self.summary_bottleneck_dim = int(bneck) if bneck else None
        _head_out = self.summary_bottleneck_dim or num_outputs
        self.source_param_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, _head_out)
        )
        # The MMD kernel bandwidth is a median heuristic over these vectors, so a drifting
        # scale is what destabilises it.
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

        # When enabled, embed() expects the packed context from variable_station_collate and
        # feeds per-sample station coordinates instead of the fixed buffer.
        self._variable_stations = bool(transformer_config.get("variable_stations", False))
        # "absolute" feeds raw (lat, lon); "relative" feeds source-relative (distance, azimuth)
        # and needs source conditioning, so there is a source vector to measure against.
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

        # Built last, so the positional encoder knows whether station coordinates are
        # source-relative; without a config it keeps the plain sinusoidal encoding.
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
        # With a pooling head the seeds live in the head and pool the final token set, so the
        # in-block query cross-attention is switched off; without one the encoder pools itself.
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
        """Build the source conditioner and the injection layers ``cond_cfg['inject']`` names; with
        no ``cond_cfg`` conditioning stays off.
        """
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

    def forward(self, x : torch.Tensor):
        """The summary vector, ``(batch, num_outputs)``, of a batch of seismograms or packed contexts."""
        # Scoped to the embedding net, so the flow still receives an fp32 context and its
        # precision-brittle transforms stay in fp32.
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
        """Pooled embedding, ``(batch, d_model)``, before the summary head.

        Unpacks a conditioning or variable-station context, encodes each station's traces, adds
        the configured conditioning and amplitude tokens and runs the axial transformer.
        """
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
            # A 4-D tensor means the full master station set: every station valid, coordinates
            # from the fixed buffer. A subset must arrive packed, with per-sample coordinates.
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
            # A packed context reached a model with no conditioning configured; fail clearly
            # rather than on the shape unpacking below.
            raise ValueError(
                "Received a packed 2-D context but this model has no conditioning configured "
                "(n_cond == 0). Did you set source_location on an unconditioned model, or load "
                "a conditioned checkpoint into a default-constructed trainer?"
            )

        if self.band_limit is not None:
            x = self.band_limit(x)

        # Model-entry decimation: after unpacking (which needs the original trace length),
        # before the encoder/amplitude paths — everything downstream sees T/k samples.
        if self.input_decimator is not None:
            x = self.input_decimator(x)

        (batch_size, num_stations, num_seismic_components, trace_length) = x.shape
        B, N = batch_size, num_stations

        # Source embedding (shared across stations/time) — only when conditioning is active.
        source_emb = self.source_conditioner(source_vec) if (source_vec is not None) else None

        # Source depth (km), RFF-encoded by the positional encoder when include_depth is set.
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
            # The transformer mask is (B, N, L) with True meaning pad, so invert the station
            # validity mask and broadcast it over the token axis as a view.
            key_padding_mask = (~var_mask).unsqueeze(-1).expand(B, N, self.L)

        # The radiation-pattern token goes in before the transformer so cross-station attention
        # sees relative amplitude; the per-event scale vector is added to the pooled embedding.
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

        # The per-station tokens carry only the array-relative pattern, so the per-event
        # amplitude vector is what gives the flow absolute scale.
        if amp_global is not None:
            pooled = pooled + amp_global
        return pooled
