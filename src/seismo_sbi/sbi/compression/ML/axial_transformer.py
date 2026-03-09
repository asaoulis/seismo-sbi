import torch
import torch.nn as nn
from typing import Optional, Tuple
import math

def sinusoidal_time_embedding(L: int, d_model: int, device=None):
    """
    Returns sinusoidal embeddings for positions [0..L-1].
    Shape: (L, d_model)
    """
    position = torch.arange(L, device=device).unsqueeze(1)        # (L, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * -(math.log(10000.0) / d_model))
    pe = torch.zeros(L, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (L, d_model)

# ----------------------------
# Utility: simple position-wise FFN
# ----------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int = 2 * 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ----------------------------
# One axial block:
#   1) station-wise self-attn (per time slice)
#   2) time-wise self-attn (per station)
#   3) query tokens cross-attend to content (global)
#   4) position-wise FFN
# All pre-norm + residual
# ----------------------------
# ...existing code...
class AxialOrFullBlock(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward=512, dropout=0.1,
                 use_query_xattn=True, mode="axial", input_dim=None,
                 temporal_pool_tokens: int = 4):
        """
        mode: "axial" (station × time attention) or "full" (station-level only)
        input_dim: only needed if mode="full" (will project L*D → d_model)
        temporal_pool_tokens: if >0, use PMA-style temporal summarization per station before station-wise attention
        """
        super().__init__()
        self.mode = mode
        self.use_query_xattn = use_query_xattn
        self.temporal_pool_tokens = temporal_pool_tokens

        if mode == "full":
            assert input_dim is not None, "Need input_dim=L*D when mode='full'"
            print(f"Using full attention with input dim {input_dim}, projecting to {d_model}")
            self.proj = nn.Linear(input_dim, d_model)
            self.ln_sta = nn.LayerNorm(d_model)
            self.attn_sta = nn.MultiheadAttention(d_model, nheads, dropout=dropout, batch_first=True)

        elif mode == "axial":
            # Always keep these for the baseline/time path
            self.ln_sta = nn.LayerNorm(d_model)
            self.attn_sta = nn.MultiheadAttention(d_model, nheads, dropout=dropout, batch_first=True)

            self.ln_tim = nn.LayerNorm(d_model)
            self.attn_tim = nn.MultiheadAttention(d_model, nheads, dropout=dropout, batch_first=True)

            # Optional PMA-style temporal summarization and time←station cross-attn
            if self.temporal_pool_tokens > 0:
                # Per-station temporal pooling by multihead attention (PMA)
                self.pma_queries = nn.Parameter(torch.randn(1, self.temporal_pool_tokens, d_model))
                self.ln_pma_in = nn.LayerNorm(d_model)
                self.attn_pma = nn.MultiheadAttention(d_model, nheads, dropout=dropout, batch_first=True)

                # Used to provide station summaries as context to time tokens
                self.ln_z = nn.LayerNorm(d_model)
                self.tim_from_sta = nn.MultiheadAttention(d_model, nheads, dropout=dropout, batch_first=True)

        # Query cross-attention
        if use_query_xattn:
            self.ln_q = nn.LayerNorm(d_model)
            self.ln_ctx = nn.LayerNorm(d_model)
            self.q_xattn = nn.MultiheadAttention(d_model, nheads, dropout=dropout, batch_first=True)

        # FFN
        self.ln_ff = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)

    def forward(self, x, q=None, key_padding_mask=None):
        """
        Axial: x (B, N, L, D)
        Full:  x (B, N, L, D) but will flatten L*D first
        key_padding_mask (axial): (B, N, L) with True=pad
        """
        if self.mode == "full":
            B, N, LD = x.shape
            x = self.proj(x)                # (B, N, d_model)

            # station-level attention
            xs_in = self.ln_sta(x)
            sta_mask = key_padding_mask.reshape(B, N) if key_padding_mask is not None else None
            xs_out, _ = self.attn_sta(xs_in, xs_in, xs_in, key_padding_mask=sta_mask, need_weights=False)
            x = x + xs_out

        elif self.mode == "axial":
            B, N, L, D = x.shape

            if self.temporal_pool_tokens > 0:
                # -------- temporal PMA per station --------
                # shape inputs as (B*N, L, D)
                xt = x.reshape(B * N, L, D)
                tim_mask = key_padding_mask.reshape(B * N, L) if key_padding_mask is not None else None

                # learned queries attend to time tokens (Set Transformer PMA / Perceiver-style)
                q_lat = self.pma_queries.expand(B * N, -1, -1)              # (B*N, K, D)
                xt_in = self.ln_pma_in(xt)                                  # (B*N, L, D)
                z, _ = self.attn_pma(q_lat, xt_in, xt_in, key_padding_mask=tim_mask, need_weights=False)  # (B*N, K, D)

                # -------- station-wise attention over summaries --------
                K = self.temporal_pool_tokens
                zs = z.reshape(B, N, K, D).permute(0, 2, 1, 3).reshape(B * K, N, D)  # (B*K, N, D)
                zs_in = self.ln_sta(zs)

                sta_mask = None
                if key_padding_mask is not None:
                    # station is padded if all its times are padded
                    sta_mask_base = key_padding_mask.all(dim=2)             # (B, N) True=pad
                    sta_mask = sta_mask_base.unsqueeze(1).expand(B, K, N).reshape(B * K, N)

                zs_out, _ = self.attn_sta(zs_in, zs_in, zs_in, key_padding_mask=sta_mask, need_weights=False)
                zs = zs + zs_out

                # back to (B*N, K, D)
                z = zs.reshape(B, K, N, D).permute(0, 2, 1, 3).reshape(B * N, K, D)

                # -------- time tokens attend to station summaries (per station) --------
                xt_in = self.ln_tim(xt)             # reuse ln over time tokens
                z_ctx = self.ln_z(z)
                xt_from_sta, _ = self.tim_from_sta(xt_in, z_ctx, z_ctx, need_weights=False)  # (B*N, L, D)
                xt = xt + xt_from_sta

                # optional time self-attention for temporal dependencies
                xt_in2 = self.ln_tim(xt)
                xt_out, _ = self.attn_tim(xt_in2, xt_in2, xt_in2, key_padding_mask=tim_mask, need_weights=False)
                xt = xt + xt_out

                x = xt.reshape(B, N, L, D)

            else:
                # -------- original axial path --------
                # station-wise across N for each time slice
                xs = x.permute(0, 2, 1, 3).reshape(B * L, N, D)
                xs_in = self.ln_sta(xs)
                sta_mask = None
                if key_padding_mask is not None:
                    sta_mask = key_padding_mask.permute(0, 2, 1).reshape(B * L, N)
                xs_out, _ = self.attn_sta(xs_in, xs_in, xs_in, key_padding_mask=sta_mask, need_weights=False)
                xs = xs + xs_out
                x = xs.reshape(B, L, N, D).permute(0, 2, 1, 3).contiguous()

                # time-wise across L for each station
                xt = x.reshape(B * N, L, D)
                xt_in = self.ln_tim(xt)
                tim_mask = None
                if key_padding_mask is not None:
                    tim_mask = key_padding_mask.reshape(B * N, L)
                xt_out, _ = self.attn_tim(xt_in, xt_in, xt_in, key_padding_mask=tim_mask, need_weights=False)
                xt = xt + xt_out
                x = xt.reshape(B, N, L, D)

        # -------- query cross-attention --------
        if self.use_query_xattn and q is not None:
            q_norm = self.ln_q(q)
            ctx = self.ln_ctx(x.reshape(x.size(0), -1, x.size(-1)))  # flatten all tokens
            ctx_mask = key_padding_mask.reshape(x.size(0), -1) if key_padding_mask is not None else None
            q_upd, _ = self.q_xattn(q_norm, ctx, ctx, key_padding_mask=ctx_mask, need_weights=False)
            q = q + q_upd

        # -------- FFN --------
        xf = self.ln_ff(x)
        x = x + self.ffn(xf)

        return x, q

class SeismogramAxialTransformer(nn.Module):
    def __init__(
        self,
        station_coords: torch.Tensor,   # (N, 2) or (1, N, 2)
        d_model: int,
        nheads: int,
        num_layers: int,
        time_steps: int,
        conv_length: int,
        max_time_steps: int = 60,
        num_query_tokens: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pool_queries: str = "mean",
        mode: str = "axial",   # "axial" or "full"
        device=None,
        time_embedding_mode: str = "add",   # "add" or "concat"
        use_cls_token: bool = False,         # New: CLS-style global token instead of query tokens
        temporal_pool_tokens: int = 0,    # New: number of PMA temporal pool tokens per station (0=disable)
    ):
        super().__init__()
        self.register_buffer(
            "station_coords",
            torch.as_tensor(station_coords, dtype=torch.float32)
        )
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_query_tokens = num_query_tokens
        self.pool_queries = pool_queries  # now supports: mean, first, attn, max, gem
        self.device = device
        self.time_embedding_mode = time_embedding_mode
        self.timesteps = time_steps
        self.conv_length = conv_length
        self.mode = mode
        self.use_cls_token = use_cls_token

        # Sinusoidal time embedding (like transformer positional encoding)
        self.register_buffer("time_embed", sinusoidal_time_embedding(max_time_steps, conv_length))

        # Persistent query tokens (used only if CLS is not enabled)
        if (num_query_tokens > 0) and (not self.use_cls_token):
            self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, d_model))
        else:
            self.query_tokens = None

        # CLS token(s)
        if self.use_cls_token:
            if self.mode == "axial":
                # acts as an extra "station"; broadcast across time
                self.cls_token_axial = nn.Parameter(torch.randn(1, 1, 1, d_model))
            elif self.mode == "full":
                input_dim = self.conv_length * self.timesteps
                self.cls_token_full = nn.Parameter(torch.randn(1, 1, input_dim))

        # Stack of axial blocks
        self.blocks = nn.ModuleList([
            AxialOrFullBlock(
                d_model=d_model,
                nheads=nheads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_query_xattn=((num_query_tokens > 0) and (not self.use_cls_token)),
                mode=self.mode,
                input_dim=self.conv_length * self.timesteps if self.mode == "full" else None,
                temporal_pool_tokens=temporal_pool_tokens,   # pass through
            )
            for _ in range(num_layers)
        ])

        # Pooling heads (for attention and GeM)
        self.pool_attn = nn.Linear(d_model, 1)
        self.gem_p = nn.Parameter(torch.ones(1) * 3.0)

        self.final_ln = nn.LayerNorm(d_model)

    def station_position_embedding(self, pos: torch.Tensor, d_model: int = 128, batch_size: int = None):
        """
        pos: (N, 2) array of station coordinates (shared across batch)
        returns: (B, N, d_model) if batch_size is given, else (N, d_model)
        """
        N = pos.shape[0]
        device = pos.device 
        pes = []
        for coord_index in range(2):
            pe = torch.zeros(N, d_model // 2, device=device)
            position = pos[:, coord_index]                        # (N,)
            position = position.unsqueeze(1).repeat(1, d_model // 4)  # (N, d_model//4)

            div_term = 1 / torch.pow(
                10000.0, torch.arange(0, d_model // 2, 2, device=device) / (d_model // 2)
            )  # (d_model//4,)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pes.append(pe)

        out = torch.cat(pes, dim=-1)   # (N, d_model)

        if batch_size is not None:
            out = out.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, d_model)

        return out

    def _masked_mean(self, x: torch.Tensor, mask: Optional[torch.Tensor]):
        # x: (B, T, D); mask: (B, T) True means keep/valid
        if mask is None:
            return x.mean(dim=1)
        m = mask.to(x.dtype).unsqueeze(-1)
        denom = m.sum(dim=1).clamp_min(1e-6)
        return (x * m).sum(dim=1) / denom

    def _masked_max(self, x: torch.Tensor, mask: Optional[torch.Tensor]):
        if mask is None:
            return x.max(dim=1).values
        neg_inf = torch.finfo(x.dtype).min
        x_masked = x.masked_fill(~mask.unsqueeze(-1), neg_inf)
        vals = x_masked.max(dim=1).values
        # handle all-masked rows -> zeros
        all_masked = (~mask).all(dim=1)
        if all_masked.any():
            vals[all_masked] = 0.0
        return vals

    def _attn_pool(self, x: torch.Tensor, mask: Optional[torch.Tensor]):
        # x: (B, T, D)
        scores = self.pool_attn(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=1)
        attn = attn.unsqueeze(-1)
        return (x * attn).sum(dim=1)

    def _gem_pool(self, x: torch.Tensor, mask: Optional[torch.Tensor], eps: float = 1e-6):
        # x: (B, T, D)
        p = torch.clamp(self.gem_p, min=1.0, max=8.0)
        x_clamped = x.clamp_min(eps)
        if mask is None:
            return torch.pow(torch.mean(torch.pow(x_clamped, p), dim=1), 1.0 / p)
        m = mask.to(x.dtype).unsqueeze(-1)
        denom = m.sum(dim=1).clamp_min(1.0)  # avoid div by zero
        pooled = torch.pow((torch.pow(x_clamped, p) * m).sum(dim=1) / denom, 1.0 / p)
        return pooled

    def _apply_pool(self, seq: torch.Tensor, mask: Optional[torch.Tensor]):
        # seq: (B, T, D)
        mode = self.pool_queries
        if mode == "first":
            return seq[:, 0, :]
        elif mode == "mean":
            return self._masked_mean(seq, mask)
        elif mode == "max":
            return self._masked_max(seq, mask)
        elif mode == "attn":
            return self._attn_pool(seq, mask)
        elif mode == "gem":
            return self._gem_pool(seq, mask)
        else:
            # fallback to mean
            return self._masked_mean(seq, mask)

    def forward(self, x: torch.Tensor, key_padding_mask=None):
        """
        x: (B, N, L, D)
        station_coords: (B, N, 2) or (1, N, 2) fixed coordinates
        key_padding_mask: axial mode -> (B, N, L) booleans (True=pad), full mode -> same input is accepted
        """
        B, N, L, D = x.shape
        # if D != self.d_model:
        #     raise ValueError(f"Expected input dim {self.d_model}, got {D}")

        # Station embeddings from coords
        sta_e = self.station_position_embedding(self.station_coords, d_model=self.conv_length, batch_size=B)
        # sta_e: (B, N, D)

        # Time embeddings
        # t_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)  # (B, L)
        t_e = self.time_embed[:L, :]

        x = x + t_e.unsqueeze(0).unsqueeze(1)  # (B, N, L, D)

        if self.mode == "axial":
            sta_e = self.station_position_embedding(self.station_coords, d_model=self.conv_length, batch_size=B)
            x = x + sta_e.unsqueeze(2)
        elif self.mode == "full":
            x = x.reshape(B, N, L * D)
            sta_e = self.station_position_embedding(self.station_coords, d_model=L*D, batch_size=B)
            x = x + sta_e

        # Optionally add CLS token (as extra station)
        mask_to_pass = key_padding_mask
        if self.use_cls_token:
            if self.mode == "axial":
                # x: (B, N, L, D) -> (B, N+1, L, D)
                cls = self.cls_token_axial.expand(B, 1, L, D)
                x = torch.cat([cls, x], dim=1)
                if key_padding_mask is None:
                    mask_to_pass = torch.zeros(B, N + 1, L, dtype=torch.bool, device=x.device)
                else:
                    cls_mask = torch.zeros(B, 1, L, dtype=torch.bool, device=x.device)
                    mask_to_pass = torch.cat([cls_mask, key_padding_mask], dim=1)
            elif self.mode == "full":
                # x: (B, N, L*D) -> (B, N+1, L*D)
                cls_pre = self.cls_token_full.expand(B, 1, -1)
                x = torch.cat([cls_pre, x], dim=1)
                if key_padding_mask is None:
                    mask_to_pass = torch.zeros(B, N + 1, dtype=torch.bool, device=x.device)
                else:
                    # collapse time mask to station-level and prepend CLS unmasked
                    sta_mask = key_padding_mask.any(dim=2)  # (B, N)
                    cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
                    mask_to_pass = torch.cat([cls_mask, sta_mask], dim=1)

        # Prepare query tokens (disabled when using CLS)
        q = None
        if (self.query_tokens is not None) and (not self.use_cls_token):
            q = self.query_tokens.expand(B, -1, -1)

        # Axial/full blocks
        for blk in self.blocks:
            x, q = blk(x, q, key_padding_mask=mask_to_pass)

        x = self.final_ln(x)

        pooled = None
        # Pooling: prefer CLS if enabled, else pool queries (backward compatible)
        if self.use_cls_token:
            if self.mode == "axial":
                # CLS is the first "station" across time
                cls_seq = x[:, 0, :, :]  # (B, L, D)
                time_keep = None
                if key_padding_mask is not None:
                    # valid if any station has valid at that time -> we pool over valid times
                    time_keep = ~key_padding_mask.any(dim=1)  # (B, L)
                pooled = self._apply_pool(cls_seq, time_keep)
            elif self.mode == "full":
                # single CLS token after station-level attention
                pooled = x[:, 0, :]  # (B, D)
        elif q is not None:
            # pool queries with selected strategy
            # q: (B, Tq, D), no mask
            if self.pool_queries == "first":
                pooled = q[:, 0, :]
            else:
                pooled = self._apply_pool(q, mask=None)
        # else: pooled remains None (backward compatible)
        return x, q, pooled
