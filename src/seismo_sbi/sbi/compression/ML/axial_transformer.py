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
class AxialOrFullBlock(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward=512, dropout=0.1,
                 use_query_xattn=True, mode="axial", input_dim=None):
        """
        mode: "axial" (station × time attention) or "full" (station-level only)
        input_dim: only needed if mode="full" (will project L*D → d_model)
        """
        super().__init__()
        self.mode = mode
        self.use_query_xattn = use_query_xattn

        if mode == "full":
            assert input_dim is not None, "Need input_dim=L*D when mode='full'"
            print(f"Using full attention with input dim {input_dim}, projecting to {d_model}")
            self.proj = nn.Linear(input_dim, d_model)
            self.ln_sta = nn.LayerNorm(d_model)
            self.attn_sta = nn.MultiheadAttention(d_model, nheads, dropout=dropout, batch_first=True)

        elif mode == "axial":
            # Station-wise self-attn
            self.ln_sta = nn.LayerNorm(d_model)
            self.attn_sta = nn.MultiheadAttention(d_model, nheads, dropout=dropout, batch_first=True)

            # Time-wise self-attn
            self.ln_tim = nn.LayerNorm(d_model)
            self.attn_tim = nn.MultiheadAttention(d_model, nheads, dropout=dropout, batch_first=True)

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

            # -------- station-wise --------
            xs = x.permute(0, 2, 1, 3).reshape(B*L, N, D)
            xs_in = self.ln_sta(xs)
            sta_mask = None
            if key_padding_mask is not None:
                sta_mask = key_padding_mask.permute(0, 2, 1).reshape(B*L, N)
            xs_out, _ = self.attn_sta(xs_in, xs_in, xs_in, key_padding_mask=sta_mask, need_weights=False)
            xs = xs + xs_out
            x = xs.reshape(B, L, N, D).permute(0, 2, 1, 3).contiguous()

            # -------- time-wise --------
            xt = x.reshape(B*N, L, D)
            xt_in = self.ln_tim(xt)
            tim_mask = None
            if key_padding_mask is not None:
                tim_mask = key_padding_mask.reshape(B*N, L)
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
    ):
        super().__init__()
        self.register_buffer(
            "station_coords",
            torch.as_tensor(station_coords, dtype=torch.float32)
        )
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_query_tokens = num_query_tokens
        self.pool_queries = pool_queries
        self.device = device
        self.time_embedding_mode = time_embedding_mode
        self.timesteps = time_steps
        self.conv_length = conv_length
        self.mode = mode

        # Sinusoidal time embedding (like transformer positional encoding)
        self.register_buffer("time_embed", sinusoidal_time_embedding(max_time_steps, conv_length))

        # Persistent query tokens
        if num_query_tokens > 0:
            self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, d_model))
        else:
            self.query_tokens = None

        # Stack of axial blocks
        self.blocks = nn.ModuleList([
            AxialOrFullBlock(
                d_model=d_model,
                nheads=nheads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_query_xattn=(num_query_tokens > 0),
                mode=self.mode,                       # or "axial"
                input_dim=self.conv_length * self.timesteps if self.mode == "full" else None,
            )
            for _ in range(num_layers)
        ])


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


    def forward(self, x: torch.Tensor, key_padding_mask=None):
        """
        x: (B, N, L, D)
        station_coords: (B, N, 2) or (1, N, 2) fixed coordinates
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
        # Prepare query tokens
        q = None
        if self.query_tokens is not None:
            q = self.query_tokens.expand(B, -1, -1)

        # Axial blocks
        for blk in self.blocks:
            x, q = blk(x, q, key_padding_mask=key_padding_mask)

        x = self.final_ln(x)

        pooled = None
        if q is not None:
            if self.pool_queries == "mean":
                pooled = q.mean(dim=1)
            elif self.pool_queries == "first":
                pooled = q[:, 0, :]
        return x, q, pooled
