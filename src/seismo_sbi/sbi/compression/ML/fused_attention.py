"""Fused multi-head attention, an opt-in drop-in for ``nn.MultiheadAttention``.

``FusedMHA`` routes the attention through ``scaled_dot_product_attention`` instead of the
unfused ``bmm -> softmax -> bmm`` path. The axial transformer and the pooling head issue many
small masked attentions, which the stock module does not fast-path while a key-padding mask is
present, so the workload is launch-bound; one fused kernel also avoids materialising the score
matrix. Parameters, initialisation and the arithmetic match, so outputs agree to floating-point
tolerance. ``build_mha(..., use_sdpa)`` returns this or the stock module.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedMHA(nn.Module):
    """SDPA-based drop-in for ``nn.MultiheadAttention(..., batch_first=True)``.

    Supports the exact call surface the axial / PMA code uses:
    ``forward(query, key, value, key_padding_mask=None, need_weights=False)`` and returns
    ``(attn_output, None)`` (weights are never requested). ``key_padding_mask`` is ``(B, S)``
    with ``True`` = pad, matching the convention fed to the stock module.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 batch_first: bool = True, bias: bool = True) -> None:
        super().__init__()
        if not batch_first:
            raise ValueError("FusedMHA only supports batch_first=True.")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = float(dropout)
        # Packed QKV projection — identical layout to nn.MultiheadAttention.
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Mirror nn.MultiheadAttention._reset_parameters: xavier-uniform on the packed in-proj,
        # zero in/out biases, and leave out_proj.weight at the default nn.Linear init.
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, None]:
        B, Lq, E = query.shape
        Lk = key.shape[1]
        if query is key and key is value:
            qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)
            if self.in_proj_bias is not None:
                b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0)
            else:
                b_q = b_k = b_v = None
            q = F.linear(query, w_q, b_q)
            k = F.linear(key, w_k, b_k)
            v = F.linear(value, w_v, b_v)

        # (B, S, E) -> (B, heads, S, head_dim)
        q = q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # Additive attention mask from the (B, Lk) key-padding mask (True=pad → -inf).
        attn_bias = attn_mask
        if key_padding_mask is not None:
            kpm = key_padding_mask.view(B, 1, 1, Lk)
            attn_bias = torch.zeros(B, 1, 1, Lk, dtype=q.dtype, device=q.device)
            attn_bias = attn_bias.masked_fill(kpm, float("-inf"))

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(B, Lq, E)
        return self.out_proj(out), None


def build_mha(embed_dim: int, num_heads: int, dropout: float = 0.0,
              use_sdpa: bool = False) -> nn.Module:
    """Return a FusedMHA (when ``use_sdpa``) or a stock ``nn.MultiheadAttention``.

    Both expose ``forward(q, k, v, key_padding_mask=, need_weights=)`` returning
    ``(output, weights)`` and the same parameter names, so callers are agnostic.
    """
    if use_sdpa:
        return FusedMHA(embed_dim, num_heads, dropout=dropout, batch_first=True)
    return nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
