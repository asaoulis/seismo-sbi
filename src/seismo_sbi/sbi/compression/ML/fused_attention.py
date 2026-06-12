"""Fused multi-head attention drop-in (opt-in performance path).

``FusedMHA`` is a faithful, numerically-equivalent replacement for
``nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)`` that routes the
attention through :func:`torch.nn.functional.scaled_dot_product_attention` (the fused
flash / memory-efficient kernel) instead of the unfused ``bmm → softmax → bmm`` math path.

Why: the axial transformer and the PMA pooling head issue many *small* masked attentions
(station axis N≈16, time axis L≈50). On torch 2.0 ``nn.MultiheadAttention`` does NOT take its
fused fast-path when training with a ``key_padding_mask``, so each attention becomes ~5 separate
CUDA kernels (two ``bmm`` + softmax + masking + projections). This module collapses the core
attention into ONE fused kernel — fewer launches (the workload is launch-overhead-bound) and
lower memory (no materialised B×h×S×S score matrix), and it is dramatically faster under bf16
autocast (flash attention).

Equivalence: same learnable parameters (``in_proj_weight``, ``in_proj_bias``, ``out_proj``) and the
same scaled-dot-product math ⇒ given identical weights the output matches ``nn.MultiheadAttention``
to floating-point tolerance (verified ~1e-7 in the benches). The parameter layout and the
``_reset_parameters`` scheme mirror ``nn.MultiheadAttention`` so a fresh module also *initialises*
the same way (xavier-uniform packed in-proj, zero biases, default-Linear out-proj weight).

``build_mha(embed_dim, num_heads, dropout, use_sdpa)`` returns a ``FusedMHA`` when ``use_sdpa`` is
set, else a stock ``nn.MultiheadAttention`` — so the surrounding code is unchanged and the feature
is fully opt-in (absent ⇒ byte-identical legacy attention).
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
