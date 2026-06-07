"""Set-Transformer PMA pooling head (review §3.4).

The legacy aggregation in ``axial_transformer.py`` is *already a light PMA*: ``num_query_tokens``
(default 8) learnable query tokens cross-attend to the flattened ``(N·L)`` token field inside
**every** axial block (``q = q + q_upd``), and are then collapsed by an **unweighted mean**
(``_apply_pool(q, "mean")``). Review §3.4 makes two precise criticisms:

1. The unweighted mean over the ``k`` seeds is **lossy** — each seed is *trained to specialise*
   (one may track overall amplitude / ``M0``, another the azimuthal polarity pattern …) and
   averaging deliberately discards that. A learned ``concat→Linear`` combination keeps it. This is
   the one component with a clear, non-trivial expected benefit.
2. Cross-attending the seeds to the full ``N·L`` soup *inside every block* is the Perceiver pattern
   (a latent array repeatedly reading the inputs through depth). The canonical Set-Transformer
   factorisation is: the axial blocks **are** the set encoder, and a single PMA head pools the
   *final* encoded token set **once** at the end (cheaper, and a clean reusable module).

This module implements that head as an **opt-in** alternative. Crucially it does **not** change the
axial encoder: in the legacy path the query/seed tokens only ever *read* from the token field ``x``
and never write back to it, so removing the per-block skim (the transformer sets
``num_query_tokens=0`` when this head is enabled) leaves ``x``'s station-then-time-then-FFN evolution
byte-identical. Absent any config ⇒ the head is never built ⇒ the legacy query-mean path is unchanged.

Two pooling modes (``pool_over``), per the user's B+C decision:

``tokens`` (Option B, default)
    ``k`` learnable seeds PMA-pool the flattened, masked ``(N·L)`` final encoded token set.

``stations`` (Option C, review §3.4b — "stations as a set on a sphere")
    A single-seed :class:`_TimePool` MAB first collapses each station's ``L`` time tokens to **one**
    token (masked over time) → ``(B, N, D)``; then ``k`` seeds PMA-pool over the ``N`` station tokens
    (masked over stations). Sharper, physically-motivated inductive bias; sets up the §3.3 geometric
    station-attention work; cheapest read-out (``k × N`` rather than ``k × N·L``).

Both modes share the seeds, the MAB pool, the optional self-attention among seeds (SAB), and the
learned combination.

Scaling discipline (the §3.1/§3.2 "well-designed and scaled" requirement — there are no RFFs to add
here; this is the analogue for an attention head):

* seeds initialised ``randn * seed_init_scale`` with ``seed_init_scale=1.0`` by default, matching the
  distribution of the legacy ``query_tokens`` so init behaviour is familiar;
* attention scaling handled by ``nn.MultiheadAttention`` (``1/√d_head``);
* **pre-LN + residual** placement matching ``AxialOrFullBlock`` so the head cannot destabilise
  training, and a final ``LayerNorm`` on the combined vector for a consistent scale into the flow;
* the ``linear`` combine is **initialised to the mean** of the seeds (each of the ``k`` ``d×d`` blocks
  ``= I/k``, bias ``0``; identity for ``k=1``), a well-conditioned, non-degenerate start the network
  then refines into a *learned* combination — it is NOT frozen as a mean;
* fully-padded rows (a sample / station whose entire key set is masked) are unmasked before attention
  so ``nn.MultiheadAttention`` never sees an all-``True`` key row and stays finite (the same
  NaN-avoidance the axial blocks use).

The optional SAB among seeds is the most "fluff-prone" knob (the Set-Transformer paper reports
diminishing returns past a small seed/head count); it defaults **off** and exists for ablation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# Keys accepted in the ``pma_pooling`` config block (``enabled`` is stripped by the config parser
# before the dict reaches the model). A single validator covers the whole block.
_CONFIG_KEYS = {
    "pool_over", "num_seeds", "num_heads", "seed_self_attention", "combine",
    "ffn", "dim_feedforward", "dropout", "seed_init_scale", "time_pool_heads",
}


def _unmask_all_true_rows(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Set fully-padded (all-``True``) rows to all-``False`` so attention stays finite.

    ``mask`` is a key-padding mask ``(..., S)`` with ``True`` = pad. A query whose entire key set is
    masked makes ``nn.MultiheadAttention`` return ``NaN``; that only happens for a fully-padded
    sample / station, which downstream code discards anyway, so blanking its mask row is safe and
    purely keeps the forward finite. Returns ``None`` for ``None`` and leaves the mask untouched
    (no copy) when no row is fully padded.
    """
    if mask is None:
        return None
    fully = mask.all(dim=-1)
    if not bool(fully.any()):
        return mask
    mask = mask.clone()
    mask[fully] = False
    return mask


class _FeedForward(nn.Module):
    """Position-wise FFN (Linear→GELU→Dropout→Linear→Dropout); mirrors ``axial_transformer``.

    Defined locally (rather than imported) so this module has no dependency on
    ``axial_transformer`` — which imports *this* module — avoiding a circular import.
    """

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.0) -> None:
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


class _MAB(nn.Module):
    """Pre-LN Multihead-Attention Block (Set Transformer): ``Q`` attends to set ``Z``.

    ``H = Q + MHA(LN_q(Q), LN_kv(Z), LN_kv(Z))``; then if ``ffn``: ``H = H + FFN(LN_ff(H))``.
    Self-attention (SAB) is just ``forward(Z, Z)``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        ffn: bool = True,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.use_ffn = bool(ffn)
        if self.use_ffn:
            self.ln_ff = nn.LayerNorm(d_model)
            self.ffn = _FeedForward(d_model, dim_feedforward or 2 * d_model, dropout)

    def forward(
        self,
        q: torch.Tensor,
        z: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z_n = self.ln_kv(z)
        attn_out, _ = self.attn(
            self.ln_q(q), z_n, z_n, key_padding_mask=key_padding_mask, need_weights=False
        )
        h = q + attn_out
        if self.use_ffn:
            h = h + self.ffn(self.ln_ff(h))
        return h


class _TimePool(nn.Module):
    """Collapse each station's ``L`` time tokens to one token via a single-seed MAB.

    ``forward(x (B,N,L,D), time_mask (B,N,L)|None) -> (B, N, D)``. A learned query attends over the
    time axis per station (batched over ``B·N``); fully-padded stations are unmasked first so the
    MAB stays finite. Used only by ``pool_over="stations"``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        ffn: bool = True,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        seed_init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, d_model) * seed_init_scale)
        self.mab = _MAB(
            d_model, num_heads, ffn=ffn, dim_feedforward=dim_feedforward, dropout=dropout
        )

    def forward(
        self, x: torch.Tensor, time_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, L, D = x.shape
        xt = x.reshape(B * N, L, D)
        kpm = None
        if time_mask is not None:
            kpm = _unmask_all_true_rows(time_mask.reshape(B * N, L))
        seed = self.seed.expand(B * N, 1, D)
        out = self.mab(seed, xt, key_padding_mask=kpm)  # (B*N, 1, D)
        return out.reshape(B, N, D)


class SetTransformerPMAHead(nn.Module):
    """Opt-in Set-Transformer PMA pooling head (review §3.4); see module docstring.

    ``forward(x (B,N,L,D), key_padding_mask (B,N,L)|None) -> (B, d_model)``. ``key_padding_mask``
    uses ``True`` = pad (the same convention the axial transformer feeds the blocks). The axial
    encoder is unchanged; this head only replaces the read-out.
    """

    def __init__(
        self,
        d_model: int,
        num_heads_default: int,
        *,
        pool_over: str = "tokens",
        num_seeds: int = 1,
        num_heads: Optional[int] = None,
        seed_self_attention: bool = False,
        combine: str = "linear",
        ffn: bool = True,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        seed_init_scale: float = 1.0,
        time_pool_heads: Optional[int] = None,
    ) -> None:
        super().__init__()
        if pool_over not in ("tokens", "stations"):
            raise ValueError(
                f"pool_over must be 'tokens' or 'stations', got '{pool_over}'."
            )
        if combine not in ("linear", "mean", "first"):
            raise ValueError(
                f"combine must be 'linear', 'mean' or 'first', got '{combine}'."
            )
        num_seeds = int(num_seeds)
        if num_seeds < 1:
            raise ValueError(f"num_seeds must be >= 1, got {num_seeds}.")
        if seed_self_attention and num_seeds < 2:
            raise ValueError(
                "seed_self_attention=True requires num_seeds >= 2 (nothing for a single seed "
                "to attend to)."
            )
        nh = int(num_heads) if num_heads else int(num_heads_default)

        self.pool_over = pool_over
        self.num_seeds = num_seeds
        self.combine = combine
        self.d_model = d_model

        # Per-station time-collapse (Option C only).
        self.time_pool: Optional[_TimePool] = None
        if pool_over == "stations":
            self.time_pool = _TimePool(
                d_model, int(time_pool_heads) if time_pool_heads else nh,
                ffn=ffn, dim_feedforward=dim_feedforward, dropout=dropout,
                seed_init_scale=seed_init_scale,
            )

        # PMA seeds + pool.
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, d_model) * seed_init_scale)
        self.pma = _MAB(
            d_model, nh, ffn=ffn, dim_feedforward=dim_feedforward, dropout=dropout
        )

        # Optional self-attention among the k seed outputs.
        self.sab: Optional[_MAB] = None
        if seed_self_attention:
            self.sab = _MAB(
                d_model, nh, ffn=ffn, dim_feedforward=dim_feedforward, dropout=dropout
            )

        # k -> 1 combination.
        self.combine_linear: Optional[nn.Linear] = None
        if combine == "linear":
            self.combine_linear = nn.Linear(num_seeds * d_model, d_model)
            self._init_combine_as_mean()

        self.final_ln = nn.LayerNorm(d_model)

    def _init_combine_as_mean(self) -> None:
        """Initialise the combine ``Linear`` so the output starts as the mean of the seeds.

        Each of the ``k`` ``d×d`` weight blocks is set to ``I/k`` and the bias to ``0`` ⇒ at init the
        head is a well-conditioned, non-degenerate aggregator (identity for ``k=1``) that the
        network then refines into a learned, weighted combination.
        """
        assert self.combine_linear is not None
        d = self.d_model
        with torch.no_grad():
            w = self.combine_linear.weight  # (d, k*d)
            w.zero_()
            eye = torch.eye(d, dtype=w.dtype, device=w.device)
            for j in range(self.num_seeds):
                w[:, j * d:(j + 1) * d] = eye / self.num_seeds
            self.combine_linear.bias.zero_()

    @classmethod
    def from_config(
        cls, d_model: int, num_heads_default: int, cfg: Dict[str, Any]
    ) -> "SetTransformerPMAHead":
        """Build from the ``pma_pooling`` config dict, rejecting unknown keys."""
        unknown = set(cfg) - _CONFIG_KEYS
        if unknown:
            raise ValueError(
                f"Unknown pma_pooling keys {sorted(unknown)}; valid: {sorted(_CONFIG_KEYS)}"
            )
        return cls(d_model, num_heads_default, **cfg)

    def _combine_seeds(self, h: torch.Tensor) -> torch.Tensor:
        """Reduce the ``k`` seed outputs ``(B, k, D)`` to one vector ``(B, D)``."""
        if self.combine == "first":
            return h[:, 0, :]
        if self.combine == "mean":
            return h.mean(dim=1)
        # learned linear combination
        B = h.shape[0]
        assert self.combine_linear is not None
        return self.combine_linear(h.reshape(B, -1))

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, L, D = x.shape
        if self.pool_over == "stations":
            z = self.time_pool(x, time_mask=key_padding_mask)        # (B, N, D)
            set_mask = (
                key_padding_mask.all(dim=2) if key_padding_mask is not None else None
            )  # (B, N) True = station fully padded
        else:  # tokens
            z = x.reshape(B, N * L, D)                               # (B, N·L, D)
            set_mask = (
                key_padding_mask.reshape(B, N * L)
                if key_padding_mask is not None else None
            )

        set_mask = _unmask_all_true_rows(set_mask)
        seeds = self.seeds.expand(B, -1, -1)                         # (B, k, D)
        h = self.pma(seeds, z, key_padding_mask=set_mask)           # (B, k, D)
        if self.sab is not None:
            h = self.sab(h, h)                                       # SAB among seeds (no mask)
        pooled = self._combine_seeds(h)                              # (B, D)
        return self.final_ln(pooled)
