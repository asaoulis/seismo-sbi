"""Reusable Random Fourier Feature (RFF) primitives for embedding scalars / small
coordinate vectors into a transformer-width token.

Why this exists
---------------
Networks have a spectral bias toward low frequencies, so a raw, unbounded scalar fed
directly into an MLP is hard to learn from (Tancik et al. 2020, *Fourier Features Let
Networks Learn High Frequency Functions in Low Dimensional Domains*). A Fourier-feature
map lifts the scalar into a handful of well-conditioned sin/cos dimensions. This module
provides a single, carefully-scaled implementation used first by the per-station
**amplitude** embedding (``amplitude_embedding.py``) and intended for reuse by the
later source-relative **geometry** positional encoding (review §3.2) — keep new Fourier
encodings here rather than re-deriving ad-hoc sinusoids.

Scaling discipline (avoiding dead weights / aliasing)
-----------------------------------------------------
The single biggest failure mode of RFF on a physical scalar is a **scale mismatch**: if
the input wanders over many orders of magnitude (e.g. ``log`` amplitude across earthquake
magnitudes), a fixed frequency bank either sees an essentially constant input (features
collapse to ``[0, 1]`` ⇒ dead, no gradient) or a pseudo-random one (aliasing ⇒ noise).
:class:`ScalarFourierEmbedding` therefore (a) **standardises the input to O(1)** before the
RFF, (b) uses a Gaussian frequency bank with a tuned ``sigma`` matched to that O(1) scale,
and (c) **concatenates a raw standardised pass-through** alongside the RFF so there is
always a non-vanishing linear gradient path even if every RFF frequency is momentarily
ill-matched.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class GaussianFourierFeatures(nn.Module):
    r"""Gaussian Random Fourier Features (Tancik et al. 2020).

    Maps ``x: (..., in_dim)`` to ``[sin(2\pi x B^T), cos(2\pi x B^T)]`` of shape
    ``(..., 2 * num_freqs)``, with the frequency matrix ``B \in R^{num_freqs x in_dim}``
    drawn from ``N(0, sigma^2)``. By default ``B`` is a **seeded, fixed buffer**
    (reproducible, no spectral drift during training); set ``learnable=True`` to make it a
    trainable ``nn.Parameter``.

    The input is expected to be ~O(1) (standardise upstream). ``sigma`` then sets how many
    cycles the bank resolves over a unit of input: ``sigma ~ 1`` resolves roughly one cycle
    per unit (a sane default), larger ``sigma`` resolves finer structure at the risk of
    aliasing, smaller ``sigma`` flattens toward a constant (dead) map.
    """

    def __init__(
        self,
        in_dim: int,
        num_freqs: int = 16,
        sigma: float = 1.0,
        learnable: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if num_freqs <= 0:
            raise ValueError(f"num_freqs must be positive, got {num_freqs}")
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.sigma = float(sigma)
        self.out_dim = 2 * num_freqs

        gen = torch.Generator().manual_seed(int(seed))
        B = torch.randn(num_freqs, in_dim, generator=gen) * self.sigma
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim) -> (..., 2*num_freqs)
        proj = 2.0 * math.pi * torch.matmul(x, self.B.t())  # (..., num_freqs)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class RunningStandardizer(nn.Module):
    """Non-trainable running standardiser (BatchNorm-style, ``affine=False``, mask-aware).

    Normalises the last-dim features to ~unit scale using **running** mean/variance buffers
    (so the forward pass — and its gradient — never couples samples within a batch, unlike
    standard BatchNorm). Running stats are updated from the *masked* batch in ``train()``
    mode only; the first update seeds them directly. Used to bring an **absolute** physical
    scalar (e.g. ``log`` amplitude, or the per-event reference) to O(1) without any
    dataset-specific config.
    """

    def __init__(self, dim: int, momentum: float = 0.01, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def _update(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> None:
        x_flat = x.reshape(-1, self.dim)
        if mask is not None:
            m = mask.reshape(-1).to(x_flat.dtype)
            denom = m.sum().clamp_min(1.0)
            if float(m.sum()) == 0.0:
                return  # nothing valid in this batch; leave stats untouched
            mean = (x_flat * m[:, None]).sum(dim=0) / denom
            var = ((x_flat - mean) ** 2 * m[:, None]).sum(dim=0) / denom
        else:
            mean = x_flat.mean(dim=0)
            var = x_flat.var(dim=0, unbiased=False)
        if not bool(self.initialized):
            self.running_mean.copy_(mean)
            self.running_var.copy_(var)
            self.initialized.fill_(True)
        else:
            self.running_mean.mul_(1.0 - self.momentum).add_(self.momentum * mean)
            self.running_var.mul_(1.0 - self.momentum).add_(self.momentum * var)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.training:
            self._update(x, mask)
        # Normalise with the (detached) running buffers → no batch coupling in the graph.
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)


class ScalarFourierEmbedding(nn.Module):
    """Embed a small scalar / coordinate vector ``(..., in_dim)`` into ``(..., out_dim)``.

    Pipeline: ``standardise -> concat([raw_passthrough, GaussianFourierFeatures]) -> MLP``.
    The raw pass-through guarantees a non-vanishing linear gradient path (dead-weight guard);
    the RFF supplies the high-frequency capacity.

    Parameters
    ----------
    standardize:
        ``"fixed"`` — subtract ``center`` and divide by ``scale`` (use when the input is
        already centred, e.g. an array-relative feature). ``"running"`` — use a
        :class:`RunningStandardizer` (use for an absolute scalar with unknown offset).
        ``"none"`` — assume the caller already standardised.
    center, scale:
        Per-feature affine constants for ``standardize="fixed"`` (broadcast scalars allowed).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        num_freqs: int = 16,
        sigma: float = 1.0,
        learnable_freqs: bool = False,
        standardize: str = "fixed",
        center: float = 0.0,
        scale: float = 1.0,
        hidden: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if standardize not in ("fixed", "running", "none"):
            raise ValueError(
                f"standardize must be 'fixed', 'running' or 'none', got '{standardize}'."
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.standardize = standardize

        if standardize == "fixed":
            if float(scale) == 0.0:
                raise ValueError("scale must be non-zero for standardize='fixed'.")
            self.register_buffer("center", torch.full((in_dim,), float(center)))
            self.register_buffer("scale", torch.full((in_dim,), float(scale)))
            self.standardizer = None
        elif standardize == "running":
            self.standardizer = RunningStandardizer(in_dim)
        else:  # "none"
            self.standardizer = None

        self.rff = GaussianFourierFeatures(
            in_dim, num_freqs=num_freqs, sigma=sigma, learnable=learnable_freqs, seed=seed
        )
        feat_dim = in_dim + self.rff.out_dim  # raw pass-through + RFF
        hidden = hidden or max(out_dim, 2 * feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def _standardize(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.standardize == "fixed":
            return (x - self.center) / self.scale
        if self.standardize == "running":
            return self.standardizer(x, mask=mask)
        return x

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        xs = self._standardize(x, mask=mask)
        feats = torch.cat([xs, self.rff(xs)], dim=-1)
        return self.mlp(feats)
