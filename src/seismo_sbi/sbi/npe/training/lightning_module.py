"""The Lightning module that trains the NPE's conditional flow.

:class:`NPELightningModule` maximises the log-probability of the true parameters under the flow,
whose embedding net encodes the data, with a linear warm-up and a cosine, constant or cyclic
learning-rate schedule, and the opt-in MMD auxiliary loss. It loads checkpoints written with the
earlier parameter names.
"""

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR, CyclicLR

from seismo_sbi.sbi.npe.training.legacy_checkpoints import remap_legacy_state_dict


def fused_adam_supported(params):
    """True if every parameter is a real-valued tensor on a device fused AdamW accepts; the unfused optimiser gives the same update otherwise."""
    params = list(params)
    if not params:
        return False
    return (all(p.is_cuda for p in params)
            and not any(p.is_complex() for p in params))


class NPELightningModule(pl.LightningModule):
    """Trains a conditional normalising flow, whose embedding net encodes the data, by maximising
    the log-probability of the true parameters; the MMD auxiliary loss is opt-in.
    """
    def __init__(self, flow, lr=1e-3, weight_decay=0.0, lr_second_stage="cosine",
                 lr_min_factor=0.1,
                 fused_adam=False, compile_forward=False, compile_flow=False, **kwargs):
        """``flow`` is an nflows ``Flow`` whose embedding net takes the data. ``lr_second_stage`` is
        the schedule after the linear warmup (``cosine``, ``constant`` or ``cyclic``) and
        ``lr_min_factor`` the cosine floor as a fraction of ``lr``; ``fused_adam``,
        ``compile_forward`` and ``compile_flow`` are speed options.
        """
        super().__init__()
        self.flow = flow
        self.lr = lr
        self.weight_decay = weight_decay
        # After the linear warmup: "cosine" anneals to lr * lr_min_factor over the remaining
        # epochs, "constant" holds the base rate, "cyclic" is a step-based triangular cycle.
        self.lr_second_stage = lr_second_stage
        # Cosine floor as a fraction of the base LR: eta_min = lr * lr_min_factor. Raising it
        # to 0.2 keeps a usefully large step in the tail of a long run. Cosine branch only.
        self.lr_min_factor = float(lr_min_factor)
        self.cyclic_period_steps = 8000
        self._fused_adam = bool(fused_adam)
        self._log_prob_fn = None
        self._flow_tail_fn = None
        if compile_forward and hasattr(torch, "compile"):
            self._log_prob_fn = torch.compile(self._log_prob)
        elif compile_flow and hasattr(torch, "compile"):
            self._flow_tail_fn = torch.compile(self._flow_log_prob_from_embedded)
        # The MMD auxiliary loss is armed by enable_mmd; None means NLL only.
        self._mmd_cfg = None
        self._mmd_psim_loader = None
        self._mmd_psim_iter = None
        self._mmd_bandwidth_ema = None
        self._mmd_last_beta = float("nan")
        self._mmd_last_beta_ema = float("nan")
        self._mmd_last_z_scale = float("nan")

    def on_load_checkpoint(self, checkpoint):
        """Rename a checkpoint's earlier station-CNN parameter names before its weights are loaded."""
        checkpoint["state_dict"] = remap_legacy_state_dict(checkpoint["state_dict"])

    def enable_mmd(self, mmd_config: dict, real_context, psim_loader):
        """Arm the summary-space MMD auxiliary loss (two-sample, after Huang et al. 2023).

        ``real_context`` is the ``(N_real, W)`` context tensor of the QA-cleaned real events;
        ``psim_loader`` yields ``(theta, context)`` batches of the posterior-matched simulation suite
        with the training-time augmentation applied, one batch per MMD step. The loss becomes
        ``nll + lambda(t) * MMD^2_u(embed(real), embed(psim))`` with ``lambda`` ramped from 0 to
        ``lambda_mmd`` after ``warmup_epochs`` over ``ramp_epochs``; checkpoint selection stays on the
        NLL-only ``val_loss`` and the MMD is logged as ``train_mmd2`` / ``val_mmd2``.
        """
        from seismo_sbi.sbi.npe.training.mmd import DEFAULT_BANDWIDTH_SCALES
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
        """Next batch of contexts from the posterior-simulation loader, restarting it when
        exhausted, on the module's device and dtype.
        """
        if self._mmd_psim_iter is None:
            self._mmd_psim_iter = iter(self._mmd_psim_loader)
        try:
            _, ctx = next(self._mmd_psim_iter)
        except StopIteration:
            self._mmd_psim_iter = iter(self._mmd_psim_loader)
            _, ctx = next(self._mmd_psim_iter)
        return ctx.to(device=self.device, dtype=self.mmd_real_context.dtype)

    def _mmd_lambda(self):
        """Weight of the MMD loss this epoch: zero during warmup, then ramped linearly to
        ``lambda_mmd``.
        """
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
        from seismo_sbi.sbi.npe.training.mmd import median_bandwidth, rbf_mixture_mmd2_unbiased
        cfg = self._mmd_cfg
        n_real = self.mmd_real_context.shape[0]
        b = min(cfg["batch_size"], n_real)
        idx = torch.randperm(n_real, device=self.mmd_real_context.device)[:b]
        # Read the bottleneck when there is one, so the kernel lives in the narrow space and
        # not in its rank-limited flow-facing expansion.
        _emb = self.flow._embedding_net
        _summarise = getattr(_emb, "summary_bottleneck", _emb)
        z_real = _summarise(self.mmd_real_context[idx]).float()
        z_psim = _summarise(self._next_psim_context()).float()
        beta = median_bandwidth(z_real, z_psim)
        ema = cfg["bandwidth_ema"]
        self._mmd_bandwidth_ema = (beta if self._mmd_bandwidth_ema is None
                                   else ema * self._mmd_bandwidth_ema + (1 - ema) * beta)
        # An embedding contracting faster than the bandwidth EMA can follow drives every kernel
        # to 1 and the statistic to 0, so log the instantaneous bandwidth beside the lagged one.
        self._mmd_last_beta = beta
        self._mmd_last_beta_ema = self._mmd_bandwidth_ema
        with torch.no_grad():
            self._mmd_last_z_scale = float(
                torch.cat([z_real, z_psim], dim=0).pow(2).mean().sqrt())
        bandwidths = [self._mmd_bandwidth_ema * s for s in cfg["bandwidth_scales"]]
        return rbf_mixture_mmd2_unbiased(z_real, z_psim, bandwidths)

    def _log_prob(self, theta, x):
        """Flow log-probability of ``theta`` given the data ``x``."""
        return self.flow.log_prob(theta, context=x)

    def _flow_log_prob_from_embedded(self, theta, embedded):
        """Flow log-probability of ``theta`` given an already embedded context."""
        # nflows.Flow.log_prob with the embedding hoisted out, so only the launch-bound
        # transform stack + base density are compiled.
        noise, logabsdet = self.flow._transform(theta, context=embedded)
        return self.flow._distribution.log_prob(noise, context=embedded) + logabsdet

    def forward(self, x, theta):
        """Log-probability of ``theta`` given the data ``x``, shape ``(batch,)``."""
        # The flow contains the embedding_net; pass x as context to be embedded internally.
        if self._log_prob_fn is not None:
            return self._log_prob_fn(theta, x)
        if self._flow_tail_fn is not None:
            embedded = self.flow._embedding_net(x)
            return self._flow_tail_fn(theta, embedded)
        return self.flow.log_prob(theta, context=x)

    def training_step(self, batch, batch_idx):
        """Negative mean log-probability of a batch, plus the weighted MMD term when it is enabled."""
        theta, x = batch
        log_prob = self.forward(x, theta)
        loss = -log_prob.mean()
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
        """Negative mean log-probability of a validation batch; the MMD term is logged, not added."""
        theta, x = batch
        log_prob = self.forward(x, theta)
        val_loss = -log_prob.mean()
        # sync_dist averages across ranks so the monitored val_loss covers the whole split. It
        # stays likelihood-only even with MMD armed: checkpoints must track posterior quality.
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_log_prob", log_prob.mean(), sync_dist=True)
        if self._mmd_cfg is not None and batch_idx == 0:
            with torch.no_grad():
                self.log("val_mmd2", self._mmd_term(), sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        """AdamW with a linear warmup over 5 % of the epochs, then the ``lr_second_stage`` schedule."""
        # The fused optimizer's preconditions -- floating-point parameters on an accelerator --
        # are checked here because torch only rejects them inside the first step().
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

        # Cosine annealing down to lr_min_factor times the base learning rate.
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
