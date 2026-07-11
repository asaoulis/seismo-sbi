# Runbook — Brustle-lomax encoder sweep (cnn / tcn / pno)

**Goal:** one 500k baked dataset, three NPE trainings (cnn, tcn, pno) off the *same* sims,
60 epochs each, on the L40S — to compare per-station encoders under the up-to-date
brustle-lomax setup with the efficiency optimisations.

- **Config:** `first_ml_npe_brustle_lomax_sweep.yaml` (encoder-agnostic; `--arch` overrides
  `ml_architecture`, so all three share one config + one sim set).
- **Launcher:** `run_brustle_encoder_sweep.sh` (wraps the orchestrator stages).
- **Dataset run:** `santorini_brustle_lomax_sweep` · **train runs:** `…_cnn / …_tcn / …_pno`.

## Unified Nyquist temporal handling (all three encoders)
The `ml_encoder` block (`input_decimate: 3`, `downsample: 2`) is applied to **all three**
encoders:
- **Nyquist `input_decimate: 3`** runs at the model entry, before every encoder (201 → 67
  samples). LOSSLESS — the 0.03-0.08 Hz band (6 s min period) sits inside the post-decimation
  Nyquist (0.167 Hz); anti-alias FIR on by default. This is the Nyquist sampling logic applied
  to every model.
- **`downsample: 2`** then does one light strided reduction in each encoder → ~33-34
  tokens/station (cnn 34, tcn/pno 33). The cnn encoder gained a `downsample` param (single
  same-padded strided conv, `ceil(L/downsample)`, short-input-safe) so it shares this block —
  **not backward-compatible** with the old fixed cnn stack (accepted). Total 6x temporal
  reduction, comparable to the original brustle ds8 (~25 tokens).

Also fixed for this sweep: the **pno SpectralConv1d FFT** had no bf16 kernel and crashed under
AMP — it now runs its FFT in fp32 inside the autocast region (regression-tested). Verified: all
three encoders build + run a fwd/bwd training step on this config (cnn 8.27M / tcn 7.97M /
pno 8.02M params, finite log-prob). Other kept speed wins: the **500k baking** (dataloader off
the critical path), the **in-RAM sim+noise caches**, **bf16 AMP + fused SDPA**. `num_transforms`
stays 8 (quality-safe, identical across all three; set 5 for ~1.17x more speed).

## Prerequisites (do once, in order)
1. **Cluster torch env** `seismo-sbi-t25` (py3.11 + torch 2.5.1+cu121). If you edited any
   `.claude/cluster/remote/*` (we did — setup script + config), redeploy first:
   ```
   ! bash .claude/cluster/remote/bootstrap_install_train.sh
   ssh hypatia-train setup-train none          # uses mamba/libmamba → fast solve
   ```
2. **Commit + push** the branch (sync does `tsync` = checkout of a *pushed* rev; pushing is
   required or tsync fails `reference is not a tree`). Includes the baked staging code,
   `InputDecimator`, the MTfit removal, and these config/script files.
3. **Ensemble DBs**: `santorini_tomo_brustle` must be present on the cluster. ⚠ The archive was
   previously **32/61 members usable** (truncated PZ on the rest) — verify/rebuild before a 500k
   gen if full theory-error diversity matters, else the baked theory error draws from 32 models.

## Run order
```bash
cd <repo root>
SWEEP=scripts/configs/santorini/run_brustle_encoder_sweep.sh
bash $SWEEP avail                 # free GPU/CPU probe + recommended --gpu
bash $SWEEP sync                  # tsync cluster repo to your pushed HEAD
bash $SWEEP push-data             # noise catalogue / event / stations / components
bash $SWEEP gen                   # submit the 500k dataset gen (CORES64, 48h) — MULTI-HOUR
bash $SWEEP status                # poll until gen is done (sims present)
bash $SWEEP train                 # AFTER gen: fire cnn + tcn + pno (60 epochs, l40s)
# single encoder: bash $SWEEP train tcn
# preview only:  bash $SWEEP gen --dry-run   /   bash $SWEEP train --dry-run
```
Overridable via env: `EPOCHS`, `GPU`, `NSIMS`, `GEN_PARTITION`, `GEN_NCPU`, `GEN_WALL_H`.

## Notes
- The three trainings share the **one** sim set (orchestrator: `--run` = dataset, `--train-name`
  = checkpoint subdir), so gen runs once. They can run concurrently if GPUs are free, or
  sequentially; each lands on whichever L40S the scheduler picks (re-submit a stuck one with a
  fresh `--train-name` per the zombie-GPU-OOM note).
- 500k float32 sim-cache ≈ 19.3 GB RAM — the L40S node (compute-gpu-0-5, ~1.5 TB) fits it.
- After training: `… eval` (posttrain/dropout) + `… fetch` per encoder via the orchestrator.
- **Auto-fire after gen:** gen can't start until the env + sync + data prereqs are done (all
  user-gated), so there's nothing to auto-poll yet. Once gen is submitted, `status` shows
  completion; run `train` then (or ask the agent to watch `status` and fire `train` on done).
