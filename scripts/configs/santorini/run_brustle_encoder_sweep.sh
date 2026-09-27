#!/bin/bash
# ============================================================================
# Brustle-lomax ENCODER SWEEP launcher — ONE 500k baked dataset, THREE encoders.
#
# Drives the remote-training orchestrator (.claude/cluster/run_remote_training.py) to:
#   1. generate ONE 500k baked dataset (run = santorini_brustle_lomax_sweep), then
#   2. train cnn / tcn / pno off that SAME dataset (distinct --train-name each), 60 epochs.
#
# Config: scripts/configs/santorini/first_ml_npe_brustle_lomax_sweep.yaml (encoder-agnostic;
# --arch overrides ml_architecture, so all three share one config + one sim set).
#
# Run from the repo root.  STAGES (run in order; gen is a multi-hour SLURM CPU job — wait for
# it to finish before `train`):
#   bash scripts/configs/santorini/run_brustle_encoder_sweep.sh avail       # free GPU/CPU probe
#   bash scripts/configs/santorini/run_brustle_encoder_sweep.sh sync        # tsync cluster to HEAD
#   bash scripts/configs/santorini/run_brustle_encoder_sweep.sh push-data   # noise/events/configs
#   bash scripts/configs/santorini/run_brustle_encoder_sweep.sh gen         # submit 500k gen
#   bash scripts/configs/santorini/run_brustle_encoder_sweep.sh status      # poll gen/train
#   bash scripts/configs/santorini/run_brustle_encoder_sweep.sh train       # FIRE all 3 (after gen)
#   bash scripts/configs/santorini/run_brustle_encoder_sweep.sh train tcn   # or one encoder
#
# PREREQUISITES (see the companion runbook RUNBOOK_brustle_encoder_sweep.md):
#   * cluster env `seismo-sbi-t25` built (ssh hypatia-train setup-train none) — torch 2.5.1+cu121
#   * branch committed + pushed (sync does tsync only; it checks out a pushed rev)
#   * santorini_tomo_brustle ensemble DBs present on the cluster (NOTE: archive was 32/61 usable —
#     verify/rebuild before a 500k gen if full theory-error diversity matters)
# Add --dry-run after the stage to preview the orchestrator call without executing.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ORCH="$REPO_ROOT/.claude/cluster/run_remote_training.py"
CFG="$REPO_ROOT/scripts/configs/santorini/first_ml_npe_brustle_lomax_sweep.yaml"
RUN="santorini_brustle_lomax_sweep"     # the shared dataset run (sims live here)
ARCHES=(cnn tcn pno)
EPOCHS="${EPOCHS:-60}"
GPU="${GPU:-l40s}"
NSIMS="${NSIMS:-500000}"
GEN_PARTITION="${GEN_PARTITION:-CORES64}"
GEN_NCPU="${GEN_NCPU:-64}"
GEN_WALL_H="${GEN_WALL_H:-48}"

stage="${1:-help}"; shift || true
EXTRA=("$@")   # passthrough (e.g. --dry-run), or a single arch for `train <arch>`

orch() { echo "+ python $ORCH --config <cfg> $*"; python "$ORCH" --config "$CFG" "$@"; }

case "$stage" in
  avail)      orch avail "${EXTRA[@]}" ;;
  sync)       orch sync "${EXTRA[@]}" ;;
  push-data)  orch push-data "${EXTRA[@]}" ;;
  status)     orch --run "$RUN" status "${EXTRA[@]}" ;;
  gen)
    orch --run "$RUN" gen --partition "$GEN_PARTITION" --ncpu "$GEN_NCPU" \
         --wall_h "$GEN_WALL_H" --nsims "$NSIMS" "${EXTRA[@]}"
    ;;
  train)
    # Optional single arch: `train tcn [--dry-run]`. Otherwise fire all three.
    sel=("${ARCHES[@]}"); passthrough=()
    if [ "${#EXTRA[@]}" -gt 0 ]; then
      case "${EXTRA[0]}" in
        cnn|tcn|pno) sel=("${EXTRA[0]}"); passthrough=("${EXTRA[@]:1}") ;;
        *)           passthrough=("${EXTRA[@]}") ;;
      esac
    fi
    for A in "${sel[@]}"; do
      echo "=== train-submit: arch=$A  train-name=${RUN}_$A  (dataset run $RUN) ==="
      orch --run "$RUN" train --train-name "${RUN}_$A" --arch "$A" \
           --epochs "$EPOCHS" --gpu "$GPU" "${passthrough[@]}"
    done
    ;;
  *)
    echo "stages: avail | sync | push-data | gen | status | train [cnn|tcn|pno] [--dry-run]"
    echo "  gen submits the 500k dataset; train (after gen completes) fires ${ARCHES[*]} @ ${EPOCHS} epochs on $GPU."
    exit 1 ;;
esac
