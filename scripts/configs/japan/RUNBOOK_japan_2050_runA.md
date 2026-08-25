# Runbook — Japan 20–50 s retrain, RUN A (band + per-trace amplitude nuisance + distance coda)

**Hand-off for an agent. Written 2026-08-25 after `japan-paper-prep/nondc-bias-forensics` (N1–N18).**
Read `.claude/runs/japan-paper-prep/tilt-origin-prestep/artifacts/PREP_next_run.md` and
`.claude/runs/japan-paper-prep/nondc-bias-forensics/artifacts/TECH_attenuation_tests.md` §7–§8 for the
evidence; this file is the *procedure*. Track the work as a `/tracked-task japan-paper-prep run-a-2050`
(heartbeat every step; checkpoint before every cluster action).

## 0. What run A is (and is not)

| setting | previous (`japan_10s50s_tcn`, b10) | run A | why |
|---|---|---|---|
| band | 10–50 s (0.02–0.1 Hz) | **20–50 s (0.02–0.05 Hz)**, `input_decimate 4` | N16c: the tilt term's lever 0.28/0.46 → 0.08/0.05 (moderate γ/δ); N1/N2; F-net's own band |
| `amplitude_error` | per-station probability gate (0.2) × U(0.7, 1.4), all components alike | **always-on, per-trace (station × component) log-normal σ = 0.30 dex** | N14 §4 measured 0.31–0.35 dex + inter-component σ 0.16/0.30 vs trained 0; N17a: this structure reproduces 0.37 of the ISO field, half the width excess and DC-evidence loss; per-station structure reproduces 0 |
| `scattering_coda` | stähler, gate 0.4, α U(0.2, 0.6), distance-blind | **`distance_mode: true`, causal, α 0.1 + 0.9/1000 km, +0.25 dex/1000 km, cap 1000 km** | N9/N10: measured incoherent Rayleigh-window excess the ensemble cannot produce; reproduces the width |
| everything else | | **UNCHANGED** (21 stations, no distance cap, per-station member draws, time shift 3 + 4 s, source_location_error, dropouts, STF, GR prior, japan10s DBs) | parsimony: dropped/deferred/future items are listed in TECH §7 and are NOT to be added |

Config: `scripts/configs/japan/first_ml_npe_japan_2050.yaml` (already written; parses through
`SBI_Configuration`; simulation chain = TimeShift + ScatteringCoda(distance_mode), augmentation chain =
AmplitudeError(lognormal, per_component, always_on) + InstrumentDropout). The new `AmplitudeErrorEffect`
options are **off by default and RNG-identical on the legacy path** (`tests/unit/test_nuisance_recalibration.py`).
Run B (later, if needed) = the same YAML with `syngine_address`/`syngine_fiducial_address` → the rebuilt
`japan_patch10s` DBs; it is **not** part of this runbook.

## 1. Prerequisite: commit + push the src bundle (USER-gated: ask before pushing)

Uncommitted in the working tree and required on the cluster: `src/seismo_sbi/instaseis_simulator/
post_processing.py` (AmplitudeErrorEffect options, TimeShiftErrorEffect distance keys — inert,
DispersionSpreadEffect — unused, ScatteringCodaEffect distance_mode from session 3), `ensemble.py`,
`multi_model.py`, `sbi/simulator_wrapper.py`, `sbi/configuration.py`, `sbi/types/parameters.py`,
`tests/unit/test_nuisance_recalibration.py`, `tests/unit/test_scattering_distance_mode.py`,
`scripts/axisem/build_japan_patched_ensemble.py`, `scripts/axisem/depthdep_perturb.py`, the two
`scripts/axisem/ensemble_config_japan_patch*.yaml`, this runbook. Gate first:
```bash
conda run -n seismo-sbi python -m pytest tests/unit tests/integration -x -q     # 1812 passed on 2026-08-25
```
Then `git add` the files above, commit on `public-lib-refactor`, and **`git push origin public-lib-refactor`**
(needs the ssh-agent — memory `github-push-ssh-agent`). Record `git rev-parse HEAD`: the cluster sync
must use that SHA (memory `commit-push-before-tsync`: a branch name resolves to the cluster's stale
local branch).

## 2. Rebuild the noise pool at 20–50 s (LOCAL, no download needed)

The band is baked into the daily cache at build time, so the pool must be rebuilt — but the continuous
2 Hz raw survived this time: `/data/alex/fnet_japan/raw_continuous_10s/` (21 station dirs, 63 verified
days in `raw_continuous_10s_ledger.json`, 5.5 GB). **Never write into an existing `_daily_*` or
`catalogue*` directory** — every band lives in its own namespace.

Reuse the b10 driver with the band changed:
```bash
cp .claude/runs/japan-paper-prep/band10s-retrain/artifacts/build_noise_pool.py \
   .claude/runs/japan-paper-prep/run-a-2050/artifacts/build_noise_pool_2050.py
# edit the copy: FILTER_JSON -> '{"freqmin":0.02,"freqmax":0.05,"corners":4,"zerophase":false}'
#                 --out default -> /data/alex/fnet_japan/catalogue_2050s
#                 --processed-dir default -> /data/alex/fnet_japan/_daily_2050s   (NEW dir)
#                 add "_daily_10s50s" and "_daily_events_10s50s" to FORBIDDEN_CACHES
# keep: --duration 800 --sampling_rate 1.0 --rolling_window_gap 100 --channel_glob 'BH?' --no_events,
#       --catalogue /data/alex/fnet_japan/events_10s50s_avoid.xml (event avoidance, same date range),
#       --stations_file scripts/configs/japan/fnet_demo_stations.txt, the QA gates and the degenerate-window purge
setsid nohup conda run -n seismo-sbi --no-capture-output python -u \
   .claude/runs/japan-paper-prep/run-a-2050/artifacts/build_noise_pool_2050.py --n-jobs 8 \
   > /data/alex/fnet_japan/catalogue_2050s_build.nohup 2>&1 &
```
It calls `scripts/build_catalogue.py` once per day (`--noise_start/--noise_end`), ~765 windows/day →
**~48k windows from 63 days** (b10 got 48,536 from the same days). Liveness: `ps`, not `pgrep -f`
(memory: pgrep matched its own shell). Gate before moving on (write `s1_noise_gate.md`):
* count ≈ 48k; all 21 stations present in > 99 % of windows, 3 components each;
* per-station median spectra show the 20 s corner (compare with `catalogue_10s50s/noise`);
* no degenerate windows (the purge in the driver), `min_completeness 0.9`, `max_flat_fraction 0.05`.

## 3. Rebuild the 655 event windows at 20–50 s (LOCAL, ~1 h)

Same recipe as the 10–50 s rebuild (`band10s-retrain/artifacts/s2_events_gate.md`), new namespaces.
`--data_dir /data/alex/fnet_japan/raw` is the recovered 2 Hz event-window archive (21 station dirs ×
485 day-dirs `YYYY.JJJ`, 2.2 GB) from which `_daily_events_10s50s` (47 GB) was derived — verify the
day-dir layout and that the 655 event dates are covered before launching.
```bash
conda run -n seismo-sbi --no-capture-output python -u scripts/build_catalogue.py \
   --catalogue /data/alex/fnet_japan/events_655_10s50s.xml \
   --data_dir /data/alex/fnet_japan/raw \
   --processed_dir /data/alex/fnet_japan/_daily_events_2050s \
   --stations_file scripts/configs/japan/fnet_demo_stations.txt \
   --output_dir /data/alex/fnet_japan/catalogue_2050s \
   --no_noise --duration 800 --sampling_rate 1.0 --pre_event_window 60 --channel_glob 'BH?' \
   --filter '{"freqmin":0.02,"freqmax":0.05,"corners":4,"zerophase":false}' --n_jobs 8
```
Gate: **655/655 h5 written**, ids = `origin_time.strftime('%Y%m%dT%H%M%S')` (they key the evaluation),
the 15–50 s and 10–50 s event dirs untouched. Update the YAML's `fnet_ev0` to an event that exists in
the new dir (it points at `catalogue_2050s/events/20250103T042045.h5` — check).

## 4. Cluster: sync, push, gen (checkpoint before EACH step)

The orchestrator is `.claude/cluster/run_remote_training.py` with `.claude/cluster/training_cluster_config.yaml`;
the `path_remap` line for `/data/alex/fnet_japan/catalogue_2050s` → `/share/gpu5/asaoulis/seismo_data/japan_fnet/catalogue_2050s`
is already present (added 2026-08-25; mirrors the 10s50s line — the dir-name flip must be reconciled
per path component). Always `--dry-run` first.

1. **Disk (USER-gated).** A 500k float64 dataset is 226 GB (451,280 B per sim). Check `/share/gpu5`
   free space via `status`/`view`; if short, the ONLY remedy is the user deleting an old `sims/` tree
   (the gatekeeper cannot, by design — memory `noise-pool-band-locked-and-gpu5-disk`). Candidates the
   user may choose: none named here — ask.
2. **Sync** the science repo to the pushed SHA:
   `python .claude/cluster/run_remote_training.py --config scripts/configs/japan/first_ml_npe_japan_2050.yaml sync --rev <SHA>`
3. **Push the events + config** (files are pushed by `push-data`; the noise pool is NOT — `view` is
   confined to RESULTS_ROOT, so the directory diff fails):
   `python .claude/cluster/run_remote_training.py --config scripts/configs/japan/first_ml_npe_japan_2050.yaml push-data`
4. **Push the noise pool in chunks** via the file-server route, with the b10 script adapted
   (`band10s-retrain/artifacts/push_noise_pool.sh`: set `SRC=/data/alex/fnet_japan/catalogue_2050s`,
   `RELDIR=japan_fnet/catalogue_2050s/noise`, a new ledger/log). Gzip is mandatory (`recv-data` does
   `tar xzf`); ~4000 windows per chunk; resumable via the ledger. Run detached (`setsid nohup … &`),
   never inside a harness background task (they are reaped at ~8 min). Gate: remote count + size match
   (`view`), 13 chunks for ~48k windows.
5. **Gen** (CPU, CORES64, full node memory for the per-station ensemble cache):
   `python .claude/cluster/run_remote_training.py --config scripts/configs/japan/first_ml_npe_japan_2050.yaml --run japan_2050_500k gen --partition CORES64 --ncpu 24 --wall_h 24 --nsims 500000`
   — b10 used `--ncpu 24` (128 G cgroup workaround unless `submit_gen.sh --mem=0` has been redeployed
   by the user via `bootstrap_install_train.sh`; check `remote/submit_gen.sh` on the cluster with `view`
   before choosing 24 vs 64). ~4 h at 34 it/s.
   **Submits are NOT idempotent** (memory `gatekeeper-submit-not-idempotent`): run the submit fully
   detached with no `timeout`, then re-verify the queue with `status`/`train-monitor` **after** it
   returns and again 30–40 min later (the pre-sbatch `find` can take that long on gpu5). A `gatekeeper:
   bad …` deny is the only failure that is safe to resubmit immediately. Never retry a 255/timeout
   blind. Gen restarts from scratch on rerun (no skip of existing files) — cancel a disk-full gen rather
   than letting it churn.
6. **Gate the dataset**: `status` shows the gen job gone from the queue and ~500k `random_event_*.h5`
   under `<RESULTS_ROOT>/japan_2050_500k/sims`; the skip guard (20 %) must not have fired. Pull ~20 sims
   (`send`) and check locally: band-limited to 0.02–0.05 Hz, 21 stations × 3 comps × 801 samples,
   the coda tail energy grows with distance (distance_mode is on), amplitudes are NOT per-trace scaled
   (amplitude_error is training_augmentation — applied in the dataloader, not baked).

## 5. Train (GPU) — same recipe as b10, new names

```bash
python .claude/cluster/run_remote_training.py --config scripts/configs/japan/first_ml_npe_japan_2050.yaml avail
python .claude/cluster/run_remote_training.py --config scripts/configs/japan/first_ml_npe_japan_2050.yaml \
    --run japan_2050_500k train --train-name japan_2050_tcn --arch tcn --epochs 250 --ngpu 4
```
b10 notes that carry over: global batch 128 (per-GPU 32 at NGPU = 4); the local smoke gate runs first;
`avail` decides the GPU type (RAM-gated); L40S zombie-GPU OOM → resubmit under a distinct train-name;
a wall-killed run is resumable (`ml_warm_start`, memory `japan-npe-warm-start-r2`) — resume, don't
restart. The 20–50 s noise pool stations must cover `stations_path` (they do: same 21).

## 6. Post-train (LOCAL after `fetch`)

1. `fetch` → `ml-checkpoints/japan_2050_tcn/`; **strict scaler gate + negative control first**
   (`band10s-retrain/artifacts/s6_scaler_gate.py`; memory `qa-resample-scaler-trap`: inference builds the
   θ-scaler from the CONFIG and only warns on mismatch).
2. Catalogue inference on the 655 events at 20–50 s (`band10s-retrain/artifacts/j3_post_qa_sequence_10s50s.sh`
   adapted: new event dir, new checkpoint, new output namespace). Same QA (tail_p90 ≤ 0.17 + contamination flag).
3. **Acceptance (the agreed rule — never one without the other):** moderate against-DC fraction ↓ vs
   b10 AND r2 on the same 526-event paired set (`nondc-bias-forensics/artifacts/n6_causal.py` machinery)
   **and** TARP / calibration honest; then re-run the injection suite on the new checkpoint
   (N6 floor, N10 coda, N11 dispersion, N14 C/D, N16 b0, N17a scatter — ~10 min per arm) to confirm the
   width is now reproduced and the per-trace scatter response is gone. Re-measure the Rayleigh tilt on
   the 20–50 s stores (r2 carried −0.146 dex inside 20–50 s; expect it reduced, not removed).
4. Report vs F-net **and** GCMT/JMA and the `fnet_emul` LSQ (TECH §7 row 15).

## 7. Do NOT do in this run
Distance cap; `dispersion_spread`; distance-scaled time-shift σ; sector member sampling; Qμ(z) or the
Vs-patched DBs (that is run B); site-gain table; any change to `source_location_error`, dropouts, STF,
prior. If run A's acceptance fails, the deferred list in TECH §7 is the menu — bring it back to the user,
do not add terms unasked.
