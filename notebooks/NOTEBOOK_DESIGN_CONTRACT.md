# Colab Notebook Design Contract (Resumable + Persistent)

Use this as the standard for every new research notebook.

## Goals
- Thin notebook orchestration, heavy logic in `src/`.
- Fully resumable on transient Colab runtimes.
- Persistent artifacts/checkpoints in Google Drive.
- Deterministic and reproducible runs.
- Fast iteration with quick probe before expensive runs.

## Non-Negotiable Rules
1. Keep notebooks thin:
   - No large algorithm functions in `.ipynb`.
   - Put reusable logic in `src/...`.
2. Every run must be resumable:
   - Training resume from latest checkpoint.
   - Experiment resume from latest completed shard/chunk.
3. Every run must write a manifest:
   - git commit, config hash, package versions, runtime info.
4. Every step must be idempotent:
   - Rerunning a cell should not corrupt state.
5. Never trust score alone:
   - Run quick probe and preflight gates before expensive loops.

## Canonical Cell Order
1. Title + objective + expected outputs (markdown).
2. Deterministic setup + auto-restart if package mismatch.
3. Drive mount + required folder check.
4. Repo sync + import setup.
5. Config block (single source of truth).
6. Run context init (run tag, resume mode, shard selection).
7. Quick probe (metric sweep + gate checks).
8. Build full simulation/training context.
9. Main run loop (with periodic checkpoint flush).
10. Evaluation + diagnostics.
11. Export + summary + next actions.

## Required Config Block (Notebook)
```python
RUN_TAG = ""  # empty means auto-select or auto-resume logic
RUN_PREFIX = "closedloop_run"
PERSIST_ROOT = "/content/drive/MyDrive/waymax_experiments/closedloop_runs/v1"

N_SHARDS = 5
SHARD_ID = 0  # optional manual override; auto-select preferred

AUTO_RUN_MAIN_LOOP_WHEN_READY = True
RUN_MAIN_LOOP_OVERRIDE = None
```

## Persistent Storage Layout
Use:
```text
{PERSIST_ROOT}/
  {RUN_PREFIX}_{RUN_TAG}/
    carry_forward.json
    env_manifest.json
    config.json
    checkpoints/
      latest.json
      ckpt_step_{k}.pt
    shards/
      shard_{i}/
        progress.json
        results.csv
        trace.csv
    exports/
      summary.csv
      plots/
```

## Resume Protocol
### Training Resume
- Save model/optimizer/scheduler/scaler/RNG states.
- Save atomic checkpoint:
  - write to temp file
  - `fsync`
  - rename to final path
- Update `latest.json` only after checkpoint is complete.

### Experiment Resume
- Track completion per shard/chunk.
- On restart:
  - scan progress files
  - skip completed shards/chunks
  - continue from first incomplete unit

## Minimal Required Metadata
Every run writes:
- `run_tag`
- `git_commit`
- `created_utc`
- `cfg_hash`
- `python_version`
- `numpy/pandas/torch/jax versions`
- `colab_runtime_type` (cpu/gpu)
- `n_shards`, `shard_id`

## Probe-First Policy
Before main run, quick probe must report:
- finite surprise rows > 0
- nonzero surprise fraction >= threshold
- realized fraction >= threshold
- raw belief/policy shift checks pass
- raw-vs-floor fraction checks pass

If not, do not run full experiment.

## What Goes in `src/` vs Notebook
Move to `src/`:
- all scoring metrics
- calibration and gates
- perturbation generators
- checkpoint/resume IO
- export/report utilities
- model training loops

Keep in notebook:
- high-level orchestration
- config values
- display/report tables
- small glue code only

## Agent Prompt Template
Give this to any coding agent:

```text
Follow notebooks/NOTEBOOK_DESIGN_CONTRACT.md exactly.
Keep notebook thin and orchestration-only.
All reusable logic must go to src/.
Implement resumable training + resumable experiment execution.
Use PERSIST_ROOT on Google Drive for durable state.
Write carry_forward/config/env manifests.
Make quick probe + preflight gates mandatory before main loop.
Use idempotent cells and atomic checkpoint writes.
```

## Optional Enhancements
- Add heartbeat writes every N minutes (`heartbeat.json`).
- Add run lock file to prevent concurrent writes.
- Add retention policy for old checkpoints.
- Add artifact upload compression for large traces.
