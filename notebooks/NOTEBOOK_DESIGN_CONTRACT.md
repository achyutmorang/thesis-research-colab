# Colab Notebook Design Contract (Generic, Resumable, Persistent)

Use this as the standard for any Colab notebook, regardless of topic.

## Purpose
- Survive transient Colab runtimes.
- Resume work without manual reconstruction.
- Keep notebooks clean, fast, and deterministic.
- Separate orchestration (notebook) from logic (`src/`).
- Keep each notebook attached to one experiment pack (`experiments/<slug>/` + `configs/experiments/<slug>.json`).

## Applicability
- This contract is the **default** for long-running, artifact-heavy, resumable experiments.
- If an experiment design is short, exploratory, and does not require persistence/resume guarantees, strict compliance is optional.
- In those cases, use the "Lightweight profile" below and document the reason in the objective cell.

## Execution Profiles
### Full Profile (default)
- Use the required order, mandatory config fields, persistent storage contract, and resume contract exactly as written below.

### Lightweight Profile (allowed exception)
- Allowed only when all are true:
  - runtime is expected to be short and restart cost is low,
  - no critical checkpoint state is required,
  - outputs can be regenerated quickly.
- Minimum requirements in lightweight mode:
  - objective + hypothesis cell,
  - explicit config cell,
  - one fast-fail sanity cell,
  - one export cell that writes at least a compact summary artifact.
- Still recommended:
  - keep logic in `src/`,
  - keep cells idempotent where practical.

## Core Principles
1. Notebook cells orchestrate; modules implement.
2. Every step is idempotent.
3. Every long-running process is resumable.
4. Persistent storage is the source of truth.
5. Fail fast with lightweight checks before expensive runs.

## Notebook Structure (Required Order)
1. **Objective cell**:
   - What this notebook does.
   - Inputs, outputs, and success criteria.
2. **Bootstrap cell**:
   - Check package/runtime versions.
   - Install only missing/mismatched dependencies.
   - Restart runtime only when required.
3. **Storage cell**:
   - Mount Drive.
   - Verify required folders exist and are writable.
4. **Repo sync cell**:
   - Clone/pull repo.
   - Set Python path/import guards.
5. **Configuration cell**:
   - Single config object / dict.
   - No hidden defaults spread across cells.
6. **Run context cell**:
   - Resolve run name/tag.
   - Resolve resume mode.
   - Resolve shard/chunk allocation.
7. **Fast-fail validation cell**:
   - Small smoke run.
   - Data/IO checks.
   - Sanity constraints.
8. **Main execution cell**:
   - Training / simulation / processing loop.
   - Periodic checkpoint and progress flush.
9. **Evaluation/reporting cell**:
   - Metrics, summaries, diagnostics.
10. **Export cell**:
   - Persist final artifacts and manifests.
   - Print next-step actions.

## Mandatory Config Fields
```python
RUN_NAME = ""
RUN_PREFIX = "experiment"
PERSIST_ROOT = "/content/drive/MyDrive/project_runs/v1"

N_SHARDS = 1
SHARD_ID = 0
RESUME_FROM_EXISTING = True

RUN_ENABLED = True
```

## Persistent Storage Contract
```text
{PERSIST_ROOT}/
  {RUN_PREFIX}_{RUN_NAME}/
    config.json
    env_manifest.json
    run_manifest.json
    progress/
      shard_{i}.json
    checkpoints/
      latest.json
      step_{k}.ckpt
    outputs/
      metrics.csv
      artifacts/
```

## Resume Contract
### Checkpointing
- Save state atomically:
  - write temp file
  - flush + fsync
  - rename
- Update `latest.json` only after full checkpoint success.

### Progress Tracking
- Persist chunk/shard completion markers.
- On restart:
  - load progress
  - skip completed units
  - continue from first incomplete unit

## Manifest Contract
Each run must persist:
- `run_name`, `run_prefix`
- `created_utc`
- `git_commit`
- `config_hash`
- `python_version`
- package versions (core dependencies)
- runtime type (CPU/GPU)
- sharding metadata

## What Goes in Notebook vs `src/`
Move to `src/`:
- business logic
- training/inference loops
- metrics and diagnostics
- checkpoint + resume I/O
- export utilities
- dataset adapters/loaders

Keep in notebook:
- orchestration flow
- config values
- concise visual checks
- calls to reusable APIs

## Agent Implementation Rules
When an agent writes a Colab notebook:
1. Follow this contract exactly.
2. Keep notebook code minimal and readable.
3. Reuse existing `src/` functions when possible.
4. Add new reusable code to `src/`, not notebook cells.
5. Include restart-safe setup and resume-safe run logic.
6. Include clear status prints after each critical step.

## Agent Prompt Template
```text
Follow notebooks/NOTEBOOK_DESIGN_CONTRACT.md exactly.
Design a thin Colab notebook with restart-safe bootstrap, Drive-backed persistence, and resumable execution.
Keep logic in src/ modules and keep notebook cells orchestration-only.
Ensure idempotent cells, manifest writing, checkpoint resume, and shard/chunk progress resume.
Add fast-fail validation before full execution.
```

## Optional Enhancements
- Heartbeat file (`heartbeat.json`) every N minutes.
- Run lock file to prevent concurrent writers.
- Checkpoint retention policy.
- Auto-compaction/compression for large outputs.
