# Waymax Simulation Experiments

Experiment-first research repository for closed-loop planning, calibration, and evaluation on Waymax/WOMD.

## Experimental Status
This codebase is intentionally experimental.

- Methods, interfaces, and defaults can change as thesis experiments evolve.
- Notebooks are orchestration/reporting surfaces; core logic is in `src/` modules.
- Results should always be cited with commit hash + run artifacts, not notebook screenshots alone.

## Open In Colab
- Closed-loop simulation notebook: [notebooks/closedloop_simulation_colab.ipynb](https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/notebooks/closedloop_simulation_colab.ipynb)
- Closed-loop evaluation notebook: [notebooks/closedloop_evaluation_colab.ipynb](https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/notebooks/closedloop_evaluation_colab.ipynb)
- Compute-normalized discovery notebook: [notebooks/compute_normalized_blindspot_discovery_colab.ipynb](https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/notebooks/compute_normalized_blindspot_discovery_colab.ipynb)
- Counterfactual sensitivity notebook: [notebooks/counterfactual_risk_sensitivity_atlas_colab.ipynb](https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/notebooks/counterfactual_risk_sensitivity_atlas_colab.ipynb)

## Research Focus
- Closed-loop search under fixed compute budget.
- Calibration + quality gates before expensive runs.
- Surprise-signal usefulness diagnostics for ranking quality.
- Counterfactual and compute-normalized post-hoc evaluation.

## Repository Layout
- `notebooks/`: thin Colab notebooks for orchestration, diagnostics, and reporting.
- `src/closedloop/`: domain logic (planner integration, calibration, search, metrics, artifact IO).
- `src/workflows/`: notebook workflow orchestration (`closedloop_flow.py`).
- `src/platform/`: Colab/runtime bootstrap (repo sync, Drive checks, deterministic setup, hot reload).
- `scripts/`: setup and utility scripts.

## Recommended Workflow
1. Open `notebooks/closedloop_simulation_colab.ipynb` in Colab.
2. Run Step 1 bootstrap cell.
3. In Step 2, set user knobs (`RUN_TAG`, `RUN_MODE`, `PERSIST_ROOT`, sharding).
4. Run quick probe (Step 3) before full dataset build.
5. Continue preflight, calibration, gate, main loop, and export cells top-to-bottom.

## Run Management Semantics
Step 2 is auto-aware and produces an explicit run plan.

- `RUN_TAG`:
  - If empty, auto-generated as `<RUN_TAG_PREFIX>_YYYYMMDD_HHMMSS` (UTC).
- `RUN_MODE`:
  - `auto`: infer `fresh`/`resume` from existing shard artifacts.
  - `fresh`: force recomputation for the selected run prefix.
  - `resume`: continue from existing artifacts when present.
- `SHARD_ID="auto"`:
  - picks the least-complete shard first.
- Config drift warning:
  - in resume mode, Step 2 compares key config/search fields against prior carry-forward config and warns on mismatches.

## Contributor Access To Persisted Artifacts (Drive Shortcut)
Ask for edit access to shared folder `waymax_experiments` (owner: Achyut Morang), then add a shortcut:

1. Open Google Drive with the same account used in Colab.
2. Go to `Shared with me` and find `waymax_experiments`.
3. Right-click folder -> `Organize` -> `Add shortcut`.
4. Choose `My Drive` (or subfolder) and confirm.

Then use:

```python
PERSIST_ROOT = "/content/drive/MyDrive/waymax_experiments/closedloop_runs/v1"
```

## WOMD Access Prerequisite
Before dataset access in Colab, register your Google account on Waymo Open:

1. Visit [waymo.com/open/terms](https://waymo.com/open/terms).
2. Sign in with your target Colab Google account.
3. Accept terms and wait for access propagation.

Optional Colab sanity check:

```bash
!gsutil ls gs://waymo_open_dataset_motion_v_1_1_0/
```

## Reproducibility Notes
- Colab runtime dependencies are pinned in `requirements-colab.txt`.
- Deterministic bootstrap is handled by `scripts/colab_setup.py`.
- Runtime setup caches healthy states and only reinstalls when required.
- Exported artifacts include runtime/config manifests for provenance.

## Key Exported Artifacts
Given run prefix `<run_prefix>`:

- `<run_prefix>_per_scenario_results.csv`
- `<run_prefix>_per_eval_trace.csv`
- `<run_prefix>_closedloop_calibration.csv`
- `<run_prefix>_thresholds.json`
- `<run_prefix>_quick_summary.csv`
- `<run_prefix>_runtime_manifest.json`
- `<run_prefix>_carry_forward_config.json`
- `<run_prefix>_artifact_schema.json`

## Testing And CI
Run locally:

```bash
pip install -r requirements-dev.txt
pytest -q
```

CI (`.github/workflows/ci.yml`) validates syntax and tests on push/PR.

## Ownership And License
This repository is thesis research code and is not open source.

- Unauthorized reuse is not permitted.
- Cite repository URL and exact commit hash when referencing results.
- License: proprietary, all rights reserved (see `LICENSE`).
