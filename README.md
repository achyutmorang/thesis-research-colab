# Thesis Research Colab

Private research repository for PRiSM Track B closed-loop simulation experiments.

## Open In Colab
- Notebook: [PRiSM_trackB_closedloop_simulation_colab.ipynb](https://colab.research.google.com/github/achyutmorang/thesis-research-colab/blob/main/PRiSM_trackB_closedloop_simulation_colab.ipynb)
- Note: because this repo is private, open the link while signed in to GitHub in Colab.

## What This Repo Contains
- `PRiSM_trackB_closedloop_simulation_colab.ipynb`: thin Colab orchestration notebook.
- `src/trackb/config.py`: dataclass configs and run artifact path helpers.
- `src/trackb/metrics.py`: risk and surprise metric primitives.
- `src/trackb/latentdriver.py`: planner integration, rollouts, predictive-KL utilities.
- `src/trackb/calibration.py`: preflight checks, calibration, surprise quality gates.
- `src/trackb/search.py`: optimization/search methods for closed-loop perturbations.
- `src/trackb/resume_io.py`: checkpoint resume and export/report artifact writing.
- `src/trackb/core.py`: top-level orchestration over split modules.

## Recommended Workflow
1. Open notebook in Colab.
2. Run the repo-sync cell (`git clone`/`git pull`) or use a Drive copy of this repo.
3. In a fresh runtime, set `RUN_SETUP=True` in the setup cell and run it once.
4. Restart runtime, set `RUN_SETUP=False`, then run all cells.
5. Keep `RUN_TAG`, `PERSIST_ROOT`, `N_SHARDS`, and `SHARD_ID` stable for resumable runs.

## Environment Reproducibility
- The notebook setup cell pins core dependencies for the known working stack (Waymax + JAX + LatentDriver-related packages).
- The setup cell also applies LatentDriver compatibility patches and fetches the expected checkpoint when missing.

## WOMD Data Access
- GCS auth is handled via `ensure_womd_gcs_access(...)` before dataset creation.
- If access is missing, the notebook triggers Colab auth and verifies bucket access for WOMD.

## Resume And Artifacts
- Intermediate and final outputs are written under `cfg.run_prefix`.
- Key artifacts include:
  - `*_per_scenario_results.csv`
  - `*_per_eval_trace.csv`
  - `*_closedloop_calibration.csv`
  - `*_thresholds.json`
  - `*_quick_summary.csv`
  - `*_runtime_manifest.json`

## Ownership And Reuse
- This repository is for thesis research and is not open source.
- Keep the repository private to minimize unauthorized reuse.
- If attribution is required in papers/slides, cite the repository URL and commit hash used for experiments.

## License
This project is distributed under a **proprietary, all-rights-reserved** license. See [LICENSE](LICENSE).
