# Thesis Research Colab

Research repository for closed-loop simulation experiments.

## Open In Colab
- Notebook: [notebooks/closedloop_simulation_colab.ipynb](https://colab.research.google.com/github/achyutmorang/thesis-research-colab/blob/main/notebooks/closedloop_simulation_colab.ipynb)
- Evaluation notebook: [notebooks/closedloop_evaluation_colab.ipynb](https://colab.research.google.com/github/achyutmorang/thesis-research-colab/blob/main/notebooks/closedloop_evaluation_colab.ipynb)
- If the repo is private, open the link while signed in to GitHub in Colab.

## What This Repo Contains
- `notebooks/closedloop_simulation_colab.ipynb`: thin Colab orchestration notebook.
- `scripts/colab_setup.py`: deterministic Colab bootstrap (dependency install + compatibility patches + checkpoint fetch).
- `scripts/merge_shards.py`: validates and merges shard outputs into consolidated artifacts.
- `src/closedloop/config.py`: dataclass configs and run artifact path helpers.
- `src/closedloop/metrics.py`: risk and surprise metric primitives.
- `src/closedloop/latentdriver.py`: planner integration, rollouts, predictive-KL utilities.
- `src/closedloop/calibration.py`: preflight checks, calibration, surprise quality gates.
- `src/closedloop/search.py`: optimization/search methods for closed-loop perturbations.
- `src/closedloop/resume_io.py`: checkpoint resume and export/report artifact writing.
- `src/closedloop/core.py`: top-level orchestration over split modules.
- `src/eval/io.py`: run-prefix discovery and artifact loading from Drive/local outputs.
- `src/eval/analysis.py`: reusable evaluation metrics/tables for conditional lift and discovery efficiency.
- `notebooks/closedloop_evaluation_colab.ipynb`: analysis notebook consuming simulation outputs (`csv` + `json`).
- `requirements-colab.txt`: pinned Colab runtime dependency lock.
- `requirements-dev.txt`: local/CI test dependency lock.
- `tests/`: unit tests for deterministic metric and sharding logic.

## Recommended Workflow
1. Open notebook in Colab.
2. Run the repo-sync cell (`git clone`/`git pull`) or use a Drive copy of this repo.
3. In a fresh runtime, set `RUN_SETUP=True` in the setup cell and run it once.
4. Restart runtime, set `RUN_SETUP=False`, then run all cells.
5. If setup is skipped, the helper now performs a runtime probe and raises early when NumPy/runtime state is stale.
5. Keep `RUN_TAG`, `PERSIST_ROOT`, and `N_SHARDS` stable for resumable runs.
6. Use `SHARD_ID="auto"` to pick the next shard automatically from existing progress files.

## Environment Reproducibility
- Core runtime dependencies are pinned in `requirements-colab.txt`.
- Notebook setup calls `scripts/colab_setup.py`, which installs from the lockfile, applies LatentDriver compatibility patches, fetches the expected checkpoint when missing, and validates NumPy runtime symbols required by the stack.
- CI/test dependencies are pinned in `requirements-dev.txt`.

## WOMD Data Access
- GCS auth is handled via `ensure_womd_gcs_access(...)` before dataset creation.
- If access is missing, the notebook triggers Colab auth and verifies bucket access for WOMD.

## Resume And Artifacts
- Intermediate and final outputs are written under `cfg.run_prefix`.
- Artifact schema is versioned (`artifact_schema.json`) to prevent incompatible resume across schema changes.
- Key artifacts include:
  - `*_per_scenario_results.csv`
  - `*_per_eval_trace.csv`
  - `*_closedloop_calibration.csv`
  - `*_thresholds.json`
  - `*_quick_summary.csv`
  - `*_runtime_manifest.json`
  - `*_artifact_schema.json`

## Shard Merge Workflow
After all shards finish, merge them with:

```bash
python scripts/merge_shards.py \
  --run-tag closedloop_v1 \
  --persist-root /content/drive/MyDrive/closedloop_runs \
  --n-shards 4
```

This writes merged CSVs and a merge manifest under the same `persist_root`.

## Testing And CI
- Run unit tests locally:

```bash
pip install -r requirements-dev.txt
pytest -q
```

- GitHub Actions workflow (`.github/workflows/ci.yml`) runs syntax checks and tests on push/PR.

## Ownership And Reuse
- This repository is for thesis research and is not open source.
- Keep the repository private to minimize unauthorized reuse.
- If attribution is required in papers/slides, cite the repository URL and commit hash used for experiments.

## License
This project is distributed under a **proprietary, all-rights-reserved** license. See [LICENSE](LICENSE).
