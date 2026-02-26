# Thesis Research Colab

Research repository for closed-loop simulation experiments.

## Open In Colab
- Notebook: [notebooks/closedloop_simulation_colab.ipynb](https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/notebooks/closedloop_simulation_colab.ipynb)
- Evaluation notebook: [notebooks/closedloop_evaluation_colab.ipynb](https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/notebooks/closedloop_evaluation_colab.ipynb)
- Compute-normalized blindspot notebook: [notebooks/compute_normalized_blindspot_discovery_colab.ipynb](https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/notebooks/compute_normalized_blindspot_discovery_colab.ipynb)
- Counterfactual sensitivity notebook: [notebooks/counterfactual_risk_sensitivity_atlas_colab.ipynb](https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/notebooks/counterfactual_risk_sensitivity_atlas_colab.ipynb)
- If the repo is private, open the link while signed in to GitHub in Colab.

## Notebook Overview
- `notebooks/closedloop_simulation_colab.ipynb`: runs closed-loop Waymax simulation and exports per-scenario/per-trace artifacts.
- `notebooks/closedloop_evaluation_colab.ipynb`: loads simulation JSON/CSV outputs and computes core thesis evaluation metrics and plots.
- `notebooks/compute_normalized_blindspot_discovery_colab.ipynb`: evaluates compute-normalized blindspot discovery efficiency under multiple definitions/thresholds.
- `notebooks/counterfactual_risk_sensitivity_atlas_colab.ipynb`: builds a counterfactual factor-response atlas to measure risk sensitivity and ranking stability.

## Source Layout
- `src/closedloop/`: closed-loop domain logic (planner integration, calibration, search, metrics, artifact IO).
- `src/workflows/`: reusable notebook orchestration flows (`closedloop_flow.py`) for thin Colab notebooks.
- `src/platform/`: Colab/runtime lifecycle helpers (repo sync, Drive mount checks, deterministic setup, hot-reload prep).
- Backward-compatible shims are kept in `src/closedloop/notebook_flow.py` and `src/closedloop/colab_runtime.py`.

## Research Notice
Everything in this repository is experimental and part of ongoing Master's research work at IIT Hyderabad. Results, methods, and interfaces may change as the thesis evolves.

## Paper-Ready Evaluation Outputs
- Both new experiment notebooks emit:
  - explicit hypothesis verdict tables (`PASS` / `INCONCLUSIVE` / `FAIL`)
  - robustness sweeps over key threshold/definition settings
  - repo-inspired evaluation views:
    - STRIVE/FREA-style plausibility-filtered discovery metrics
    - SEAL-style realism-gap metrics (distribution-distance based)
    - VerifAI-style rulebook-based multi-objective ranking
  - plot artifacts (`.png`) and analysis tables (`.csv`, `.json`) under experiment-specific export folders in Drive
- Recommended for reporting:
  - include hypothesis verdict table in main paper body
  - include full definition/threshold robustness tables in appendix

## Recommended Workflow
1. Open notebook in Colab.
2. Run the single bootstrap setup cell (repo sync + Drive validation + deterministic runtime setup + import hot-reload).
3. Keep `AUTO_RESTART_AFTER_SETUP=True` so Colab restarts automatically if compiled dependencies changed.
4. After setup completes, run the remaining cells top-to-bottom.
5. Keep `RUN_TAG`, `PERSIST_ROOT`, and `N_SHARDS` stable for resumable runs.
6. Use `SHARD_ID="auto"` to pick the next shard automatically from existing progress files.

## Contributor Access To Logged Artifacts (Drive Shortcut)
If you want to contribute and inspect logged simulation artifacts, ask for access to the shared Drive folder `waymax_experiments` (owner: Achyut Morang), then add a shortcut in your own Drive:

1. Open Google Drive with the same Google account you use in Colab.
2. Go to `Shared with me` and open `waymax_experiments`.
3. Right-click the `waymax_experiments` folder.
4. Select `Organize` -> `Add shortcut`.
5. In the dialog, choose `My Drive` (or a subfolder inside it) and click `Add`.
6. Confirm the shortcut now appears under `My Drive` as `waymax_experiments`.
7. In Colab, mount Drive and keep:
   - `PERSIST_ROOT = "/content/drive/MyDrive/waymax_experiments/closedloop_runs/v1"`

Notes:
- This shortcut does not duplicate files; it points to the shared folder.
- To write/update artifacts, your shared-folder permission must include edit access.
- If the mount check in notebook Step 2 fails, verify the shortcut exists under `My Drive` for the active Colab account.

## Environment Reproducibility
- Core runtime dependencies are pinned in `requirements-colab.txt`.
- Notebook setup calls `scripts/colab_setup.py`, which uses the active kernel interpreter (`sys.executable -m pip`), probes the runtime first, attempts targeted numeric-stack repair for common NumPy mismatch states, skips heavy installs when possible, applies LatentDriver compatibility patches, fetches the expected checkpoint when missing, and validates core imports.
- CI/test dependencies are pinned in `requirements-dev.txt`.

## WOMD Data Access
- Before running WOMD/Waymax notebooks, register your Google account for Waymo Open Dataset access:
  1. Go to [https://waymo.com/open/terms](https://waymo.com/open/terms).
  2. Click **Access Waymo Open Dataset** (Google sign-in is required).
  3. Sign in with the same Gmail account you plan to use in Colab.
  4. Accept the Waymo Open Dataset non-commercial license terms.
  5. Wait a few minutes for access propagation, then open/restart your Colab session.
- In Colab, the notebook handles Google auth via `ensure_womd_gcs_access(...)` before dataset creation.
- If needed, manually authenticate in Colab with:

```python
from google.colab import auth
auth.authenticate_user()
```

- Optional sanity check in a notebook cell:

```bash
!gsutil ls gs://waymo_open_dataset_motion_v_1_1_0/
```

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
