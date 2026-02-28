# Waymax Simulation Experiments

Experiment-first repository for reproducible Waymax/WOMD research on closed-loop simulation, surprise diagnostics, and risk/UQ evaluation.

## Why This Repo Exists
- Replicate paper methods quickly on Colab.
- Test new paper ideas with minimal notebook boilerplate.
- Keep runs resumable despite transient GPU runtimes.
- Keep notebooks thin and move implementation into reusable `src/` modules.

## Design Contract
- Notebook contract: `notebooks/NOTEBOOK_DESIGN_CONTRACT.md`
- One experiment pack per research track: `experiments/<slug>/`
- Pack config defaults: `configs/experiments/<slug>.json`
- Reusable orchestration APIs: `src/workflows/`
- Reusable core methods: `src/closedloop/`, `src/risk_model/`, `src/eval*/`

## Repository Layout
- `experiments/`
  - `closedloop-simulation/`
  - `surprise-potential/`
  - `closedloop-evaluation/`
  - `risk-uq-suite/`
- `configs/experiments/`: versioned runtime defaults per pack.
- `src/experiments/`: experiment-pack registry and scaffolding utilities.
- `src/workflows/`: notebook-facing orchestration entrypoints.
- `src/closedloop/`: planner integration, calibration, search, metrics, resume/export.
- `src/risk_model/`: risk model training, calibration, inference, artifacts.
- `src/eval*`: post-hoc analysis tracks.
- `notebooks/templates/`: starter template for new notebook orchestration surfaces.
- `scripts/new_experiment.py`: scaffold a new pack.

## Experiment Packs
### 1) Closed-Loop Simulation
- Notebook: `experiments/closedloop-simulation/notebooks/closedloop_simulation_colab.ipynb`
- Colab: <https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/closedloop-simulation/notebooks/closedloop_simulation_colab.ipynb>
- Goal: run calibration-gated, resumable closed-loop search.

### 2) Surprise Potential
- Notebook: `experiments/surprise-potential/notebooks/surprise_potential_closedloop_colab.ipynb`
- Colab: <https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/surprise-potential/notebooks/surprise_potential_closedloop_colab.ipynb>
- Goal: compare surprise definitions and feasibility before full-budget runs.

### 3) Closed-Loop Evaluation
- Notebooks:
  - `experiments/closedloop-evaluation/notebooks/closedloop_evaluation_colab.ipynb`
  - `experiments/closedloop-evaluation/notebooks/compute_normalized_blindspot_discovery_colab.ipynb`
  - `experiments/closedloop-evaluation/notebooks/counterfactual_risk_sensitivity_atlas_colab.ipynb`
- Colab:
  - <https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/closedloop-evaluation/notebooks/closedloop_evaluation_colab.ipynb>
  - <https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/closedloop-evaluation/notebooks/compute_normalized_blindspot_discovery_colab.ipynb>
  - <https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/closedloop-evaluation/notebooks/counterfactual_risk_sensitivity_atlas_colab.ipynb>
- Goal: analyze artifacts for compute-normalized and counterfactual findings.

### 4) Risk-UQ Suite
- Notebooks:
  - `experiments/risk-uq-suite/notebooks/risk_model_training_colab.ipynb`
  - `experiments/risk-uq-suite/notebooks/uq_benchmark_colab.ipynb`
  - `experiments/risk-uq-suite/notebooks/paper_tables_figures_colab.ipynb`
- Colab:
  - <https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/risk-uq-suite/notebooks/risk_model_training_colab.ipynb>
  - <https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/risk-uq-suite/notebooks/uq_benchmark_colab.ipynb>
  - <https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/risk-uq-suite/notebooks/paper_tables_figures_colab.ipynb>
- Goal: train calibrated risk models and benchmark robustness/calibration.

## Pack Discovery And Validation
```python
from src.workflows import list_experiment_packs, validate_registry

for pack in list_experiment_packs():
    print(pack.slug, pack.notebooks)

print(validate_registry('.'))
```

## Create A New Paper Pack
```bash
python scripts/new_experiment.py \
  --slug social-lstm-replication \
  --title "Social LSTM Replication" \
  --objective "Replicate and extend Social LSTM style interaction modeling on WOMD."
```

Scaffold output:
- `experiments/<slug>/README.md`
- `experiments/<slug>/notebooks/<slug>_colab.ipynb`
- `configs/experiments/<slug>.json`
- `src/workflows/<slug>_flow.py`
- `src/experiments/papers/<slug>/__init__.py`

## Runtime, Persistence, Resume
- Deterministic setup: `scripts/colab_setup.py`
- Recommended persistent root (Colab Drive):
  - `/content/drive/MyDrive/waymax_experiments/...`
- Run manifests/checkpoints are written by workflow/export modules.
- Resume behavior is run-tag and shard aware; keep `RUN_TAG` stable to resume the same run.

## WOMD Access Prerequisite
- Register your Google account at <https://waymo.com/open/terms>.
- Use the same account in Colab runtime authentication.

## Local Testing
```bash
pip install -r requirements-dev.txt
PYTHONPATH=. pytest -q
```

## License
Proprietary thesis research code. All rights reserved. See `LICENSE`.
