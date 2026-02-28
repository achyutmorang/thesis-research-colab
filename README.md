# Waymax Simulation Experiments

Experiment-first repository for reproducible Waymax/WOMD research using thin Colab notebooks and reusable `src/` modules.

## What This Repo Optimizes For
- Fast replication of related papers.
- Fast prototyping of new ideas on Colab.
- Resumable runs with Drive-backed persistence.
- Minimal notebook logic; maximal reusable Python modules.

## Core Layout
- `experiments/`: experiment packs (`<slug>/README.md` + notebooks)
- `configs/experiments/`: per-pack runtime defaults
- `src/experiments/`: pack registry + scaffolding
- `src/workflows/`: notebook-facing orchestration APIs
- `src/closedloop/`, `src/eval*`, `src/risk_model/`: implementation modules
- `notebooks/templates/`: notebook starter template
- `notebooks/NOTEBOOK_DESIGN_CONTRACT.md`: notebook design standard

## Experiment Packs
### 1) closedloop-core
Unified pack for simulation and evaluation.

Notebooks:
- `experiments/closedloop-core/notebooks/closedloop_simulation_colab.ipynb`
- `experiments/closedloop-core/notebooks/closedloop_evaluation_colab.ipynb`
- `experiments/closedloop-core/notebooks/compute_normalized_blindspot_discovery_colab.ipynb`
- `experiments/closedloop-core/notebooks/counterfactual_risk_sensitivity_atlas_colab.ipynb`

Colab links:
- <https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/closedloop-core/notebooks/closedloop_simulation_colab.ipynb>
- <https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/closedloop-core/notebooks/closedloop_evaluation_colab.ipynb>
- <https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/closedloop-core/notebooks/compute_normalized_blindspot_discovery_colab.ipynb>
- <https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/closedloop-core/notebooks/counterfactual_risk_sensitivity_atlas_colab.ipynb>

### 2) surprise-potential
- Notebook: `experiments/surprise-potential/notebooks/surprise_potential_closedloop_colab.ipynb`
- Colab: <https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/surprise-potential/notebooks/surprise_potential_closedloop_colab.ipynb>

### 3) risk-uq-suite
Notebooks:
- `experiments/risk-uq-suite/notebooks/risk_model_training_colab.ipynb`
- `experiments/risk-uq-suite/notebooks/uq_benchmark_colab.ipynb`
- `experiments/risk-uq-suite/notebooks/paper_tables_figures_colab.ipynb`

Colab links:
- <https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/risk-uq-suite/notebooks/risk_model_training_colab.ipynb>
- <https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/risk-uq-suite/notebooks/uq_benchmark_colab.ipynb>
- <https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/risk-uq-suite/notebooks/paper_tables_figures_colab.ipynb>

## Programmatic Pack Discovery
```python
from src.workflows import list_experiment_packs, validate_registry

for pack in list_experiment_packs():
    print(pack.slug, pack.notebooks)

print(validate_registry('.'))
```

## Scaffold A New Pack
```bash
python scripts/new_experiment.py \
  --slug my-paper-replication \
  --title "My Paper Replication" \
  --objective "Replicate and extend XYZ under the notebook design contract."
```

Creates:
- `experiments/<slug>/README.md`
- `experiments/<slug>/notebooks/<slug>_colab.ipynb`
- `configs/experiments/<slug>.json`
- `src/workflows/<slug>_flow.py`
- `src/experiments/papers/<slug>/__init__.py`

## Runtime and Persistence
- Deterministic setup: `scripts/colab_setup.py`
- Recommended Drive root: `/content/drive/MyDrive/waymax_experiments/...`
- Resume is run-tag + shard aware. Keep `RUN_TAG` stable to continue a run.

## WOMD Access
- Register at <https://waymo.com/open/terms>
- Authenticate from the same Google account in Colab

## Local Tests
```bash
pip install -r requirements-dev.txt
PYTHONPATH=. pytest -q
```

## License
Proprietary thesis research code. All rights reserved. See `LICENSE`.
