# Waymax Simulation Experiments

Experiment-first repository for reproducible Waymax/WOMD research with thin Colab notebooks and reusable `src/` modules.

## Repository Goals
- Reproduce related papers quickly.
- Prototype new ideas with minimal notebook boilerplate.
- Keep runs resumable on transient Colab runtimes.
- Keep implementation in versioned Python modules, not notebook cells.

## Experiment Index
| Pack | Notebook | GitHub | Open in Colab | Main Research Question | Objective |
|---|---|---|---|---|---|
| `closedloop-core` | `closedloop_simulation_colab.ipynb` | [GitHub](experiments/closedloop-core/notebooks/closedloop_simulation_colab.ipynb) | [Open in Colab](https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/closedloop-core/notebooks/closedloop_simulation_colab.ipynb) | Which closed-loop scenarios expose planner failure-prone behavior under fixed compute? | Run calibration-gated, resumable closed-loop search and export artifacts. |
| `closedloop-core` | `closedloop_evaluation_colab.ipynb` | [GitHub](experiments/closedloop-core/notebooks/closedloop_evaluation_colab.ipynb) | [Open in Colab](https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/closedloop-core/notebooks/closedloop_evaluation_colab.ipynb) | Do discovered scenarios generalize and remain stable under evaluation metrics? | Evaluate closed-loop outputs, summarize diagnostics, and produce reporting tables. |
| `closedloop-core` | `compute_normalized_blindspot_discovery_colab.ipynb` | [GitHub](experiments/closedloop-core/notebooks/compute_normalized_blindspot_discovery_colab.ipynb) | [Open in Colab](https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/closedloop-core/notebooks/compute_normalized_blindspot_discovery_colab.ipynb) | Which method finds more high-value blindspots per compute unit? | Measure discovery efficiency under compute-normalized constraints. |
| `closedloop-core` | `counterfactual_risk_sensitivity_atlas_colab.ipynb` | [GitHub](experiments/closedloop-core/notebooks/counterfactual_risk_sensitivity_atlas_colab.ipynb) | [Open in Colab](https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/closedloop-core/notebooks/counterfactual_risk_sensitivity_atlas_colab.ipynb) | How sensitive are risk outcomes to structured counterfactual perturbation families? | Build counterfactual sensitivity maps across methods and scenario subsets. |
| `surprise-potential` | `surprise_potential_closedloop_colab.ipynb` | [GitHub](experiments/surprise-potential/notebooks/surprise_potential_closedloop_colab.ipynb) | [Open in Colab](https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/surprise-potential/notebooks/surprise_potential_closedloop_colab.ipynb) | Which surprise instantiation is non-collapsing and usable as an optimization signal? | Sweep surprise metrics/perturbation settings and run feasibility diagnostics. |
| `risk-uq-suite` | `miscalibration_probe_colab.ipynb` | [GitHub](experiments/risk-uq-suite/notebooks/miscalibration_probe_colab.ipynb) | [Open in Colab](https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/risk-uq-suite/notebooks/miscalibration_probe_colab.ipynb) | Is planner-side confidence miscalibrated enough to justify calibrated risk modeling? | Measure pre-training over/under-confidence and budget-threshold false-safe diagnostics. |
| `risk-uq-suite` | `risk_model_training_colab.ipynb` | [GitHub](experiments/risk-uq-suite/notebooks/risk_model_training_colab.ipynb) | [Open in Colab](https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/risk-uq-suite/notebooks/risk_model_training_colab.ipynb) | Can we train a calibrated risk model from closed-loop candidate outcomes? | Build training dataset, train ensemble model, and save calibrated checkpoints. |
| `risk-uq-suite` | `uq_benchmark_colab.ipynb` | [GitHub](experiments/risk-uq-suite/notebooks/uq_benchmark_colab.ipynb) | [Open in Colab](https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/risk-uq-suite/notebooks/uq_benchmark_colab.ipynb) | How robust and well-calibrated is predictive uncertainty under shifts? | Run UQ benchmarks, calibration curves, and robustness metrics. |
| `risk-uq-suite` | `paper_tables_figures_colab.ipynb` | [GitHub](experiments/risk-uq-suite/notebooks/paper_tables_figures_colab.ipynb) | [Open in Colab](https://colab.research.google.com/github/achyutmorang/waymax-simulation-experiments/blob/main/experiments/risk-uq-suite/notebooks/paper_tables_figures_colab.ipynb) | Are results presentation-ready and reproducible for reporting/paper writing? | Export final paper-quality tables and figures from saved artifacts. |

## Core Structure
- `experiments/`: pack-level notebooks and pack docs.
- `configs/experiments/`: pack-level runtime defaults.
- `src/closedloop/`, `src/eval*`, `src/risk_model/`: reusable method implementations.
- `src/workflows/`: notebook-facing orchestration APIs.
- `src/experiments/`: pack registry and scaffolding utilities.
- `notebooks/NOTEBOOK_DESIGN_CONTRACT.md`: notebook design standard.
- `notebooks/templates/`: starter template for new notebooks.

## Programmatic Pack Discovery
```python
from src.workflows import list_experiment_packs, validate_registry

for pack in list_experiment_packs():
    print(pack.slug, pack.notebooks)

print(validate_registry('.'))
```

## Scaffold a New Experiment Pack
```bash
python scripts/new_experiment.py \
  --slug my-paper-replication \
  --title "My Paper Replication" \
  --objective "Replicate and extend XYZ under the notebook design contract."
```

Generated files:
- `experiments/<slug>/README.md`
- `experiments/<slug>/notebooks/<slug>_colab.ipynb`
- `configs/experiments/<slug>.json`
- `src/workflows/<slug>_flow.py`
- `src/experiments/papers/<slug>/__init__.py`

## Runtime and Persistence
- Deterministic setup: `scripts/colab_setup.py`
- Recommended Drive root: `/content/drive/MyDrive/waymax_experiments/...`
- Resume behavior is run-tag + shard aware.

## WOMD Access
- Register at [Waymo Open Terms](https://waymo.com/open/terms)
- Authenticate with the same Google account in Colab runtime

## Local Tests
```bash
pip install -r requirements-dev.txt
PYTHONPATH=. pytest -q
```

## License
Proprietary thesis research code. All rights reserved. See `LICENSE`.
