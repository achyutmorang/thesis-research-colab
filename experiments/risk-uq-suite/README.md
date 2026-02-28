# risk-uq-suite

## Objective
Train/calibrate risk models and run uncertainty-quantification benchmarks with paper tables/figures.

## Notebooks
- `experiments/risk-uq-suite/notebooks/risk_model_training_colab.ipynb`
- `experiments/risk-uq-suite/notebooks/uq_benchmark_colab.ipynb`
- `experiments/risk-uq-suite/notebooks/paper_tables_figures_colab.ipynb`

## Experiment Config
- `configs/experiments/risk-uq-suite.json`
- Notebooks load mandatory run fields from this config:
  - `RUN_NAME`
  - `RUN_PREFIX`
  - `PERSIST_ROOT`
  - `N_SHARDS`
  - `SHARD_ID`
  - `RESUME_FROM_EXISTING`
  - `RUN_ENABLED`

## Workflow Entrypoints
- `src/workflows/risk_training_flow.py`
- `src/workflows/uq_benchmark_flow.py`
- `src/workflows/paper_export_flow.py`

## Core Modules
- `src/risk_model/`
- `src/workflows/`
