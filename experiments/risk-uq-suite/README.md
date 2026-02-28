# risk-uq-suite

## Objective
Train/calibrate risk models and run uncertainty-quantification benchmarks with paper tables/figures.

## Problem Statement
In closed-loop planning, action selection depends on predicted risk/confidence. If these probabilities are miscalibrated, thresholded decisions can become false-safe (unsafe actions accepted) or overly conservative (safe actions rejected), especially under distribution shift. This track tests that failure mode, then evaluates whether calibrated risk estimates improve robustness and safety-progress tradeoffs.

## Notebooks
- `experiments/risk-uq-suite/notebooks/miscalibration_probe_colab.ipynb`
- `experiments/risk-uq-suite/notebooks/risk_model_training_colab.ipynb`
- `experiments/risk-uq-suite/notebooks/uq_benchmark_colab.ipynb`
- `experiments/risk-uq-suite/notebooks/paper_tables_figures_colab.ipynb`

## Suggested Run Order
1. `miscalibration_probe_colab.ipynb`
2. `risk_model_training_colab.ipynb`
3. `uq_benchmark_colab.ipynb`
4. `paper_tables_figures_colab.ipynb`

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

## Rigorous Literature Survey (Gap Justification)
| Paper | Type | What it established | Limits for this project | Code |
|---|---|---|---|---|
| [Guo et al., ICML 2017, *On Calibration of Modern Neural Networks*](https://proceedings.mlr.press/v70/guo17a.html) | Foundational calibration | Demonstrated modern deep nets can be poorly calibrated; temperature scaling is a strong post-hoc baseline. | Not autonomous-driving specific; not closed-loop control integrated. | [reference impl](https://github.com/gpleiss/temperature_scaling) |
| [Lakshminarayanan et al., NeurIPS 2017, *Deep Ensembles*](https://papers.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles) | Baseline UQ method | Practical uncertainty baseline with strong empirical behavior and simple training recipe. | Not tied to candidate-level action reranking in simulation loops. | method paper (commonly re-implemented) |
| [Ovadia et al., NeurIPS 2019, *Evaluating Uncertainty Under Dataset Shift*](https://arxiv.org/abs/1906.02530) | Benchmark evidence | Showed uncertainty quality can degrade under shift; static calibration alone can fail. | Classification-centric benchmark; not AV closed-loop action selection. | [uncertainty-baselines](https://github.com/google/uncertainty-baselines) |
| [Gulino et al., 2023, *Waymax*](https://arxiv.org/abs/2310.08710) | Simulator foundation | Large-scale, accelerator-friendly closed-loop simulator for data-driven AV research. | Provides simulation substrate, not a calibrated risk-aware controller method. | [waymo-research/waymax](https://github.com/waymo-research/waymax) |
| [*Waymo Open Sim Agents Challenge* (NeurIPS D\&B 2023)](https://papers.nips.cc/paper_files/paper/2023/hash/b96ce67b2f2d45e4ab315e13a6b5b9c5-Abstract-Datasets_and_Benchmarks.html) | Task/metric benchmark | Public benchmark and realism-focused evaluation for interactive simulation. | Focus is simulator realism ranking, not calibrated risk-to-control pipelines. | challenge/eval server |
| [Muthali et al., 2023, *Multi-Agent Reachability Calibration with Conformal Prediction*](https://arxiv.org/abs/2304.00432) | Safety-UQ method | Combines forecasting uncertainty calibration with reachability-style safety assurances. | Does not provide a Waymax-native, notebook-first, reproducible closed-loop reranking stack. | paper link (public code not prominent) |
| [Huang et al., 2024, *CUQDS*](https://arxiv.org/abs/2406.12100) | Shift-aware UQ | Addresses conformal UQ under shift for trajectory prediction. | Primarily trajectory-prediction UQ; no direct candidate-action closed-loop controller integration in Waymax workflows. | paper link (code status unclear) |
| [*Adversarially Robust Conformal Prediction for Interactive Safe Planning* (2025)](https://arxiv.org/abs/2511.10586) | Recent robust CP direction | Targets interactive settings where policy updates induce shift in environment response. | Early-stage direction; not a standardized Waymax closed-loop benchmarking package with end-to-end artifacts. | paper link |

## What This Track Adds Relative to Prior Work
1. Candidate-action-level risk modeling directly from Waymax closed-loop rollouts.
2. Explicit calibration-to-control bridge (raw vs calibrated risk used in action selection diagnostics).
3. Standardized nominal-plus-shift evaluation pack with reliability and threshold-budget diagnostics.
4. Reproducible Colab-first orchestration (resume, manifests, persistent artifacts, paper export path).

## Inspiration References
- `experiments/risk-uq-suite/references/INSPIRATION_REPOS.md`
- `experiments/risk-uq-suite/references/fetch_inspiration_repos.sh`
