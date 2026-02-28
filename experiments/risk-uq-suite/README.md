# risk-uq-suite

## Objective
Train/calibrate risk models and run uncertainty-quantification benchmarks with paper tables/figures.

## Living Report
- `experiments/risk-uq-suite/LIVING_REPORT.md`
- Auto-updated by the risk-uq workflows to capture evolving problem framing, hypothesis status, and evidence snapshots.

## Independence From Closedloop-Core Notebook Logic
This track now uses risk-UQ-specific notebook orchestration and smoke gates:

- `initialize_risk_uq_run_context(...)`
- `build_risk_uq_simulation_context(...)`
- `run_risk_uq_smoke_gates(...)`

These replace inherited closedloop-core notebook flow gates and avoid surprise/counterfactual preflight coupling in the risk-UQ notebook path.

## Problem Statement
In closed-loop planning, action selection depends on predicted risk/confidence. If these probabilities are miscalibrated, thresholded decisions can become false-safe (unsafe actions accepted) or overly conservative (safe actions rejected), especially under distribution shift. This track tests that failure mode, then evaluates whether calibrated risk estimates improve robustness and safety-progress tradeoffs.

## Formal Problem Formulation
At scenario `s` and timestep `t`, we construct `K` candidate ego actions `a_{s,t,k}` from the planner action distribution. Each candidate is rolled out for a short horizon and mapped to a feature vector:

- `x_{s,t,k} = [distribution stats, belief stats, short-rollout stats]`

We predict event risk for events `e in {collision, offroad, failure_proxy}` and horizons `h in {5,10,15}`:

- `p_raw^{e,h}(x_{s,t,k}) = f_theta(x_{s,t,k})`

The model is trained with summed binary cross-entropy across all heads:

- `L(theta) = sum_{(s,t,k)} sum_{e,h} BCE(p_raw^{e,h}(x_{s,t,k}), y_{s,t,k}^{e,h})`

Post-hoc per-head calibration (temperature scaling on validation split):

- `p_cal^{e,h} = sigmoid(logit^{e,h} / T_{e,h})`

Risk-aware candidate selection uses calibrated failure probability at `h=15` with epistemic and comfort penalties:

- `score_k = w_p * progress_k - w_f * p_cal_k - w_u * epistemic_k - w_c * comfort_k`

Budgeted control rule:

- choose highest-score candidate among those with `p_cal_k <= tau` (fail budget),
- otherwise fallback to candidate with minimum calibrated risk.

## Research Questions
1. Are planner-side confidence/risk proxies miscalibrated in closed-loop rollouts, and does this worsen under shift?
2. Does post-hoc calibration improve reliability metrics (ECE/NLL/Brier) over raw risk probabilities?
3. Does calibrated risk-aware reranking reduce failures with limited progress loss relative to base planner behavior?
4. Are gains robust across nominal and shift suites (`nominal_clean`, `hist_prim_shift`, `fut_prim_shift`, `hist_rmv_shift`, `high_interaction_holdout`)?

## Research Gap (Why This Work)
Most prior work contributes one piece (calibration method, UQ benchmark, or simulator infrastructure). This track targets the missing integration:

1. Candidate-action-level risk modeling in a closed-loop Waymax setting.
2. Explicit calibration-to-control bridge (probabilities used in real selection constraints).
3. Standardized, reproducible shift benchmark with threshold-budget diagnostics (`false-safe` vs `over-conservative` behavior).
4. Notebook-first reproducibility with manifests, resume support, and paper-export artifacts.

## Novelty Boundary (Claim Discipline)
This work does **not** claim novelty for deep ensembles, temperature scaling, or conformal-style thresholds individually.  
It claims novelty in the integrated pipeline and evaluation protocol for closed-loop risk-aware planning in Waymax.

## Feasibility and Evaluation Criteria
Engineering feasibility:

1. End-to-end notebooks: probe -> train/calibrate -> benchmark -> paper export.
2. Persistent artifacts and resume-aware flows for Colab.
3. Regression-safe implementation with tests in this repository.

Research feasibility:

1. Miscalibration probe must show non-trivial calibration gap for at least one proxy/shift.
2. Calibrated monitor should improve or preserve reliability on nominal and most shifts.
3. Risk-aware controller should improve failure metrics under a bounded progress tradeoff target.

## Implementation Alignment (Checked Against Current Code)
- Candidate count: `K=8`
- Control horizon: `H=6`
- Label horizons: `5/10/15`
- Events: `collision/offroad/failure_proxy`
- Ensemble: `5` members, hidden dims `(128,128)`, dropout `0.10`
- Training defaults: `lr=1e-3`, `batch=1024`, `max_epochs=50`, `patience=8`
- Controller weights: `w_p=1.0`, `w_f=2.0`, `w_u=0.5`, `w_c=0.25`
- Fail budget: `tau=0.20`
- Miscalibration probe variants: `planner_top1_proxy`, `planner_entropy_proxy`, `planner_combo_proxy`

## Notebooks
- `experiments/risk-uq-suite/notebooks/miscalibration_probe_colab.ipynb`
- `experiments/risk-uq-suite/notebooks/miscalibration_interpretation_colab.ipynb`
- `experiments/risk-uq-suite/notebooks/risk_model_training_colab.ipynb`
- `experiments/risk-uq-suite/notebooks/uq_benchmark_colab.ipynb`
- `experiments/risk-uq-suite/notebooks/paper_tables_figures_colab.ipynb`

## Suggested Run Order
1. `miscalibration_probe_colab.ipynb`
2. `miscalibration_interpretation_colab.ipynb`
3. `risk_model_training_colab.ipynb`
4. `uq_benchmark_colab.ipynb`
5. `paper_tables_figures_colab.ipynb`

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
