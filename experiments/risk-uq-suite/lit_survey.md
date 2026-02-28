# Risk-UQ Suite: Targeted Literature Survey

This survey is curated to justify the framing of the `risk-uq-suite` track:
1. Uncertainty use in planning/decision-making
2. Calibration of probabilistic outputs
3. Risk/safety decisions under thresholds or budgets
4. Closed-loop AV benchmark design

Local PDFs are stored in `experiments/risk-uq-suite/references/pdfs/`.

## Paper Index (15 papers)

| Title | Year | Area | Why relevant to risk-uq-suite |
|---|---:|---|---|
| Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles | 2017 | Uncertainty | Strong practical baseline for predictive uncertainty quality. |
| Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS) | 2018 | Uncertainty + Planning | Demonstrates uncertainty-aware planning with ensemble dynamics + MPC. |
| MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction | 2020 | AV Uncertainty | AV-specific probabilistic trajectory outputs used for downstream decisions. |
| On Calibration of Modern Neural Networks | 2017 | Calibration | Canonical evidence of overconfidence and effectiveness of temperature scaling. |
| Predicting Good Probabilities with Supervised Learning | 2005/2012 | Calibration | Classic comparative study of Platt scaling and isotonic calibration. |
| Beyond Temperature Scaling: Dirichlet Calibration | 2019 | Calibration | Strong multiclass post-hoc calibration baseline beyond simple temperature. |
| Can You Trust Your Model’s Uncertainty? Under Dataset Shift | 2019 | Calibration under Shift | Shows calibration degrades under shift; motivates shift-suite evaluation. |
| Selective Classification for Deep Neural Networks | 2017 | Threshold decisions | Formal risk-coverage tradeoff under confidence thresholds. |
| Constrained Policy Optimization | 2017 | Budgeted safety control | Constrained optimization view for safety budgets in decision policies. |
| Distribution-Free, Risk-Controlling Prediction Sets | 2021 | Risk control | Distribution-free control of task-defined risk levels. |
| Conformal Risk Control | 2022 | Risk control | Finite-sample conformal guarantees for risk-controlled decisions. |
| Multi-Agent Reachability Calibration with Conformal Prediction | 2023 | AV Safety + Conformal | AV-tailored calibration plus safety reachability constraints. |
| Waymax: Accelerated Data-Driven Simulator for AV Research | 2023 | Closed-loop benchmark infra | Simulation substrate used directly by this project. |
| The Waymo Open Sim Agents Challenge | 2023 | Closed-loop benchmark | Standardized interactive closed-loop AV benchmark and metrics. |
| nuPlan: A Closed-Loop ML-Based Planning Benchmark for AVs | 2021 | Closed-loop benchmark | Strong closed-loop planning benchmark with realistic scenario taxonomy. |

## 1) Uncertainty in Planning / Decision-Making

- **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles** (Lakshminarayanan et al., NeurIPS 2017)  
  Links: [paper](https://papers.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html), [pdf](https://arxiv.org/pdf/1612.01474.pdf), [local pdf](references/pdfs/deep_ensembles_2017.pdf)
  - Problem: Obtain useful uncertainty estimates from deep models without Bayesian complexity.
  - Method: Train multiple independently initialized models; use ensemble mean/variance.
  - Key result: Strong uncertainty quality and robustness vs single-model confidence.
  - Relation to our work: Supports our use of ensemble-style uncertainty as a practical baseline for decision-time risk signals.

- **Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS)** (Chua et al., NeurIPS 2018)  
  Links: [paper](https://papers.nips.cc/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html), [pdf](https://arxiv.org/pdf/1805.12114.pdf), [local pdf](references/pdfs/pets_2018.pdf)
  - Problem: Sample-efficient control/planning under model uncertainty.
  - Method: Probabilistic ensemble dynamics + trajectory sampling MPC.
  - Key result: Competitive performance with high sample efficiency on control tasks.
  - Relation to our work: Reinforces uncertainty-aware candidate evaluation in a planning loop (our case: Waymax closed-loop candidate actions).

- **MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction** (Chai et al., CoRL 2020)  
  Links: [paper](https://proceedings.mlr.press/v100/chai20a.html), [pdf](https://proceedings.mlr.press/v100/chai20a/chai20a.pdf), [local pdf](references/pdfs/multipath_2020.pdf)
  - Problem: Multi-modal trajectory uncertainty for AV behavior prediction.
  - Method: Anchor-based multimodal trajectory heads with probabilistic outputs.
  - Key result: Strong trajectory prediction quality on large-scale AV data.
  - Relation to our work: Supports modeling/using uncertainty from multimodal predictive distributions for AV decisions.

## 2) Calibration of Probabilities

- **On Calibration of Modern Neural Networks** (Guo et al., ICML 2017)  
  Links: [paper](https://proceedings.mlr.press/v70/guo17a.html), [pdf](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), [local pdf](references/pdfs/guo_2017.pdf)
  - Problem: Neural network probabilities are often overconfident and miscalibrated.
  - Method: Empirical analysis + post-hoc temperature scaling.
  - Key result: Temperature scaling is a strong simple baseline.
  - Relation to our work: Core justification for post-hoc calibration before using probabilities in control constraints.

- **Predicting Good Probabilities with Supervised Learning** (Niculescu-Mizil & Caruana, ICML 2005; arXiv mirror 2012)  
  Links: [paper](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf), [pdf](https://arxiv.org/pdf/1207.1403.pdf), [local pdf](references/pdfs/obtaining_calibrated_probabilities_boosting_2012.pdf)
  - Problem: Classifier scores are not necessarily calibrated probabilities.
  - Method: Comparative study including Platt-style sigmoid and isotonic regression.
  - Key result: Calibration methods can materially improve probability quality depending on model family.
  - Relation to our work: Direct methodological support for simple post-hoc calibrators on risk scores/proxies.

- **Beyond Temperature Scaling: Obtaining Well-Calibrated Multiclass Probabilities with Dirichlet Calibration** (Kull et al., NeurIPS 2019)  
  Links: [paper](https://papers.nips.cc/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html), [pdf](https://arxiv.org/pdf/1910.12656.pdf), [local pdf](references/pdfs/dirichlet_2019.pdf)
  - Problem: Temperature scaling can be too limited for some multiclass outputs.
  - Method: Dirichlet calibration (low-parameter post-hoc mapping).
  - Key result: Better calibration than TS in several settings.
  - Relation to our work: Useful ablation alternative if TS/Platt is insufficient under shift.

- **Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift** (Ovadia et al., NeurIPS 2019)  
  Links: [paper](https://papers.nips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html), [pdf](https://arxiv.org/pdf/1906.02530.pdf), [local pdf](references/pdfs/ovadia_2019.pdf)
  - Problem: Uncertainty and calibration methods often degrade under distribution shift.
  - Method: Large benchmark across uncertainty methods and shift settings.
  - Key result: In-domain gains do not guarantee robust OOD calibration.
  - Relation to our work: Strong justification for our nominal + shift-suite calibration analysis in Waymax.

## 3) Risk / Safety Decisions Under Thresholds or Budgets

- **Selective Classification for Deep Neural Networks** (Geifman & El-Yaniv, NeurIPS 2017)  
  Links: [paper](https://papers.nips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html), [pdf](https://arxiv.org/pdf/1705.08500.pdf), [local pdf](references/pdfs/selective_classification_2017.pdf)
  - Problem: Decide when to abstain to control decision risk.
  - Method: Confidence-thresholded selective prediction with risk-coverage tradeoff.
  - Key result: Explicit tradeoff curves for safety vs coverage.
  - Relation to our work: Mirrors our threshold-budget framing (`accept` vs `reject/fallback`) and selective-risk analysis.

- **Constrained Policy Optimization** (Achiam et al., ICML 2017)  
  Links: [paper](https://proceedings.mlr.press/v70/achiam17a.html), [pdf](https://arxiv.org/pdf/1705.10528.pdf), [local pdf](references/pdfs/cpo_2017.pdf)
  - Problem: Optimize return while respecting explicit safety constraints.
  - Method: Trust-region constrained updates with near-constraint satisfaction.
  - Key result: Better safety-constraint handling than unconstrained policy updates.
  - Relation to our work: Supports explicit budget-constrained decision rules instead of raw score optimization only.

- **Distribution-Free, Risk-Controlling Prediction Sets** (Bates et al., 2021)  
  Links: [paper](https://arxiv.org/abs/2101.02703), [pdf](https://arxiv.org/pdf/2101.02703.pdf), [local pdf](references/pdfs/rcps_2021.pdf)
  - Problem: Control task-level risk without distributional assumptions.
  - Method: Distribution-free set construction controlling expected risk.
  - Key result: Risk control guarantees with finite-sample validity.
  - Relation to our work: Gives formal grounding for risk-budgeted decisions and coverage-risk diagnostics.

- **Conformal Risk Control** (Angelopoulos et al., 2022/2023)  
  Links: [paper](https://arxiv.org/abs/2208.02814), [pdf](https://arxiv.org/pdf/2208.02814.pdf), [local pdf](references/pdfs/crc_2022.pdf)
  - Problem: Guarantee user-specified risk levels with conformal methods.
  - Method: Risk-controlling conformal wrappers for flexible tasks.
  - Key result: Finite-sample risk guarantees under exchangeability.
  - Relation to our work: Aligns with our threshold-budget objective and motivates conformal-style safety gating.

- **Multi-Agent Reachability Calibration with Conformal Prediction** (Muthali et al., 2023)  
  Links: [paper](https://arxiv.org/abs/2304.00432), [pdf](https://arxiv.org/pdf/2304.00432.pdf), [local pdf](references/pdfs/marc_2023.pdf)
  - Problem: Safety-critical AV interaction requires calibrated uncertainty tied to safety envelopes.
  - Method: Conformal calibration integrated with multi-agent reachability reasoning.
  - Key result: Improved safety-calibrated prediction in interactive settings.
  - Relation to our work: Closest conceptual bridge to AV-specific calibrated risk decisions.

## 4) Closed-Loop AV Benchmarks and Simulation

- **Waymax: An Accelerated, Data-Driven Simulator for Large-Scale Autonomous Driving Research** (Gulino et al., 2023)  
  Links: [paper](https://arxiv.org/abs/2310.08710), [pdf](https://arxiv.org/pdf/2310.08710.pdf), [local pdf](references/pdfs/waymax_2023.pdf)
  - Problem: Scalable closed-loop simulation infrastructure for AV research.
  - Method: JAX-based simulator with WOMD integration and accelerated rollouts.
  - Key result: Efficient, reproducible, large-scale simulation suitable for research workflows.
  - Relation to our work: Direct platform basis for all experiments in this project.

- **The Waymo Open Sim Agents Challenge** (Ettinger et al., NeurIPS Datasets & Benchmarks 2023)  
  Links: [paper](https://papers.nips.cc/paper_files/paper/2023/hash/b96ce67b2f2d45e4ab315e13a6b5b9c5-Abstract-Datasets_and_Benchmarks.html), [pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/b96ce67b2f2d45e4ab315e13a6b5b9c5-Paper-Datasets_and_Benchmarks.pdf), [local pdf](references/pdfs/wosac_2023.pdf)
  - Problem: Standardized evaluation for realistic interactive simulation agents.
  - Method: Public benchmark with realism-focused metrics and challenge protocol.
  - Key result: Common reference point for closed-loop simulation agent evaluation.
  - Relation to our work: Supports benchmark-oriented evaluation design and reporting rigor.

- **nuPlan: A Closed-Loop ML-Based Planning Benchmark for Autonomous Vehicles** (Caesar et al., 2021)  
  Links: [paper](https://arxiv.org/abs/2106.11810), [pdf](https://arxiv.org/pdf/2106.11810.pdf), [local pdf](references/pdfs/nuplan_2021.pdf)
  - Problem: Realistic closed-loop planning benchmark with diverse driving scenarios.
  - Method: Scenario taxonomy, simulation stack, and planner evaluation metrics.
  - Key result: Establishes reproducible closed-loop planner evaluation at scale.
  - Relation to our work: Provides external benchmark precedent for closed-loop safety/progress tradeoff reporting.

## Notes for our project framing

- This survey supports the claim that **uncertainty alone is not enough**; it must be **calibrated and evaluated under shift** before using it for thresholded safety decisions.
- It also supports a paper story built around **risk-quality -> threshold decisions -> closed-loop outcomes**.
- None of the papers requires our exact proxy design; they justify the framing and evaluation protocol choices.
