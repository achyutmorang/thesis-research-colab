# Risk-UQ Suite: Comprehensive Literature Survey (Planner Uncertainty, Calibration, Risk Decisions, Closed-Loop AV)

This update expands the survey specifically for your question:
- Do strong prior papers use uncertainty or risk proxies for planner decision-making?
- What is known about uncertainty-aware / risk-based planning in AVs?
- Have proxies been used before?

Short answer: **yes**. Prior work uses uncertainty and surrogate risk signals (e.g., confidence/entropy, RSS margins, TTC/reachability proxies, chance constraints) to guide planning decisions, but robust calibration-to-control evaluation in closed-loop AV settings remains less standardized.

Local PDFs are in `experiments/risk-uq-suite/references/pdfs/`.

## What We Can Defend From Prior Art

1. **Uncertainty-informed planning is established**: belief-space/POMDP, game-theoretic, robust MPC, and chance-constrained planners explicitly use predictive uncertainty.
2. **Risk proxies are common in practice**: RSS safety margins, TTC-like or reachability surrogates, and proxy confidence thresholds are used to control acceptance/rejection and fallback behavior.
3. **Calibration is necessary before using probabilities in decisions**: many papers show raw confidence is miscalibrated, especially under shift.
4. **Closed-loop benchmark rigor is increasingly expected**: Waymax, WOSAC, nuPlan support standardized safety/progress tradeoff reporting.

## Master Table (25 papers)

| Title | Year | Area | Why relevant to our questions |
|---|---:|---|---|
| MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction | 2020 | AV uncertainty prediction | Canonical multimodal uncertainty output used downstream in AV stacks. |
| On a Formal Model of Safe and Scalable Self-driving Cars | 2017 | Safety proxies / AV policy | Defines RSS safety rules as decision-grade safety surrogates. |
| PORCA: Modeling and Planning for Autonomous Driving among Many Pedestrians | 2018 | Uncertainty-aware AV planning | Probabilistic model for interaction uncertainty in planning. |
| Hierarchical Game-Theoretic Planning for Autonomous Vehicles | 2018 | Interactive AV planning | Uncertainty-aware interaction planning through game-theoretic reasoning. |
| Constrained Iterative LQG for Real-Time Chance-Constrained Gaussian Belief Space Planning | 2021 | Belief-space planning | Chance constraints directly map uncertainty to safety budgets. |
| Game-Theoretic Planning for Autonomous Driving among Risk-Aware Human Drivers | 2022 | Risk-aware interaction planning | Explicit risk-aware human models influence AV planner decisions. |
| Potential risk assessment for safe driving of autonomous vehicles under occluded vision | 2022 | Risk proxy under occlusion | Uses risk proxy estimation under partial observability. |
| Occlusion-aware Risk Assessment and Driving Strategy for AVs Using Simplified Reachability Quantification | 2023 | Reachability risk proxy | Reachability-based risk surrogate for occluded interaction. |
| Predictive Control for Autonomous Driving with Uncertain, Multi-modal Predictions | 2023 | MPC with uncertainty | Integrates multimodal predictive uncertainty in control optimization. |
| Recursively Feasible Chance-constrained MPC under Gaussian Mixture Model Uncertainty | 2024 | Risk-constrained MPC | Safety constraints under multimodal/GMM uncertainty. |
| RACP: Risk-Aware Contingency Planning with Multi-Modal Predictions | 2024 | Contingency planning | Planner switches/hedges using predicted risk contingencies. |
| Uncertainty-Aware Prediction and Application in Planning for AV: Definitions, Methods, and Comparison | 2024 | Survey | Consolidates uncertainty-use patterns in AV planning pipelines. |
| Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles | 2017 | UQ baseline | Strong practical uncertainty baseline for downstream risk estimation. |
| On Calibration of Modern Neural Networks | 2017 | Calibration | Canonical evidence + temperature scaling baseline. |
| Predicting Good Probabilities with Supervised Learning | 2005/2012 | Calibration | Classic Platt/isotonic calibration comparison. |
| Beyond Temperature Scaling: Dirichlet Calibration | 2019 | Calibration | Strong post-hoc multiclass calibration alternative. |
| Can You Trust Your Model’s Uncertainty? Under Dataset Shift | 2019 | Calibration under shift | In-domain calibration can fail under shift. |
| Selective Classification for Deep Neural Networks | 2017 | Threshold decisions | Formal risk-coverage decision tradeoff under confidence thresholds. |
| Constrained Policy Optimization | 2017 | Budgeted safety control | Optimization with explicit constraints/budgets. |
| Distribution-Free, Risk-Controlling Prediction Sets | 2021 | Risk control | Distribution-free risk control framework. |
| Conformal Risk Control | 2022 | Risk control | Conformal finite-sample risk guarantees. |
| Multi-Agent Reachability Calibration with Conformal Prediction | 2023 | AV safety calibration | AV-specific bridge between calibration and safety control. |
| Waymax: An Accelerated, Data-Driven Simulator for Large-Scale AV Research | 2023 | Closed-loop benchmark infra | Direct simulator foundation for this project. |
| The Waymo Open Sim Agents Challenge | 2023 | Closed-loop benchmark | Standardized interactive simulation benchmark. |
| nuPlan: A Closed-Loop ML-Based Planning Benchmark for AVs | 2021 | Closed-loop benchmark | Strong closed-loop planning benchmark with realistic scenarios. |

---

## A) Uncertainty-Aware / Risk-Based Planning in AV (and direct proxy usage)

- **MultiPath** (Chai et al., 2020, CoRL)  
  Links: [paper](https://proceedings.mlr.press/v100/chai20a.html), [pdf](https://proceedings.mlr.press/v100/chai20a/chai20a.pdf), [local pdf](references/pdfs/multipath_2020.pdf)
  - Problem: Predicting multimodal future trajectories for interactive AV planning.
  - Method: Anchor-based probabilistic multimodal trajectory prediction.
  - Key result: Strong multimodal prediction quality on AV datasets.
  - Relation: Supports our use of multimodal uncertainty signals before decision rules.

- **On a Formal Model of Safe and Scalable Self-driving Cars (RSS)** (Shalev-Shwartz et al., 2017)  
  Links: [paper](https://arxiv.org/abs/1708.06374), [pdf](https://arxiv.org/pdf/1708.06374.pdf), [local pdf](references/pdfs/rss_2017.pdf)
  - Problem: Formalizing AV safety decision rules.
  - Method: Responsibility-Sensitive Safety (distance/time-margin style safety constraints).
  - Key result: Formal policy-level safety model widely referenced in AV safety discussions.
  - Relation: Explicit evidence that **risk proxies/safety surrogates** are used in planning policies.

- **PORCA: Modeling and Planning for Autonomous Driving among Many Pedestrians** (Luber et al., 2018)  
  Links: [paper](https://arxiv.org/abs/1805.11833), [pdf](https://arxiv.org/pdf/1805.11833.pdf), [local pdf](references/pdfs/porca_2018.pdf)
  - Problem: Planning in highly interactive pedestrian-rich settings under uncertainty.
  - Method: Probabilistic ORCA extension for interaction-aware planning.
  - Key result: Improved behavior in crowded scenarios with uncertain human motion.
  - Relation: Shows uncertainty-aware planning with tractable surrogate interaction models.

- **Hierarchical Game-Theoretic Planning for Autonomous Vehicles** (Fisac et al., 2018)  
  Links: [paper](https://arxiv.org/abs/1810.05766), [pdf](https://arxiv.org/pdf/1810.05766.pdf), [local pdf](references/pdfs/hierarchical_game_theoretic_planning_2018.pdf)
  - Problem: Interactive planning with strategic uncertainty about other agents.
  - Method: Hierarchical game-theoretic planning over interaction models.
  - Key result: Better interactive decision quality than non-interactive baselines.
  - Relation: Supports decision-making driven by uncertainty over other-agent responses.

- **Constrained Iterative LQG for Real-Time Chance-Constrained Gaussian Belief Space Planning** (Heiden et al., 2021)  
  Links: [paper](https://arxiv.org/abs/2108.06533), [pdf](https://arxiv.org/pdf/2108.06533.pdf), [local pdf](references/pdfs/cilqg_2021.pdf)
  - Problem: Real-time planning with uncertainty and strict probabilistic safety constraints.
  - Method: Belief-space iterative LQG with chance constraints.
  - Key result: Real-time feasible chance-constrained planning in uncertain domains.
  - Relation: Theoretical and algorithmic precedent for threshold/budget constraints on risk.

- **Game-Theoretic Planning for AVs among Risk-Aware Human Drivers** (Chandra et al., 2022)  
  Links: [paper](https://arxiv.org/abs/2205.00562), [pdf](https://arxiv.org/pdf/2205.00562.pdf), [local pdf](references/pdfs/risk_aware_human_drivers_2022.pdf)
  - Problem: Human drivers exhibit heterogeneous risk attitudes, affecting AV safety.
  - Method: Game-theoretic planner with risk-aware human behavior modeling.
  - Key result: Better planning outcomes in mixed-risk interaction scenarios.
  - Relation: Reinforces risk-aware decision criteria for planner action choices.

- **Potential risk assessment for safe driving of AVs under occluded vision** (Yu et al., 2022, Scientific Reports)  
  Links: [paper](https://www.nature.com/articles/s41598-022-08810-z), [pdf](https://www.nature.com/articles/s41598-022-08810-z.pdf), [local pdf](references/pdfs/potential_risk_occluded_vision_2022.pdf)
  - Problem: Occlusion causes latent hazards that standard planner confidence can miss.
  - Method: Potential-risk estimation under occluded visibility.
  - Key result: Risk-aware behavior improves safety handling under occlusion.
  - Relation: Direct example of **proxy risk signals** used for AV decisions.

- **Occlusion-aware Risk Assessment and Driving Strategy via Reachability** (Huang et al., 2023)  
  Links: [paper](https://arxiv.org/abs/2306.07004), [pdf](https://arxiv.org/pdf/2306.07004.pdf), [local pdf](references/pdfs/occlusion_reachability_risk_2023.pdf)
  - Problem: Quantifying risk when key agents are partially/unobserved.
  - Method: Simplified reachability-based risk quantification integrated with strategy.
  - Key result: Safer strategy under occlusion compared with naive planning.
  - Relation: Strong precedent for surrogate risk quantification driving planner strategy.

- **Predictive Control for AV with Uncertain, Multi-modal Predictions** (Nair et al., 2023)  
  Links: [paper](https://arxiv.org/abs/2310.20561), [pdf](https://arxiv.org/pdf/2310.20561.pdf), [local pdf](references/pdfs/predictive_control_multimodal_2023.pdf)
  - Problem: MPC decisions degrade if predictive uncertainty is ignored.
  - Method: MPC integrating multimodal uncertain forecasts.
  - Key result: Better safety-performance tradeoffs in interactive scenarios.
  - Relation: Very close to our framing: uncertainty estimate -> constrained/control decision.

- **Recursively Feasible Chance-constrained MPC under GMM Uncertainty** (Koller et al., 2024)  
  Links: [paper](https://arxiv.org/abs/2401.03799), [pdf](https://arxiv.org/pdf/2401.03799.pdf), [local pdf](references/pdfs/safe_ccmpc_gmm_2024.pdf)
  - Problem: Maintaining feasibility/safety with multimodal uncertainty in control.
  - Method: Chance-constrained MPC with Gaussian mixture uncertainty model.
  - Key result: Recursive feasibility plus controlled risk under uncertainty.
  - Relation: Connects multimodal uncertainty to explicit risk budgets similar to our thresholded controller logic.

- **RACP: Risk-Aware Contingency Planning with Multi-Modal Predictions** (Rosen et al., 2024)  
  Links: [paper](https://arxiv.org/abs/2402.17387), [pdf](https://arxiv.org/pdf/2402.17387.pdf), [local pdf](references/pdfs/racp_2024.pdf)
  - Problem: Single-plan execution can fail under multimodal interaction uncertainty.
  - Method: Risk-aware contingency planning over multiple candidate futures.
  - Key result: Improved robustness in uncertain interactions.
  - Relation: Strong support for candidate-level risk-guided reranking/selection ideas.

- **Uncertainty-Aware Prediction and Application in Planning for AV: Definitions, Methods, and Comparison** (Liu et al., 2024)  
  Links: [paper](https://arxiv.org/abs/2403.02297), [pdf](https://arxiv.org/pdf/2403.02297.pdf), [local pdf](references/pdfs/uap_planning_comparison_2024.pdf)
  - Problem: Lack of unified understanding of uncertainty roles in AV prediction/planning.
  - Method: Taxonomy and comparative study.
  - Key result: Clarifies where uncertainty helps vs fails in planning pipelines.
  - Relation: High-level support that our problem statement is meaningful and current.

## B) Calibration and Uncertainty Quality

- **Deep Ensembles** (Lakshminarayanan et al., NeurIPS 2017)  
  Links: [paper](https://papers.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html), [pdf](https://arxiv.org/pdf/1612.01474.pdf), [local pdf](references/pdfs/deep_ensembles_2017.pdf)
  - Problem: Practical predictive uncertainty from deep models.
  - Method: Independent ensemble members + predictive variance.
  - Key result: Strong and robust uncertainty baseline.
  - Relation: Directly supports ensemble uncertainty features and epistemic terms.

- **On Calibration of Modern Neural Networks** (Guo et al., ICML 2017)  
  Links: [paper](https://proceedings.mlr.press/v70/guo17a.html), [pdf](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), [local pdf](references/pdfs/guo_2017.pdf)
  - Problem: Neural confidence miscalibration.
  - Method: Post-hoc temperature scaling.
  - Key result: Simple calibrator often works well in-domain.
  - Relation: Core support for post-hoc calibration stage before control use.

- **Predicting Good Probabilities with Supervised Learning** (Niculescu-Mizil & Caruana, ICML 2005 / arXiv mirror)  
  Links: [paper](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf), [pdf](https://arxiv.org/pdf/1207.1403.pdf), [local pdf](references/pdfs/obtaining_calibrated_probabilities_boosting_2012.pdf)
  - Problem: Raw model scores are not reliable probabilities.
  - Method: Platt sigmoid and isotonic calibration comparisons.
  - Key result: Calibration materially changes probability quality.
  - Relation: Direct basis for our simple proxy calibrator baselines.

- **Beyond Temperature Scaling: Dirichlet Calibration** (Kull et al., NeurIPS 2019)  
  Links: [paper](https://papers.nips.cc/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html), [pdf](https://arxiv.org/pdf/1910.12656.pdf), [local pdf](references/pdfs/dirichlet_2019.pdf)
  - Problem: TS may be too rigid for multiclass settings.
  - Method: Dirichlet post-hoc mapping.
  - Key result: Improved calibration over TS in multiple tasks.
  - Relation: Strong alternative calibrator for ablations if needed.

- **Can You Trust Your Model’s Uncertainty?** (Ovadia et al., NeurIPS 2019)  
  Links: [paper](https://papers.nips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html), [pdf](https://arxiv.org/pdf/1906.02530.pdf), [local pdf](references/pdfs/ovadia_2019.pdf)
  - Problem: UQ methods can fail under dataset shift.
  - Method: Systematic large-scale shift benchmark.
  - Key result: Calibration degrades OOD despite strong in-domain behavior.
  - Relation: Justifies our shift-suite reliability and threshold diagnostics.

## C) Risk / Safety Decisions Under Thresholds or Budgets

- **Selective Classification for Deep Neural Networks** (Geifman & El-Yaniv, NeurIPS 2017)  
  Links: [paper](https://papers.nips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html), [pdf](https://arxiv.org/pdf/1705.08500.pdf), [local pdf](references/pdfs/selective_classification_2017.pdf)
  - Problem: Control risk by abstaining below confidence thresholds.
  - Method: Selective prediction with risk-coverage metrics.
  - Key result: Formal threshold tradeoff and guarantees.
  - Relation: Mirrors our accept/reject budget diagnostics.

- **Constrained Policy Optimization** (Achiam et al., ICML 2017)  
  Links: [paper](https://proceedings.mlr.press/v70/achiam17a.html), [pdf](https://arxiv.org/pdf/1705.10528.pdf), [local pdf](references/pdfs/cpo_2017.pdf)
  - Problem: Optimize reward under explicit constraints.
  - Method: Trust-region constrained updates.
  - Key result: Better empirical constraint satisfaction.
  - Relation: Supports budgeted safety constraints in control design.

- **Distribution-Free, Risk-Controlling Prediction Sets (RCPS)** (Bates et al., 2021)  
  Links: [paper](https://arxiv.org/abs/2101.02703), [pdf](https://arxiv.org/pdf/2101.02703.pdf), [local pdf](references/pdfs/rcps_2021.pdf)
  - Problem: Guarantee risk control without distribution assumptions.
  - Method: Distribution-free risk-controlling sets.
  - Key result: Finite-sample risk control properties.
  - Relation: Conceptual basis for thresholded risk control with guarantees.

- **Conformal Risk Control (CRC)** (Angelopoulos et al., 2022/2023)  
  Links: [paper](https://arxiv.org/abs/2208.02814), [pdf](https://arxiv.org/pdf/2208.02814.pdf), [local pdf](references/pdfs/crc_2022.pdf)
  - Problem: Enforce user-defined risk levels with statistical validity.
  - Method: Conformal wrappers with risk controls.
  - Key result: Finite-sample validity under exchangeability assumptions.
  - Relation: Relevant for robust risk-budget thresholding beyond simple calibration.

- **Multi-Agent Reachability Calibration with Conformal Prediction** (Muthali et al., 2023)  
  Links: [paper](https://arxiv.org/abs/2304.00432), [pdf](https://arxiv.org/pdf/2304.00432.pdf), [local pdf](references/pdfs/marc_2023.pdf)
  - Problem: Convert uncertain prediction into AV-relevant safety guarantees.
  - Method: Conformal calibration + reachability constraints.
  - Key result: Better calibrated safety envelopes in multi-agent settings.
  - Relation: Closest direct precedent to calibration-to-safety-control bridge.

## D) Closed-Loop AV Benchmarks / Evaluation Substrate

- **Waymax** (Gulino et al., 2023)  
  Links: [paper](https://arxiv.org/abs/2310.08710), [pdf](https://arxiv.org/pdf/2310.08710.pdf), [local pdf](references/pdfs/waymax_2023.pdf)
  - Problem: Scalable closed-loop AV simulation.
  - Method: JAX-accelerated simulator + WOMD ecosystem.
  - Key result: Efficient large-scale research runs.
  - Relation: Core experimental substrate for this project.

- **Waymo Open Sim Agents Challenge (WOSAC)** (Ettinger et al., NeurIPS D&B 2023)  
  Links: [paper](https://papers.nips.cc/paper_files/paper/2023/hash/b96ce67b2f2d45e4ab315e13a6b5b9c5-Abstract-Datasets_and_Benchmarks.html), [pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/b96ce67b2f2d45e4ab315e13a6b5b9c5-Paper-Datasets_and_Benchmarks.pdf), [local pdf](references/pdfs/wosac_2023.pdf)
  - Problem: Standardized interactive simulator benchmark.
  - Method: Challenge protocol + realism-oriented evaluation.
  - Key result: Community benchmark for simulation quality/performance.
  - Relation: External benchmark precedent for closed-loop reporting discipline.

- **nuPlan** (Caesar et al., NeurIPS D&B 2021)  
  Links: [paper](https://arxiv.org/abs/2106.11810), [pdf](https://arxiv.org/pdf/2106.11810.pdf), [local pdf](references/pdfs/nuplan_2021.pdf)
  - Problem: Closed-loop planner benchmarking at scale.
  - Method: Scenario-rich benchmark + planner metrics.
  - Key result: Broad adoption for closed-loop planner evaluation.
  - Relation: Supports our emphasis on closed-loop, not only offline calibration metrics.

---

## Direct Answer to “Have others used proxies?”

Yes. Prior AV and safety-control work commonly uses proxies/surrogates such as:
- **Safety margin proxies**: RSS distance/time margins (Shalev-Shwartz et al., 2017).
- **Reachability risk surrogates**: occlusion/reachability quantification (Huang et al., 2023; Muthali et al., 2023).
- **Uncertainty/confidence surrogates**: multimodal distribution confidence/entropy as decision features (MultiPath, uncertainty-aware MPC papers).
- **Threshold/coverage proxies**: confidence-threshold accept/reject with selective-risk tradeoffs (Geifman & El-Yaniv, 2017).

This validates the core research question in `risk-uq-suite`: whether planner-derived uncertainty/risk proxies are calibrated enough to be decision-grade under thresholds and shift.

## Research Gap Still Open (supports our project)

- Many papers provide either uncertainty prediction, calibration methods, or risk-constrained planning, but fewer provide a **reproducible end-to-end closed-loop pipeline** that demonstrates:
  1. miscalibration evidence,
  2. calibration improvement,
  3. decision-level threshold diagnostics,
  4. closed-loop safety/progress impact under shift,
all in one Waymax-native experimental track.
