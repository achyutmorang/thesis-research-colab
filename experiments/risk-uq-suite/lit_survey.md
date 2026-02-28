# Risk-UQ Suite Literature Survey (Methodology-Centric, Top-Venue Core)

This survey is narrowed to papers that directly inform a stronger method for:
1. converting uncertainty into a risk signal,
2. using that signal in a decision rule,
3. enforcing safety via thresholds/constraints,
4. validating outcomes in closed-loop settings.

Venue scope (as requested): **NeurIPS, ICLR, ICML, ICRA, CoRL, CVPR, RSS**.

Local PDFs: `experiments/risk-uq-suite/references/pdfs/`.

## Screening Criteria

A paper is included in the core set only if it has:
1. an explicit decision variable (action selection, accept/reject, or constrained optimization),
2. uncertainty/risk that directly changes the decision,
3. reported decision-impact outcomes (safety, collision, violation, efficiency, or coverage-risk).

---

## Core Paper Set (14 papers)

| Title | Venue | Year | Bucket | Decision Granularity | Why it matters for our method gap |
|---|---|---:|---|---|---|
| PETS: Deep RL in a Handful of Trials using Probabilistic Dynamics Models | NeurIPS | 2018 | Uncertainty->Control | Step/trajectory | Canonical uncertainty-aware MPC without calibrated decision risk. |
| MATS: An Interpretable Trajectory Forecasting Representation for Planning and Control | CoRL | 2020 | Uncertainty->Control | Trajectory -> action | Joint forecasting-planning; uncertainty enters planning scores. |
| RAP: Risk-Aware Prediction for Planning | CoRL | 2022 | Uncertainty->Control | Candidate/trajectory | Explicit risk-sensitive planning objective under multimodality. |
| Hierarchical Game-Theoretic Planning for Autonomous Vehicles | ICRA | 2019 | Risk Proxy / Surrogate | Step/trajectory | Interaction risk enters planner utility via surrogate terms. |
| Safe Occlusion-Aware Autonomous Driving via Interactive Game-Theoretic Active Perception Planning | RSS | 2021 | Risk Proxy / Surrogate | Step-level | Occlusion-induced uncertainty converted to risk penalties/constraints. |
| MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction | CoRL | 2020 | Risk Proxy / Surrogate | Trajectory-level | Multimodal probabilities used downstream as planning risk surrogates. |
| Selective Classification for Deep Neural Networks | NeurIPS | 2017 | Threshold/Budget | Candidate-level accept/reject | Direct tau-like threshold framework for decision risk. |
| Constrained Policy Optimization (CPO) | ICML | 2017 | Threshold/Budget | Policy-level | Explicit safety budget constraints in closed-loop control. |
| Distribution-Free, Risk-Controlling Prediction Sets (RCPS) | NeurIPS | 2021 | Threshold/Budget | Sample/set-level | Risk control under target alpha with finite-sample logic. |
| CARLA: An Open Urban Driving Simulator | CoRL | 2017 | Closed-loop Benchmarks | Closed-loop step-level | Widely used AV closed-loop decision evaluation substrate. |
| SafeBench: A Benchmarking Platform for Safety Evaluation of Autonomous Vehicles | NeurIPS (Datasets/Benchmarks) | 2022 | Closed-loop Benchmarks | Closed-loop scenario-level | Stress tests and safety-centric evaluation under challenging conditions. |
| Waymo Open Sim Agents Challenge (WOSAC) | NeurIPS (Datasets/Benchmarks) | 2023 | Closed-loop Benchmarks | Closed-loop scenario-level | Standardized interactive closed-loop AV benchmark protocol. |
| On Calibration of Modern Neural Networks | ICML | 2017 | Calibration Background | Probability mapping | Canonical post-hoc temperature scaling. |
| Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift | NeurIPS | 2019 | Calibration Background | Probability reliability under shift | Shows calibration collapse under shift. |

---

## 1) Uncertainty-to-Control Integration

### 1.1 PETS (NeurIPS 2018)
Links: [paper](https://papers.nips.cc/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html), [pdf](https://arxiv.org/pdf/1805.12114.pdf), [local pdf](references/pdfs/pets_2018.pdf)
- Risk definition: predictive uncertainty from probabilistic ensemble dynamics.
- Uncertainty use: enters rollout evaluation in MPC objective.
- Decision rule: \(u^*_{0:H-1}=\arg\min_{u_{0:H-1}} \mathbb{E}_{\hat p(x'|x,u)}\left[\sum_t \ell(x_t,u_t)\right]\).
- Calibrated probability assumption: yes (implicitly); no explicit post-hoc calibration of safety risk.
- Threshold/constraint usage: generally soft optimization, no explicit tau gating.
- Evaluation type: closed-loop control.
- Key limitation vs our setting: no candidate-level tau decision diagnostics; no calibration->decision causal test.

### 1.2 MATS (CoRL 2020)
Links: [paper](https://proceedings.mlr.press/v155/ivanovic21a.html), [pdf](https://arxiv.org/pdf/2009.07517.pdf), [local pdf](references/pdfs/mats_corl_2020.pdf)
- Risk definition: multimodal predictive distribution over other-agent futures.
- Uncertainty use: weighted plan evaluation under forecast modes.
- Decision rule (canonicalized): \(a^*=\arg\min_a \sum_m w_m C(a,\tau^{(m)})\).
- Calibrated probability assumption: yes (mode weights treated as decision-relevant).
- Threshold/constraint usage: mostly cost-based, not explicit tau accept/reject.
- Evaluation type: prediction + control integration (closed-loop style planning impact).
- Key limitation vs our setting: does not isolate probability calibration quality from control policy quality.

### 1.3 RAP (CoRL 2022)
Links: [paper](https://openreview.net/forum?id=8r5Q2q7s8Q), [pdf](https://arxiv.org/pdf/2210.01368.pdf), [local pdf](references/pdfs/rap_corl_2022.pdf)
- Risk definition: risk-aware objective over uncertain multi-agent futures.
- Uncertainty use: risk-sensitive terms in planning objective.
- Decision rule (canonicalized): \(a^*=\arg\min_a \mathbb{E}[C(a)] + \lambda\,\mathrm{Risk}(C(a))\) (e.g., CVaR-like emphasis on bad outcomes).
- Calibrated probability assumption: yes (risk score treated as decision-grade).
- Threshold/constraint usage: mostly objective shaping, not tau-threshold gating.
- Evaluation type: closed-loop planner comparisons.
- Key limitation vs our setting: no explicit calibration-to-threshold decision correctness analysis.

---

## 2) Risk Proxy / Surrogate Safety Signals

### 2.1 Hierarchical Game-Theoretic Planning for AVs (ICRA 2019)
Links: [paper](https://arxiv.org/abs/1810.05766), [pdf](https://arxiv.org/pdf/1810.05766.pdf), [local pdf](references/pdfs/hierarchical_game_theoretic_icra_2019.pdf)
- Risk definition: surrogate interaction risk in game-theoretic utilities (e.g., collision/proximity penalties).
- Uncertainty use: uncertainty over human responses enters strategic utility.
- Decision rule (canonicalized): \(u_e^*=\arg\max_{u_e} U_e(u_e, u_h^*(u_e))\).
- Calibrated probability assumption: not probability-calibrated; surrogate utilities assumed meaningful.
- Threshold/constraint usage: safety via utility penalties/constraints, not probabilistic tau gating.
- Evaluation type: closed-loop interaction scenarios.
- Key limitation vs our setting: risk is not an explicit calibrated probability for candidate accept/reject.

### 2.2 Safe Occlusion-Aware Active Perception Planning (RSS 2021)
Links: [paper](https://roboticsconference.org/2021/program/papers/033/index.html), [pdf](https://arxiv.org/pdf/2105.08169.pdf), [local pdf](references/pdfs/safe_occlusion_active_perception_rss_2021.pdf)
- Risk definition: occlusion-induced hazard surrogate (reachable hidden-agent risk).
- Uncertainty use: active perception balances exploration and safety under partial observability.
- Decision rule (canonicalized): \(a^*=\arg\max_a R_{task}(a)-\lambda R_{occ}(a)\), with safety guard constraints.
- Calibrated probability assumption: yes (surrogate risk treated as valid control signal).
- Threshold/constraint usage: yes, typically via safety guards/constraints.
- Evaluation type: closed-loop AV interactions under occlusion.
- Key limitation vs our setting: surrogate-risk calibration and threshold decision error are not explicitly audited.

### 2.3 MultiPath (CoRL 2020)
Links: [paper](https://proceedings.mlr.press/v100/chai20a.html), [pdf](https://proceedings.mlr.press/v100/chai20a/chai20a.pdf), [local pdf](references/pdfs/multipath_2020.pdf)
- Risk definition: multimodal trajectory probabilities (surrogate for downstream interaction risk).
- Uncertainty use: forecast mode weights and dispersion become planner-side uncertainty features.
- Decision rule: predictor outputs \(\{(w_k,\tau_k)\}_{k=1}^K\); downstream planners typically minimize \(\sum_k w_k C(a,\tau_k)\).
- Calibrated probability assumption: usually yes; forecast probabilities often consumed directly.
- Threshold/constraint usage: indirect (depends on downstream planner).
- Evaluation type: mostly open-loop prediction metrics with downstream planning relevance.
- Key limitation vs our setting: weak direct evidence on candidate-level threshold decisions in closed-loop.

---

## 3) Threshold / Budgeted Decision Frameworks

### 3.1 Selective Classification (NeurIPS 2017)
Links: [paper](https://papers.nips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html), [pdf](https://arxiv.org/pdf/1705.08500.pdf), [local pdf](references/pdfs/selective_classification_2017.pdf)
- Risk definition: error risk conditioned on acceptance coverage.
- Uncertainty use: confidence score directly drives abstain/accept decision.
- Decision rule: accept if \(s(x)\ge\theta\), reject otherwise.
- Calibrated probability assumption: often implicit; score-quality governs decision quality.
- Threshold/constraint usage: yes (explicit threshold theta).
- Evaluation type: open-loop decision risk/coverage.
- Key limitation vs our setting: static i.i.d. setting; no closed-loop feedback or planner fallback dynamics.

### 3.2 Constrained Policy Optimization (ICML 2017)
Links: [paper](https://proceedings.mlr.press/v70/achiam17a.html), [pdf](https://arxiv.org/pdf/1705.10528.pdf), [local pdf](references/pdfs/cpo_2017.pdf)
- Risk definition: expected safety cost \(J_C(\pi)\).
- Uncertainty use: enters via stochastic policy optimization under constraint estimates.
- Decision rule: \(\max_\pi J_R(\pi)\) s.t. \(J_C(\pi)\le d\).
- Calibrated probability assumption: no explicit calibration layer.
- Threshold/constraint usage: yes (hard budget \(d\)).
- Evaluation type: closed-loop constrained RL.
- Key limitation vs our setting: policy-level optimization, not per-step candidate-level tau filtering with calibrated probabilities.

### 3.3 RCPS (NeurIPS 2021)
Links: [paper](https://papers.nips.cc/paper_files/paper/2021/hash/32c0fdfc4f8f5f8f2f9ebf58f4a6ef08-Abstract.html), [pdf](https://arxiv.org/pdf/2101.02703.pdf), [local pdf](references/pdfs/rcps_2021.pdf)
- Risk definition: user-chosen risk functional controlled below \(\alpha\).
- Uncertainty use: nonconformity/risk scores define prediction sets.
- Decision rule: choose \(\lambda\) on calibration data so \(\hat R(\lambda)\le\alpha\), deploy set/rule with \(\lambda\).
- Calibrated probability assumption: replaces strict calibration with finite-sample risk control assumptions.
- Threshold/constraint usage: yes (alpha-level risk budget).
- Evaluation type: mostly open-loop statistical risk control.
- Key limitation vs our setting: exchangeability assumptions can fail under adaptive closed-loop AV rollout.

---

## 4) Closed-Loop AV Benchmarks (Decision-Impact Evaluation Substrate)

### 4.1 CARLA (CoRL 2017)
Links: [paper](https://proceedings.mlr.press/v78/dosovitskiy17a.html), [pdf](https://proceedings.mlr.press/v78/dosovitskiy17a/dosovitskiy17a.pdf), [local pdf](references/pdfs/carla_corl_2017.pdf)
- Risk definition: benchmark metrics (collisions, infractions, route completion) rather than one fixed probability model.
- Uncertainty use: method-dependent (participants).
- Decision rule: generic closed-loop policy \(a_t=\pi(o_t)\).
- Calibrated probability assumption: benchmark does not enforce calibration checks.
- Threshold/constraint usage: task metrics/violations; no standard tau-risk protocol.
- Evaluation type: closed-loop AV simulation.
- Key limitation vs our setting: no explicit calibration->decision correctness analysis.

### 4.2 SafeBench (NeurIPS D&B 2022)
Links: [paper](https://openreview.net/forum?id=ANQ3LafS8h), [pdf](https://arxiv.org/pdf/2206.09655.pdf), [local pdf](references/pdfs/safebench_2022.pdf)
- Risk definition: attack/stress scenario outcomes as safety failures.
- Uncertainty use: benchmark allows uncertainty-aware agents but does not prescribe calibration protocol.
- Decision rule: policy-dependent closed-loop action generation under stressors.
- Calibrated probability assumption: not a benchmark requirement.
- Threshold/constraint usage: typically method-defined; benchmark-level constraints are task metrics.
- Evaluation type: closed-loop robustness/safety evaluation.
- Key limitation vs our setting: no standardized candidate-level tau decision audit.

### 4.3 WOSAC (NeurIPS D&B 2023)
Links: [paper](https://papers.nips.cc/paper_files/paper/2023/hash/b96ce67b2f2d45e4ab315e13a6b5b9c5-Abstract-Datasets_and_Benchmarks.html), [pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/b96ce67b2f2d45e4ab315e13a6b5b9c5-Paper-Datasets_and_Benchmarks.pdf), [local pdf](references/pdfs/wosac_2023.pdf)
- Risk definition: simulation realism/safety outcomes in interactive rollouts.
- Uncertainty use: method-dependent.
- Decision rule: policy-defined closed-loop action sequence.
- Calibrated probability assumption: no explicit requirement.
- Threshold/constraint usage: method-dependent.
- Evaluation type: large-scale closed-loop benchmark.
- Key limitation vs our setting: benchmark does not isolate probability calibration effects on decision correctness.

---

## Canonical Calibration Background (for methodological grounding)

### C1) Guo et al., ICML 2017 (Temperature Scaling)
Links: [paper](https://proceedings.mlr.press/v70/guo17a.html), [pdf](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), [local pdf](references/pdfs/guo_2017.pdf)
- Core equation: \(p_{cal}=\mathrm{softmax}(z/T)\), with \(T\) fit on validation NLL.
- Relevance: simplest strong baseline for post-hoc calibration in our pipeline.

### C2) Ovadia et al., NeurIPS 2019 (Calibration Under Shift)
Links: [paper](https://papers.nips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html), [pdf](https://arxiv.org/pdf/1906.02530.pdf), [local pdf](references/pdfs/ovadia_2019.pdf)
- Core finding: in-domain calibration can degrade substantially under distribution shift.
- Relevance: directly motivates our shift-suite calibration and decision-audit protocol.

---

## Critical Methodological Analysis

## 1) Do existing methods ensure decision-grade risk?

Short answer: **rarely**.
- Most uncertainty-aware planners optimize with surrogate uncertainty or risk penalties but do not verify that risk values are calibrated for threshold decisions.
- Threshold/budget methods provide guarantees in their own assumptions (e.g., exchangeability, stationary data), but those assumptions are often violated in closed-loop interactive driving.

## 2) Do papers evaluate calibration -> decision correctness causally?

Short answer: **almost never in AV closed-loop candidate selection**.
- Calibration papers evaluate probability quality (ECE/NLL/Brier), usually not closed-loop controller outcomes.
- Planning papers evaluate collisions/progress, usually without a controlled calibration intervention.
- Missing bridge: proving that fixing calibration specifically reduces false-safe and safe-reject decision errors at a fixed tau in closed-loop.

## 3) Comparison of approach families

| Family | Typical risk definition | Strength | Limitation for our setting |
|---|---|---|---|
| Heuristic proxies | entropy, top-1 weight, surrogate margins, reachability scores | cheap and deployable | often uncalibrated; threshold behavior unstable under shift |
| Learned risk models | explicit \(P(\text{failure}|x)\) or score heads | can be discriminative and adaptable | usually evaluated open-loop; calibration under shift often weak |
| Constrained optimization / budgets | chance constraints, expected safety cost, alpha-risk sets | direct safety-efficiency tradeoff control | often policy/trajectory-level; limited candidate-level tau auditing |

## 4) Where existing methods fail for our target setting

Target setting: **candidate-level selection + tau-threshold + closed-loop execution + distribution shift**.

1. Candidate-level selection gap: most works optimize trajectories/policies, not per-step candidate reranking with explicit accept/reject.
2. Tau-threshold decision gap: few papers report false-safe and safe-reject rates at operational tau.
3. Closed-loop causal gap: calibration is rarely linked causally to downstream control outcomes.
4. Shift robustness gap: calibration and budget behavior under controlled shift suites are under-reported.

## 5) Why existing methods are insufficient

Existing methods typically assume the risk signal is already trustworthy for decisions, but they do not jointly validate:
1. probability reliability,
2. threshold decision correctness,
3. closed-loop impact,
4. robustness under shift,
in one unified protocol.

## 6) What kind of method is missing

A method that couples:
1. candidate-level risk estimation,
2. post-hoc (or adaptive) calibration targeted to decision operating points,
3. tau-budgeted selection with explicit fallback accounting,
4. closed-loop, shift-aware causal evaluation.

---

## Refined Methodology Gap (for paper framing)

Current literature demonstrates uncertainty-aware planning and risk-constrained optimization, but lacks an end-to-end **decision-grade risk pipeline** for closed-loop AV candidate selection. Specifically, prior work does not systematically test whether calibrated probabilities improve thresholded decisions (false-safe vs safe-reject) and translate into better closed-loop safety-efficiency tradeoffs under distribution shift.

---

## Promising Method Directions Suggested by the Literature

1. **Decision-aware calibration at operating tau**
- Train/calibrate not only for global NLL/ECE but also for decision metrics at tau (false-safe, safe-reject, feasible-set rate).

2. **Conformalized risk-budget gating for candidate selection**
- Combine calibrated probabilities with alpha-controlled conformal thresholds to reduce unsafe acceptance under shift.

3. **Controller bottleneck decomposition (oracle analysis)**
- Separate failures due to risk estimation vs candidate set quality vs controller rule to prove causal source of failure.

---

## Explicit Statement for the Paper

**Existing methods assume uncertainty-derived risk is decision-ready, but fail when candidate-level tau-threshold decisions are made in closed-loop under shift, motivating a method that calibrates risk for decision correctness and evaluates calibration->decision->outcome causally.**
