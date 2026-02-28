# Risk-UQ Suite: Decision-Causal Core Literature Survey

This version is intentionally narrowed to papers where uncertainty/risk **causes a decision change**.

## Inclusion Criteria Used
1. Explicit decision variable exists (accept/reject, action selection, or constraint satisfaction).
2. Uncertainty/risk directly influences that decision.
3. Prefer closed-loop/control-integrated evidence (allow strong open-loop papers only if decision rule is explicit).
4. Decision-impact outcomes are reported (safety/collision/constraint/efficiency/coverage).

Notes:
- Equations below are written in canonical form for comparability across papers.
- Local PDFs are in `experiments/risk-uq-suite/references/pdfs/`.

## Quick Decision-Core Matrix

| Bucket | Paper | Year | Decision granularity | Decision variable | Why directly relevant |
|---|---|---:|---|---|---|
| Uncertainty -> control | PETS | 2018 | step-level control sequence | `u_{0:H-1}` | Uncertainty in model dynamics directly changes MPC choice. |
| Uncertainty -> control | C-iLQG chance-constrained belief-space planning | 2021 | step-level control sequence | `u_{0:H-1}` with chance constraints | Explicit probabilistic safety constraints in planner optimization. |
| Uncertainty -> control | Predictive Control with uncertain multimodal predictions | 2023 | trajectory/step-level controls | `u_{0:H-1}` under multimodal forecast | Multi-modal predictive uncertainty enters control objective/constraints. |
| Risk proxy / surrogate | RSS formal AV safety model | 2017 | step-level rule | proper response (brake/limit accel) | Safety proxy margin determines action constraints. |
| Risk proxy / surrogate | Potential risk under occluded vision | 2022 | step-level tactical | speed/behavior adjustment | Occlusion risk score drives conservative maneuvering. |
| Risk proxy / surrogate | Occlusion-aware reachability risk + strategy | 2023 | step-level strategy/control | strategy + speed profile | Reachability risk proxy directly gates/penalizes decisions. |
| Threshold / budget | Selective Classification | 2017 | sample/candidate-level accept-reject | accept/reject via threshold | Canonical threshold decision with risk-coverage tradeoff. |
| Threshold / budget | Constrained Policy Optimization (CPO) | 2017 | policy-level (closed-loop) | policy update under safety budget | Explicit budgeted safety decision in control optimization. |
| Threshold / budget | Conformal Risk Control (CRC) | 2022 | sample/set decision | threshold/parameter `lambda` for risk target | Risk target (`alpha`) directly controls decision sets. |
| Closed-loop benchmarks | Waymax | 2023 | closed-loop step-level rollouts | evaluated policy actions `a_t` | Simulation substrate for controlled decision-impact studies. |
| Closed-loop benchmarks | WOSAC | 2023 | scenario-level closed-loop | agent policy outputs | Standardized closed-loop AV evaluation protocol. |
| Closed-loop benchmarks | nuPlan | 2021 | scenario-level closed-loop | planner actions along route | Widely used closed-loop AV planning benchmark. |
| Calibration background | Guo et al. (Temperature scaling) | 2017 | probability calibration | scalar `T` | Canonical post-hoc calibration baseline. |
| Calibration background | Niculescu-Mizil & Caruana (Platt/isotonic) | 2005/2012 | probability calibration | sigmoid/isotonic map | Canonical score->probability calibration comparison. |

---

## 1) Uncertainty-to-Control Integration (max 3)

### 1.1 PETS: Deep RL in a Handful of Trials using Probabilistic Dynamics Models (NeurIPS 2018)
Links: [paper](https://papers.nips.cc/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html), [pdf](https://arxiv.org/pdf/1805.12114.pdf), [local pdf](references/pdfs/pets_2018.pdf)
- Decision rule equation:  
  \(u_{0:H-1}^* = \arg\min_{u_{0:H-1}} \mathbb{E}_{\hat p(x_{t+1}|x_t,u_t)}\left[\sum_t \ell(x_t,u_t)\right]\) with ensemble uncertainty propagation.
- Risk/uncertainty signal: ensemble predictive uncertainty of dynamics.
- Where it enters planner: directly inside MPC rollout evaluation.
- Decision-level outcomes: task success and sample efficiency vs model-free baselines.
- Key assumptions: learned dynamics are locally valid; uncertainty estimates are informative.
- Limitation vs our setting: not AV-specific candidate-action safety thresholding; no explicit `tau` reject/fallback logic.
- Granularity: step-level control-sequence optimization.

### 1.2 Constrained iLQG for Chance-Constrained Gaussian Belief-Space Planning (2021)
Links: [paper](https://arxiv.org/abs/2108.06533), [pdf](https://arxiv.org/pdf/2108.06533.pdf), [local pdf](references/pdfs/cilqg_2021.pdf)
- Decision rule equation:  
  \(\min_{u_{0:H-1}} \mathbb{E}[\sum_t \ell(x_t,u_t)]\) s.t. \(\Pr(g_j(x_t,u_t) \le 0) \ge 1-\delta_j\).
- Risk/uncertainty signal: Gaussian belief-state uncertainty.
- Where it enters planner: chance constraints in trajectory optimization.
- Decision-level outcomes: constraint satisfaction and control performance.
- Key assumptions: approximately Gaussian beliefs; local linearization quality.
- Limitation vs our setting: assumptions weaker for multimodal AV interactions; no candidate-level discrete reranking.
- Granularity: trajectory/step-level controls.

### 1.3 Predictive Control for Autonomous Driving with Uncertain, Multi-modal Predictions (2023)
Links: [paper](https://arxiv.org/abs/2310.20561), [pdf](https://arxiv.org/pdf/2310.20561.pdf), [local pdf](references/pdfs/predictive_control_multimodal_2023.pdf)
- Decision rule equation (canonical form):  
  \(u^* = \arg\min_u \sum_m w_m \; J(u; \hat\tau^{(m)})\) (often with risk/robust penalties).
- Risk/uncertainty signal: multimodal forecast weights and trajectory dispersion.
- Where it enters planner: objective and/or safety terms of predictive controller.
- Decision-level outcomes: safety-performance tradeoff in interactive scenes.
- Key assumptions: forecast mode probabilities are useful and sufficiently calibrated.
- Limitation vs our setting: generally trajectory-level optimization; not explicit candidate-level `tau` acceptance diagnostics.
- Granularity: trajectory-level (applied step-wise in closed-loop).

---

## 2) Risk Proxy / Surrogate Safety Signals (max 3)

### 2.1 RSS: On a Formal Model of Safe and Scalable Self-driving Cars (2017)
Links: [paper](https://arxiv.org/abs/1708.06374), [pdf](https://arxiv.org/pdf/1708.06374.pdf), [local pdf](references/pdfs/rss_2017.pdf)
- Decision rule equation (longitudinal canonical form):  
  dangerous if \(d < d_{\text{safe}}(v_r,v_f,\rho,a_{\max},b_{\min},b_{\max})\); then enforce proper response.
- Risk/uncertainty signal: surrogate safety margin proxy (safe distance/time margin).
- Where it enters planner: rule-based action constraints (brake/accel bounds).
- Decision-level outcomes: formal safety compliance and rule-consistent behavior.
- Key assumptions: bounded response/braking model captures relevant safety dynamics.
- Limitation vs our setting: deterministic conservative surrogate; no probabilistic calibration or shift diagnostics.
- Granularity: step-level action constraints.

### 2.2 Potential Risk Assessment under Occluded Vision (Scientific Reports 2022)
Links: [paper](https://www.nature.com/articles/s41598-022-08810-z), [pdf](https://www.nature.com/articles/s41598-022-08810-z.pdf), [local pdf](references/pdfs/potential_risk_occluded_vision_2022.pdf)
- Decision rule equation (canonical):  
  \(u^* = \arg\min_u J_{\text{drive}}(u) + \lambda R_{\text{occ}}(u)\).
- Risk/uncertainty signal: occlusion-derived potential risk proxy.
- Where it enters planner: additional risk term in tactical/control decision.
- Decision-level outcomes: safer behavior under occlusions.
- Key assumptions: occlusion risk proxy correlates with true hazard.
- Limitation vs our setting: proxy validity/calibration under shift not deeply quantified.
- Granularity: step-level tactical/behavior decision.

### 2.3 Occlusion-aware Risk Assessment using Simplified Reachability (2023)
Links: [paper](https://arxiv.org/abs/2306.07004), [pdf](https://arxiv.org/pdf/2306.07004.pdf), [local pdf](references/pdfs/occlusion_reachability_risk_2023.pdf)
- Decision rule equation (canonical):  
  \(u^* = \arg\min_u J_{\text{efficiency}}(u) + \lambda R_{\text{reach}}(u)\) or reject if \(R_{\text{reach}} > \tau\).
- Risk/uncertainty signal: reachability-based risk surrogate from occluded hypotheses.
- Where it enters planner: cost penalty or gating condition.
- Decision-level outcomes: reduced hazardous interactions in occluded settings.
- Key assumptions: simplified reachability approximates true future occupancy risk.
- Limitation vs our setting: surrogate model mismatch can dominate; candidate-level calibration not central.
- Granularity: step-level strategy and control.

---

## 3) Threshold/Budgeted Decision Frameworks (max 3)

### 3.1 Selective Classification for DNNs (NeurIPS 2017)
Links: [paper](https://papers.nips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html), [pdf](https://arxiv.org/pdf/1705.08500.pdf), [local pdf](references/pdfs/selective_classification_2017.pdf)
- Decision rule equation:  
  accept if \(s(x) \ge \theta\), else abstain; optimize risk at target coverage.
- Risk/uncertainty signal: confidence score (often max prob).
- Where it enters planner: accept/reject policy (analogous to candidate filtering).
- Decision-level outcomes: risk-coverage tradeoff curves.
- Key assumptions: score ranking is aligned with correctness risk.
- Limitation vs our setting: static i.i.d. setting; no closed-loop feedback.
- Granularity: sample/candidate-level decision.

### 3.2 Constrained Policy Optimization (ICML 2017)
Links: [paper](https://proceedings.mlr.press/v70/achiam17a.html), [pdf](https://arxiv.org/pdf/1705.10528.pdf), [local pdf](references/pdfs/cpo_2017.pdf)
- Decision rule equation:  
  \(\max_\pi J(\pi)\) s.t. \(J_C(\pi) \le d\).
- Risk/uncertainty signal: expected constraint cost as safety budget term.
- Where it enters planner: policy update objective with hard budget.
- Decision-level outcomes: improved constraint satisfaction with competitive reward.
- Key assumptions: on-policy estimates sufficiently accurate; constraint surrogate meaningful.
- Limitation vs our setting: policy-level RL framework, not per-step candidate reranking with calibrated probabilities.
- Granularity: policy-level closed-loop control.

### 3.3 Conformal Risk Control (2022/2023)
Links: [paper](https://arxiv.org/abs/2208.02814), [pdf](https://arxiv.org/pdf/2208.02814.pdf), [local pdf](references/pdfs/crc_2022.pdf)
- Decision rule equation (canonical):  
  choose \(\hat\lambda\) so empirical calibration risk \(\hat R_{\text{cal}}(\lambda)\le\alpha\), then deploy decision set/rule at \(\hat\lambda\).
- Risk/uncertainty signal: conformalized nonconformity/risk scores.
- Where it enters planner: threshold parameter controlling acceptance region size/risk.
- Decision-level outcomes: target risk control with finite-sample guarantees.
- Key assumptions: exchangeability (or suitable approximation).
- Limitation vs our setting: strict exchangeability can break in adaptive closed-loop interaction.
- Granularity: sample/set-level threshold decision.

---

## 4) Closed-Loop AV Benchmarks (max 3)

### 4.1 Waymax (2023)
Links: [paper](https://arxiv.org/abs/2310.08710), [pdf](https://arxiv.org/pdf/2310.08710.pdf), [local pdf](references/pdfs/waymax_2023.pdf)
- Decision rule equation: evaluated policy uses \(a_t = \pi(o_t)\) in closed-loop simulator dynamics.
- Risk/uncertainty signal used: framework-level (paper is infrastructure, not one fixed risk rule).
- Where it enters planner: user-defined policy/control module in simulation loop.
- Decision-level outcomes: closed-loop rollout metrics and scale/reproducibility properties.
- Key assumptions: simulator realism is adequate for comparative research conclusions.
- Limitation vs our setting: does not prescribe calibrated risk decision protocols itself.
- Granularity: step-level policy action in closed-loop rollouts.

### 4.2 WOSAC (Waymo Open Sim Agents Challenge, 2023)
Links: [paper](https://papers.nips.cc/paper_files/paper/2023/hash/b96ce67b2f2d45e4ab315e13a6b5b9c5-Abstract-Datasets_and_Benchmarks.html), [pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/b96ce67b2f2d45e4ab315e13a6b5b9c5-Paper-Datasets_and_Benchmarks.pdf), [local pdf](references/pdfs/wosac_2023.pdf)
- Decision rule equation: agent policy outputs per-step actions in interactive simulation.
- Risk/uncertainty signal used: benchmark allows any method; uncertainty use varies by participant.
- Where it enters planner: participant-defined policy internals.
- Decision-level outcomes: realism/simulation performance metrics over closed-loop scenarios.
- Key assumptions: benchmark metrics align with deployment-relevant quality.
- Limitation vs our setting: not focused on calibration-to-threshold decision diagnostics.
- Granularity: scenario-level aggregated closed-loop performance.

### 4.3 nuPlan (2021)
Links: [paper](https://arxiv.org/abs/2106.11810), [pdf](https://arxiv.org/pdf/2106.11810.pdf), [local pdf](references/pdfs/nuplan_2021.pdf)
- Decision rule equation: planner policy produces trajectory/actions in closed-loop stack.
- Risk/uncertainty signal used: method-dependent (benchmark supports many planner classes).
- Where it enters planner: planner internals and scenario-level evaluation.
- Decision-level outcomes: safety, comfort, progress, rule compliance metrics.
- Key assumptions: scenario set sufficiently captures relevant driving complexity.
- Limitation vs our setting: no built-in causal decomposition of calibration vs controller bottleneck.
- Granularity: scenario-level closed-loop planning outcomes.

---

## Canonical Calibration Background (1–2 papers)

### C1) On Calibration of Modern Neural Networks (Guo et al., 2017)
Links: [paper](https://proceedings.mlr.press/v70/guo17a.html), [pdf](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), [local pdf](references/pdfs/guo_2017.pdf)
- Equation: \(p_{\text{cal}} = \text{softmax}(z/T)\), fit `T` on validation NLL.
- Why background: establishes that raw confidence can be miscalibrated even when accuracy is good.

### C2) Predicting Good Probabilities with Supervised Learning (Niculescu-Mizil & Caruana)
Links: [paper](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf), [pdf](https://arxiv.org/pdf/1207.1403.pdf), [local pdf](references/pdfs/obtaining_calibrated_probabilities_boosting_2012.pdf)
- Equation (Platt): \(p = \sigma(a s + b)\).
- Why background: canonical evidence that score-to-probability mapping strongly affects decision quality.

---

## Common Assumptions Across Decision-Causal Papers

1. **Ranking assumption**: higher uncertainty/risk score corresponds to higher failure probability.
2. **Model-form assumption**: Gaussian/GMM or limited-mode approximations are adequate for decision constraints.
3. **Stationarity/exchangeability assumption**: calibration set statistics transfer to deployment/evaluation conditions.
4. **Proxy sufficiency assumption**: surrogate risk captures enough of true safety-critical outcomes.
5. **Metric alignment assumption**: optimized surrogate objective aligns with actual safety-progress goals.

## Where These Methods Can Fail in *Our* Setting
Target setting: **candidate-level selection + `tau` threshold + closed-loop + shift**.

1. **Candidate-level mismatch**: many methods optimize trajectory-level or policy-level decisions, not per-candidate reranking at each step.
2. **Threshold fragility**: even calibrated global probabilities can fail at local operating threshold (`tau`) due to sparse accepted samples.
3. **Closed-loop distribution drift**: planner decisions alter future states, breaking static calibration assumptions.
4. **Shift amplification**: uncertainty estimates calibrated on nominal may become anti-conservative or over-conservative on shift suites.
5. **Proxy aliasing**: entropy/top1/reachability proxies may correlate with complexity rather than true failure risk for certain scenario classes.
6. **Feasible-set collapse**: strict `tau` can drive accept-rate to zero, turning controller into fallback policy regardless of score quality.

## Precise Research Gap Statement

Existing literature establishes uncertainty-aware control, proxy-based risk surrogates, and threshold/budget decision frameworks, but lacks a standardized **closed-loop, candidate-level, calibration-to-decision causal evaluation protocol** under distribution shift.

Specifically, the missing piece is an end-to-end methodology that jointly quantifies:
1. whether planner-derived risk proxies are probability-calibrated,
2. whether miscalibration causes decision errors at a fixed operating threshold (`false-safe` vs `safe-reject`),
3. whether correcting calibration improves feasible-set/fallback behavior and safety-progress tradeoff in closed-loop,
4. how these effects change across controlled shift suites.

This is exactly the target of the `risk-uq-suite` track.
