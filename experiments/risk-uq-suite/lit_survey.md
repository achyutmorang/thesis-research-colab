# Risk-UQ Suite Literature Survey (Decision-Core, Citation-Strengthened)

This pass tightens the survey to maximize citation quality and methodological support for our exact claim.

## Scope

- Focus: papers where uncertainty/risk directly changes a decision variable.
- Decision variables considered valid: `action selection`, `threshold accept/reject`, `constraint satisfaction`, `fallback/abstain`.
- Venue focus: NeurIPS, ICLR, ICML, ICRA, CoRL, CVPR, RSS (plus benchmark tracks where needed).

## Operational Definition: Decision-Grade Risk

We define a risk score as **decision-grade** if, at the operating threshold `tau`, it supports correct candidate-level choices in closed-loop:
1. calibrated near `tau` (probability meaning preserved),
2. discriminative across candidates (ranking quality),
3. robust under shift (no collapse/flip in behavior),
4. decision-useful (`false_safe` down without infeasible conservatism/fallback explosion).

Scope qualifier for claims in this survey:
- We target **candidate-level tau-threshold reranking in closed-loop AV simulation under shift**. This is narrower than generic risk-aware control.

---

## Curated Portfolio (22 Papers)

- Recent SOTA (last ~1-4 years): 6
- Canonical/foundational: 11
- Closest prior to our method setting: 3
- Benchmark substrate papers: 3

## Master Evidence Table

| ID | Paper | Venue/Year | Citation Strength | Role | Decision Rule (canonical) | How uncertainty/risk enters decision | Assumptions | Evaluation | Limitation vs our setting |
|---|---|---|---|---|---|---|---|---|---|
| P01 | On Calibration of Modern Neural Networks | ICML 2017 | Very high | Canonical | `p_cal = softmax(z/T)`, `T` fit on val-NLL | Probability map used by downstream thresholds | Calibration split representative, quasi-IID | Open-loop (ECE/NLL/Brier) | No closed-loop causal effect on control |
| P02 | Predicting Good Probabilities with Supervised Learning | ICML 2005 | High | Canonical | `p = sigma(a s + b)` (Platt) / isotonic map | Score-to-probability conversion changes threshold decisions | Monotonic score-risk relation | Open-loop | No shift/closed-loop analysis |
| P03 | Dropout as a Bayesian Approximation | ICML 2016 | Very high | Canonical | `a* = argmin_a E[C(a)]` under MC-dropout | Epistemic uncertainty from stochastic passes | Dropout approximates posterior uncertainty | Open-loop + control demos | No calibration-to-decision audit |
| P04 | Deep Ensembles | NeurIPS 2017 | Very high | Canonical | `p = (1/M) sum_m p_m`; decisions from mean/variance | Ensemble spread as uncertainty signal | Ensemble diversity approximates epistemic | Open-loop UQ/OOD | No tau-threshold decision metrics |
| P05 | Can You Trust Your Model's Uncertainty Under Shift? | NeurIPS 2019 | High | Canonical | Reliability under controlled shifts | OOD miscalibration diagnosis | Shift benchmarks approximate deployment | Open-loop shift benchmarks | No planner/action-level decision loop |
| P06 | Beyond Temperature Scaling: Dirichlet Calibration | NeurIPS 2019 | High | Canonical | `p_cal = DirichletMap(p_raw; theta)` | Alternative calibration map for decisions | Val distribution representative | Open-loop calibration | No closed-loop safety impact analysis |
| P07 | Selective Classification for DNNs | NeurIPS 2017 | High | Canonical | Accept if `s(x) >= theta`, reject otherwise | Confidence directly controls abstain/accept | Ranking quality of confidence | Open-loop risk-coverage | No sequential control coupling |
| P08 | SelectiveNet | ICML 2019 | Medium | Canonical | Joint predictor-selector with target coverage | Uncertainty integrated in selector head | Coverage surrogate transfers OOD | Open-loop selective prediction | No AV closed-loop shift study |
| P09 | Deep Gamblers | NeurIPS 2019 | Medium | Canonical | Bet/reject via reservation utility | Confidence controls abstention behavior | Utility captures risk preference | Open-loop abstention metrics | No control-level causal chain |
| P10 | Constrained Policy Optimization (CPO) | ICML 2017 | High | Canonical | `max_pi J_R(pi)` s.t. `J_C(pi) <= d` | Safety budget constrains policy decision updates | Cost surrogate correctly encodes safety | Closed-loop RL constraints | Policy-level, not candidate-level tau gating |
| P11 | RCPS: Distribution-Free Risk-Controlling Prediction Sets | NeurIPS 2021 | Medium | Canonical | choose `lambda` s.t. `R_hat(lambda)<=alpha` | Alpha-risk budget controls acceptance set | Exchangeability | Open-loop finite-sample guarantees | Exchangeability fails in adaptive closed-loop |
| P12 | Conformal Risk Control (CRC) | 2022 | Medium | Canonical | Calibrate threshold/parameter to satisfy target risk | Risk budget enters decision set construction | Calibration data representative/exchangeable | Open-loop risk control | Closed-loop interaction not central |
| P13 | Policy Gradient for Coherent Risk Measures | NeurIPS 2015 | High | Canonical formal-risk anchor | optimize `max_theta rho(G_theta)` for coherent risk `rho` (e.g., CVaR) | Risk measure directly optimizes policy choice | Risk estimator stable; policy gradient validity | RL control tasks | Not candidate-level tau filtering |
| P14 | PETS | NeurIPS 2018 | High | Canonical uncertainty-control | `u* = argmin_u E_{p_hat}[sum_t l(x_t,u_t)]` | Dynamics uncertainty inside MPC rollouts | Model validity near visited states | Closed-loop control | No decision-threshold calibration audit |
| P15 | RAP: Risk-Aware Prediction for Planning | CoRL 2022 | Emerging/Medium | Closest prior + SOTA | `a* = argmin_a E[C(a)] + lambda Risk(C(a))` | Risk-sensitive objective over uncertain futures | Risk functional stable under scene shifts | Closed-loop planning | No explicit calibrated tau-decision analysis |
| P16 | Hierarchical Game-Theoretic Planning for AVs | ICRA 2019 | Medium | Closest prior | `u_e* = argmax_{u_e} U_e(u_e, u_h*(u_e))` | Interaction uncertainty in utility/game response | Human-response model fidelity | Closed-loop interaction metrics | Surrogate risk, not calibrated probability |
| P17 | Safe Occlusion-Aware Active Perception Planning | RSS 2021 | Medium | Closest prior + SOTA precursor | `a* = argmax_a R_task(a)-lambda R_occ(a)` with safety guards | Occlusion uncertainty -> risk penalty/constraints | Occlusion proxy captures true hazard | Closed-loop occlusion scenarios | No tau-level false-safe/safe-reject audit |
| P18 | CARLA | CoRL 2017 | Very high | Benchmark | `a_t = pi(o_t)` in simulator | Supports risk-aware policies (method-dependent) | Simulator realism | Closed-loop collisions/infractions/success | No standard calibration-to-decision protocol |
| P19 | nuPlan | NeurIPS D&B 2021 | High | Benchmark | planner action/trajectory each step | Supports constrained/risk-aware planners | Scenario suite representativeness | Closed-loop safety/progress/comfort | No required tau-threshold diagnostics |
| P20 | SafeBench | NeurIPS D&B 2022 | Medium | Benchmark + SOTA | policy under stress/adversarial scenarios | Stress exposes uncertainty failures | Stress scenarios approximate real OOD | Closed-loop safety robustness | Does not isolate calibration as causal factor |
| P21 | WOSAC | NeurIPS D&B 2023 | Emerging/Medium | Benchmark + SOTA | multi-agent closed-loop rollout policy | Method-specific uncertainty use | Challenge metrics map to realism/safety | Closed-loop interactive metrics | No candidate-level decision audit |
| P22 | Waymax | arXiv 2023 | Emerging/Medium | Benchmark substrate | `a_t = pi(o_t)` in accelerated sim | Enables controlled shift and candidate-level experiments | Simulator shifts reflect relevant disturbances | Closed-loop simulator metrics | Not itself a decision-grade risk method |

---

## Paper Links

- P01: [paper](https://proceedings.mlr.press/v70/guo17a.html), [pdf](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), [local](references/pdfs/guo_2017.pdf)
- P02: [pdf](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf), [local](references/pdfs/obtaining_calibrated_probabilities_boosting_2012.pdf)
- P03: [paper](https://proceedings.mlr.press/v48/gal16.html), [pdf](https://arxiv.org/pdf/1506.02142.pdf), [local](references/pdfs/dropout_bayesian_2016.pdf)
- P04: [paper](https://papers.nips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html), [pdf](https://arxiv.org/pdf/1612.01474.pdf), [local](references/pdfs/deep_ensembles_2017.pdf)
- P05: [paper](https://papers.nips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html), [pdf](https://arxiv.org/pdf/1906.02530.pdf), [local](references/pdfs/ovadia_2019.pdf)
- P06: [paper](https://papers.nips.cc/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html), [pdf](https://arxiv.org/pdf/1910.12656.pdf), [local](references/pdfs/dirichlet_2019.pdf)
- P07: [paper](https://papers.nips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html), [pdf](https://arxiv.org/pdf/1705.08500.pdf), [local](references/pdfs/selective_classification_2017.pdf)
- P08: [paper](https://proceedings.mlr.press/v97/geifman19a.html), [pdf](https://arxiv.org/pdf/1901.09192.pdf), [local](references/pdfs/selectivenet_2019.pdf)
- P09: [paper](https://arxiv.org/abs/1905.09786), [pdf](https://arxiv.org/pdf/1905.09786.pdf), [local](references/pdfs/deep_gamblers_2019.pdf)
- P10: [paper](https://proceedings.mlr.press/v70/achiam17a.html), [pdf](https://arxiv.org/pdf/1705.10528.pdf), [local](references/pdfs/cpo_2017.pdf)
- P11: [paper](https://papers.nips.cc/paper_files/paper/2021/hash/32c0fdfc4f8f5f8f2f9ebf58f4a6ef08-Abstract.html), [pdf](https://arxiv.org/pdf/2101.02703.pdf), [local](references/pdfs/rcps_2021.pdf)
- P12: [paper](https://arxiv.org/abs/2208.02814), [pdf](https://arxiv.org/pdf/2208.02814.pdf), [local](references/pdfs/crc_2022.pdf)
- P13: [paper](https://papers.nips.cc/paper/2015/hash/024d7f84fff11dd7e8d9c510137a2381-Abstract.html), [pdf](https://arxiv.org/pdf/1502.03919.pdf), [local](references/pdfs/policy_gradient_coherent_risk_2015.pdf)
- P14: [paper](https://papers.nips.cc/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html), [pdf](https://arxiv.org/pdf/1805.12114.pdf), [local](references/pdfs/pets_2018.pdf)
- P15: [paper](https://openreview.net/forum?id=8r5Q2q7s8Q), [pdf](https://arxiv.org/pdf/2210.01368.pdf), [local](references/pdfs/rap_corl_2022.pdf)
- P16: [paper](https://arxiv.org/abs/1810.05766), [pdf](https://arxiv.org/pdf/1810.05766.pdf), [local](references/pdfs/hierarchical_game_theoretic_icra_2019.pdf)
- P17: [paper](https://roboticsconference.org/2021/program/papers/033/index.html), [pdf](https://arxiv.org/pdf/2105.08169.pdf), [local](references/pdfs/safe_occlusion_active_perception_rss_2021.pdf)
- P18: [paper](https://proceedings.mlr.press/v78/dosovitskiy17a.html), [pdf](https://proceedings.mlr.press/v78/dosovitskiy17a/dosovitskiy17a.pdf), [local](references/pdfs/carla_corl_2017.pdf)
- P19: [paper](https://arxiv.org/abs/2106.11810), [pdf](https://arxiv.org/pdf/2106.11810.pdf), [local](references/pdfs/nuplan_2021.pdf)
- P20: [paper](https://openreview.net/forum?id=ANQ3LafS8h), [pdf](https://arxiv.org/pdf/2206.09655.pdf), [local](references/pdfs/safebench_2022.pdf)
- P21: [paper](https://papers.nips.cc/paper_files/paper/2023/hash/b96ce67b2f2d45e4ab315e13a6b5b9c5-Abstract-Datasets_and_Benchmarks.html), [pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/b96ce67b2f2d45e4ab315e13a6b5b9c5-Paper-Datasets_and_Benchmarks.pdf), [local](references/pdfs/wosac_2023.pdf)
- P22: [paper](https://arxiv.org/abs/2310.08710), [pdf](https://arxiv.org/pdf/2310.08710.pdf), [local](references/pdfs/waymax_2023.pdf)

---

## Critical Cross-Paper Findings

## 1) Common Patterns

1. Uncertainty is mostly injected as a soft cost or risk regularizer.
2. Constraints/budgets (`d`, `alpha`) are common at policy/trajectory level.
3. Calibration is typically assessed separately from controller outcomes.
4. Benchmark studies report safety/performance but rarely decision-threshold diagnostics.

## 2) Common Assumptions

1. `uncertainty score` is a monotonic surrogate of true failure probability.
2. Risk scores are assumed usable without operating-point calibration.
3. Validation distribution is close to deployment distribution.
4. Open-loop reliability is assumed to transfer to closed-loop behavior.

## 3) What Is Missing Repeatedly

1. Candidate-level decision auditing at fixed `tau`.
2. Calibration -> decision correctness -> closed-loop outcome causal chain.
3. Explicit `false_safe`, `safe_reject`, `feasible_set_rate`, `fallback_rate` under shift.
4. Bottleneck decomposition between risk model error and controller-rule error.

## 4) Stress-Test of Prior Work in Our Setting

Setting: candidate-level selection + tau-threshold + closed-loop + shift.

1. Candidate mismatch:
- Policy-level success can hide poor candidate-level filtering decisions.

2. Tau sensitivity:
- Small calibration error near `tau` changes accept/reject sets discontinuously.

3. Closed-loop feedback:
- Early over/under-conservative choices alter state visitation, invalidating static assumptions.

4. Shift fragility:
- Same `tau` can become unsafe (false-safe spike) or unusable (feasible-set collapse) under OOD shift.

---

## Refined Gap Statements

1. Conservative:
- Existing calibration and risk-aware control methods are strong individually, but not jointly validated for candidate-level tau-threshold AV decisions in closed-loop.

2. Strong:
- Prior work does not systematically test whether calibration improves decision correctness (`false_safe` / `safe_reject`) at operational `tau` under shift.

3. Bold:
- The field lacks a decision-grade risk methodology that proves calibration-to-decision-to-outcome causality for candidate-level closed-loop AV control under distribution shift.

---

## Methodology Justification (Why New Method Is Needed)

1. Calibration papers optimize probability quality but usually stop before control outcomes.
2. Planning papers optimize safety/progress but often assume risk signals are decision-ready.
3. Benchmark papers expose failures but do not prescribe decision-threshold diagnostics.

Required methodology shape:
1. Candidate-level risk estimation for each action option.
2. Operating-point calibration near `tau` (plus optional conformal wrapping).
3. Decision metrics at `tau`: `accept_rate`, `false_safe`, `safe_reject`, `feasible_set_rate`, `fallback_rate`.
4. Closed-loop shift-suite evaluation with explicit safety-progress tradeoff.

---

## Closest Prior Work: Direct Comparison

| Prior work | What they do | What we add | Why it matters |
|---|---|---|---|
| RAP (CoRL 2022) | Risk-sensitive planning objective under uncertain predictions | Explicit post-hoc calibration + tau-threshold decision diagnostics | Separates risk estimation quality from controller behavior |
| Safe Occlusion-Aware Planning (RSS 2021) | Uses occlusion-risk surrogate in planning decisions | Tests whether surrogate risk is decision-grade at tau under shift | Prevents hidden over/under-confidence at deployment threshold |
| Hierarchical Game-Theoretic Planning (ICRA 2019) | Utility-based risk-aware interactive decisions | Candidate-level accept/reject and fallback feasibility auditing | Converts aggregate safety claims into decision-causal evidence |

---

## Anchor Papers (2-3 strongest)

1. **Guo et al., ICML 2017 (P01)**: canonical calibration baseline.
2. **CPO, ICML 2017 (P10)**: canonical budgeted safety-constrained decision optimization.
3. **RAP, CoRL 2022 (P15)**: closest AV uncertainty-to-decision prior.

---

## Final Positioning Statement

Existing methods assume uncertainty-derived risk is decision-ready, but fail to validate candidate-level tau-threshold behavior in closed-loop under shift. This motivates a methodology that calibrates risk for decision correctness and evaluates the full calibration -> decision -> outcome chain.
