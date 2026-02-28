# Risk-UQ Suite Literature Survey (High-Citation, Methodology-Focused)

This revision prioritizes citation quality, decision relevance, and methodological support for our paper direction.

## Scope and Curation Rules

- Focus: uncertainty/risk that directly changes a decision (action choice, threshold accept/reject, or constrained optimization).
- Venue priority: NeurIPS, ICLR, ICML, ICRA, CoRL, CVPR, RSS (plus benchmark Datasets/Benchmarks tracks).
- Balance target: recent SOTA + canonical foundations + closest prior + closed-loop benchmarks.
- Citation strength is reported qualitatively:
  - `Very high`: field-defining, typically multi-thousand citations.
  - `High`: widely used, typically several hundred to >1000.
  - `Medium`: visible and growing impact.
  - `Emerging`: recent work, still accruing citations.

## Portfolio Summary (22 papers)

- Recent SOTA (last ~1-4 years): 6
- Canonical/foundational: 10
- Closest prior to our setup: 3
- Benchmarks/substrate: 3

Note: Some papers play multiple roles.

---

## Master Evidence Table

| ID | Paper | Venue/Year | Citation Strength | Role | Decision Rule (equation/canonical form) | How uncertainty/risk enters decision | Key assumptions | Evaluation | Limitation vs our setting |
|---|---|---|---|---|---|---|---|---|---|
| P01 | On Calibration of Modern Neural Networks | ICML 2017 | Very high | Canonical | `p_cal = softmax(z / T)` with `T = argmin NLL_val` | Calibrated probabilities for downstream thresholds | Val IID and stationary | Open-loop, ECE/NLL/Brier | No closed-loop decision impact test |
| P02 | Predicting Good Probabilities with Supervised Learning | ICML 2005 | High | Canonical | `p = sigma(a s + b)` (Platt) / isotonic map | Score-to-probability map changes threshold outcomes | Monotonic score-risk relation | Open-loop classification/calibration | No shift/closed-loop analysis |
| P03 | Dropout as a Bayesian Approximation | ICML 2016 | Very high | Canonical | `a* = argmin E[C(a)]` under MC-dropout predictive distribution | Epistemic uncertainty from stochastic forward passes | Dropout posterior approximation quality | Open-loop and RL examples | Calibration-to-decision not audited |
| P04 | Deep Ensembles | NeurIPS 2017 | Very high | Canonical | `p(y|x)=1/M sum_m p_m(y|x)`; act by risk/confidence | Ensemble spread used as uncertainty in decisions | Ensemble diversity approximates epistemic uncertainty | Open-loop, OOD/UQ metrics | No candidate-level tau control protocol |
| P05 | Can You Trust Your Model's Uncertainty Under Shift? | NeurIPS 2019 | High | Canonical | Decision-oriented reliability checks under shift | Tests whether uncertainty remains valid OOD | Shift suite approximates deployment | Open-loop shift benchmarks | Does not connect to planner decisions |
| P06 | Beyond Temperature Scaling: Dirichlet Calibration | NeurIPS 2019 | High | Canonical | `p_cal = DirichletMap(p_raw; theta)` | Better probability map for thresholding | Calibration split representative | Open-loop calibration metrics | No closed-loop policy consequences |
| P07 | Selective Classification for DNNs | NeurIPS 2017 | High | Canonical | Accept if `s(x) >= theta`, else abstain | Uncertainty/confidence directly drives accept/reject | Score ranking aligns with risk | Open-loop risk-coverage | No sequential control feedback |
| P08 | SelectiveNet | ICML 2019 | Medium | Canonical | Jointly learn predictor + selection function under coverage target | Uncertainty integrated into selection head | Coverage objective transfers to deployment | Open-loop selective prediction | No AV closed-loop or shift-control coupling |
| P09 | Deep Gamblers | NeurIPS 2019 | Medium | Canonical | Bet/reject via reservation utility in objective | Confidence controls abstention budget | Utility surrogate matches risk preference | Open-loop abstention metrics | No causal calibration->control link |
| P10 | Constrained Policy Optimization (CPO) | ICML 2017 | High | Canonical | `max_pi J_R(pi) s.t. J_C(pi) <= d` | Safety cost budget constrains policy updates | Cost surrogate accurately captures safety | Closed-loop RL constraints | Policy-level, not candidate-level tau gating |
| P11 | Distribution-Free Risk-Controlling Prediction Sets (RCPS) | NeurIPS 2021 | Medium | Canonical | Choose `lambda` s.t. `R_hat(lambda) <= alpha` on calibration set | Risk budget controls acceptance set size | Exchangeability | Open-loop risk-control guarantees | Closed-loop adaptive dynamics break assumptions |
| P12 | PETS | NeurIPS 2018 | High | Canonical | `u* = argmin_u E_{p_hat}[sum_t l(x_t,u_t)]` | Dynamics uncertainty enters MPC rollout cost | Learned dynamics valid near rollout states | Closed-loop control tasks | No explicit tau accept/reject evaluation |
| P13 | MultiPath | CoRL 2020 | High | Closest prior | Downstream planners use `argmin_a sum_k w_k C(a,tau_k)` | Multimodal forecast probabilities used as risk proxy | Forecast mode probs are decision-grade | Mostly open-loop prediction + planning relevance | Candidate-level safety thresholding not validated |
| P14 | MATS | CoRL 2020 | Medium | Closest prior | `a* = argmin_a sum_m w_m C(a,tau^(m))` | Forecast uncertainty affects planning score | Mode weights are usable as risk weights | Planning-control integration experiments | Calibration->decision causality not isolated |
| P15 | RAP (Risk-Aware Prediction for Planning) | CoRL 2022 | Emerging/Medium | Closest prior + SOTA | `a* = argmin_a E[C(a)] + lambda Risk(C(a))` | Risk-sensitive objective over uncertain futures | Risk functional tuned and stable across scenes | Closed-loop planning outcomes | No explicit calibrated tau-budget diagnostics |
| P16 | Hierarchical Game-Theoretic Planning for AVs | ICRA 2019 | Medium | SOTA precursor | `u_e* = argmax_{u_e} U_e(u_e, u_h*(u_e))` | Interaction uncertainty enters utility/game response | Human response model fidelity | Closed-loop interaction metrics | Surrogate utilities, not calibrated probabilities |
| P17 | Safe Occlusion-Aware Active Perception Planning | RSS 2021 | Medium | SOTA precursor | `a* = argmax_a R_task(a) - lambda R_occ(a)` with safety guards | Occlusion uncertainty converted to risk penalty/constraint | Occlusion risk proxy tracks true hazard | Closed-loop occlusion scenarios | Little calibration/threshold error analysis |
| P18 | CARLA | CoRL 2017 | Very high | Benchmark | `a_t = pi(o_t)` in closed-loop simulator | Supports uncertainty-aware policies; method-defined | Simulator realism approximates deployment | Closed-loop collisions/infractions/success | No standard calibration-to-decision protocol |
| P19 | nuPlan | NeurIPS D&B 2021 | High | Benchmark | Planner chooses trajectory/action at each step | Supports risk/cost-aware planners | Scenario benchmark coverage adequate | Closed-loop safety/progress/comfort | No required tau-threshold diagnostics |
| P20 | SafeBench | NeurIPS D&B 2022 | Medium | Benchmark + SOTA | Policy under adversarial/stress scenarios | Robustness stressors expose uncertainty failures | Stress scenarios represent safety-critical shift | Closed-loop safety robustness metrics | Does not isolate calibration as causal factor |
| P21 | WOSAC | NeurIPS D&B 2023 | Emerging/Medium | Benchmark + SOTA | Interactive policy rollout in multi-agent sim | Method-specific uncertainty handling | Challenge metrics proxy deployment behavior | Closed-loop interactive realism and safety | No explicit candidate-level decision audit |
| P22 | Waymax | arXiv 2023 | Emerging/Medium | Benchmark substrate | `a_t = pi(o_t)` in accelerated closed-loop sim | Enables controlled uncertainty/risk experiments at scale | Simulator shift knobs reflect meaningful disturbances | Closed-loop simulator metrics | Not itself a risk-calibration methodology |

---

## Paper Links

- P01: [paper](https://proceedings.mlr.press/v70/guo17a.html), [pdf](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), [local](references/pdfs/guo_2017.pdf)
- P02: [pdf](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf), [local](references/pdfs/obtaining_calibrated_probabilities_boosting_2012.pdf)
- P03: [paper](https://proceedings.mlr.press/v48/gal16.html), [pdf](https://arxiv.org/pdf/1506.02142.pdf), [local](references/pdfs/dropout_bayesian_2016.pdf)
- P04: [paper](https://papers.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html), [pdf](https://arxiv.org/pdf/1612.01474.pdf), [local](references/pdfs/deep_ensembles_2017.pdf)
- P05: [paper](https://papers.nips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html), [pdf](https://arxiv.org/pdf/1906.02530.pdf), [local](references/pdfs/ovadia_2019.pdf)
- P06: [paper](https://papers.nips.cc/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html), [pdf](https://arxiv.org/pdf/1910.12656.pdf), [local](references/pdfs/dirichlet_2019.pdf)
- P07: [paper](https://papers.nips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html), [pdf](https://arxiv.org/pdf/1705.08500.pdf), [local](references/pdfs/selective_classification_2017.pdf)
- P08: [paper](https://proceedings.mlr.press/v97/geifman19a.html), [pdf](https://arxiv.org/pdf/1901.09192.pdf), [local](references/pdfs/selectivenet_2019.pdf)
- P09: [paper](https://arxiv.org/abs/1905.09786), [pdf](https://arxiv.org/pdf/1905.09786.pdf), [local](references/pdfs/deep_gamblers_2019.pdf)
- P10: [paper](https://proceedings.mlr.press/v70/achiam17a.html), [pdf](https://arxiv.org/pdf/1705.10528.pdf), [local](references/pdfs/cpo_2017.pdf)
- P11: [paper](https://papers.nips.cc/paper_files/paper/2021/hash/32c0fdfc4f8f5f8f2f9ebf58f4a6ef08-Abstract.html), [pdf](https://arxiv.org/pdf/2101.02703.pdf), [local](references/pdfs/rcps_2021.pdf)
- P12: [paper](https://papers.nips.cc/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html), [pdf](https://arxiv.org/pdf/1805.12114.pdf), [local](references/pdfs/pets_2018.pdf)
- P13: [paper](https://proceedings.mlr.press/v100/chai20a.html), [pdf](https://proceedings.mlr.press/v100/chai20a/chai20a.pdf), [local](references/pdfs/multipath_2020.pdf)
- P14: [paper](https://proceedings.mlr.press/v155/ivanovic21a.html), [pdf](https://arxiv.org/pdf/2009.07517.pdf), [local](references/pdfs/mats_corl_2020.pdf)
- P15: [paper](https://openreview.net/forum?id=8r5Q2q7s8Q), [pdf](https://arxiv.org/pdf/2210.01368.pdf), [local](references/pdfs/rap_corl_2022.pdf)
- P16: [paper](https://arxiv.org/abs/1810.05766), [pdf](https://arxiv.org/pdf/1810.05766.pdf), [local](references/pdfs/hierarchical_game_theoretic_icra_2019.pdf)
- P17: [paper](https://roboticsconference.org/2021/program/papers/033/index.html), [pdf](https://arxiv.org/pdf/2105.08169.pdf), [local](references/pdfs/safe_occlusion_active_perception_rss_2021.pdf)
- P18: [paper](https://proceedings.mlr.press/v78/dosovitskiy17a.html), [pdf](https://proceedings.mlr.press/v78/dosovitskiy17a/dosovitskiy17a.pdf), [local](references/pdfs/carla_corl_2017.pdf)
- P19: [paper](https://arxiv.org/abs/2106.11810), [pdf](https://arxiv.org/pdf/2106.11810.pdf), [local](references/pdfs/nuplan_2021.pdf)
- P20: [paper](https://openreview.net/forum?id=ANQ3LafS8h), [pdf](https://arxiv.org/pdf/2206.09655.pdf), [local](references/pdfs/safebench_2022.pdf)
- P21: [paper](https://papers.nips.cc/paper_files/paper/2023/hash/b96ce67b2f2d45e4ab315e13a6b5b9c5-Abstract-Datasets_and_Benchmarks.html), [pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/b96ce67b2f2d45e4ab315e13a6b5b9c5-Paper-Datasets_and_Benchmarks.pdf), [local](references/pdfs/wosac_2023.pdf)
- P22: [paper](https://arxiv.org/abs/2310.08710), [pdf](https://arxiv.org/pdf/2310.08710.pdf), [local](references/pdfs/waymax_2023.pdf)

---

## Global Refinement Decisions

Removed or deprioritized from the core argument:
1. Pure prediction papers without explicit downstream decision rule.
2. Niche low-citation papers with overlapping claims but weaker evidence.
3. Methods that report aggregate accuracy only, without decision-impact or safety metrics.

Replaced with stronger references that directly support claims about:
1. calibration quality,
2. threshold/budget decisions,
3. constrained optimization,
4. closed-loop safety evaluation.

---

## Critical Analysis

## A) Common Patterns Across Literature

1. Uncertainty is widely used as a **cost weight** or **regularizer** in planning.
2. Safety is often enforced via **constraints/budgets** (`d`, `alpha`, chance constraints).
3. Evaluation commonly reports aggregate safety metrics (collision, violations, completion), but rarely decision-level threshold errors.
4. Calibration work focuses on probability quality metrics, while planning work focuses on trajectory outcomes; these are usually disconnected.

## B) Common Assumptions

1. Uncertainty is treated as a monotonic proxy for risk.
2. Raw probabilities/scores are assumed decision-ready after minimal post-processing.
3. Validation and deployment distributions are assumed sufficiently similar.
4. Open-loop risk quality is assumed to transfer to closed-loop control behavior.

## C) What Is Consistently Missing

1. Candidate-level decision auditing at a fixed operating threshold `tau`.
2. Causal chain evaluation: calibration -> decision correctness -> closed-loop outcome.
3. Explicit measurement of `false-safe` and `safe-reject` under distribution shift.
4. Bottleneck decomposition to separate model-risk error from controller-rule error.

## D) Stress-Test in Our Setting (candidate-level + tau + closed-loop + shift)

Where prior methods fail concretely:
1. Candidate-level mismatch:
   - Policy/trajectory-level methods can show good average safety but still make poor per-candidate accept/reject decisions.
2. Tau-threshold fragility:
   - A mildly miscalibrated score can cause feasible-set collapse (`no candidate <= tau`) and excessive fallback.
3. Closed-loop coupling:
   - Early conservative or unsafe choices change future states, invalidating static calibration assumptions.
4. Shift amplification:
   - OOD shifts can flip behavior from over-conservative to false-safe at the same `tau`.

---

## Refined Research Gap Statements (3 versions)

1. Conservative:
- Existing uncertainty-aware planners and calibration methods are strong individually, but they are not jointly validated for candidate-level threshold decisions in closed-loop AV simulation.

2. Strong:
- Prior work rarely evaluates whether calibrated risk probabilities actually improve decision correctness (`false-safe` / `safe-reject`) at operational `tau` in closed-loop, especially under shift.

3. Bold:
- The field lacks a decision-grade risk methodology: current pipelines assume risk scores are decision-ready, yet fail to prove calibration-to-decision-to-outcome causality for candidate-level AV control under distribution shift.

---

## Methodology Justification

Why existing methods are insufficient:
1. Calibration-only papers do not test control consequences.
2. Planning-only papers do not test probability reliability at the decision threshold.
3. Benchmark papers do not prescribe causal diagnostics linking uncertainty quality to decision errors.

What method is needed:
1. Candidate-level risk estimation for each action option.
2. Post-hoc calibration (and optionally conformal wrapping) tuned to operating `tau`.
3. Decision-level diagnostics (`accept_rate`, `false_safe`, `safe_reject`, `fallback_rate`).
4. Closed-loop and shift-aware evaluation that quantifies tradeoff: safety gain vs progress loss.

---

## Closest Prior Work Comparison (Direct)

| Prior work | What they do | What we do differently | Why it matters |
|---|---|---|---|
| RAP (CoRL 2022) | Risk-sensitive planning objective over uncertain predictions | Add explicit post-hoc calibration and tau-threshold decision auditing | Separates risk-model quality from controller behavior |
| Safe Occlusion-Aware Planning (RSS 2021) | Uses occlusion risk surrogate in planning constraints/objective | Evaluate whether surrogate risk is calibrated for threshold decisions under shift | Prevents hidden over/under-confidence at deployment threshold |
| Hierarchical Game-Theoretic Planning (ICRA 2019) | Utility/game-based risk-aware interactive control | Add candidate-level accept/reject analysis with fallback and feasibility diagnostics | Converts qualitative risk-aware behavior into measurable decision correctness |

---

## Anchor Papers Defining the Space

1. **Guo et al., ICML 2017 (P01)**: establishes modern neural miscalibration and temperature scaling baseline.
2. **CPO, ICML 2017 (P10)**: establishes budgeted safety-constrained decision optimization.
3. **RAP, CoRL 2022 (P15)**: closest AV planning precedent for uncertainty-to-risk decision integration.

---

## Final Positioning Statement

Existing methods assume uncertainty-derived risk is decision-ready, but fail to validate candidate-level tau-threshold decisions in closed-loop under shift; this motivates a methodology that explicitly calibrates risk for decision correctness and tests the full causal chain from probability quality to control outcomes.
