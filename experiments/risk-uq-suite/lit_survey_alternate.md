# Alternate Literature Survey: Decision-Causal Risk for Closed-Loop AV (Risk-UQ Suite)

This is an alternate, gap-focused survey designed to complement `lit_survey.md`.
It keeps the same paper universe (P01-P30), but emphasizes paper-by-paper critical transferability to our setting:
- candidate-level action selection,
- thresholded decision rule `D(p_hat, tau)`,
- closed-loop execution,
- distribution shift.

## 0) Reference Availability Check

All 30 referenced PDFs were verified locally under:
`experiments/risk-uq-suite/references/pdfs/`

Validation status:
- `total references`: 30
- `present`: 30
- `missing`: 0
- `invalid header`: 0

---

## 1) Analytical Lens Used in This Alternate Survey

We evaluate each paper through the same decision-causal interface:

- latent risk: `p(x) = P(Y=1 | x)`
- predicted risk used in control: `p_hat(x)`
- decision operator: `D(p_hat, tau) = 1[p_hat <= tau]`

Primary question per paper:
- does the method provide decision-grade evidence for `D(p_hat, tau)` under closed-loop and shift, or only partial evidence (signal/calibration/rule in isolation)?

---

## 2) Paper-by-Paper Analysis

## 2.1 Calibration Foundations (P01-P06)

### P01 - Guo et al. (ICML 2017)
- Problem solved: modern neural nets produce overconfident probabilities.
- Method: post-hoc temperature scaling.
- What works: reliable and simple probability correction in many IID-like settings.
- Key assumption: validation distribution reflects deployment.
- Transfer to us: strong baseline for post-hoc calibration, but does not by itself validate threshold decisions in closed-loop control.
- Limitation vs gap: no candidate-level `false_safe/safe_reject` evidence.

### P02 - Niculescu-Mizil and Caruana (ICML 2005)
- Problem solved: convert scores to calibrated probabilities.
- Method: Platt scaling and isotonic regression.
- What works: practical score-to-probability mapping for threshold decisions.
- Key assumption: score-risk monotonicity and enough calibration data.
- Transfer to us: directly useful for proxy calibration variants in decision-audit notebooks.
- Limitation vs gap: static supervised framing; no closed-loop feedback effects.

### P03 - Gal and Ghahramani (ICML 2016)
- Problem solved: approximate epistemic uncertainty in deep models.
- Method: MC-dropout as Bayesian approximation.
- What works: useful uncertainty estimates when model uncertainty dominates.
- Key assumption: dropout posterior approximation is informative.
- Transfer to us: supports uncertainty feature design and uncertainty-aware penalties.
- Limitation vs gap: does not connect uncertainty quality to tau-threshold decision correctness.

### P04 - Lakshminarayanan et al. (NeurIPS 2017)
- Problem solved: robust predictive uncertainty from ensembles.
- Method: deep ensembles.
- What works: strong empirical uncertainty baseline.
- Key assumption: ensemble diversity approximates epistemic uncertainty.
- Transfer to us: supports ensemble-style risk modeling and epistemic terms.
- Limitation vs gap: no direct candidate-level decision audit in AV control.

### P05 - Ovadia et al. (NeurIPS 2019)
- Problem solved: uncertainty methods can degrade under distribution shift.
- Method: large benchmark across uncertainty methods under controlled shifts.
- What works: establishes shift fragility as real, not anecdotal.
- Key assumption: benchmark shifts are representative of deployment changes.
- Transfer to us: central justification for nominal-vs-shift decision audits.
- Limitation vs gap: classification-centric; no planner decision operator.

### P06 - Kull et al., Dirichlet Calibration (NeurIPS 2019)
- Problem solved: multiclass calibration beyond temperature scaling.
- Method: Dirichlet calibration map.
- What works: improved calibration in multiclass regimes.
- Key assumption: calibration split quality and representative class simplex geometry.
- Transfer to us: supports trying multiple calibrators, not just one.
- Limitation vs gap: no closed-loop control impact analysis.

## 2.2 Thresholded / Budgeted Decision Frameworks (P07-P12, P30)

### P07 - Geifman and El-Yaniv, Selective Classification (NeurIPS 2017)
- Problem solved: abstain when uncertain under risk constraints.
- Method: confidence thresholding with coverage-risk tradeoff.
- What works: clean accept/reject formalism.
- Key assumption: confidence has stable ranking relation to error.
- Transfer to us: directly aligned to `tau` gating and accept-rate diagnostics.
- Limitation vs gap: sample-level classification; no sequential dynamics.

### P08 - SelectiveNet (ICML 2019)
- Problem solved: jointly learn prediction and rejection.
- Method: architecture with selection head and coverage objective.
- What works: better selective behavior than post-hoc rejection in some settings.
- Key assumption: training objective matches deployment utility.
- Transfer to us: motivates integrated decision heads.
- Limitation vs gap: no candidate-level closed-loop AV demonstration.

### P09 - Deep Gamblers (NeurIPS 2019)
- Problem solved: selective prediction via betting/reservation mechanism.
- Method: utility-based abstention objective.
- What works: robust reject behavior under uncertainty.
- Key assumption: betting utility aligns with downstream decision cost.
- Transfer to us: conceptual support for decision rule design sensitivity.
- Limitation vs gap: no closed-loop safety-progress diagnostics.

### P10 - CPO (ICML 2017)
- Problem solved: constrained policy optimization with safety costs.
- Method: optimize reward subject to expected-cost constraints.
- What works: principled safe RL formulation.
- Key assumption: cost surrogate and trust-region approximations are faithful.
- Transfer to us: supports constrained control framing.
- Limitation vs gap: policy-level constraints, not candidate-step tau audits.

### P11 - RCPS (NeurIPS 2021)
- Problem solved: risk control with prediction sets.
- Method: conformal risk control over set parameters.
- What works: finite-sample style risk guarantees in exchangeable settings.
- Key assumption: calibration assumptions (exchangeability-like) hold.
- Transfer to us: formal basis for budgeted risk control.
- Limitation vs gap: guarantees weaken in adaptive closed-loop dependence.

### P12 - Conformal Risk Control (Angelopoulos et al., ICLR 2024)
- Problem solved: control task-level risk via conformalized thresholds.
- Method: calibration of decision parameter to satisfy risk target.
- What works: explicit risk budget interpretation.
- Key assumption: calibration/deployment alignment and non-adaptive conditions.
- Transfer to us: supports conformal threshold variants.
- Limitation vs gap: limited candidate-level closed-loop evaluation.

### P30 - Planning on a (Risk) Budget (Huang et al., 2021)
- Problem solved: planning under uncertainty with explicit safety-performance compromise.
- Method: risk-budgeted planning objective/constraints to target non-conservative behavior under bounded risk.
- What works: demonstrates explicit budget tuning can avoid both naive optimism and excessive conservatism.
- Key assumption: chosen risk surrogate and budget parameter reflect true operational hazards/costs.
- Transfer to us: direct conceptual anchor for tau-threshold/budget framing and conservatism-efficiency analysis.
- Limitation vs gap: does not provide candidate-level `false_safe/safe_reject/feasible_set` audit under closed-loop shift suites.

## 2.3 Uncertainty-to-Control / Risk-Aware Planning (P13-P17)

### P13 - Tamar et al., Policy Gradient for Coherent Risk (NeurIPS 2015)
- Problem solved: optimize policies for coherent risk measures (e.g., CVaR-like criteria).
- Method: risk-sensitive policy gradient.
- What works: formal risk-sensitive control objective.
- Key assumption: risk functional and return estimates are stable.
- Transfer to us: foundational rationale for risk-penalized objectives.
- Limitation vs gap: does not address probability calibration near operating thresholds.

### P14 - PETS (NeurIPS 2018)
- Problem solved: model-based control with uncertainty-aware trajectory evaluation.
- Method: probabilistic ensembles + MPC.
- What works: sample-efficient control with uncertainty-aware planning.
- Key assumption: learned dynamics are adequate in rollout region.
- Transfer to us: supports uncertainty-aware candidate evaluation concepts.
- Limitation vs gap: no explicit threshold-correctness diagnostics.

### P15 - RAP (CoRL 2022)
- Problem solved: risk-aware planning in interactive driving.
- Method: objective combines expected cost with risk-sensitive terms.
- What works: improved safety/interaction behavior in tested settings.
- Key assumption: risk surrogate and interaction model are faithful.
- Transfer to us: very close in spirit to risk-aware reranking.
- Limitation vs gap: does not explicitly audit `FS/SR/feasibility` at candidate-level `tau`.

### P16 - Hierarchical game-theoretic planning (ICRA 2019)
- Problem solved: interactive planning with strategic multi-agent reasoning.
- Method: game-theoretic decision hierarchy using interaction utilities.
- What works: structured behavior in strategic interactions.
- Key assumption: behavior model quality and utility design.
- Transfer to us: supports interaction-risk surrogates in planning.
- Limitation vs gap: surrogate-risk calibration usually untested.

### P17 - Safe occlusion-aware active perception (RSS 2021)
- Problem solved: planning under occlusion uncertainty and hidden hazards.
- Method: active sensing + safety-aware planning objective.
- What works: handles hidden-risk scenarios better than naive planning.
- Key assumption: occlusion proxy quality and map/sensing assumptions.
- Transfer to us: supports proxy-risk usage when direct risk labels are hard.
- Limitation vs gap: no candidate-threshold calibration-to-decision protocol.

## 2.4 Closed-Loop Benchmarks and Substrates (P18-P22)

### P18 - CARLA (CoRL 2017)
- Problem solved: closed-loop AV simulation benchmark substrate.
- Method: simulator + task suites.
- What works: reproducible closed-loop testing.
- Key assumption: simulator realism and scenario relevance.
- Transfer to us: supports experimental methodology but not decision-grade risk by itself.
- Limitation vs gap: no built-in tau-level correctness metrics.

### P19 - nuPlan (NeurIPS D&B 2021)
- Problem solved: standardized planning benchmark.
- Method: scenario-based closed-loop/offline planning evaluation.
- What works: broad planner comparability.
- Key assumption: benchmark protocol reflects deployment priorities.
- Transfer to us: benchmark inspiration for reproducibility standards.
- Limitation vs gap: calibration-to-decision linkage generally external.

### P20 - SafeBench (NeurIPS D&B 2022)
- Problem solved: stress-testing AV safety under adversarial/shift conditions.
- Method: closed-loop robustness benchmarks and stressors.
- What works: exposes brittleness that nominal testing misses.
- Key assumption: stressors map to real failure modes.
- Transfer to us: supports shift-suite necessity and robustness reporting.
- Limitation vs gap: does not isolate signal vs calibration vs decision-rule bottlenecks.

### P21 - WOSAC (NeurIPS D&B 2023)
- Problem solved: realism-oriented interactive simulation challenge.
- Method: challenge benchmark for sim-agents in closed-loop.
- What works: pushes quality of interaction modeling at scale.
- Key assumption: challenge metrics align with practical planning goals.
- Transfer to us: strong benchmark context for publication relevance.
- Limitation vs gap: no direct decision-grade risk evaluation protocol.

### P22 - Waymax (2023)
- Problem solved: accelerator-friendly large-scale closed-loop simulation.
- Method: scalable simulator with differentiable-friendly design and tooling.
- What works: high-throughput controlled closed-loop experiments.
- Key assumption: simulator dynamics and data interfaces are suitable for planning research.
- Transfer to us: core substrate for our reproducible pipeline.
- Limitation vs gap: method-level decision calibration must be added by user workflows.

## 2.5 Recent 2023-2025 Methods Most Relevant to the Gap (P23-P29)

### P23 - RACP (2024)
- Problem solved: contingency planning with multimodal uncertainty and risk penalties.
- Method: risk-aware contingency planning objective over multimodal predictions.
- What works: improves safety behavior in uncertain interaction regimes.
- Key assumption: belief-weighted multimodal risk surrogate is informative.
- Transfer to us: closest style to risk-penalized reranking baseline.
- Limitation vs gap: limited explicit calibration and fixed-`tau` decision correctness auditing.

### P24 - Chance-constrained MPC with GMM uncertainty (2024/2025)
- Problem solved: trajectory optimization under chance constraints with multimodal predictions.
- Method: chance-constrained MPC with feasibility analysis.
- What works: principled constraint handling under forecast uncertainty.
- Key assumption: uncertainty propagation and model form (GMM) are valid.
- Transfer to us: strongest formal analogue to thresholded budget constraints.
- Limitation vs gap: model-dependent constraints; candidate-step local calibration not central.

### P25 - Localized Adaptive Risk Control (NeurIPS 2024)
- Problem solved: local risk control under heterogeneity/non-stationarity.
- Method: localized adaptive thresholding / risk control.
- What works: better subgroup/region risk behavior than global thresholds.
- Key assumption: localization signal is informative and stable enough.
- Transfer to us: motivates local (near-`tau`) calibration and local diagnostics.
- Limitation vs gap: not directly validated for AV candidate-level closed-loop control.

### P26 - RADIUS (RSS 2023)
- Problem solved: real-time risk-aware motion planning using reachability bounds.
- Method: reachable-set style risk constraints.
- What works: safety-focused planning under explicit risk bounds.
- Key assumption: reachable set approximations are conservative but useful.
- Transfer to us: supports risk-constrained control framing.
- Limitation vs gap: risk is often upper-bound surrogate, not calibrated probability at operating threshold.

### P27 - MARC (2023)
- Problem solved: calibrate multi-agent prediction uncertainty for reachability/safety.
- Method: conformalized prediction sets + safety envelopes.
- What works: safety certification improves with calibrated uncertainty sets.
- Key assumption: calibration assumptions and set construction validity.
- Transfer to us: strong evidence that uncertainty calibration can matter for downstream safety.
- Limitation vs gap: set-level guarantees, not candidate-level `false_safe/safe_reject` decomposition.

### P28 - CUQDS (2024)
- Problem solved: conformal uncertainty quantification under distribution shift for trajectories.
- Method: shift-aware conformal calibration/coverage adjustments.
- What works: improves uncertainty reliability under shift in prediction tasks.
- Key assumption: shift adaptation mechanism captures deployment drift.
- Transfer to us: supports shift-aware calibration research direction.
- Limitation vs gap: generally open-loop prediction focus, limited closed-loop control linkage.

### P29 - Adversarially Robust Conformal Prediction for Interactive Safe Planning (2025)
- Problem solved: robust conformal safety filtering under interactive/adversarial uncertainty.
- Method: adversarially robust conformal-style set/threshold mechanisms.
- What works: stronger robustness framing than static calibration.
- Key assumption: adversarial model and robustness design reflect actual interaction shifts.
- Transfer to us: motivates robust thresholding extensions beyond static `tau`.
- Limitation vs gap: early-stage evidence; standardized candidate-step protocol still missing.

---

## 3) Cross-Paper Synthesis (Critical, Not Confirmatory)

## 3.1 What Prior Work Gets Right

1. Uncertainty-aware and risk-constrained control can improve safety-efficiency tradeoffs when risk signals are informative.
2. Calibration methods are practical and often materially improve probability quality.
3. Shift stress tests are necessary; nominal performance alone is unreliable.
4. Benchmark infrastructure for large-scale closed-loop testing is now mature enough to evaluate these ideas rigorously.

## 3.2 What Remains Under-tested for Our Exact Problem

1. Candidate-step threshold correctness (`false_safe`, `safe_reject`) is rarely reported as a primary metric.
2. Local reliability near operating `tau` is rarely separated from global calibration.
3. Few works jointly test the full chain:
   - quality of `p_hat(x)`,
   - correctness of `D(p_hat, tau)`,
   - final closed-loop outcomes under shift.
4. Failure attribution (signal vs calibration vs decision-rule) is often not isolated by causal ablations.

## 3.3 Common Assumptions That Can Break in Our Setting

1. Calibration split is representative of deployment distribution.
2. Risk surrogate is monotone enough with true event probability for thresholding.
3. Policy/data dependence does not strongly violate static calibration assumptions.
4. Trajectory-level gains imply candidate-step decision correctness.

When these assumptions break, expected failure modes are:
- high safe-reject and fallback (over-conservatism),
- hidden false-safe pockets near boundary,
- shift-induced instability of decision metrics.

---

## 4) Refined Gap Justification (Alternate Formulation)

Conservative gap claim:
- Prior work provides strong pieces (calibration, constrained control, robust benchmarks), but the joint candidate-level decision-audit protocol remains sparse.

Stronger gap claim:
- We do not find established protocols that jointly evaluate, in one closed-loop pipeline, all of:
  1. candidate-level risk discrimination,
  2. local calibration near `tau`,
  3. threshold decision correctness (`false_safe/safe_reject/feasible_set/fallback`),
  4. robustness under shift suites,
  5. causal bottleneck attribution (signal vs calibration vs rule).

Bold but bounded claim:
- The key missing methodological primitive is not another isolated predictor or calibrator, but a decision-grade validation stack that makes thresholded planner decisions scientifically auditable under closed-loop shift.

---

## 5) Hypothesis and Falsification Conditions

Primary hypothesis:
- In this candidate-level closed-loop setting, non-decision-grade risk signals produce measurable threshold decision errors (unsafe acceptance and/or over-conservative rejection), especially under shift.

Null hypothesis:
- Existing uncertainty-derived signals are already sufficient for threshold decisions; additional calibration/auditing does not materially improve decision correctness or closed-loop outcomes.

Evidence supporting null:
1. low `false_safe` and low `safe_reject` across shifts,
2. high feasible-set rate and low fallback,
3. negligible improvement from calibration/rule variants,
4. oracle variants do not materially outperform current controller.

Evidence refuting null:
1. local miscalibration near `tau`,
2. significant `false_safe` or `safe_reject`,
3. shift deltas that increase decision errors,
4. improvements after calibration or rule redesign with fixed candidate sets.

---

## 6) Closest Prior Anchors for Implementation Comparisons

If we need direct method-style baselines in code for this project track, strongest anchors from this corpus are:
1. P23 (RACP-style): risk-penalized reranking objective.
2. P24 (chance-constrained MPC spirit): threshold-gated feasibility controller.
3. P29/P12 family (conformal-style): calibrated threshold/budget controller.
4. P30 (risk-budget planning): explicit safety-performance budget baseline.

These anchor three controller families that map cleanly to current notebooks:
- risk-penalized score controllers,
- hard-threshold chance-style controllers,
- conformal/budgeted controllers.

---

## 7) Practical Readout for the Risk-UQ Suite

For our repository workflow, this alternate survey implies a concrete sequence:
1. Build candidate-level artifacts on shift-rich data.
2. Run cross-signal decision audit for calibration + threshold diagnostics.
3. Run oracle bottleneck analysis to isolate dominant failure source.
4. Compare RACP/chance/conformal controller families on identical candidate sets.
5. Report failure-progress tradeoff with data-sufficiency gates and confidence intervals.

This sequence is the minimum needed to make the gap claim evidence-backed rather than narrative.
