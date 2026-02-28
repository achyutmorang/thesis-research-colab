# Risk-UQ Suite Literature Survey (Critical, Falsifiable, Unbiased)

This version is intentionally non-confirmatory.

## Target Hypothesis (to test, not assume)

In candidate-level AV action selection with a fixed risk threshold `tau`, uncertainty-derived risk scores are often not decision-grade under shift; therefore calibration and explicit decision auditing may be needed.

This hypothesis can be false in settings where:
1. the risk signal is already strong and stable,
2. simple calibration is sufficient,
3. the decision rule is already robust.

## Decision-Grade Risk (Operational)

A risk score is decision-grade at `tau` if it is:
1. predictive (candidate ranking aligns with outcomes),
2. calibrated near `tau`,
3. stable under shift,
4. useful for decisions (lower `false_safe` without infeasible `safe_reject`/fallback growth).

---

## Critical Evidence Matrix (22 papers)

Legend for “Contradiction to hypothesis”:
- `No`: supports hypothesis or is consistent with it.
- `Partial`: shows scenarios where our hypothesis may not hold.
- `Untested`: paper does not directly test this hypothesis.

| ID | Paper | Venue/Year | What it solves successfully | Core assumptions | Works well when | Would it transfer to our setting? | Contradiction to hypothesis? |
|---|---|---|---|---|---|---|---|
| P01 | Guo et al. (Temperature Scaling) | ICML 2017 | Improves confidence calibration in many in-domain models | Validation data reflects deployment | In-domain, moderate shift | Partially; helps calibration but not candidate closed-loop causality | Partial (calibration may be enough in stable IID) |
| P02 | Platt/Isotonic calibration | ICML 2005 | Converts scores to better probabilities | Monotonic score-risk relation | Static supervised tasks | Partially; useful post-hoc baseline only | Partial |
| P03 | Dropout as Bayesian Approximation | ICML 2016 | Cheap epistemic uncertainty estimate | Dropout approximates posterior | Moderate model misspecification | Unclear; may provide signal, not decision guarantees | Untested |
| P04 | Deep Ensembles | NeurIPS 2017 | Strong uncertainty quality and robustness baseline | Ensemble diversity approximates epistemic | Many supervised tasks, some OOD | Potentially useful signal source; no tau decision proof | Untested |
| P05 | Ovadia et al. (UQ under shift) | NeurIPS 2019 | Demonstrates OOD calibration degradation | Shift suite representative | Distribution shifts present | Yes; directly warns calibration can fail under shift | No |
| P06 | Dirichlet Calibration | NeurIPS 2019 | Better post-hoc calibration vs TS in some regimes | Calibration set representativeness | Multiclass in-domain | Helps calibration stage, not control coupling | Partial |
| P07 | Selective Classification | NeurIPS 2017 | Formal risk-coverage thresholding | Confidence ranks true risk | Static prediction settings | Conceptually yes for tau gating; dynamics missing | Partial |
| P08 | SelectiveNet | ICML 2019 | Joint prediction + abstention at target coverage | Coverage objective generalizes | Similar train/test distributions | Useful for threshold ideas, not closed-loop AV | Partial |
| P09 | Deep Gamblers | NeurIPS 2019 | Abstention via utility-based objective | Reservation utility matches risk objective | Controlled classification tasks | Only conceptual transfer; no AV dynamics | Untested |
| P10 | CPO | ICML 2017 | Safety-constrained policy optimization | Constraint cost captures true risk | RL tasks with good cost signal | Partial; policy-level constraints can work but not candidate-level tau | Partial |
| P11 | RCPS | NeurIPS 2021 | Distribution-free risk-control sets | Exchangeability | Static or near-static data | Weak direct transfer; adaptive closed-loop may violate assumptions | No |
| P12 | CRC | 2022 | Risk-level control via conformal calibration | Calibration assumptions (exchangeability-like) | Non-adaptive decision settings | Potentially useful wrapper; guarantees may weaken in closed-loop | Partial |
| P13 | Policy Gradient for Coherent Risk | NeurIPS 2015 | Optimizes policies for formal risk measures (e.g., CVaR) | Risk estimator quality and policy gradient validity | RL tasks with stable return estimates | Conceptually relevant formal risk anchor; not candidate tau auditing | Untested |
| P14 | PETS | NeurIPS 2018 | Uncertainty-aware MPC improves control/sample efficiency | Dynamics model validity in visited regions | Low-to-mid complexity control tasks | Partial transfer; no calibrated probability or tau decision metrics | Partial |
| P15 | RAP | CoRL 2022 | Risk-aware planning objective with uncertain prediction | Risk functional and forecasts remain informative | Interactive planning scenes tested | Closest; still lacks calibration-to-decision causal tests | No |
| P16 | Hierarchical Game-Theoretic Planning | ICRA 2019 | Interactive risk-aware behavior from strategic modeling | Human response model fidelity | Structured multi-agent interactions | Partial; surrogate utilities, no probability calibration | Partial |
| P17 | Safe Occlusion-Aware Active Perception | RSS 2021 | Better safety under occlusion with risk-aware planning | Occlusion proxy tracks true hazard | Occlusion-dominated scenarios | Closest family; still no tau-level calibration audit | Partial |
| P18 | CARLA | CoRL 2017 | Closed-loop AV benchmark for safety/infractions | Simulator realism | Scenario set aligns with tested behavior | Useful evaluation substrate, not evidence for/against hypothesis | Untested |
| P19 | nuPlan | NeurIPS D&B 2021 | Standardized closed-loop planner evaluation | Scenario diversity sufficient | Planner-comparison pipelines | Good substrate; no built-in decision-calibration diagnostics | Untested |
| P20 | SafeBench | NeurIPS D&B 2022 | Stress-testing AV safety robustness | Stressors represent deployment risk | Adversarial/stress regimes | Supports need to test under shift | No |
| P21 | WOSAC | NeurIPS D&B 2023 | Large-scale interactive closed-loop evaluation | Benchmark metrics correlate with deployment quality | Interactive multi-agent simulation | Useful substrate; no direct tau audit | Untested |
| P22 | Waymax | arXiv 2023 | Fast, scalable closed-loop sim for controlled experiments | Simulator shifts are meaningful proxies | Large-scale scenario sweeps | High transfer for our experiments; methodology still external | Untested |

---

## What Prior Work Gets Right (Balanced)

## A) Cases where uncertainty/risk works well

1. Risk-aware objectives improve behavior in interactive planning (P15, P16, P17).
2. Uncertainty-aware control can improve control quality/sample efficiency (P14).
3. Formal risk-sensitive optimization is viable (P10, P13).

## B) Cases where calibration can be sufficient

1. In stable in-domain settings, simple post-hoc calibration often improves probabilities enough for threshold decisions (P01, P02, P06).
2. Selective prediction frameworks show thresholding can control risk when confidence ranking is strong and stationary (P07, P08).

## C) Cases where decision constraints succeed

1. Budgeted constraints can reduce violations when risk/cost surrogates are faithful (P10).
2. Conformal/RCPS style methods can enforce risk targets in suitable non-adaptive settings (P11, P12).

---

## Failure-Source Decomposition (Literature-Grounded)

| Failure source | Definition | Evidence in literature | Support strength | Status for our setting |
|---|---|---|---|---|
| Signal failure | Risk/uncertainty has weak predictive power for outcome | Many methods assume signal quality; fewer directly test candidate-level predictive power in AV closed-loop | Medium | Unclear; must test directly via AUROC/AUPRC + ranking at candidate level |
| Calibration failure | Scores are predictive but numerically wrong probabilities (esp. near tau) | Strong evidence calibration errors exist and worsen under shift (P01, P05, P06) | High | Likely relevant; needs tau-local calibration checks |
| Decision-rule failure | Even good probabilities mapped by a poor rule produce bad outcomes | Selective and constrained methods show rules matter (P07, P10), but closed-loop AV candidate-level evidence is sparse | Medium | Unclear; requires ablations/oracle tests |

Interpretation:
1. Calibration failure is strongly supported as a general risk.
2. Signal failure and decision-rule failure are both plausible in our setting but under-tested in prior AV literature.
3. Therefore, negative outcomes cannot be attributed to calibration alone without decomposition tests.

---

## Counter-Evidence and Boundary Conditions

Counter-evidence to a strong version of our hypothesis:
1. In stable IID regimes, temperature scaling or Platt scaling may be enough for threshold decisions (P01, P02).
2. Constraint-based control can work well without explicit calibration if constraint costs are already faithful (P10).
3. Some risk-aware planners improve safety with surrogate risks despite no probability calibration (P15, P17).

Boundary conditions where our hypothesis may fail:
1. High-quality signal + mild shift: calibration-to-decision gap may be small.
2. Very low positive event rate at operating `tau`: false-safe estimates become statistically weak.
3. Dominant controller bottleneck: better calibration may not improve outcomes if candidate set/rule is the real failure.

---

## Refined, Defensible Research Gap

Limited and precise gap:
- Prior work has strong components (calibration, risk-aware objectives, constraints, benchmarks), but rarely evaluates whether risk scores are decision-grade for **candidate-level tau-threshold decisions in closed-loop AV under shift**, while separating signal vs calibration vs decision-rule failure.

This avoids claiming all existing methods fail. It claims the **evaluation and decomposition protocol** is missing.

---

## Explicit Limitations of Our Hypothesis

Where hypothesis may not hold:
1. If candidate-level risk signal is already strong and well-ranked, simple calibration may suffice.
2. If decision failures are dominated by candidate generation or controller logic, calibration improvements may not help.
3. If shift is weak, measured benefits of robust calibration may be small.

Evidence required to validate or falsify our hypothesis:
1. Signal tests: candidate-level discriminative metrics (AUROC/AUPRC, ranking consistency).
2. Calibration tests: tau-local reliability and shift-wise ECE/NLL/Brier.
3. Decision tests: `false_safe`, `safe_reject`, `feasible_set_rate`, `fallback_rate` vs `tau`.
4. Causal ablations: raw vs calibrated risk under same rule; oracle-risk under same rule; same risk under alternative rules.

---

## Balanced Conclusion

What prior work gets right:
1. Uncertainty/risk can improve planning and constraints can improve safety.
2. Calibration can materially improve probability reliability.
3. Closed-loop AV benchmarks are mature enough for rigorous testing.

What remains unclear:
1. Whether calibration improvements consistently translate to better candidate-level closed-loop decisions at `tau`.
2. Whether failures arise from weak signal, poor calibration, or poor decision rules.
3. How stable threshold behavior is under realistic shift suites.

This uncertainty is exactly why a falsifiable, decomposition-first methodology is warranted.

---

## Links to Local PDFs

All referenced PDFs are stored under:
`experiments/risk-uq-suite/references/pdfs/`
