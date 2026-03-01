# First-Principles Guide: Risk, Uncertainty, Calibration, and Decision Failures in AD Planning

This document is a reasoning scaffold for the main risk-UQ research idea.  
Goal: help you test whether the problem framing is true, not just plausible.

## 1) Core Problem in One Line

At each timestep, a controller accepts/rejects candidate actions using a risk signal.  
If that signal is poorly calibrated or weakly informative, threshold decisions can become unsafe or overly conservative.

---

## 2) Minimal Formal Setup

Let `x` be candidate context (scene + candidate action + short rollout summaries + uncertainty features).

- Latent true risk: `p(x) = P(Y=1 | x)`.
- Estimated risk used in control: `p_hat(x)`.
- Threshold decision: `D(p_hat, tau) = 1[p_hat <= tau]` where `1` means accepted as safe.

Decision error metrics:
- `false_safe(tau) = P(Y=1 | p_hat <= tau)` (unsafe accepted).
- `safe_reject(tau) = P(Y=0 | p_hat > tau)` (safe rejected).
- `feasible_set_rate(tau) = P(exists candidate with p_hat <= tau)`.
- `fallback_rate(tau) = 1 - feasible_set_rate(tau)` for hard-gate controllers.

Ideal operating-point target (not a guarantee):
- `P(Y=1 | p_hat <= tau) <= tau`.

---

## 3) AD Planner + Controller Architecture (Who Estimates Risk vs Who Decides?)

Typical decomposition in modern stacks:

1. `Perception/State`:
- Builds scene representation and tracked-agent state.

2. `Prediction/Planner model`:
- Produces future hypotheses, trajectory candidates, or action distributions.
- Often also emits uncertainty/confidence-style outputs.

3. `Risk estimator (explicit or implicit)`:
- Converts model outputs into a risk quantity.
- Can be:
  - direct probability head,
  - surrogate proxy (TTC, distance, entropy, top1 weight),
  - uncertainty set or chance-constraint bound.

4. `Decision controller`:
- Uses risk in a rule:
  - risk penalty in objective,
  - hard threshold gate (`p_hat <= tau`),
  - constrained optimization (chance/risk budget),
  - reject/fallback mechanism.

5. `Dynamics + closed loop`:
- Selected action changes next state distribution, so data are policy-dependent.

Key distinction:
- Planner/predictor often provides candidate actions + uncertainty.
- Controller applies the decision rule that turns those into one committed action.

---

## 4) How Prior Work Defines Risk Proxies and Uncertainty

Common definitions in literature:

1. `Explicit probabilistic risk`:
- Failure/collision probability predicted directly.
- Seen in risk-aware forecasting/planning pipelines and calibrated classifiers.

2. `Surrogate safety proxy`:
- TTC-based hazard, minimum distance, occupancy overlap, reachability margin.
- Often not a calibrated probability; still used as decision score/constraint.

3. `Uncertainty-derived proxy`:
- Entropy, ensemble variance, confidence/top-1 mass, belief divergence.
- Assumption: higher uncertainty correlates with higher failure risk.

4. `Set-valued uncertainty`:
- Conformal sets, reachable sets, robust uncertainty sets.
- Used to gate decisions or enforce coverage/risk bounds.

Important:
- In many papers, risk used in control is a surrogate/proxy, not necessarily a calibrated probability.

---

## 5) Are Risk Proxies/Uncertainty Used for Control Decisions in Most Implementations?

Yes, widely, via three controller styles:

1. `Penalty style`:
- `score = progress - lambda * risk`.
- Example family: risk-aware contingency planning and risk-sensitive objectives.

2. `Constraint/budget style`:
- Accept only if `risk <= tau` or satisfy chance constraint.
- Example family: chance-constrained MPC, risk-budget planning, conformal risk control.

3. `Reject/fallback style`:
- If no candidate passes safety filter, fallback to safer default.
- Common in selective and safety-filtered controllers.

So the field already uses risk in decisions; the open question is whether that risk is decision-grade at the operating threshold.

---

## 6) Does Miscalibration Really Exist? What Strong Evidence Exists?

Strong evidence in the broader ML/UQ literature:

1. Guo et al. (ICML 2017):
- Modern deep nets can be miscalibrated; temperature scaling is a strong baseline.

2. Ovadia et al. (NeurIPS 2019):
- Uncertainty quality degrades under distribution shift.

3. Conformal/CRC lines (RCPS, CRC, robust conformal papers):
- Explicitly motivated by controlling risk under uncertainty and distributional mismatch.

What this means for AD:
- Miscalibration is a real, documented phenomenon.
- But AD-specific closed-loop, candidate-level threshold correctness is still under-tested.

---

## 7) Is Miscalibration Planner-Independent?

Partly.

Planner-independent component:
- Any learned score/probability can be miscalibrated under mismatch or shift.

Planner-dependent component:
- Severity and shape of miscalibration depend on:
  - planner architecture,
  - candidate generator,
  - feature representation,
  - environment distribution.

Conclusion:
- Miscalibration is not unique to one planner.
- Its practical impact is planner/controller/policy dependent and must be measured per stack.

---

## 8) Is Our Risk Formulation LatentDriver-Dependent?

Current implementation:
- Uses LatentDriver outputs and uncertainty traces to build candidate-level signals.

What is specific:
- Some raw features come from LatentDriver predictive distribution/belief traces.

What is general:
- Decision-audit protocol:
  - calibrate score,
  - evaluate `false_safe/safe_reject/feasibility/fallback`,
  - test nominal vs shift,
  - run oracle bottleneck ablations.

So method logic is portable; feature extractors are backend-specific.

---

## 9) Do We Have a Strong Paper Risk Definition We Can Adopt?

Yes. Strong anchors:

1. `Planning on a (Risk) Budget` (Huang et al., 2021):
- Directly frames safety-performance compromise under a risk budget.
- Good conceptual anchor for `tau`-style decisions.

2. `Safe Chance-Constrained MPC under GMM Uncertainty` (Ren et al., 2024):
- Strong formal chance-constraint framing.

3. `RACP` (Mustafa et al., 2024):
- Risk-penalized objective with controllable conservatism-efficiency tradeoff.

Adoption strategy:
- Recreate these controller styles on the same candidate set.
- Compare with your current controller under identical artifacts.

---

## 10) Why Train a Separate Risk Model?

Because planner confidence/uncertainty is not automatically a calibrated failure probability.

Training a risk model provides:

1. Mapping from features/proxies to event probabilities (`collision/offroad/failure_proxy`).
2. Multi-horizon event outputs aligned with controller decision horizon.
3. A calibratable object (`p_hat`) for threshold decisions.
4. Better decomposition:
- Is failure due to weak signal, poor calibration, or poor rule?

Without a trained/calibrated risk model, you can still run proxy baselines, but causal attribution is weaker.

---

## 11) How to Show Miscalibration Causes Decision Failures

Use causal-style ablations with fixed candidate sets:

1. `Raw vs calibrated (same rule, same candidates)`:
- Isolates calibration effect.

2. `Current vs oracle-risk (same rule, same candidates)`:
- Tests signal/calibration ceiling.

3. `Oracle-risk vs oracle-best`:
- Tests rule quality vs candidate quality.

4. `Tau sweep + CI`:
- Evaluate decision behavior across `tau in [0.05, 0.8]`.

5. `Nominal vs shift`:
- Check instability and error amplification.

If calibration improves reliability but not decisions:
- bottleneck is likely rule/candidate quality, not calibration alone.

---

## 12) How to Show Miscalibrated Risk Leads to Higher Failure

Minimum evidence chain:

1. `Calibration failure`:
- Reliability/ECE/NLL/Brier and local near-`tau` error.

2. `Decision error`:
- Elevated `false_safe` or `safe_reject` at operating `tau`.

3. `System outcome`:
- Closed-loop failure/progress changes under matched conditions.

4. `Attribution`:
- Oracle ablations rule out “it was only candidate quality” explanations.

Claim discipline:
- If false-safe is statistically inconclusive due to low positives, state that explicitly and avoid over-claiming.

---

## 13) Why Decision Errors Worsen Under Shift (What Shift Means)

Shift = deployment distribution differs from calibration/training distribution.

Common shift types in AD simulation:

1. `Covariate shift`:
- Scene/context statistics differ (traffic density, interaction intensity, road topology).

2. `Behavioral/interaction shift`:
- Agent behavior policies differ from train/calibration regime.

3. `Temporal/horizon shift`:
- Future trajectories less predictable in new regimes.

4. `Pipeline/simulator shift`:
- Different perturbation regimes, planner settings, or model versions.

Why worse decisions happen:
- Threshold decisions are local near `tau`; small probability distortions can flip accept/reject outcomes.
- Closed-loop feedback amplifies these flips over time.

---

## 14) Strong Evidence We Already Have vs What Is Still Needed

Already strong from literature:

1. Miscalibration exists in modern models.
2. Shift harms uncertainty reliability.
3. Risk/uncertainty is widely used in downstream decision rules.

Still needed for your paper:

1. Candidate-level, closed-loop evidence in your stack.
2. Decisive `tau`-level FS/SR/feasibility curves with CIs.
3. Bottleneck attribution (signal vs calibration vs rule vs candidate quality).

---

## 15) First-Principles Research Checklist (Before Making Claims)

Use this checklist to keep framing scientifically tight:

1. Did we show risk signal has discrimination (AUROC/AUPRC/within-step AUC)?
2. Did we test global and local calibration near `tau`?
3. Did we report `false_safe`, `safe_reject`, `feasible_set`, `fallback` with uncertainty intervals?
4. Did we compare nominal and shift suites?
5. Did we run oracle bottleneck ablations?
6. Are accepted-sample and positive counts sufficient for decisive claims?
7. Are claims limited to what metrics statistically support?

If any answer is no, problem framing is still plausible but not yet decisively established.

---

## 16) Practical Positioning for the Main Idea

A defensible statement:

`We study whether uncertainty-derived risk used in candidate-level threshold decisions is decision-grade in closed-loop AD under shift, and whether calibration and controller redesign improve safety-efficiency outcomes.`

This is strong, falsifiable, and aligned with current literature and your implementation path.
