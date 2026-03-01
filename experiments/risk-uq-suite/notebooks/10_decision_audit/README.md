# 10_decision_audit

## Objective
Audit causal links from risk estimates to action decisions using fixed candidate sets and multiple controller/risk formulations.

## Notebooks
- `decision_audit_artifact_builder_colab.ipynb`: Runs the closed-loop probe producer stage and writes all prerequisite artifacts to persistent Drive storage for audit notebooks (fresh on first launch, then auto-resume after Colab restart via Drive run-tag sentinel).
- `decision_audit_artifact_builder_harder_colab.ipynb`: Harder producer preset with larger scenario/eval support and stronger high-interaction pressure so downstream tau-sweep and bottleneck estimates are more statistically decisive.
- `cross_signal_decision_audit_colab.ipynb`: Compares multiple risk/uncertainty signals (raw + calibrated) with tau-sweep diagnostics.
- `oracle_bottleneck_colab.ipynb`: Decomposes bottlenecks into signal/calibration vs decision rule vs candidate quality.
- `planner_practice_method_benchmark_colab.ipynb`: Future-facing benchmark for paper-inspired controller families (chance-gate, RACP-style rerank, conformal gate).

## Inputs
- Probe/cross-signal artifacts from prior runs.

## Outputs
- Signal-level and controller-level tradeoff tables, bottleneck diagnosis, conformal threshold summaries.

## Role In Risk-UQ-Suite
Provides decision-grade evidence before closed-loop paper claims.

## Recommended Run Sequence (Decision-Audit-Only Path)
1. `decision_audit_artifact_builder_harder_colab.ipynb` (recommended for decisive evidence)
2. `cross_signal_decision_audit_colab.ipynb`
3. `oracle_bottleneck_colab.ipynb`
4. `planner_practice_method_benchmark_colab.ipynb`

## Alternative Fast Path
1. `decision_audit_artifact_builder_colab.ipynb`
2. `cross_signal_decision_audit_colab.ipynb`
3. `oracle_bottleneck_colab.ipynb`

## Latest Output Interpretation (from committed notebook cell outputs)

### Evidence snapshot
- `cross_signal_decision_audit_colab.ipynb` has saved outputs for focus label `failure_proxy_h15` at `tau=0.2`.
- Signals evaluated in saved run:
  - `belief_kl`, `combo`, `entropy`, `inv_distance`, `inv_ttc`, `stdmax`, `top1`
- Split sizes in saved outputs:
  - `calibration_rows = 520`
  - `evaluation_rows = 520`
  - `variant_count = 21`
- Tau-sweep diagnostics are heavily data-limited in the saved run:
  - `tau_sweep_df rows = 672`
  - `inconclusive tau rows = 635`
- Near-budget pattern (from final summary tables):
  - Raw combo can collapse acceptance (`accept_rate ~ 0.0`, `fallback ~ 1.0` in reported rows).
  - Calibrated combo variants show materially higher acceptance (`~0.60-0.65`) with lower fallback (`~0.35-0.40`) in reported rows.

### Evolved understanding at decision-audit stage
- Current evidence points to a conservative-decision failure mode under raw proxy thresholds.
- Calibration can improve decision usability (feasibility/acceptance), but many rows remain statistically inconclusive due to low event/support counts.
- Saved outputs in this repo do not yet contain executed results for:
  - `oracle_bottleneck_colab.ipynb`
  - `planner_practice_method_benchmark_colab.ipynb`
  Their interpretation remains pending committed execution outputs.
