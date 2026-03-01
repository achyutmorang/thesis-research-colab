# 10_decision_audit

## Objective
Audit causal links from risk estimates to action decisions using fixed candidate sets and multiple controller/risk formulations.

## Notebooks
- `cross_signal_decision_audit_colab.ipynb`: Compares multiple risk/uncertainty signals (raw + calibrated) with tau-sweep diagnostics.
- `oracle_bottleneck_colab.ipynb`: Decomposes bottlenecks into signal/calibration vs decision rule vs candidate quality.
- `planner_practice_method_benchmark_colab.ipynb`: Future-facing benchmark for paper-inspired controller families (chance-gate, RACP-style rerank, conformal gate).

## Inputs
- Probe/cross-signal artifacts from prior runs.

## Outputs
- Signal-level and controller-level tradeoff tables, bottleneck diagnosis, conformal threshold summaries.

## Role In Risk-UQ-Suite
Provides decision-grade evidence before closed-loop paper claims.
