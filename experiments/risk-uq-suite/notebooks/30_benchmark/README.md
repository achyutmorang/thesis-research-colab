# 30_benchmark

## Objective
Evaluate uncertainty calibration and decision behavior across nominal and shift suites.

## Notebooks
- `uq_benchmark_colab.ipynb`: Runs benchmark suites and exports shift-wise calibration/robustness diagnostics.

## Inputs
- Trained risk model artifacts and scenario run artifacts.

## Outputs
- Benchmark summary tables, per-shift diagnostics, reliability bins, selective-risk curves.

## Role In Risk-UQ-Suite
Validates robustness and generalization of calibrated risk estimates under shift.

## Latest Output Interpretation (from committed notebook cell outputs)

### Evidence snapshot
- `uq_benchmark_colab.ipynb` currently has no committed executed code-cell outputs in this repository snapshot.

### Evolved understanding at benchmark stage
- Benchmark-stage claims are pending committed execution outputs.
- Once executed and committed, this README should summarize:
  - nominal vs shift calibration gaps
  - selective-risk and feasibility/fallback behavior across suites
  - confidence intervals and statistically inconclusive regions
