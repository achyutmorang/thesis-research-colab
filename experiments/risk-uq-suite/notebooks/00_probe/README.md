# 00_probe

## Objective
Establish whether planner-side confidence/risk estimates are miscalibrated and whether thresholded decisions are over/under-confident.

## Notebooks
- `miscalibration_probe_colab.ipynb`: Generates probe artifacts from candidate-level data and computes initial calibration/decision diagnostics.
- `miscalibration_interpretation_colab.ipynb`: Loads probe artifacts and produces interpretable plots/verdicts.

## Inputs
- Existing run artifacts under `PERSIST_ROOT` (or current run prefix).

## Outputs
- Probe summaries, reliability bins, threshold diagnostics, interpretation plots and summary tables.

## Role In Risk-UQ-Suite
Defines whether the core problem exists before model/controller method development.
