# 20_modeling

## Objective
Train the risk model, calibrate outputs, and export reusable artifacts for downstream control and benchmarking.

## Notebooks
- `risk_model_training_colab.ipynb`: Dataset build, model training, calibration fitting, checkpoint/artifact export.

## Inputs
- Candidate-level dataset artifacts and run config.

## Outputs
- Model checkpoints, calibration scalers/thresholds, reliability summaries, validation predictions.

## Role In Risk-UQ-Suite
Creates the decision-grade risk model used by benchmark and controller evaluations.
