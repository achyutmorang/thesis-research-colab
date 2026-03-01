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

## Latest Output Interpretation (from committed notebook cell outputs)

### Evidence snapshot
- `risk_model_training_colab.ipynb` currently has no committed executed code-cell outputs in this repository snapshot.

### Evolved understanding at modeling stage
- Modeling-stage conclusions are pending a committed executed run with saved outputs.
- Once executed and committed, this README should summarize:
  - training convergence and early-stopping behavior
  - raw vs calibrated validation reliability (ECE/NLL/Brier)
  - artifact completeness and resume behavior
