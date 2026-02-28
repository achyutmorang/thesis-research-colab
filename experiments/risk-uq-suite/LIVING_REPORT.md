# Risk-UQ Living Report

This is the living report for `risk-uq-suite`. It captures evolving problem framing, hypotheses, and evidence.

The file is auto-updated by workflow runs:

- `src/workflows/miscalibration_probe_flow.py`
- `src/workflows/risk_training_flow.py`
- `src/workflows/uq_benchmark_flow.py`
- `src/workflows/paper_export_flow.py`

## Problem Framing
Goal: build a calibrated risk-to-control pipeline in Waymax that reduces unsafe decisions without collapsing progress.

Causal chain being tested:

`miscalibration -> wrong threshold decisions -> false-safe action acceptance -> closed-loop failures`

## Research Questions
1. Are planner-side risk proxies miscalibrated, especially under shift?
2. Does post-hoc calibration improve reliability of learned risk probabilities?
3. Does calibrated risk-aware reranking reduce failures with bounded progress loss?
4. Do gains transfer across nominal and shifted suites?

## Hypotheses To Track
- `H1`: Planner-side risk proxies are miscalibrated under nominal/shift.
- `H2`: Calibration improves reliability metrics over raw risk.
- `H3`: Calibrated reranking reduces failures with <=5% relative progress loss.
- `H4`: Improvements remain on non-nominal shift suites.

## Experiment Sequence
1. Run `miscalibration_probe_colab.ipynb` to validate H1 and threshold-level failure modes.
2. Run `risk_model_training_colab.ipynb` to train/calibrate and validate H2.
3. Run `uq_benchmark_colab.ipynb` to evaluate H2/H3/H4 under shift.
4. Run `paper_tables_figures_colab.ipynb` to export paper-ready tables and figures.

## What Gets Auto-Written
- Hypothesis status updates with evidence snippets.
- Stage snapshots with key metrics and artifact paths.
- Update history across probe/training/benchmark/export runs.

On the first completed workflow run, this file is replaced by a fully structured auto-generated report and then continually refreshed.
