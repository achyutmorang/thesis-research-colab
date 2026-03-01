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

## Latest Output Interpretation (from committed notebook cell outputs)

### Evidence snapshot
- `miscalibration_probe_colab.ipynb` gates passed in the saved run:
  - `risk_probe_pass = True`
  - `preflight_pass = True`
  - `overall_pass = True`
- Saved run evidence (prefix: `risk_uq_20260228_165758`) shows large raw-to-calibrated reliability gains for the combo proxy on `failure_proxy_h15`:
  - Raw: high calibration error / poor probabilistic quality.
  - Platt-calibrated: substantially improved ECE/NLL/Brier.
- `miscalibration_interpretation_colab.ipynb` claim table in saved outputs reports:
  - Miscalibration exists: `supported`
  - Calibration improves reliability: `supported`
  - Over-confidence at `tau=0.2`: `inconclusive`
  - Under-confidence (safe-reject tendency): `supported`
  - Shift worsening in this run: `not_supported`
  - Overall problem framing validity: `supported`

### Evolved understanding at probe stage
- The strongest evidence so far is not "unsafe over-confidence"; it is "raw proxy conservatism + miscalibration".
- Calibration appears to make thresholded decisions less degenerate (non-zero acceptance where raw can reject nearly everything).
- Current probe supports continuing to decision-level audits, but not yet making broad shift-robust over-confidence claims.
