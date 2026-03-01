# Notebooks Layout

This folder is organized by experiment stage:

- `00_probe/`: miscalibration existence checks
- `10_decision_audit/`: closed-loop probe artifact builder + signal/controller causal diagnostics
- `20_modeling/`: risk model training and calibration
- `30_benchmark/`: shift robustness and UQ benchmark evaluation
- `40_paper_exports/`: final publication tables/figures

Run stages in ascending order unless you are intentionally re-running a later stage from existing artifacts.

## Current Interpretation Status (from committed notebook outputs)
- `00_probe`: interpreted from saved outputs.
- `10_decision_audit`: interpreted from `cross_signal_decision_audit_colab.ipynb` saved outputs; oracle/planner-practice notebook outputs not yet committed.
- `20_modeling`: interpretation pending committed execution outputs.
- `30_benchmark`: interpretation pending committed execution outputs.
- `40_paper_exports`: interpretation pending committed execution outputs.
