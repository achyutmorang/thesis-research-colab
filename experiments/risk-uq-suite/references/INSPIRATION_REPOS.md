# Risk/UQ Inspiration Repositories

This paper track reuses ideas from established open repositories while keeping implementation native to this repo.

## Fetched References (local snapshot used)

| Repository | URL | Commit |
|---|---|---|
| `waymax` | https://github.com/waymo-research/waymax | `a64dfec9be8576b60d9cecc94f406d9812d4a7d0` |
| `temperature_scaling` | https://github.com/gpleiss/temperature_scaling | `ce1154ec7d5235eee046f28875c3dd8421f11d45` |
| `uncertainty_baselines` | https://github.com/google/uncertainty-baselines | `f79fcc2b6d3669028258f0e8837bb2f672a81af0` |
| `plancp` | https://github.com/Jiankai-Sun/PlanCP | `98eac96b129661506c5057c857a0bf10c2438fbc` |
| `icp` | https://github.com/tedhuang96/icp | `7e0eaac5b8398481e8b35617063c929a1a1ee5c3` |

## Practical Patterns Adopted

1. **Validation-only temperature calibration**
- Source inspiration: `temperature_scaling/temperature_scaling.py`
- Applied here: fit post-hoc temperature on validation logits only, then freeze for evaluation/control.

2. **Reliability-first diagnostics**
- Source inspiration: uncertainty baselines metric style and reliability reporting.
- Applied here: ECE/NLL/Brier + reliability bins and shift-wise gap summaries.

3. **Conformal risk-threshold diagnostics**
- Source inspiration: PlanCP and ICP conformal-style quantile thresholding workflows.
- Applied here: budgeted accept/reject diagnostics around `risk_control_fail_budget`.

4. **Simulator-native safety metric decomposition**
- Source inspiration: Waymax metric decomposition.
- Applied here: candidate-level rollout features and event labels tied to closed-loop outcomes.

## Fetch Script

Run from repository root:

```bash
bash experiments/risk-uq-suite/references/fetch_inspiration_repos.sh
```

Optional custom destination:

```bash
bash experiments/risk-uq-suite/references/fetch_inspiration_repos.sh /path/to/external_refs
```
