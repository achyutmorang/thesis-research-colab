# thesis-research-colab

Modular Colab-first repo for PRiSM Track B closed-loop experiments.

## Open in Colab
- Notebook: [PRiSM_trackB_closedloop_simulation_colab.ipynb](https://colab.research.google.com/github/achyutmorang/thesis-research-colab/blob/main/PRiSM_trackB_closedloop_simulation_colab.ipynb)

## Structure
- `PRiSM_trackB_closedloop_simulation_colab.ipynb`: thin orchestration notebook
- `src/trackb/config.py`: config dataclasses and persistence/checkpoint utilities
- `src/trackb/metrics.py`: risk/surprise metric primitives and scaling
- `src/trackb/latentdriver.py`: planner wiring, rollout, predictive-KL utilities
- `src/trackb/calibration.py`: preflight checks, calibration, surprise quality gate
- `src/trackb/search.py`: method objective evaluation and optimization loops
- `src/trackb/resume_io.py`: checkpoint resume and artifact export helpers
- `src/trackb/core.py`: high-level orchestration over split modules

## Colab workflow
1. Open notebook from the Colab link above.
2. Run repo sync cell (`git clone`/`git pull`).
3. Install dependencies only when needed (fresh runtime).
4. Keep `PERSIST_ROOT`, `RUN_TAG`, `N_SHARDS`, `SHARD_ID` stable for resume.

## Notes
- Runtime artifacts (`*.csv`, `*.json`, checkpoints) are intentionally not versioned.
- Save executed notebook outputs back to GitHub only when needed.
