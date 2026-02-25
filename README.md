# thesis-research-colab

Modular Colab-first repo for PRiSM Track B closed-loop experiments.

## Open in Colab
- Notebook: [PRiSM_trackB_closedloop_simulation_colab.ipynb](https://colab.research.google.com/github/achyutmorang/thesis-research-colab/blob/main/PRiSM_trackB_closedloop_simulation_colab.ipynb)

## Structure
- `PRiSM_trackB_closedloop_simulation_colab.ipynb`: thin orchestration notebook
- `src/trackb/config.py`: config dataclasses and persistence/checkpoint utilities
- `src/trackb/core.py`: loader, metrics, closed-loop planner integration, calibration, simulation, summaries, export

## Colab workflow
1. Open notebook from the Colab link above.
2. Run repo sync cell (`git clone`/`git pull`).
3. On a fresh runtime set `RUN_SETUP=True` once, then restart and rerun top-to-bottom.
4. Keep `persist_root`, `run_tag`, `n_shards`, `shard_id` stable for resume.

## Notes
- Runtime artifacts (`*.csv`, `*.json`, checkpoints) are intentionally not versioned.
- Save executed notebook outputs back to GitHub only when needed.
