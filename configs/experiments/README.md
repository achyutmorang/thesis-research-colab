# Experiment Configs

Small JSON configs that keep notebook parameters centralized and versionable.

## Conventions
- One file per experiment pack: `configs/experiments/<pack-slug>.json`.
- Keep only orchestration/runtime knobs here.
- Put algorithmic implementation in `src/` modules, not JSON.
- Track config changes with commit hash and run artifacts.

## Typical fields
- `slug`, `title`, `objective`
- `run.run_tag_prefix`
- `run.persist_root`
- `run.n_shards`, `run.shard_id`

