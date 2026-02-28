# Experiment Packs

This folder is the paper/idea-facing organization layer for the repository.

Each pack describes one research track and points to:
- thin Colab notebook(s)
- workflow entrypoints
- reusable `src/` modules
- pack-specific default config JSON

## Existing Packs
- `closedloop-simulation`
- `surprise-potential`
- `closedloop-evaluation`
- `risk-uq-suite`

## Add A New Paper Idea
Use:

```bash
python scripts/new_experiment.py \
  --slug my-paper-replication \
  --title "My Paper Replication" \
  --objective "Replicate and extend XYZ under the repository design contract."
```

This scaffolds:
- `experiments/<slug>/README.md`
- `configs/experiments/<slug>.json`
- `experiments/<slug>/notebooks/<slug>_colab.ipynb`
- `src/workflows/<slug>_flow.py`
- `src/experiments/papers/<slug>/__init__.py`

For manual starts, use:
- `notebooks/templates/paper_experiment_colab_template.ipynb`
