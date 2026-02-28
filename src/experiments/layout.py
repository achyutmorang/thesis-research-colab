from __future__ import annotations

from pathlib import Path
from typing import Dict

from .spec import normalize_slug


def recommended_repo_layout() -> Dict[str, str]:
    return {
        'experiments/': 'Experiment packs (one folder per paper or idea).',
        'configs/experiments/': 'Small JSON configs used by notebooks and workflows.',
        'notebooks/': 'Thin orchestration notebooks for Colab.',
        'src/': 'Reusable Python modules and workflow entry points.',
        'src/workflows/': 'Notebook-facing orchestration APIs.',
        'src/experiments/': 'Pack registry, scaffolding, and shared experiment contracts.',
    }


def experiment_pack_paths(repo_root: str | Path, slug: str) -> Dict[str, Path]:
    root = Path(repo_root).expanduser().resolve()
    pack_slug = normalize_slug(slug)
    return {
        'pack_dir': root / 'experiments' / pack_slug,
        'config_file': root / 'configs' / 'experiments' / f'{pack_slug}.json',
        'notebook_file': root / 'experiments' / pack_slug / 'notebooks' / f'{pack_slug}_colab.ipynb',
        'workflow_file': root / 'src' / 'workflows' / f'{pack_slug.replace("-", "_")}_flow.py',
        'module_dir': root / 'src' / 'experiments' / 'papers' / pack_slug.replace('-', '_'),
    }
