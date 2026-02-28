from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .layout import experiment_pack_paths
from .spec import normalize_slug


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, content: str, *, overwrite: bool) -> bool:
    if path.exists() and (not overwrite):
        return False
    _ensure_parent(path)
    path.write_text(content)
    return True


def _default_notebook(title: str, slug: str) -> Dict[str, Any]:
    return {
        'cells': [
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    f'# {title}\n',
                    '\n',
                    '## Objective\n',
                    '- Reproduce and extend a paper-inspired experiment on Waymax/WOMD.\n',
                    '- Keep notebook orchestration-only and move logic to `src/` modules.\n',
                    '- Persist all outputs/checkpoints to Drive-backed storage.\n',
                ],
            },
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '## Methodology Snapshot\n',
                    '- Define baseline and counterfactual variants.\n',
                    '- Run quick diagnostics before full-budget execution.\n',
                    '- Export artifact manifests with run metadata for reproducibility.\n',
                ],
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    '# Bootstrap + imports\n',
                    "from src.workflows import load_notebook_contract_manifest\n",
                    f"EXPERIMENT_SLUG = '{slug}'\n",
                    "print('Experiment:', EXPERIMENT_SLUG)\n",
                ],
            },
        ],
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


def scaffold_experiment_pack(
    *,
    repo_root: str | Path,
    slug: str,
    title: str,
    objective: str,
    overwrite: bool = False,
) -> Dict[str, List[str]]:
    pack_slug = normalize_slug(slug)
    paths = experiment_pack_paths(repo_root=repo_root, slug=pack_slug)

    created: List[str] = []
    skipped: List[str] = []

    pack_readme = f"""# {title}

## Objective
{objective}

## Contents
- Notebook: `experiments/{pack_slug}/notebooks/{pack_slug}_colab.ipynb`
- Workflow: `src/workflows/{pack_slug.replace('-', '_')}_flow.py`
- Config: `configs/experiments/{pack_slug}.json`
- Paper-specific module: `src/experiments/papers/{pack_slug.replace('-', '_')}/`
"""
    if _write_text(paths['pack_dir'] / 'README.md', pack_readme, overwrite=overwrite):
        created.append(str(paths['pack_dir'] / 'README.md'))
    else:
        skipped.append(str(paths['pack_dir'] / 'README.md'))

    config_payload = {
        'slug': pack_slug,
        'title': title,
        'objective': objective,
        'run': {
            'run_tag_prefix': pack_slug.replace('-', '_'),
            'persist_root': '/content/drive/MyDrive/waymax_experiments',
            'n_shards': 1,
            'shard_id': 0,
        },
    }
    cfg_text = json.dumps(config_payload, indent=2, sort_keys=True) + '\n'
    if _write_text(paths['config_file'], cfg_text, overwrite=overwrite):
        created.append(str(paths['config_file']))
    else:
        skipped.append(str(paths['config_file']))

    module_init = (
        '"""Paper-specific reusable code for this experiment pack."""\n\n'
        '__all__ = []\n'
    )
    if _write_text(paths['module_dir'] / '__init__.py', module_init, overwrite=overwrite):
        created.append(str(paths['module_dir'] / '__init__.py'))
    else:
        skipped.append(str(paths['module_dir'] / '__init__.py'))

    workflow_stub = f"""from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class {pack_slug.replace('-', ' ').title().replace(' ', '')}Bundle:
    summary: Dict[str, Any]


def run_{pack_slug.replace('-', '_')}_flow(**kwargs: Any) -> {pack_slug.replace('-', ' ').title().replace(' ', '')}Bundle:
    return {pack_slug.replace('-', ' ').title().replace(' ', '')}Bundle(summary={{'status': 'todo', 'kwargs': dict(kwargs)}})
"""
    if _write_text(paths['workflow_file'], workflow_stub, overwrite=overwrite):
        created.append(str(paths['workflow_file']))
    else:
        skipped.append(str(paths['workflow_file']))

    notebook_obj = _default_notebook(title=title, slug=pack_slug)
    notebook_text = json.dumps(notebook_obj, indent=1) + '\n'
    if _write_text(paths['notebook_file'], notebook_text, overwrite=overwrite):
        created.append(str(paths['notebook_file']))
    else:
        skipped.append(str(paths['notebook_file']))

    return {'created': created, 'skipped': skipped}
