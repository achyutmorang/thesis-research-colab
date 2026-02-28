from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from src.experiments import (
    ExperimentPack,
    experiment_pack_paths,
    get_experiment_pack,
    list_experiment_packs,
)


@dataclass
class ExperimentBootstrapBundle:
    pack: ExperimentPack
    config: Dict[str, Any]
    paths: Dict[str, str]
    summary_df: pd.DataFrame


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_experiment_config(
    *,
    slug: str,
    repo_root: str | Path = '.',
    default_on_missing: bool = True,
) -> Dict[str, Any]:
    paths = experiment_pack_paths(repo_root=repo_root, slug=slug)
    payload = _load_json(paths['config_file'])
    if payload:
        return payload
    if not default_on_missing:
        raise FileNotFoundError(f'Config not found for experiment slug={slug!r}: {paths["config_file"]}')
    pack = get_experiment_pack(slug)
    return {
        'slug': pack.slug,
        'title': pack.title,
        'objective': pack.objective,
        'run': {'run_tag_prefix': pack.slug.replace('-', '_')},
    }


def bootstrap_experiment_pack(
    *,
    slug: str,
    repo_root: str | Path = '.',
    overrides: Optional[Mapping[str, Any]] = None,
) -> ExperimentBootstrapBundle:
    pack = get_experiment_pack(slug)
    cfg = load_experiment_config(slug=slug, repo_root=repo_root, default_on_missing=True)
    cfg = dict(cfg)
    if overrides:
        cfg.update(dict(overrides))

    paths = experiment_pack_paths(repo_root=repo_root, slug=slug)
    path_map = {k: str(v) for k, v in paths.items()}
    summary_df = pd.DataFrame(
        [
            {'field': 'slug', 'value': pack.slug},
            {'field': 'title', 'value': pack.title},
            {'field': 'objective', 'value': pack.objective},
            {'field': 'notebook', 'value': ', '.join(pack.notebooks)},
            {'field': 'workflow', 'value': ', '.join(pack.workflows)},
            {'field': 'config_file', 'value': path_map['config_file']},
        ]
    )
    return ExperimentBootstrapBundle(
        pack=pack,
        config=cfg,
        paths=path_map,
        summary_df=summary_df,
    )


def list_experiment_pack_table() -> pd.DataFrame:
    rows = []
    for pack in list_experiment_packs():
        rows.append(
            {
                'slug': pack.slug,
                'title': pack.title,
                'n_notebooks': len(pack.notebooks),
                'n_workflows': len(pack.workflows),
                'n_modules': len(pack.modules),
                'tags': ','.join(pack.tags),
            }
        )
    return pd.DataFrame(rows).sort_values(['slug']).reset_index(drop=True)

