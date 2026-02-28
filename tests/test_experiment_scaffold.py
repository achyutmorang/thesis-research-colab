from __future__ import annotations

import json
from pathlib import Path

from src.experiments import experiment_pack_paths, scaffold_experiment_pack


def test_scaffold_creates_pack_files(tmp_path: Path) -> None:
    summary = scaffold_experiment_pack(
        repo_root=tmp_path,
        slug='my-paper-pack',
        title='My Paper Pack',
        objective='Check scaffold output.',
        overwrite=False,
    )
    assert summary['created']
    assert summary['skipped'] == []

    paths = experiment_pack_paths(tmp_path, 'my-paper-pack')
    assert paths['pack_dir'].exists()
    assert paths['config_file'].exists()
    assert paths['workflow_file'].exists()
    assert paths['module_dir'].exists()
    assert paths['notebook_file'].exists()

    nb = json.loads(paths['notebook_file'].read_text())
    assert nb['nbformat'] == 4
    assert len(nb['cells']) >= 3


def test_scaffold_is_idempotent_without_overwrite(tmp_path: Path) -> None:
    scaffold_experiment_pack(
        repo_root=tmp_path,
        slug='my-paper-pack',
        title='My Paper Pack',
        objective='Check scaffold output.',
        overwrite=False,
    )
    summary = scaffold_experiment_pack(
        repo_root=tmp_path,
        slug='my-paper-pack',
        title='My Paper Pack',
        objective='Check scaffold output.',
        overwrite=False,
    )
    assert summary['created'] == []
    assert len(summary['skipped']) >= 5

