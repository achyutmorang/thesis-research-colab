from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(module_name: str, rel_path: str):
    root = Path(__file__).resolve().parents[1]
    path = root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load module {module_name} from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


drive_sync = _load_module('trackb_drive_api_sync', 'src/trackb/drive_api_sync.py')


def test_coerce_path_parts_from_string():
    parts = drive_sync._coerce_path_parts('/alpha/beta//gamma/')
    assert parts == ['alpha', 'beta', 'gamma']


def test_coerce_path_parts_from_sequence():
    parts = drive_sync._coerce_path_parts(['alpha', ' ', 'beta', '', 'gamma'])
    assert parts == ['alpha', 'beta', 'gamma']


def test_md5sum_stable(tmp_path: Path):
    p = tmp_path / 'sample.txt'
    p.write_text('trackb-drive-api-sync', encoding='utf-8')
    digest1 = drive_sync._md5sum(p)
    digest2 = drive_sync._md5sum(p)
    assert digest1 == digest2
    assert len(digest1) == 32

