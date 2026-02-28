from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _to_serializable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in list(value)]
    if hasattr(value, '__dict__'):
        return _to_serializable(vars(value))
    return str(value)


def _safe_version(dist_name: str) -> str:
    try:
        from importlib.metadata import version
        return str(version(dist_name))
    except Exception:
        return 'not_installed'


def _detect_colab_runtime_type() -> str:
    if str(os.environ.get('COLAB_TPU_ADDR', '')).strip():
        return 'tpu'
    if str(os.environ.get('COLAB_GPU', '')).strip():
        return 'gpu'
    if str(os.environ.get('NVIDIA_VISIBLE_DEVICES', '')).strip() not in {'', 'void', 'none'}:
        return 'gpu'
    return 'cpu'


def _resolve_git_commit(repo_dir: Optional[str], fallback: Optional[str]) -> str:
    if isinstance(fallback, str) and fallback.strip():
        return str(fallback).strip()
    if not repo_dir:
        return 'unknown'
    try:
        out = subprocess.check_output(
            ['git', '-C', str(repo_dir), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
        )
        commit = out.decode('utf-8', errors='ignore').strip()
        return commit or 'unknown'
    except Exception:
        return 'unknown'


def _manifest_path(run_prefix: str) -> Path:
    return Path(f'{run_prefix}_notebook_contract_manifest.json')


def _cfg_hash(cfg: Any, search_cfg: Any) -> str:
    payload = {
        'cfg': _to_serializable(cfg),
        'search_cfg': _to_serializable(search_cfg),
    }
    wire = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(wire.encode('utf-8')).hexdigest()


def load_notebook_contract_manifest(run_prefix: str) -> Dict[str, Any]:
    p = _manifest_path(run_prefix)
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text())
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def manifest_has_stage(manifest: Mapping[str, Any], stage: str) -> bool:
    target = str(stage).strip().lower()
    if not target:
        return False
    events = manifest.get('events', [])
    if isinstance(events, list):
        for evt in events:
            if isinstance(evt, dict) and str(evt.get('stage', '')).strip().lower() == target:
                return True
    if str(manifest.get('stage', '')).strip().lower() == target:
        return True
    return False


def validate_notebook_contract_manifest(
    manifest: Mapping[str, Any],
    *,
    require_quick_probe: bool = True,
    require_preflight: bool = True,
    required_stages: Optional[Sequence[str]] = None,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not manifest:
        return False, ['manifest_missing']

    for key in (
        'run_tag',
        'git_commit',
        'created_utc',
        'cfg_hash',
        'python_version',
        'package_versions',
        'colab_runtime_type',
        'n_shards',
        'shard_id',
    ):
        if key not in manifest:
            reasons.append(f'missing_{key}')

    if require_quick_probe and (not bool(manifest.get('quick_probe_pass', False))):
        reasons.append('quick_probe_not_passed')
    if require_preflight and (not bool(manifest.get('preflight_pass', False))):
        reasons.append('preflight_not_passed')

    for stage in list(required_stages or ()):
        if not manifest_has_stage(manifest, str(stage)):
            reasons.append(f'missing_stage:{stage}')

    return len(reasons) == 0, reasons


def write_notebook_contract_manifest(
    *,
    run_prefix: str,
    run_tag: str,
    cfg: Any,
    search_cfg: Any,
    n_shards: int,
    shard_id: int,
    notebook_name: str,
    stage: str,
    repo_dir: Optional[str] = None,
    git_commit: Optional[str] = None,
    quick_probe_pass: Optional[bool] = None,
    preflight_pass: Optional[bool] = None,
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> str:
    p = _manifest_path(run_prefix)
    prior = load_notebook_contract_manifest(run_prefix)

    commit = _resolve_git_commit(repo_dir=repo_dir, fallback=git_commit)
    created_utc = str(prior.get('created_utc', '')).strip() or _utc_now_iso()

    package_versions = {
        'numpy': _safe_version('numpy'),
        'pandas': _safe_version('pandas'),
        'torch': _safe_version('torch'),
        'jax': _safe_version('jax'),
    }
    event = {
        'at_utc': _utc_now_iso(),
        'stage': str(stage),
        'notebook_name': str(notebook_name),
        'quick_probe_pass': bool(quick_probe_pass) if quick_probe_pass is not None else None,
        'preflight_pass': bool(preflight_pass) if preflight_pass is not None else None,
    }
    if isinstance(extra_fields, Mapping) and extra_fields:
        event['extra_fields'] = _to_serializable(dict(extra_fields))

    events = prior.get('events', [])
    if not isinstance(events, list):
        events = []
    events.append(event)
    if len(events) > 200:
        events = events[-200:]

    manifest = {
        'run_prefix': str(run_prefix),
        'run_tag': str(run_tag),
        'git_commit': str(commit),
        'created_utc': created_utc,
        'updated_utc': _utc_now_iso(),
        'cfg_hash': _cfg_hash(cfg=cfg, search_cfg=search_cfg),
        'python_version': str(sys.version.split()[0]),
        'platform': str(platform.platform()),
        'package_versions': package_versions,
        'colab_runtime_type': _detect_colab_runtime_type(),
        'n_shards': int(max(1, int(n_shards))),
        'shard_id': int(shard_id),
        'notebook_name': str(notebook_name),
        'stage': str(stage),
        'quick_probe_pass': bool(quick_probe_pass) if quick_probe_pass is not None else bool(prior.get('quick_probe_pass', False)),
        'preflight_pass': bool(preflight_pass) if preflight_pass is not None else bool(prior.get('preflight_pass', False)),
        'events': events,
    }
    if isinstance(extra_fields, Mapping) and extra_fields:
        manifest['extra_fields'] = _to_serializable(dict(extra_fields))
    elif 'extra_fields' in prior:
        manifest['extra_fields'] = prior['extra_fields']

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True))
    return str(p)

