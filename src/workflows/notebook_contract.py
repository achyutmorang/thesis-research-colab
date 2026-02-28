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


def run_risk_training_notebook_gates(
    *,
    runner: Any,
    cfg: Any,
    eval_idx: Optional[Iterable[int]] = None,
    probe_shift_suite: str = 'nominal_clean',
) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd
    from src.closedloop.calibration import run_closedloop_preflight_checks
    from src.closedloop.planner_backends import make_closed_loop_components
    from src.closedloop.risk_candidates import build_candidate_risk_dataset_rows

    scenarios = list(getattr(runner, 'data', {}).get('scenarios', []))
    candidate_ids: List[int] = []
    if eval_idx is not None:
        for sid in eval_idx:
            sid_int = int(sid)
            if 0 <= sid_int < len(scenarios):
                rec = scenarios[sid_int]
                if isinstance(rec, dict) and ('state' in rec):
                    candidate_ids.append(sid_int)
    if len(candidate_ids) == 0:
        for sid_int, rec in enumerate(scenarios):
            if isinstance(rec, dict) and ('state' in rec):
                candidate_ids.append(int(sid_int))
    candidate_ids = sorted(set(int(x) for x in candidate_ids))

    failure_reasons: List[str] = []
    if len(candidate_ids) == 0:
        failure_reasons.append('no_scenarios_with_state')
        return {
            'overall_pass': False,
            'risk_probe_pass': False,
            'preflight_pass': False,
            'failure_reasons': failure_reasons,
            'risk_probe_summary_df': pd.DataFrame(),
            'risk_probe_rows_df': pd.DataFrame(),
            'preflight_df': pd.DataFrame(),
        }

    preflight_idx = np.asarray(candidate_ids[: max(1, min(8, len(candidate_ids)))], dtype=np.int32)
    preflight_df = run_closedloop_preflight_checks(runner=runner, cfg=cfg, eval_idx=preflight_idx)
    preflight_pass = bool((not preflight_df.empty) and ('pass' in preflight_df.columns) and bool(preflight_df['pass'].all()))
    if not preflight_pass:
        failure_reasons.append('preflight_failed')

    sid = int(candidate_ids[0])
    rec = scenarios[sid]
    selected_idx = np.asarray(rec.get('selected_indices', []), dtype=np.int32)
    planner_bundle = make_closed_loop_components(
        rec['state'],
        planner_kind=getattr(cfg, 'planner_kind', 'latentdriver'),
        planner_name=getattr(cfg, 'planner_name', 'latentdriver_waypoint_sdc'),
        cfg=cfg,
    )
    seed = int(getattr(cfg, 'global_seed', 17) + sid * max(1, int(getattr(cfg, 'rollout_seed_stride', 10000))))
    rows = build_candidate_risk_dataset_rows(
        scenario_id=sid,
        state=rec['state'],
        selected_idx=selected_idx,
        planner_bundle=planner_bundle,
        cfg=cfg,
        seed=seed,
        shift_suite=str(probe_shift_suite),
    )
    probe_df = pd.DataFrame(rows)
    finite_numeric = False
    required_columns_ok = False
    if not probe_df.empty:
        numeric_cols = [c for c in probe_df.columns if pd.api.types.is_numeric_dtype(probe_df[c])]
        if len(numeric_cols) > 0:
            finite_numeric = bool(np.isfinite(probe_df[numeric_cols].to_numpy(dtype=float)).all())
        horizon = int(max(1, int(getattr(cfg, 'risk_dataset_control_horizon_steps', 6))))
        required = [
            'dist_entropy',
            f'progress_h{horizon}',
            'collision_h5',
            'offroad_h5',
            'failure_proxy_h15',
        ]
        required_columns_ok = all(col in probe_df.columns for col in required)
    risk_probe_pass = bool((not probe_df.empty) and finite_numeric and required_columns_ok)
    if not risk_probe_pass:
        if probe_df.empty:
            failure_reasons.append('risk_probe_empty')
        if (not finite_numeric) and (not probe_df.empty):
            failure_reasons.append('risk_probe_non_finite_numeric')
        if (not required_columns_ok) and (not probe_df.empty):
            failure_reasons.append('risk_probe_missing_required_columns')

    summary_rows = [
        {'check': 'risk_probe_rows_nonempty', 'pass': int(not probe_df.empty), 'detail': f'rows={len(probe_df)}'},
        {'check': 'risk_probe_numeric_finite', 'pass': int(finite_numeric), 'detail': 'all numeric columns finite'},
        {'check': 'risk_probe_required_columns', 'pass': int(required_columns_ok), 'detail': 'dist_entropy/progress/labels present'},
        {'check': 'preflight_all_checks_pass', 'pass': int(preflight_pass), 'detail': f'n_checks={len(preflight_df)}'},
    ]
    risk_probe_summary_df = pd.DataFrame(summary_rows)
    overall_pass = bool(risk_probe_pass and preflight_pass)
    return {
        'overall_pass': bool(overall_pass),
        'risk_probe_pass': bool(risk_probe_pass),
        'preflight_pass': bool(preflight_pass),
        'failure_reasons': list(failure_reasons),
        'risk_probe_summary_df': risk_probe_summary_df,
        'risk_probe_rows_df': probe_df,
        'preflight_df': preflight_df,
    }


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
