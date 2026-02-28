from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.closedloop.config import configure_persistent_run_prefix, initialize_configs
from src.closedloop.core import build_closedloop_runner_and_splits, make_waymax_data_iter
from src.closedloop.planner_backends import make_closed_loop_components
from src.closedloop.risk_candidates import build_candidate_risk_dataset_rows


@dataclass
class RiskUQRunContextBundle:
    cfg: Any
    run_tag: str
    run_prefix: str
    run_mode: str
    persist_root: str
    n_shards: int
    shard_id: int
    planner_backend: str = 'latentdriver'
    ckpt_scan_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class RiskUQSimulationContextBundle:
    dataset_config: Any = None
    data_iter: Optional[Iterable[Any]] = None
    runner: Any = None
    data: Any = None
    reference_idx: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    candidate_eval_idx: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    eval_idx_all: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    eval_idx: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    reference_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    base_eval_openloop_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def _auto_generate_run_tag(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    p = ''.join(ch if (ch.isalnum() or ch in {'-', '_'}) else '_' for ch in str(prefix).strip().lower())
    p = p.strip('_') or 'risk_uq'
    return f'{p}_{stamp}'


def _resolve_shard_id(shard_id: Any, n_shards: int) -> int:
    n = int(max(1, int(n_shards)))
    if isinstance(shard_id, str) and shard_id.strip().lower() == 'auto':
        return 0
    sid = int(shard_id)
    if sid < 0 or sid >= n:
        raise ValueError(f'Invalid shard_id={sid} for n_shards={n}.')
    return sid


def initialize_risk_uq_run_context(
    *,
    run_tag: str,
    run_tag_prefix: str,
    persist_root: str,
    n_shards: int,
    shard_id: Any,
    resume_mode: str = 'auto',
    resume_from_existing: bool = True,
    planner_backend: str = 'latentdriver',
) -> RiskUQRunContextBundle:
    cfg, _search_cfg, ckpt_scan_df = initialize_configs(planner_kind_override=planner_backend)
    n_shards = int(max(1, int(n_shards)))
    shard_id_int = _resolve_shard_id(shard_id, n_shards=n_shards)
    run_tag_clean = str(run_tag).strip() or _auto_generate_run_tag(str(run_tag_prefix))
    run_prefix = configure_persistent_run_prefix(
        cfg=cfg,
        run_tag=run_tag_clean,
        persist_root=str(persist_root),
        shard_id=int(shard_id_int),
        n_shards=int(n_shards),
    )
    cfg.resume_from_existing = bool(resume_from_existing)

    summary_df = pd.DataFrame(
        [
            {'field': 'run_tag', 'value': str(run_tag_clean)},
            {'field': 'run_prefix', 'value': str(run_prefix)},
            {'field': 'run_mode', 'value': str(resume_mode)},
            {'field': 'planner_backend', 'value': str(planner_backend)},
            {'field': 'persist_root', 'value': str(persist_root)},
            {'field': 'n_shards', 'value': int(n_shards)},
            {'field': 'shard_id', 'value': int(shard_id_int)},
            {'field': 'resume_from_existing', 'value': int(bool(resume_from_existing))},
            {'field': 'latentdriver_ckpt_path', 'value': str(getattr(cfg, 'latentdriver_ckpt_path', ''))},
        ]
    )
    return RiskUQRunContextBundle(
        cfg=cfg,
        run_tag=str(run_tag_clean),
        run_prefix=str(run_prefix),
        run_mode=str(resume_mode),
        persist_root=str(persist_root),
        n_shards=int(n_shards),
        shard_id=int(shard_id_int),
        planner_backend=str(planner_backend),
        ckpt_scan_df=ckpt_scan_df,
        summary_df=summary_df,
    )


def report_risk_uq_run_context(bundle: RiskUQRunContextBundle, display_fn: Optional[Any] = None) -> None:
    if display_fn is not None:
        try:
            display_fn(bundle.summary_df)
            if isinstance(bundle.ckpt_scan_df, pd.DataFrame) and (not bundle.ckpt_scan_df.empty):
                display_fn(bundle.ckpt_scan_df.head(20))
            return
        except Exception:
            pass
    print(bundle.summary_df.to_string(index=False))
    if isinstance(bundle.ckpt_scan_df, pd.DataFrame) and (not bundle.ckpt_scan_df.empty):
        print(bundle.ckpt_scan_df.head(20).to_string(index=False))


def build_risk_uq_simulation_context(
    *,
    cfg: Any,
    n_shards: int,
    shard_id: int,
) -> RiskUQSimulationContextBundle:
    dataset_config, data_iter = make_waymax_data_iter(cfg)
    (
        runner,
        data,
        reference_idx,
        candidate_eval_idx,
        eval_idx_all,
        eval_idx,
        reference_df,
        base_eval_openloop_df,
    ) = build_closedloop_runner_and_splits(
        cfg=cfg,
        data_iter=data_iter,
        dataset_config=dataset_config,
        n_shards=int(max(1, int(n_shards))),
        shard_id=int(shard_id),
    )
    return RiskUQSimulationContextBundle(
        dataset_config=dataset_config,
        data_iter=data_iter,
        runner=runner,
        data=data,
        reference_idx=np.asarray(reference_idx, dtype=np.int32),
        candidate_eval_idx=np.asarray(candidate_eval_idx, dtype=np.int32),
        eval_idx_all=np.asarray(eval_idx_all, dtype=np.int32),
        eval_idx=np.asarray(eval_idx, dtype=np.int32),
        reference_df=reference_df,
        base_eval_openloop_df=base_eval_openloop_df,
    )


def run_risk_uq_smoke_gates(
    *,
    runner: Any,
    cfg: Any,
    eval_idx: Optional[Iterable[int]] = None,
    probe_shift_suite: str = 'nominal_clean',
) -> Dict[str, Any]:
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

    sid = int(candidate_ids[0])
    rec = scenarios[sid]
    planner_bundle = None
    planner_ok = False
    planner_detail = ''
    try:
        planner_bundle = make_closed_loop_components(
            rec['state'],
            planner_kind=getattr(cfg, 'planner_kind', 'latentdriver'),
            planner_name=getattr(cfg, 'planner_name', 'latentdriver_waypoint_sdc'),
            cfg=cfg,
        )
        planner_ok = True
        planner_detail = str(planner_bundle.get('planner_used', 'ok'))
    except Exception as exc:
        planner_ok = False
        planner_detail = str(exc)

    probe_df = pd.DataFrame()
    if planner_ok:
        selected_idx = np.asarray(rec.get('selected_indices', []), dtype=np.int32)
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
    non_finite_numeric_cols: List[str] = []
    required_columns_ok = False
    if not probe_df.empty:
        numeric_cols = [c for c in probe_df.columns if pd.api.types.is_numeric_dtype(probe_df[c])]
        if len(numeric_cols) > 0:
            optional_nan_cols = {'action_2', 'planner_action_2'}
            for col in numeric_cols:
                vals = pd.to_numeric(probe_df[col], errors='coerce').to_numpy(dtype=float)
                if col in optional_nan_cols:
                    vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                if not bool(np.isfinite(vals).all()):
                    non_finite_numeric_cols.append(str(col))
            finite_numeric = len(non_finite_numeric_cols) == 0
        horizon = int(max(1, int(getattr(cfg, 'risk_dataset_control_horizon_steps', 6))))
        required = [
            'dist_entropy',
            f'progress_h{horizon}',
            'collision_h5',
            'offroad_h5',
            'failure_proxy_h15',
        ]
        required_columns_ok = all(col in probe_df.columns for col in required)

    risk_probe_pass = bool(planner_ok and (not probe_df.empty) and finite_numeric and required_columns_ok)
    if not risk_probe_pass:
        if not planner_ok:
            failure_reasons.append('planner_bundle_failed')
        if probe_df.empty:
            failure_reasons.append('risk_probe_empty')
        if (not finite_numeric) and (not probe_df.empty):
            failure_reasons.append('risk_probe_non_finite_numeric')
        if (not required_columns_ok) and (not probe_df.empty):
            failure_reasons.append('risk_probe_missing_required_columns')

    summary_rows = [
        {'check': 'planner_bundle_constructed', 'pass': int(planner_ok), 'detail': planner_detail},
        {'check': 'risk_probe_rows_nonempty', 'pass': int(not probe_df.empty), 'detail': f'rows={len(probe_df)}'},
        {
            'check': 'risk_probe_numeric_finite',
            'pass': int(finite_numeric),
            'detail': (
                'all numeric columns finite'
                if finite_numeric
                else f'non_finite_cols={";".join(non_finite_numeric_cols[:10])}'
            ),
        },
        {'check': 'risk_probe_required_columns', 'pass': int(required_columns_ok), 'detail': 'dist_entropy/progress/labels present'},
        {'check': 'preflight_all_checks_pass', 'pass': int(risk_probe_pass), 'detail': 'risk-uq smoke gates only'},
    ]
    risk_probe_summary_df = pd.DataFrame(summary_rows)
    return {
        'overall_pass': bool(risk_probe_pass),
        'risk_probe_pass': bool(risk_probe_pass),
        'preflight_pass': bool(risk_probe_pass),
        'failure_reasons': list(failure_reasons),
        'risk_probe_summary_df': risk_probe_summary_df,
        'risk_probe_rows_df': probe_df,
        'preflight_df': pd.DataFrame(summary_rows),
    }

