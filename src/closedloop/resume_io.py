from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import jax
except Exception:  # pragma: no cover - optional in lightweight test environments
    jax = None
import numpy as np
import pandas as pd

from .config import SearchConfig, ClosedLoopConfig, build_run_artifact_paths, build_uq_artifact_paths
from .calibration import diagnose_surprise_root_cause
from .signal_analysis import (
    analyze_surprise_signal_usefulness,
    save_surprise_signal_usefulness_artifacts,
)

ARTIFACT_SCHEMA_VERSION = '1.0.0'
UQ_ARTIFACT_SCHEMA_VERSION = '1.0.0'
RESULTS_REQUIRED_COLUMNS = ['scenario_id', 'method']
TRACE_REQUIRED_COLUMNS = ['scenario_id', 'method', 'eval_index']
SURPRISE_COL_CANDIDATES = ('delta_surprise', 'delta_surprise_pd', 'surprise_pd')


def _resolve_surprise_col(df: pd.DataFrame) -> str:
    for col in SURPRISE_COL_CANDIDATES:
        if isinstance(df, pd.DataFrame) and (col in df.columns):
            return col
    return 'delta_surprise'


def _ensure_surprise_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or len(df.columns) == 0:
        return df
    if 'delta_surprise' not in df.columns:
        if 'delta_surprise_pd' in df.columns:
            df['delta_surprise'] = df['delta_surprise_pd']
        elif 'surprise_pd' in df.columns:
            df['delta_surprise'] = df['surprise_pd']
    if ('delta_surprise_pd' not in df.columns) and ('delta_surprise' in df.columns):
        df['delta_surprise_pd'] = df['delta_surprise']
    if ('surprise_pd' not in df.columns) and ('delta_surprise' in df.columns):
        df['surprise_pd'] = df['delta_surprise']
    return df


def _artifact_schema_manifest_path(run_prefix: str) -> str:
    return f'{run_prefix}_artifact_schema.json'


def artifact_schema_spec() -> Dict[str, Any]:
    return {
        'schema_version': ARTIFACT_SCHEMA_VERSION,
        'required_columns': {
            'per_scenario_results': list(RESULTS_REQUIRED_COLUMNS),
            'per_eval_trace': list(TRACE_REQUIRED_COLUMNS),
        },
    }


def write_artifact_schema_manifest(run_prefix: str) -> str:
    path = _artifact_schema_manifest_path(run_prefix)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(
            {
                **artifact_schema_spec(),
                'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            },
            f,
            indent=2,
        )
    return path


def write_uq_artifact_schema_manifest(run_prefix: str, artifact_names: Optional[List[str]] = None) -> str:
    path = f'{run_prefix}_uq_artifact_schema.json'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema_version': UQ_ARTIFACT_SCHEMA_VERSION,
        'artifacts': artifact_names or sorted(build_uq_artifact_paths(run_prefix).keys()),
        'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    return path


def validate_artifact_schema_manifest(run_prefix: str, strict: bool = True) -> bool:
    path = _artifact_schema_manifest_path(run_prefix)
    if not Path(path).exists():
        # Backward compatibility: older runs may not have this manifest.
        return True

    try:
        with open(path, 'r') as f:
            payload = json.load(f)
    except Exception as e:
        msg = f'[resume] failed to parse artifact schema manifest ({path}): {e}'
        if strict:
            raise RuntimeError(msg)
        print(msg)
        return False

    got = str(payload.get('schema_version', ''))
    expected = str(ARTIFACT_SCHEMA_VERSION)
    if got != expected:
        msg = (
            f'[resume] artifact schema version mismatch for {run_prefix}: '
            f'found={got}, expected={expected}.'
        )
        if strict:
            raise RuntimeError(msg)
        print(msg)
        return False
    return True


def _validate_required_columns(
    df: pd.DataFrame,
    required_cols: Optional[List[str]],
    artifact_name: str,
    path: str,
) -> bool:
    if required_cols is None:
        return True
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) == 0:
        return True
    print(
        f'[resume] schema mismatch in {artifact_name} ({path}): '
        f'missing required columns {missing}.'
    )
    return False


def _load_existing_results(
    path: str,
    required_cols: Optional[List[str]] = None,
    artifact_name: str = 'artifact',
) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if not _validate_required_columns(df, required_cols, artifact_name, path):
                return pd.DataFrame()
            print(f'[resume] loaded existing results: {path} ({len(df)} rows)')
            return df
        except Exception as e:
            print(f'[resume] failed to read existing results ({path}): {e}')
    return pd.DataFrame()

def _completed_scenarios(df: pd.DataFrame, methods: List[str]) -> set:
    if df.empty or ('scenario_id' not in df.columns) or ('method' not in df.columns):
        return set()
    sub = df[df['method'].isin(methods)].copy()
    if sub.empty:
        return set()
    counts = sub.groupby('scenario_id')['method'].nunique()
    return set(counts[counts >= len(methods)].index.astype(int).tolist())

def _flush_checkpoint(
    rows_buffer: List[Dict[str, Any]],
    existing_df: pd.DataFrame,
    out_path: str,
    dedup_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    if len(rows_buffer) == 0:
        return existing_df
    new_df = pd.DataFrame(rows_buffer)
    if existing_df.empty:
        combined = new_df
    else:
        combined = pd.concat([existing_df, new_df], ignore_index=True)

    if dedup_cols is None:
        dedup_cols = [c for c in ['scenario_id', 'method'] if c in combined.columns]
    else:
        dedup_cols = [c for c in dedup_cols if c in combined.columns]
    if len(dedup_cols) > 0:
        combined = combined.drop_duplicates(subset=dedup_cols, keep='last')

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    return combined

def _safe_float_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float, int)) and k not in ['source', 'surprise_metric_name']:
            out[k] = float(v)
        else:
            out[k] = v
    return out

def _compute_progress_tables(df: pd.DataFrame, trace_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    methods = ['random', 'risk_only', 'surprise_only', 'joint']
    usable = _ensure_surprise_alias_columns(df[df['method'].isin(methods)].copy())

    if len(usable) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    surprise_col = _resolve_surprise_col(usable)

    quick_summary = (
        usable.groupby('method', as_index=False)
        .agg(
            n=('scenario_id', 'size'),
            mean_risk=('risk_sks', 'mean'),
            mean_surprise=(surprise_col, 'mean'),
            failure_rate=('failure_proxy', 'mean'),
            feasibility_violation_rate=('feasibility_violation', 'mean'),
            q1_rate=('q1_hit', 'mean'),
            q4_rate=('q4_hit', 'mean'),
            bsdr_proxy_rate=('blind_spot_proxy_hit', 'mean'),
            mean_objective_gain=('objective_gain', 'mean'),
            mean_budget_used=('budget_units_used', 'mean'),
        )
    )

    sanity_rows = []
    for method in ['risk_only', 'surprise_only', 'joint']:
        sub = usable[usable['method'] == method]
        if len(sub) == 0:
            continue
        sanity_rows.append({
            'method': method,
            'objective_gain_positive_rate': float((sub['objective_gain'] > 0).mean()),
            'delta_risk_positive_rate': float((sub['delta_risk'] > 0).mean()),
            'delta_surprise_positive_rate': float((sub['delta_surprise'] > 0).mean()),
        })
    sanity = pd.DataFrame(sanity_rows)

    fairness = (
        usable.groupby('method', as_index=False)
        .agg(
            n=('scenario_id', 'size'),
            mean_budget_used=('budget_units_used', 'mean'),
            mean_delta_l2=('delta_l2', 'mean'),
            feasibility_violation_rate=('feasibility_violation', 'mean'),
            common_random_numbers_rate=('common_random_numbers_used', 'mean'),
        )
    )

    if isinstance(trace_df, pd.DataFrame) and len(trace_df) > 0:
        trace_diag = (
            trace_df.groupby('method', as_index=False)
            .agg(
                n_eval_rows=('eval_index', 'size'),
                final_eval_index_mean=('eval_index', 'mean'),
                accepted_rate=('accepted', 'mean'),
                best_hit_rate=('is_best_so_far', 'mean'),
                surprise_finite_rate=('surprise_finite', 'mean'),
                dist_non_null_ratio_mean=('dist_non_null_ratio', 'mean'),
                dist_mean_components_mean=('dist_mean_components', 'mean'),
            )
        )
    else:
        trace_diag = pd.DataFrame()

    return quick_summary, sanity, fairness, trace_diag

def _write_progress_artifacts(
    run_prefix: str,
    results_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    cfg: ClosedLoopConfig,
    search_cfg: SearchConfig,
    thresholds: Dict[str, Any],
    static_frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    paths = build_run_artifact_paths(run_prefix)

    per_scenario_path = paths['per_scenario_results']
    per_eval_trace_path = paths['per_eval_trace']
    thresholds_path = paths['thresholds']
    quick_summary_path = f'{run_prefix}_quick_summary.csv'
    sanity_path = f'{run_prefix}_sanity_checks.csv'
    fairness_path = f'{run_prefix}_fairness_checks.csv'
    trace_diag_path = f'{run_prefix}_trace_diagnostics.csv'
    seed_map_path = f'{run_prefix}_eval_seed_map.csv'
    carry_path = f'{run_prefix}_carry_forward_config.json'
    schema_manifest_path = _artifact_schema_manifest_path(run_prefix)

    for p in [
        per_scenario_path,
        per_eval_trace_path,
        thresholds_path,
        quick_summary_path,
        sanity_path,
        fairness_path,
        trace_diag_path,
        seed_map_path,
        carry_path,
        schema_manifest_path,
    ]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    if not _validate_required_columns(
        results_df,
        RESULTS_REQUIRED_COLUMNS,
        'per_scenario_results',
        per_scenario_path,
    ):
        raise ValueError(
            'Cannot write per_scenario_results: required columns are missing.'
        )
    results_df.to_csv(per_scenario_path, index=False)
    if bool(cfg.save_per_eval_trace) and isinstance(trace_df, pd.DataFrame):
        if len(trace_df) > 0 and (not _validate_required_columns(
            trace_df,
            TRACE_REQUIRED_COLUMNS,
            'per_eval_trace',
            per_eval_trace_path,
        )):
            raise ValueError(
                'Cannot write per_eval_trace: required columns are missing.'
            )
        trace_df.to_csv(per_eval_trace_path, index=False)

    with open(thresholds_path, 'w') as f:
        json.dump(_safe_float_dict(thresholds), f, indent=2)

    quick_summary_df, sanity_df, fairness_df, trace_diag_df = _compute_progress_tables(results_df, trace_df)
    quick_summary_df.to_csv(quick_summary_path, index=False)
    sanity_df.to_csv(sanity_path, index=False)
    fairness_df.to_csv(fairness_path, index=False)
    trace_diag_df.to_csv(trace_diag_path, index=False)

    if not results_df.empty and {'scenario_id', 'method', 'seed_used'}.issubset(results_df.columns):
        seed_map_df = results_df[
            results_df['method'].isin(['random', 'risk_only', 'surprise_only', 'joint'])
        ][['scenario_id', 'seed_used']].drop_duplicates().sort_values('scenario_id')
        seed_map_df.to_csv(seed_map_path, index=False)

    carry_forward_config = {
        'experiment_track': 'B_closed_loop_simulation_only',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'planner': {
            'planner_kind': cfg.planner_kind,
            'planner_name_config': cfg.planner_name,
        },
        'run_controls': {
            'run_chunk_size': int(cfg.run_chunk_size),
            'checkpoint_every_scenarios': int(cfg.checkpoint_every_scenarios),
            'resume_from_existing': bool(cfg.resume_from_existing),
            'save_per_eval_trace': bool(cfg.save_per_eval_trace),
            'rollout_seed_stride': int(cfg.rollout_seed_stride),
        },
        'thresholds': _safe_float_dict(thresholds),
        'optimization': dataclasses.asdict(search_cfg),
    }
    with open(carry_path, 'w') as f:
        json.dump(carry_forward_config, f, indent=2)
    write_artifact_schema_manifest(run_prefix)

    static_frames = static_frames or {}
    static_pairs = [
        ('base_eval_openloop_df', f'{run_prefix}_base_eval_openloop_scores.csv'),
        ('reference_df', f'{run_prefix}_reference_openloop_scores.csv'),
        ('closedloop_calib_df', paths['closedloop_calibration']),
        ('preflight_df', f'{run_prefix}_preflight_checks.csv'),
        ('calib_diag_df', paths['calibration_diagnostics']),
        ('calib_quant_df', paths['calibration_quantiles']),
    ]
    for var_name, out_path in static_pairs:
        obj = static_frames.get(var_name)
        if isinstance(obj, pd.DataFrame):
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            obj.to_csv(out_path, index=False)

def summarize_method_outputs(closedloop_results_df: pd.DataFrame, closedloop_trace_df: pd.DataFrame):
    usable_df = _ensure_surprise_alias_columns(closedloop_results_df[
        closedloop_results_df['method'].isin(['random', 'risk_only', 'surprise_only', 'joint'])
    ].copy())
    surprise_col = _resolve_surprise_col(usable_df)

    quick_summary_df = (
        usable_df.groupby('method', as_index=False)
        .agg(
            n=('scenario_id', 'size'),
            mean_risk=('risk_sks', 'mean'),
            mean_surprise=(surprise_col, 'mean'),
            failure_rate=('failure_proxy', 'mean'),
            feasibility_violation_rate=('feasibility_violation', 'mean'),
            q1_rate=('q1_hit', 'mean'),
            q4_rate=('q4_hit', 'mean'),
            bsdr_proxy_rate=('blind_spot_proxy_hit', 'mean'),
            mean_objective_gain=('objective_gain', 'mean'),
            mean_budget_used=('budget_units_used', 'mean'),
        )
    )

    sanity_rows = []
    for method in ['risk_only', 'surprise_only', 'joint']:
        sub = usable_df[usable_df['method'] == method]
        if len(sub) == 0:
            continue
        sanity_rows.append({
            'method': method,
            'objective_gain_positive_rate': float((sub['objective_gain'] > 0).mean()),
            'delta_risk_positive_rate': float((sub['delta_risk'] > 0).mean()),
            'delta_surprise_positive_rate': float((sub['delta_surprise'] > 0).mean()),
        })
    sanity_df = pd.DataFrame(sanity_rows)

    fairness_checks_df = (
        usable_df.groupby('method', as_index=False)
        .agg(
            n=('scenario_id', 'size'),
            mean_budget_used=('budget_units_used', 'mean'),
            mean_delta_l2=('delta_l2', 'mean'),
            feasibility_violation_rate=('feasibility_violation', 'mean'),
            common_random_numbers_rate=('common_random_numbers_used', 'mean'),
        )
    )

    if isinstance(closedloop_trace_df, pd.DataFrame) and len(closedloop_trace_df) > 0:
        trace_diag_df = (
            closedloop_trace_df.groupby('method', as_index=False)
            .agg(
                n_eval_rows=('eval_index', 'size'),
                final_eval_index_mean=('eval_index', 'mean'),
                accepted_rate=('accepted', 'mean'),
                best_hit_rate=('is_best_so_far', 'mean'),
                surprise_finite_rate=('surprise_finite', 'mean'),
                dist_non_null_ratio_mean=('dist_non_null_ratio', 'mean'),
                dist_mean_components_mean=('dist_mean_components', 'mean'),
            )
        )
    else:
        trace_diag_df = pd.DataFrame()

    return quick_summary_df, sanity_df, fairness_checks_df, trace_diag_df

def export_closedloop_artifacts(
    cfg: ClosedLoopConfig,
    search_cfg: SearchConfig,
    eval_idx: np.ndarray,
    closedloop_results_df: pd.DataFrame,
    closedloop_trace_df: pd.DataFrame,
    base_eval_openloop_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    closedloop_calib_df: pd.DataFrame,
    preflight_df: pd.DataFrame,
    calib_diag_df: pd.DataFrame,
    calib_quant_df: pd.DataFrame,
    closedloop_thresholds: Dict[str, Any],
    quick_summary_df: pd.DataFrame,
    sanity_df: pd.DataFrame,
    fairness_checks_df: pd.DataFrame,
    trace_diag_df: pd.DataFrame,
) -> Dict[str, str]:
    run_prefix = cfg.run_prefix
    artifact_paths = build_run_artifact_paths(run_prefix)

    per_scenario_path = artifact_paths['per_scenario_results']
    per_eval_trace_path = artifact_paths['per_eval_trace']
    base_eval_openloop_path = f'{run_prefix}_base_eval_openloop_scores.csv'
    reference_openloop_path = f'{run_prefix}_reference_openloop_scores.csv'
    closedloop_calib_path = artifact_paths['closedloop_calibration']
    preflight_path = f'{run_prefix}_preflight_checks.csv'
    calib_diag_path = artifact_paths['calibration_diagnostics']
    calib_quant_path = artifact_paths['calibration_quantiles']
    seed_map_path = f'{run_prefix}_eval_seed_map.csv'
    thresholds_path = artifact_paths['thresholds']
    carry_path = f'{run_prefix}_carry_forward_config.json'
    quick_summary_path = f'{run_prefix}_quick_summary.csv'
    sanity_path = f'{run_prefix}_sanity_checks.csv'
    fairness_path = f'{run_prefix}_fairness_checks.csv'
    trace_diag_path = f'{run_prefix}_trace_diagnostics.csv'
    root_cause_summary_path = f'{run_prefix}_surprise_root_cause_summary.csv'
    root_cause_findings_path = f'{run_prefix}_surprise_root_cause_findings.csv'
    runtime_manifest_path = f'{run_prefix}_runtime_manifest.json'
    schema_manifest_path = _artifact_schema_manifest_path(run_prefix)

    all_paths = {
        'per_scenario_results': per_scenario_path,
        'per_eval_trace': per_eval_trace_path,
        'base_eval_openloop': base_eval_openloop_path,
        'reference_openloop': reference_openloop_path,
        'closedloop_calibration': closedloop_calib_path,
        'preflight_checks': preflight_path,
        'calibration_diagnostics': calib_diag_path,
        'calibration_quantiles': calib_quant_path,
        'seed_map': seed_map_path,
        'thresholds': thresholds_path,
        'carry_forward_config': carry_path,
        'quick_summary': quick_summary_path,
        'sanity_checks': sanity_path,
        'fairness_checks': fairness_path,
        'trace_diagnostics': trace_diag_path,
        'surprise_root_cause_summary': root_cause_summary_path,
        'surprise_root_cause_findings': root_cause_findings_path,
        'runtime_manifest': runtime_manifest_path,
        'artifact_schema': schema_manifest_path,
    }

    for p in all_paths.values():
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    if not _validate_required_columns(
        closedloop_results_df,
        RESULTS_REQUIRED_COLUMNS,
        'per_scenario_results',
        per_scenario_path,
    ):
        raise ValueError(
            'Cannot export per_scenario_results: required columns are missing.'
        )
    closedloop_results_df.to_csv(per_scenario_path, index=False)
    if isinstance(closedloop_trace_df, pd.DataFrame):
        if len(closedloop_trace_df) > 0 and (not _validate_required_columns(
            closedloop_trace_df,
            TRACE_REQUIRED_COLUMNS,
            'per_eval_trace',
            per_eval_trace_path,
        )):
            raise ValueError(
                'Cannot export per_eval_trace: required columns are missing.'
            )
        closedloop_trace_df.to_csv(per_eval_trace_path, index=False)
    base_eval_openloop_df.to_csv(base_eval_openloop_path, index=False)
    reference_df.to_csv(reference_openloop_path, index=False)
    closedloop_calib_df.to_csv(closedloop_calib_path, index=False)
    preflight_df.to_csv(preflight_path, index=False)
    calib_diag_df.to_csv(calib_diag_path, index=False)
    calib_quant_df.to_csv(calib_quant_path, index=False)
    try:
        root_cause_summary_df, root_cause_findings_df = diagnose_surprise_root_cause(
            preflight_df=preflight_df,
            closedloop_calib_df=closedloop_calib_df,
        )
        root_cause_summary_df.to_csv(root_cause_summary_path, index=False)
        root_cause_findings_df.to_csv(root_cause_findings_path, index=False)
    except Exception as e:
        print(f'[export] surprise root-cause diagnostics skipped: {e}')
    quick_summary_df.to_csv(quick_summary_path, index=False)
    sanity_df.to_csv(sanity_path, index=False)
    fairness_checks_df.to_csv(fairness_path, index=False)
    if isinstance(trace_diag_df, pd.DataFrame):
        trace_diag_df.to_csv(trace_diag_path, index=False)

    signal_paths: Dict[str, str] = {}
    try:
        signal_summary_df, signal_method_corr_df, signal_bins_df, signal_topk_df, signal_within_scenario_df = (
            analyze_surprise_signal_usefulness(closedloop_results_df=closedloop_results_df)
        )
        signal_paths = save_surprise_signal_usefulness_artifacts(
            run_prefix=run_prefix,
            summary_df=signal_summary_df,
            method_corr_df=signal_method_corr_df,
            bin_df=signal_bins_df,
            topk_df=signal_topk_df,
            within_scenario_df=signal_within_scenario_df,
        )
        all_paths.update(signal_paths)
    except Exception as e:
        print(f'[export] surprise signal usefulness diagnostics skipped: {e}')

    seed_map_df = closedloop_results_df[
        closedloop_results_df['method'].isin(['random', 'risk_only', 'surprise_only', 'joint'])
    ][['scenario_id', 'seed_used']].drop_duplicates().sort_values('scenario_id')
    seed_map_df.to_csv(seed_map_path, index=False)

    with open(thresholds_path, 'w') as f:
        json.dump(
            {
                k: (
                    float(v)
                    if isinstance(v, (np.floating, float, int)) and k not in ['source', 'surprise_metric_name']
                    else v
                )
                for k, v in closedloop_thresholds.items()
            },
            f,
            indent=2,
        )

    surprise_name = str(getattr(cfg, 'planner_surprise_name', 'predictive_seq_w2')).strip().lower()
    surprise_type = 'counterfactual_composite'
    surprise_formula = (
        'S(delta)=R_eff*(wB*log(1+B_eff)+wP*log(1+P_eff)); '
        'B_eff=max(B_raw,response_floor,signal_floor), P_eff=max(P_raw,response_floor,signal_floor), '
        'R_eff=max(R_raw,response_floor,signal_floor), '
        'signal_floor=floor_weight*max(proposal_delta_ratio,response_ratio,R_raw), '
        'response_floor=response_weight*response_ratio; '
        f'base divergence metric hint={surprise_name}'
    )

    carry_forward_config = {
        'experiment_track': 'closed_loop_simulation_only',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'planner': {
            'planner_kind': cfg.planner_kind,
            'planner_name_config': cfg.planner_name,
        },
        'surprise_definition': {
            'name': cfg.planner_surprise_name,
            'type': surprise_type,
            'formula': surprise_formula,
            'predictive_kl_estimator': cfg.predictive_kl_estimator,
            'predictive_kl_mc_samples': int(cfg.predictive_kl_mc_samples),
            'predictive_kl_mc_seed': int(cfg.predictive_kl_mc_seed),
            'predictive_kl_eps': float(cfg.predictive_kl_eps),
            'latentdriver_repo_path': cfg.latentdriver_repo_path,
            'latentdriver_ckpt_path': cfg.latentdriver_ckpt_path,
            'latentdriver_action_type': cfg.latentdriver_action_type,
            'latentdriver_context_len': int(cfg.latentdriver_context_len),
            'latentdriver_yaw_sigma': float(cfg.latentdriver_yaw_sigma),
        },
        'run_controls': {
            'run_chunk_size': int(cfg.run_chunk_size),
            'checkpoint_every_scenarios': int(cfg.checkpoint_every_scenarios),
            'resume_from_existing': bool(cfg.resume_from_existing),
            'save_per_eval_trace': bool(cfg.save_per_eval_trace),
            'rollout_seed_stride': int(cfg.rollout_seed_stride),
        },
        'dataset': {
            'waymax_path': cfg.waymax_path,
            'waymax_max_rg_points': int(cfg.waymax_max_rg_points),
            'waymax_batch_dims': [int(x) for x in cfg.waymax_batch_dims],
        },
        'data_split': {
            'n_total_scenarios': int(cfg.n_total_scenarios),
            'n_eval_scenarios': int(len(eval_idx)),
            'reference_fraction': float(cfg.train_fraction),
            'train_fraction': float(cfg.train_fraction),
            'eval_scenario_ids': [int(x) for x in eval_idx],
        },
        'optimization': dataclasses.asdict(search_cfg),
        'thresholds': closedloop_thresholds,
        'risk_failure_thresholds': {
            'collision_distance': float(cfg.collision_distance),
            'ttc_fail_seconds': float(cfg.ttc_fail_seconds),
            'no_hazard_ttc_seconds': float(cfg.no_hazard_ttc_seconds),
            'no_hazard_dist_m': float(cfg.no_hazard_dist_m),
            'hard_brake_mps2': float(cfg.hard_brake_mps2),
            'hard_jerk_mps3': float(cfg.hard_jerk_mps3),
        },
        'method_labels': ['random', 'risk_only', 'surprise_only', 'joint'],
    }
    with open(carry_path, 'w') as f:
        json.dump(carry_forward_config, f, indent=2)
    write_artifact_schema_manifest(run_prefix)

    import platform

    def _package_version(pkg_name: str) -> str:
        try:
            import importlib.metadata as im
            return str(im.version(pkg_name))
        except Exception:
            try:
                mod = __import__(pkg_name)
                return str(getattr(mod, '__version__', 'unknown'))
            except Exception:
                return 'not_installed'

    def _file_meta(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {'exists': False}
        st = os.stat(path)
        return {
            'exists': True,
            'size_bytes': int(st.st_size),
            'mtime_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(st.st_mtime)),
        }

    runtime_manifest = {
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'python_version': str(sys.version),
        'platform': platform.platform(),
        'packages': {
            'jax': _package_version('jax'),
            'jaxlib': _package_version('jaxlib'),
            'numpy': _package_version('numpy'),
            'pandas': _package_version('pandas'),
            'matplotlib': _package_version('matplotlib'),
            'seaborn': _package_version('seaborn'),
            'tensorflow': _package_version('tensorflow'),
            'waymax': _package_version('waymo-waymax'),
            'torch': _package_version('torch'),
        },
        'jax_backend': str(jax.default_backend()) if jax is not None else 'not_available',
        'jax_devices': [str(d) for d in jax.devices()] if jax is not None else [],
        'planner': {
            'planner_kind': cfg.planner_kind,
            'planner_name': cfg.planner_name,
            'latentdriver_repo': cfg.latentdriver_repo_path,
            'latentdriver_ckpt': cfg.latentdriver_ckpt_path,
            'latentdriver_ckpt_meta': _file_meta(cfg.latentdriver_ckpt_path),
        },
    }

    with open(runtime_manifest_path, 'w') as f:
        json.dump(runtime_manifest, f, indent=2)

    return all_paths
