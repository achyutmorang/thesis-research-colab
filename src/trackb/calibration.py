from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import SearchConfig, TrackBConfig
from .latentdriver import (
    _choose_target_non_ego,
    closed_loop_rollout_selected,
    dist_trace_change_stats,
    dist_trace_diagnostics,
    latentdriver_observation_contract,
    make_closed_loop_components,
    predictive_kl_from_dist_traces,
    project_delta_vec,
)
from .metrics import compute_risk_metrics, planner_action_surprise_kl, risk_kwargs_from_cfg, robust_scale

def run_trackb_preflight_checks(runner: Any, cfg: TrackBConfig, eval_idx: np.ndarray) -> pd.DataFrame:
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str):
        checks.append({'check': name, 'pass': bool(passed), 'detail': str(detail)})

    if cfg.planner_kind == 'latentdriver':
        repo_ok = Path(cfg.latentdriver_repo_path).exists()
        ckpt_ok = Path(cfg.latentdriver_ckpt_path).exists()
        add('latentdriver_repo_exists', repo_ok, cfg.latentdriver_repo_path)
        add('latentdriver_ckpt_exists', ckpt_ok, cfg.latentdriver_ckpt_path)
        contract = latentdriver_observation_contract()
        add('latentdriver_expected_feature_dim', contract['feature_dim'] == 7, str(contract))

    try:
        sid = None
        for i in eval_idx:
            i = int(i)
            if 'state' in runner.data['scenarios'][i]:
                sid = i
                break
        if sid is None:
            add('eval_state_available', False, 'No eval scenario with retained simulator state.')
            return pd.DataFrame(checks)

        rec = runner.data['scenarios'][sid]
        selected_idx = np.asarray(rec['selected_indices'], dtype=np.int32)
        target_idx = _choose_target_non_ego(rec['state'], selected_idx)

        planner_bundle = make_closed_loop_components(rec['state'], cfg.planner_kind, cfg.planner_name, cfg)
        add('planner_bundle_constructed', True, planner_bundle['planner_used'])

        xy, valid, actions, action_valid, dist_trace, feasible, note = closed_loop_rollout_selected(
            base_state=rec['state'],
            selected_idx=selected_idx,
            target_obj_idx=target_idx,
            delta_xy=np.zeros((2,), dtype=float),
            cfg=cfg,
            planner_bundle=planner_bundle,
            seed=int(cfg.global_seed + sid),
        )

        add('smoke_rollout_feasible', bool(feasible), note if note else 'ok')
        add('smoke_rollout_finite', bool(np.isfinite(xy).all() and np.isfinite(actions).all()), f'xy={xy.shape}, action={actions.shape}')
        add('smoke_action_valid_rate_positive', bool(np.mean(action_valid.astype(float)) > 0.0), f'action_valid_rate={float(np.mean(action_valid.astype(float))):.3f}')

        if planner_bundle.get('planner_type') == 'latentdriver':
            non_null = int(sum(d is not None for d in dist_trace))
            add('latentdriver_distribution_trace_nonempty', non_null > 0, f'non_null_steps={non_null}/{len(dist_trace)}')
            if non_null > 0:
                first = next(d for d in dist_trace if d is not None)
                add('latentdriver_dist_fields', all(k in first for k in ['weights', 'means', 'stds']), str(list(first.keys())))
            dist_diag = dist_trace_diagnostics(dist_trace)
            fallback_ratio = float(dist_diag.get('dist_fallback_ratio', np.nan))
            actor_fallback_ratio = float(dist_diag.get('dist_actor_fallback_ratio', np.nan))
            finite_ratio = float(dist_diag.get('dist_finite_ratio', np.nan))
            max_fallback = float(getattr(cfg, 'latentdriver_preflight_max_fallback_ratio', 0.95))
            add(
                'latentdriver_dist_fallback_ratio_ok',
                bool(np.isfinite(fallback_ratio) and fallback_ratio < max_fallback),
                f'fallback_ratio={fallback_ratio:.4f}, actor_fallback_ratio={actor_fallback_ratio:.4f}, max={max_fallback:.4f}',
            )
            add(
                'latentdriver_dist_finite_ratio_positive',
                bool(np.isfinite(finite_ratio) and finite_ratio > 0.0),
                f'dist_finite_ratio={finite_ratio:.4f}',
            )
            obs_info = planner_bundle['ld_adapter'].last_obs_info
            feat_dim_ok = int(obs_info.get('feature_dim', -1)) == int(latentdriver_observation_contract()['feature_dim'])
            add('latentdriver_obs_feature_dim_ok', feat_dim_ok, str(obs_info))
            add('latentdriver_obs_finite', bool(obs_info.get('finite', False)), str(obs_info))
            ld_adapter = planner_bundle['ld_adapter']
            route = str(getattr(ld_adapter, '_last_forward_route', 'unknown'))
            err = str(getattr(ld_adapter, '_last_forward_error', ''))
            route_ok = (route != 'failed') and (len(err.strip()) == 0)
            add(
                'latentdriver_forward_route_ok',
                bool(route_ok),
                f'route={route}; last_error={err[:240]}',
            )

    except Exception as e:
        add('smoke_rollout_exception', False, str(e))

    return pd.DataFrame(checks)

def make_calibration_delta_proposal(rng: np.random.Generator, k: int, search_cfg: SearchConfig) -> np.ndarray:
    # Deterministic directional proposals + jitter improve calibration coverage.
    dirs = np.asarray([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
    ], dtype=float)
    direction = dirs[int(k) % int(len(dirs))]
    direction = direction / max(1e-12, float(np.linalg.norm(direction)))

    base_scale = float(max(search_cfg.random_scale, 0.35))
    scales = np.asarray([base_scale, 1.5 * base_scale, 2.0 * base_scale], dtype=float)
    scale = float(scales[(int(k) // int(len(dirs))) % int(len(scales))])

    jitter = rng.normal(loc=0.0, scale=0.15 * base_scale, size=(2,))
    prop = scale * direction + jitter
    return project_delta_vec(prop, search_cfg.delta_clip, search_cfg.delta_l2_budget)

def calibrate_closed_loop_thresholds(
    runner: TrackBRunner,
    eval_idx: np.ndarray,
    cfg: TrackBConfig,
    search_cfg: SearchConfig,
    reference_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    calib_ids = np.asarray(eval_idx[: min(len(eval_idx), cfg.n_closedloop_calib)], dtype=int)
    rows: List[Dict[str, Any]] = []

    for sid in tqdm(calib_ids, desc='Closed-loop calibration (base + random proposals)', total=len(calib_ids)):
        rec = runner.data['scenarios'][int(sid)]
        if 'state' not in rec:
            continue

        selected_idx = np.asarray(rec['selected_indices'], dtype=np.int32)
        base_state = rec['state']
        rng = np.random.default_rng(cfg.global_seed + int(sid))

        try:
            target_idx = _choose_target_non_ego(base_state, selected_idx)
            planner_bundle = make_closed_loop_components(base_state, cfg.planner_kind, cfg.planner_name, cfg)

            base_xy, base_valid, base_actions, base_action_valid, base_dist_trace, base_feasible, base_note = closed_loop_rollout_selected(
                base_state=base_state,
                selected_idx=selected_idx,
                target_obj_idx=target_idx,
                delta_xy=np.zeros((2,), dtype=float),
                cfg=cfg,
                planner_bundle=planner_bundle,
                seed=cfg.global_seed + int(sid),
            )
            base_risk = compute_risk_metrics(base_xy, base_valid, **risk_kwargs_from_cfg(cfg))
            if planner_bundle['planner_type'] == 'latentdriver':
                base_dist_diag = dist_trace_diagnostics(base_dist_trace)
            else:
                base_dist_diag = {
                    'dist_non_null_steps': np.nan,
                    'dist_non_null_ratio': np.nan,
                    'dist_mean_components': np.nan,
                    'dist_min_weight': np.nan,
                    'dist_min_std': np.nan,
                    'dist_max_std': np.nan,
                    'dist_finite_ratio': np.nan,
                    'dist_fallback_steps': np.nan,
                    'dist_fallback_ratio': np.nan,
                    'dist_actor_fallback_steps': np.nan,
                    'dist_actor_fallback_ratio': np.nan,
                }

            for k in range(int(max(1, cfg.n_surprise_calib_proposals))):
                prop = make_calibration_delta_proposal(rng, k, search_cfg)

                p_xy, p_valid, p_actions, p_action_valid, p_dist_trace, p_feasible, p_note = closed_loop_rollout_selected(
                    base_state=base_state,
                    selected_idx=selected_idx,
                    target_obj_idx=target_idx,
                    delta_xy=prop,
                    cfg=cfg,
                    planner_bundle=planner_bundle,
                    seed=cfg.global_seed + int(sid) + 1000 + k,
                )
                p_risk = compute_risk_metrics(p_xy, p_valid, **risk_kwargs_from_cfg(cfg))

                if planner_bundle['planner_type'] == 'latentdriver':
                    p_surprise = predictive_kl_from_dist_traces(
                        p_dist_trace,
                        base_dist_trace,
                        estimator=cfg.predictive_kl_estimator,
                        n_mc_samples=cfg.predictive_kl_mc_samples,
                        seed=int(cfg.predictive_kl_mc_seed + int(sid) * 1000 + k),
                        eps=float(cfg.predictive_kl_eps),
                        symmetric=bool(cfg.predictive_kl_symmetric),
                        skip_fallback_steps=bool(cfg.predictive_kl_skip_fallback_steps),
                    )
                    p_dist_diag = dist_trace_diagnostics(p_dist_trace)
                    trace_change_diag = dist_trace_change_stats(p_dist_trace, base_dist_trace)

                    trace_pair_ratio = float(trace_change_diag.get('trace_pair_ratio', 0.0))
                    if trace_pair_ratio <= 0.0:
                        # No valid non-fallback trace pairs -> treat surprise as unusable for calibration.
                        p_surprise = np.nan
                else:
                    p_surprise = planner_action_surprise_kl(
                        p_actions,
                        p_action_valid,
                        base_actions,
                        base_action_valid,
                        sigma=0.25,
                    )
                    p_dist_diag = {
                        'dist_non_null_steps': np.nan,
                        'dist_non_null_ratio': np.nan,
                        'dist_mean_components': np.nan,
                        'dist_min_weight': np.nan,
                        'dist_min_std': np.nan,
                        'dist_max_std': np.nan,
                        'dist_finite_ratio': np.nan,
                        'dist_fallback_steps': np.nan,
                        'dist_fallback_ratio': np.nan,
                        'dist_actor_fallback_steps': np.nan,
                        'dist_actor_fallback_ratio': np.nan,
                    }
                    trace_change_diag = {
                        'trace_pair_steps': np.nan,
                        'trace_pair_ratio': np.nan,
                        'trace_pair_steps_all': np.nan,
                        'trace_pair_ratio_all': np.nan,
                        'trace_fallback_pair_ratio': np.nan,
                        'step_mean_l2_mean': np.nan,
                        'step_mean_l2_p50': np.nan,
                        'step_mean_l2_p95': np.nan,
                        'step_std_l2_mean': np.nan,
                        'step_moment_kl_mean': np.nan,
                        'step_moment_kl_p50': np.nan,
                        'step_moment_kl_p95': np.nan,
                        'step_moment_kl_nonzero_ratio': np.nan,
                    }

                rows.append({
                    'scenario_id': int(sid),
                    'planner_used': planner_bundle['planner_used'],
                    'proposal_id': int(k),
                    'base_risk_sks': float(base_risk['risk_sks']),
                    'proposal_risk_sks': float(p_risk['risk_sks']),
                    'surprise_pd': float(p_surprise),
                    'base_failure_proxy': float(base_risk['failure_extended_proxy']),
                    'proposal_failure_proxy': float(p_risk['failure_extended_proxy']),
                    'base_rollout_feasible': int(base_feasible),
                    'proposal_rollout_feasible': int(p_feasible),
                    'base_rollout_note': base_note,
                    'proposal_rollout_note': p_note,
                    'delta_x': float(prop[0]),
                    'delta_y': float(prop[1]),
                    'delta_l2': float(np.linalg.norm(prop)),
                    'base_dist_non_null_ratio': float(base_dist_diag.get('dist_non_null_ratio', np.nan)),
                    'base_dist_fallback_ratio': float(base_dist_diag.get('dist_fallback_ratio', np.nan)),
                    'base_dist_actor_fallback_ratio': float(base_dist_diag.get('dist_actor_fallback_ratio', np.nan)),
                    'proposal_dist_non_null_ratio': float(p_dist_diag.get('dist_non_null_ratio', np.nan)),
                    'proposal_dist_fallback_ratio': float(p_dist_diag.get('dist_fallback_ratio', np.nan)),
                    'proposal_dist_actor_fallback_ratio': float(p_dist_diag.get('dist_actor_fallback_ratio', np.nan)),
                    'step_mean_l2_mean': float(trace_change_diag.get('step_mean_l2_mean', np.nan)),
                    'step_mean_l2_p50': float(trace_change_diag.get('step_mean_l2_p50', np.nan)),
                    'step_mean_l2_p95': float(trace_change_diag.get('step_mean_l2_p95', np.nan)),
                    'step_std_l2_mean': float(trace_change_diag.get('step_std_l2_mean', np.nan)),
                    'step_moment_kl_mean': float(trace_change_diag.get('step_moment_kl_mean', np.nan)),
                    'step_moment_kl_p50': float(trace_change_diag.get('step_moment_kl_p50', np.nan)),
                    'step_moment_kl_p95': float(trace_change_diag.get('step_moment_kl_p95', np.nan)),
                    'step_moment_kl_nonzero_ratio': float(trace_change_diag.get('step_moment_kl_nonzero_ratio', np.nan)),
                    'trace_pair_steps': float(trace_change_diag.get('trace_pair_steps', np.nan)),
                    'trace_pair_ratio': float(trace_change_diag.get('trace_pair_ratio', np.nan)),
                    'trace_pair_steps_all': float(trace_change_diag.get('trace_pair_steps_all', np.nan)),
                    'trace_pair_ratio_all': float(trace_change_diag.get('trace_pair_ratio_all', np.nan)),
                    'trace_fallback_pair_ratio': float(trace_change_diag.get('trace_fallback_pair_ratio', np.nan)),
                })
        except Exception as e:
            rows.append({
                'scenario_id': int(sid),
                'planner_used': 'error',
                'proposal_id': -1,
                'base_risk_sks': np.nan,
                'proposal_risk_sks': np.nan,
                'surprise_pd': np.nan,
                'base_failure_proxy': np.nan,
                'proposal_failure_proxy': np.nan,
                'base_rollout_feasible': 0,
                'proposal_rollout_feasible': 0,
                'base_rollout_note': f'calibration_exception: {e}',
                'proposal_rollout_note': f'calibration_exception: {e}',
                'delta_x': np.nan,
                'delta_y': np.nan,
                'delta_l2': np.nan,
                'base_dist_non_null_ratio': np.nan,
                'base_dist_fallback_ratio': np.nan,
                'base_dist_actor_fallback_ratio': np.nan,
                'proposal_dist_non_null_ratio': np.nan,
                'proposal_dist_fallback_ratio': np.nan,
                'proposal_dist_actor_fallback_ratio': np.nan,
                'step_mean_l2_mean': np.nan,
                'step_mean_l2_p50': np.nan,
                'step_mean_l2_p95': np.nan,
                'step_std_l2_mean': np.nan,
                'step_moment_kl_mean': np.nan,
                'step_moment_kl_p50': np.nan,
                'step_moment_kl_p95': np.nan,
                'step_moment_kl_nonzero_ratio': np.nan,
                'trace_pair_steps': np.nan,
                'trace_pair_ratio': np.nan,
                'trace_pair_steps_all': np.nan,
                'trace_pair_ratio_all': np.nan,
                'trace_fallback_pair_ratio': np.nan,
            })

    calib_df = pd.DataFrame(rows)

    usable = calib_df[
        np.isfinite(calib_df['base_risk_sks'])
        & np.isfinite(calib_df['surprise_pd'])
    ].copy()

    if len(usable) >= 20:
        thresholds = {
            'source': 'closed_loop_random_calibration',
            'surprise_metric_name': cfg.planner_surprise_name,
            'risk_high_threshold': float(usable['base_risk_sks'].quantile(cfg.high_quantile)),
            'risk_low_threshold': float(usable['base_risk_sks'].quantile(1.0 - cfg.high_quantile)),
            'surprise_high_threshold': float(usable['surprise_pd'].quantile(cfg.high_quantile)),
            'risk_scale': float(robust_scale(usable['base_risk_sks'].to_numpy(), min_scale=search_cfg.min_scale)),
            'surprise_scale': float(robust_scale(usable['surprise_pd'].to_numpy(), min_scale=search_cfg.min_scale)),
            'high_quantile': float(cfg.high_quantile),
            'n_calibration_used': int(len(usable)),
        }
    else:
        fallback_risk = (
            reference_df['risk_sks'].to_numpy()
            if (isinstance(reference_df, pd.DataFrame) and 'risk_sks' in reference_df.columns and len(reference_df) > 0)
            else np.array([1.0])
        )
        thresholds = {
            'source': 'open_loop_risk_fallback',
            'surprise_metric_name': cfg.planner_surprise_name,
            'risk_high_threshold': float(np.quantile(fallback_risk, cfg.high_quantile)),
            'risk_low_threshold': float(np.quantile(fallback_risk, 1.0 - cfg.high_quantile)),
            'surprise_high_threshold': float(np.quantile(usable['surprise_pd'].to_numpy(), cfg.high_quantile)) if len(usable) > 0 else 0.0,
            'risk_scale': float(robust_scale(fallback_risk, min_scale=search_cfg.min_scale)),
            'surprise_scale': float(robust_scale(usable['surprise_pd'].to_numpy(), min_scale=search_cfg.min_scale)) if len(usable) > 0 else 1.0,
            'high_quantile': float(cfg.high_quantile),
            'n_calibration_used': int(len(usable)),
        }

    return calib_df, thresholds

def build_calibration_diagnostics(calib_df: pd.DataFrame, thresholds: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if calib_df.empty:
        summary = pd.DataFrame([{
            'n_rows': 0,
            'n_usable': 0,
            'usable_rate': 0.0,
            'base_feasible_rate': 0.0,
            'proposal_feasible_rate': 0.0,
            'surprise_std': np.nan,
            'risk_std': np.nan,
            'threshold_source': thresholds.get('source', 'unknown'),
        }])
        return summary, pd.DataFrame()

    usable_mask = np.isfinite(calib_df['base_risk_sks']) & np.isfinite(calib_df['surprise_pd'])
    usable = calib_df[usable_mask].copy()

    summary = pd.DataFrame([{
        'n_rows': int(len(calib_df)),
        'n_usable': int(len(usable)),
        'usable_rate': float(len(usable) / max(len(calib_df), 1)),
        'base_feasible_rate': float(calib_df['base_rollout_feasible'].mean()) if 'base_rollout_feasible' in calib_df else np.nan,
        'proposal_feasible_rate': float(calib_df['proposal_rollout_feasible'].mean()) if 'proposal_rollout_feasible' in calib_df else np.nan,
        'surprise_std': float(np.std(usable['surprise_pd'])) if len(usable) > 1 else np.nan,
        'nonzero_surprise_fraction': float(np.mean(np.asarray(usable['surprise_pd']) > 1e-9)) if len(usable) > 0 else 0.0,
        'risk_std': float(np.std(usable['base_risk_sks'])) if len(usable) > 1 else np.nan,
        'proposal_dist_fallback_ratio_mean': float(np.nanmean(usable['proposal_dist_fallback_ratio'])) if ('proposal_dist_fallback_ratio' in usable and len(usable) > 0) else np.nan,
        'proposal_dist_actor_fallback_ratio_mean': float(np.nanmean(usable['proposal_dist_actor_fallback_ratio'])) if ('proposal_dist_actor_fallback_ratio' in usable and len(usable) > 0) else np.nan,
        'step_mean_l2_mean': float(np.nanmean(usable['step_mean_l2_mean'])) if ('step_mean_l2_mean' in usable and len(usable) > 0) else np.nan,
        'step_moment_kl_mean': float(np.nanmean(usable['step_moment_kl_mean'])) if ('step_moment_kl_mean' in usable and len(usable) > 0) else np.nan,
        'step_moment_kl_nonzero_ratio_mean': float(np.nanmean(usable['step_moment_kl_nonzero_ratio'])) if ('step_moment_kl_nonzero_ratio' in usable and len(usable) > 0) else np.nan,
        'trace_pair_ratio_mean': float(np.nanmean(usable['trace_pair_ratio'])) if ('trace_pair_ratio' in usable and len(usable) > 0) else np.nan,
        'trace_pair_ratio_all_mean': float(np.nanmean(usable['trace_pair_ratio_all'])) if ('trace_pair_ratio_all' in usable and len(usable) > 0) else np.nan,
        'trace_fallback_pair_ratio_mean': float(np.nanmean(usable['trace_fallback_pair_ratio'])) if ('trace_fallback_pair_ratio' in usable and len(usable) > 0) else np.nan,
        'threshold_source': thresholds.get('source', 'unknown'),
        'risk_high_threshold': float(thresholds.get('risk_high_threshold', np.nan)),
        'risk_low_threshold': float(thresholds.get('risk_low_threshold', np.nan)),
        'surprise_high_threshold': float(thresholds.get('surprise_high_threshold', np.nan)),
        'risk_scale': float(thresholds.get('risk_scale', np.nan)),
        'surprise_scale': float(thresholds.get('surprise_scale', np.nan)),
    }])

    quant = pd.DataFrame({
        'metric': ['base_risk_sks', 'surprise_pd'],
        'q05': [float(usable['base_risk_sks'].quantile(0.05)) if len(usable) else np.nan,
                float(usable['surprise_pd'].quantile(0.05)) if len(usable) else np.nan],
        'q50': [float(usable['base_risk_sks'].quantile(0.50)) if len(usable) else np.nan,
                float(usable['surprise_pd'].quantile(0.50)) if len(usable) else np.nan],
        'q95': [float(usable['base_risk_sks'].quantile(0.95)) if len(usable) else np.nan,
                float(usable['surprise_pd'].quantile(0.95)) if len(usable) else np.nan],
    })

    return summary, quant

def run_surprise_quality_gate(
    closedloop_calib_df: pd.DataFrame,
    surprise_gate_enabled: bool = True,
    surprise_min_std: float = 1e-8,
    surprise_min_nonzero_fraction: float = 0.01,
    surprise_fallback_hard_stop_ratio: float = 0.95,
    surprise_min_trace_pair_ratio: float = 0.05,
):
    usable_calib = closedloop_calib_df[
        np.isfinite(closedloop_calib_df['base_risk_sks']) & np.isfinite(closedloop_calib_df['surprise_pd'])
    ].copy() if isinstance(closedloop_calib_df, pd.DataFrame) and len(closedloop_calib_df) > 0 else pd.DataFrame()

    if len(usable_calib) == 0:
        raise RuntimeError('No usable closed-loop calibration rows. Cannot validate surprise signal.')

    def _col_mean(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns or len(df) == 0:
            return float('nan')
        return float(np.nanmean(df[col].to_numpy(dtype=float)))

    def _col_q(df: pd.DataFrame, col: str, q: float) -> float:
        if col not in df.columns or len(df) == 0:
            return float('nan')
        arr = df[col].to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float('nan')
        return float(np.quantile(arr, q))

    surprise_std = float(np.nanstd(usable_calib['surprise_pd'].to_numpy(dtype=float)))
    nonzero_surprise_fraction = float(np.mean(usable_calib['surprise_pd'].to_numpy(dtype=float) > 1e-9))
    fallback_usage_rate = _col_mean(usable_calib, 'proposal_dist_fallback_ratio')
    actor_fallback_usage_rate = _col_mean(usable_calib, 'proposal_dist_actor_fallback_ratio')
    trace_pair_ratio_mean = _col_mean(usable_calib, 'trace_pair_ratio')
    trace_pair_ratio_all_mean = _col_mean(usable_calib, 'trace_pair_ratio_all')
    trace_fallback_pair_ratio_mean = _col_mean(usable_calib, 'trace_fallback_pair_ratio')

    dist_change_summary = pd.DataFrame([{
        'trace_pair_ratio_mean': _col_mean(usable_calib, 'trace_pair_ratio'),
        'trace_pair_ratio_all_mean': _col_mean(usable_calib, 'trace_pair_ratio_all'),
        'trace_fallback_pair_ratio_mean': _col_mean(usable_calib, 'trace_fallback_pair_ratio'),
        'step_mean_l2_mean': _col_mean(usable_calib, 'step_mean_l2_mean'),
        'step_mean_l2_p50': _col_q(usable_calib, 'step_mean_l2_mean', 0.50),
        'step_mean_l2_p95': _col_q(usable_calib, 'step_mean_l2_mean', 0.95),
        'step_moment_kl_mean': _col_mean(usable_calib, 'step_moment_kl_mean'),
        'step_moment_kl_p50': _col_q(usable_calib, 'step_moment_kl_mean', 0.50),
        'step_moment_kl_p95': _col_q(usable_calib, 'step_moment_kl_mean', 0.95),
        'step_moment_kl_nonzero_ratio_mean': _col_mean(usable_calib, 'step_moment_kl_nonzero_ratio'),
    }])

    gate_summary = pd.DataFrame([{
        'usable_calibration_rows': int(len(usable_calib)),
        'surprise_std': surprise_std,
        'nonzero_surprise_fraction': nonzero_surprise_fraction,
        'fallback_usage_rate': fallback_usage_rate,
        'actor_fallback_usage_rate': actor_fallback_usage_rate,
        'trace_pair_ratio_mean': trace_pair_ratio_mean,
        'trace_pair_ratio_all_mean': trace_pair_ratio_all_mean,
        'trace_fallback_pair_ratio_mean': trace_fallback_pair_ratio_mean,
    }])

    reasons = []
    if not np.isfinite(surprise_std) or surprise_std <= float(surprise_min_std):
        reasons.append(f'surprise_std too small: {surprise_std:.3e} <= {surprise_min_std:.3e}')
    if nonzero_surprise_fraction < float(surprise_min_nonzero_fraction):
        reasons.append(
            f'nonzero_surprise_fraction too low: {nonzero_surprise_fraction:.4f} < {surprise_min_nonzero_fraction:.4f}'
        )
    if np.isfinite(fallback_usage_rate) and fallback_usage_rate >= float(surprise_fallback_hard_stop_ratio):
        reasons.append(
            f'fallback_usage_rate too high: {fallback_usage_rate:.4f} >= {surprise_fallback_hard_stop_ratio:.4f}'
        )
    if np.isfinite(trace_pair_ratio_mean) and trace_pair_ratio_mean < float(surprise_min_trace_pair_ratio):
        reasons.append(
            f'trace_pair_ratio_mean too low: {trace_pair_ratio_mean:.4f} < {surprise_min_trace_pair_ratio:.4f}'
        )

    if surprise_gate_enabled and len(reasons) > 0:
        raise RuntimeError('Surprise diagnostics gate FAILED:\n- ' + '\n- '.join(reasons))

    return gate_summary, dist_change_summary
