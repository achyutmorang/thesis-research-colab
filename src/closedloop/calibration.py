from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import SearchConfig, ClosedLoopConfig
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

def run_closedloop_preflight_checks(runner: Any, cfg: ClosedLoopConfig, eval_idx: np.ndarray) -> pd.DataFrame:
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
            model_source_ratio = float(dist_diag.get('dist_source_model_ratio', np.nan))
            proxy_source_ratio = float(dist_diag.get('dist_source_proxy_ratio', np.nan))
            max_fallback = float(getattr(cfg, 'latentdriver_preflight_max_fallback_ratio', 0.95))
            add(
                'latentdriver_dist_fallback_ratio_ok',
                bool(np.isfinite(fallback_ratio) and fallback_ratio < max_fallback),
                f'fallback_ratio={fallback_ratio:.4f}, actor_fallback_ratio={actor_fallback_ratio:.4f}, max={max_fallback:.4f}',
            )
            add(
                'latentdriver_dist_model_source_ratio_positive',
                bool(np.isfinite(model_source_ratio) and model_source_ratio > 0.0),
                f'model_source_ratio={model_source_ratio:.4f}, proxy_source_ratio={proxy_source_ratio:.4f}',
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
            token_count_expected = int(getattr(ld_adapter, '_expected_token_count', 0))
            token_align_info = dict(getattr(ld_adapter, '_token_align_info', {}))
            add(
                'latentdriver_expected_token_count_known',
                True,
                f'expected_tokens={token_count_expected}, source={getattr(ld_adapter, "_expected_token_count_source", "unknown")}',
            )
            add(
                'latentdriver_token_alignment_ready',
                bool(token_align_info.get('enabled', 1) == 1),
                str(token_align_info),
            )
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

    base_scale = float(max(search_cfg.random_scale, 0.55))
    scales = np.asarray([base_scale, 0.95, 1.35], dtype=float)
    scale = float(scales[(int(k) // int(len(dirs))) % int(len(scales))])

    jitter = rng.normal(loc=0.0, scale=0.15 * base_scale, size=(2,))
    prop = scale * direction + jitter
    return project_delta_vec(prop, search_cfg.delta_clip, search_cfg.delta_l2_budget)


def make_sensitivity_grid_proposals(
    search_cfg: SearchConfig,
    n_angles: int = 8,
    scales: Tuple[float, ...] = (0.45, 0.9, 1.2),
) -> np.ndarray:
    n_angles = int(max(4, n_angles))
    raw_scales = np.asarray(scales, dtype=float).reshape(-1)
    raw_scales = raw_scales[np.isfinite(raw_scales) & (raw_scales > 0.0)]
    if raw_scales.size == 0:
        raw_scales = np.asarray([0.45, 0.9, 1.2], dtype=float)

    angles = np.linspace(0.0, 2.0 * np.pi, num=n_angles, endpoint=False)
    proposals: List[np.ndarray] = []
    for scale in raw_scales:
        for ang in angles:
            vec = np.asarray([np.cos(float(ang)), np.sin(float(ang))], dtype=float) * float(scale)
            proposals.append(project_delta_vec(vec, search_cfg.delta_clip, search_cfg.delta_l2_budget))
    return np.asarray(proposals, dtype=float)


def trajectory_effect_l2_mean(
    base_xy: np.ndarray,
    base_valid: np.ndarray,
    prop_xy: np.ndarray,
    prop_valid: np.ndarray,
) -> float:
    xb = np.asarray(base_xy, dtype=float)
    xp = np.asarray(prop_xy, dtype=float)
    vb = np.asarray(base_valid, dtype=bool)
    vp = np.asarray(prop_valid, dtype=bool)

    if xb.ndim != 3 or xp.ndim != 3:
        return float('nan')

    n_obj = int(min(xb.shape[0], xp.shape[0], vb.shape[0], vp.shape[0]))
    n_t = int(min(xb.shape[1], xp.shape[1], vb.shape[1], vp.shape[1]))
    if n_obj <= 0 or n_t <= 0:
        return float('nan')

    xb = xb[:n_obj, :n_t, :]
    xp = xp[:n_obj, :n_t, :]
    vb = vb[:n_obj, :n_t]
    vp = vp[:n_obj, :n_t]
    diff_l2 = np.linalg.norm(xp - xb, axis=-1)
    mask = vb & vp
    if np.any(mask):
        return float(np.nanmean(diff_l2[mask]))
    return float(np.nanmean(diff_l2))

def calibrate_closed_loop_thresholds(
    runner: ClosedLoopRunner,
    eval_idx: np.ndarray,
    cfg: ClosedLoopConfig,
    search_cfg: SearchConfig,
    reference_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    calib_ids = np.asarray(eval_idx[: min(len(eval_idx), cfg.n_closedloop_calib)], dtype=int)
    rows: List[Dict[str, Any]] = []
    sensitivity_rows: List[Dict[str, Any]] = []
    min_effect_l2 = float(max(0.0, getattr(cfg, 'surprise_min_effect_l2_mean', 0.05)))
    max_scan_scenarios = int(max(0, getattr(cfg, 'sensitivity_scan_max_scenarios', 20)))
    scan_sid_set = set(int(x) for x in calib_ids[:max_scan_scenarios].tolist())
    sensitivity_grid = make_sensitivity_grid_proposals(
        search_cfg=search_cfg,
        n_angles=int(max(4, getattr(cfg, 'sensitivity_scan_num_angles', 8))),
        scales=tuple(getattr(cfg, 'sensitivity_scan_scales', (0.45, 0.9, 1.2))),
    )

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
                    'dist_source_model_ratio': np.nan,
                    'dist_source_fallback_ratio': np.nan,
                    'dist_source_proxy_ratio': np.nan,
                }

            def _compute_surprise_and_diags(
                prop_actions: np.ndarray,
                prop_action_valid: np.ndarray,
                prop_dist_trace: List[Optional[Dict[str, np.ndarray]]],
                seed_offset: int,
            ) -> Tuple[float, Dict[str, float], Dict[str, float], str]:
                if planner_bundle['planner_type'] == 'latentdriver':
                    surprise_val = predictive_kl_from_dist_traces(
                        prop_dist_trace,
                        base_dist_trace,
                        estimator=cfg.predictive_kl_estimator,
                        n_mc_samples=cfg.predictive_kl_mc_samples,
                        seed=int(cfg.predictive_kl_mc_seed + int(sid) * 1000 + int(seed_offset)),
                        eps=float(cfg.predictive_kl_eps),
                        symmetric=bool(cfg.predictive_kl_symmetric),
                        skip_fallback_steps=bool(cfg.predictive_kl_skip_fallback_steps),
                    )
                    prop_dist_diag = dist_trace_diagnostics(prop_dist_trace)
                    trace_diag = dist_trace_change_stats(prop_dist_trace, base_dist_trace)
                    source = 'predictive_kl'
                    trace_pair_ratio = float(trace_diag.get('trace_pair_ratio', 0.0))
                    if trace_pair_ratio <= 0.0:
                        action_surprise = planner_action_surprise_kl(
                            prop_actions,
                            prop_action_valid,
                            base_actions,
                            base_action_valid,
                            sigma=0.25,
                        )
                        if np.isfinite(action_surprise) and float(action_surprise) > 1e-12:
                            surprise_val = float(action_surprise)
                            source = 'action_kl_no_dist_pairs'
                        else:
                            surprise_val = np.nan
                    elif (not np.isfinite(surprise_val)) or (float(surprise_val) <= 1e-12):
                        action_surprise = planner_action_surprise_kl(
                            prop_actions,
                            prop_action_valid,
                            base_actions,
                            base_action_valid,
                            sigma=0.25,
                        )
                        if np.isfinite(action_surprise) and float(action_surprise) > 1e-12:
                            surprise_val = float(action_surprise)
                            source = 'action_kl_fallback'
                else:
                    surprise_val = planner_action_surprise_kl(
                        prop_actions,
                        prop_action_valid,
                        base_actions,
                        base_action_valid,
                        sigma=0.25,
                    )
                    prop_dist_diag = {
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
                        'dist_source_model_ratio': np.nan,
                        'dist_source_fallback_ratio': np.nan,
                        'dist_source_proxy_ratio': np.nan,
                    }
                    trace_diag = {
                        'trace_pair_steps': np.nan,
                        'trace_pair_ratio': np.nan,
                        'trace_pair_steps_all': np.nan,
                        'trace_pair_ratio_all': np.nan,
                        'trace_fallback_pair_ratio': np.nan,
                        'step_mean_l2_mean': np.nan,
                        'step_mean_l2_p50': np.nan,
                        'step_mean_l2_p95': np.nan,
                        'step_mean_l2_all_mean': np.nan,
                        'step_std_l2_mean': np.nan,
                        'step_std_l2_all_mean': np.nan,
                        'step_moment_kl_mean': np.nan,
                        'step_moment_kl_p50': np.nan,
                        'step_moment_kl_p95': np.nan,
                        'step_moment_kl_nonzero_ratio': np.nan,
                        'step_moment_kl_all_mean': np.nan,
                        'step_moment_kl_all_nonzero_ratio': np.nan,
                        'step_logit_l1_mean': np.nan,
                        'step_logit_l1_p50': np.nan,
                        'step_logit_l1_p95': np.nan,
                        'step_logit_l1_nonzero_ratio': np.nan,
                        'step_logit_l1_all_mean': np.nan,
                        'step_logit_l1_all_nonzero_ratio': np.nan,
                    }
                    source = 'action_kl'
                return float(surprise_val), prop_dist_diag, trace_diag, source

            if (int(sid) in scan_sid_set) and (len(sensitivity_grid) > 0):
                for g, grid_prop in enumerate(sensitivity_grid):
                    s_xy, s_valid, s_actions, s_action_valid, s_dist_trace, s_feasible, _ = closed_loop_rollout_selected(
                        base_state=base_state,
                        selected_idx=selected_idx,
                        target_obj_idx=target_idx,
                        delta_xy=grid_prop,
                        cfg=cfg,
                        planner_bundle=planner_bundle,
                        seed=cfg.global_seed + int(sid) + 50000 + int(g),
                    )
                    s_surprise, _, s_trace_change_diag, s_source = _compute_surprise_and_diags(
                        prop_actions=s_actions,
                        prop_action_valid=s_action_valid,
                        prop_dist_trace=s_dist_trace,
                        seed_offset=50000 + int(g),
                    )
                    s_effect_l2 = trajectory_effect_l2_mean(base_xy, base_valid, s_xy, s_valid)
                    s_effect_ok = bool(np.isfinite(s_effect_l2) and (s_effect_l2 >= min_effect_l2))
                    if not s_effect_ok:
                        s_surprise = np.nan
                    sensitivity_rows.append({
                        'scenario_id': int(sid),
                        'grid_id': int(g),
                        'delta_l2': float(np.linalg.norm(grid_prop)),
                        'surprise_pd': float(s_surprise),
                        'surprise_source': str(s_source),
                        'effect_l2_mean': float(s_effect_l2),
                        'effect_ok': int(s_effect_ok),
                        'rollout_feasible': int(s_feasible),
                        'step_mean_l2_mean': float(s_trace_change_diag.get('step_mean_l2_mean', np.nan)),
                        'step_std_l2_mean': float(s_trace_change_diag.get('step_std_l2_mean', np.nan)),
                        'step_moment_kl_mean': float(s_trace_change_diag.get('step_moment_kl_mean', np.nan)),
                        'step_logit_l1_mean': float(s_trace_change_diag.get('step_logit_l1_mean', np.nan)),
                    })

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
                p_surprise, p_dist_diag, trace_change_diag, surprise_source = _compute_surprise_and_diags(
                    prop_actions=p_actions,
                    prop_action_valid=p_action_valid,
                    prop_dist_trace=p_dist_trace,
                    seed_offset=1000 + int(k),
                )

                effect_l2_mean = trajectory_effect_l2_mean(base_xy, base_valid, p_xy, p_valid)
                effect_ok = bool(np.isfinite(effect_l2_mean) and (effect_l2_mean >= min_effect_l2))
                if not effect_ok:
                    p_surprise = np.nan

                rows.append({
                    'scenario_id': int(sid),
                    'planner_used': planner_bundle['planner_used'],
                    'proposal_id': int(k),
                    'base_risk_sks': float(base_risk['risk_sks']),
                    'proposal_risk_sks': float(p_risk['risk_sks']),
                    'surprise_pd': float(p_surprise),
                    'surprise_source': str(surprise_source),
                    'proposal_effect_l2_mean': float(effect_l2_mean),
                    'proposal_effect_valid': int(effect_ok),
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
                    'base_dist_source_model_ratio': float(base_dist_diag.get('dist_source_model_ratio', np.nan)),
                    'base_dist_source_fallback_ratio': float(base_dist_diag.get('dist_source_fallback_ratio', np.nan)),
                    'base_dist_source_proxy_ratio': float(base_dist_diag.get('dist_source_proxy_ratio', np.nan)),
                    'proposal_dist_non_null_ratio': float(p_dist_diag.get('dist_non_null_ratio', np.nan)),
                    'proposal_dist_fallback_ratio': float(p_dist_diag.get('dist_fallback_ratio', np.nan)),
                    'proposal_dist_actor_fallback_ratio': float(p_dist_diag.get('dist_actor_fallback_ratio', np.nan)),
                    'proposal_dist_source_model_ratio': float(p_dist_diag.get('dist_source_model_ratio', np.nan)),
                    'proposal_dist_source_fallback_ratio': float(p_dist_diag.get('dist_source_fallback_ratio', np.nan)),
                    'proposal_dist_source_proxy_ratio': float(p_dist_diag.get('dist_source_proxy_ratio', np.nan)),
                    'step_mean_l2_mean': float(trace_change_diag.get('step_mean_l2_mean', np.nan)),
                    'step_mean_l2_p50': float(trace_change_diag.get('step_mean_l2_p50', np.nan)),
                    'step_mean_l2_p95': float(trace_change_diag.get('step_mean_l2_p95', np.nan)),
                    'step_mean_l2_all_mean': float(trace_change_diag.get('step_mean_l2_all_mean', np.nan)),
                    'step_std_l2_mean': float(trace_change_diag.get('step_std_l2_mean', np.nan)),
                    'step_std_l2_all_mean': float(trace_change_diag.get('step_std_l2_all_mean', np.nan)),
                    'step_moment_kl_mean': float(trace_change_diag.get('step_moment_kl_mean', np.nan)),
                    'step_moment_kl_p50': float(trace_change_diag.get('step_moment_kl_p50', np.nan)),
                    'step_moment_kl_p95': float(trace_change_diag.get('step_moment_kl_p95', np.nan)),
                    'step_moment_kl_nonzero_ratio': float(trace_change_diag.get('step_moment_kl_nonzero_ratio', np.nan)),
                    'step_moment_kl_all_mean': float(trace_change_diag.get('step_moment_kl_all_mean', np.nan)),
                    'step_moment_kl_all_nonzero_ratio': float(trace_change_diag.get('step_moment_kl_all_nonzero_ratio', np.nan)),
                    'step_logit_l1_mean': float(trace_change_diag.get('step_logit_l1_mean', np.nan)),
                    'step_logit_l1_p50': float(trace_change_diag.get('step_logit_l1_p50', np.nan)),
                    'step_logit_l1_p95': float(trace_change_diag.get('step_logit_l1_p95', np.nan)),
                    'step_logit_l1_nonzero_ratio': float(trace_change_diag.get('step_logit_l1_nonzero_ratio', np.nan)),
                    'step_logit_l1_all_mean': float(trace_change_diag.get('step_logit_l1_all_mean', np.nan)),
                    'step_logit_l1_all_nonzero_ratio': float(trace_change_diag.get('step_logit_l1_all_nonzero_ratio', np.nan)),
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
                'surprise_source': 'error',
                'proposal_effect_l2_mean': np.nan,
                'proposal_effect_valid': 0,
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
                'base_dist_source_model_ratio': np.nan,
                'base_dist_source_fallback_ratio': np.nan,
                'base_dist_source_proxy_ratio': np.nan,
                'proposal_dist_non_null_ratio': np.nan,
                'proposal_dist_fallback_ratio': np.nan,
                'proposal_dist_actor_fallback_ratio': np.nan,
                'proposal_dist_source_model_ratio': np.nan,
                'proposal_dist_source_fallback_ratio': np.nan,
                'proposal_dist_source_proxy_ratio': np.nan,
                'step_mean_l2_mean': np.nan,
                'step_mean_l2_p50': np.nan,
                'step_mean_l2_p95': np.nan,
                'step_mean_l2_all_mean': np.nan,
                'step_std_l2_mean': np.nan,
                'step_std_l2_all_mean': np.nan,
                'step_moment_kl_mean': np.nan,
                'step_moment_kl_p50': np.nan,
                'step_moment_kl_p95': np.nan,
                'step_moment_kl_nonzero_ratio': np.nan,
                'step_moment_kl_all_mean': np.nan,
                'step_moment_kl_all_nonzero_ratio': np.nan,
                'step_logit_l1_mean': np.nan,
                'step_logit_l1_p50': np.nan,
                'step_logit_l1_p95': np.nan,
                'step_logit_l1_nonzero_ratio': np.nan,
                'step_logit_l1_all_mean': np.nan,
                'step_logit_l1_all_nonzero_ratio': np.nan,
                'trace_pair_steps': np.nan,
                'trace_pair_ratio': np.nan,
                'trace_pair_steps_all': np.nan,
                'trace_pair_ratio_all': np.nan,
                'trace_fallback_pair_ratio': np.nan,
            })

    calib_df = pd.DataFrame(rows)

    if len(sensitivity_rows) > 0:
        sensitivity_df = pd.DataFrame(sensitivity_rows)
        sid_summary = (
            sensitivity_df.groupby('scenario_id')
            .agg(
                sensitivity_surprise_std=('surprise_pd', lambda x: float(np.nanstd(np.asarray(x, dtype=float)))),
                sensitivity_nonzero_fraction=('surprise_pd', lambda x: float(np.nanmean(np.asarray(x, dtype=float) > 1e-9))),
                sensitivity_effect_l2_mean=('effect_l2_mean', lambda x: float(np.nanmean(np.asarray(x, dtype=float)))),
                sensitivity_logit_l1_mean=('step_logit_l1_mean', lambda x: float(np.nanmean(np.asarray(x, dtype=float)))),
                sensitivity_rows=('surprise_pd', 'size'),
            )
            .reset_index()
        )
        sid_summary['sensitivity_flat'] = (
            np.asarray(sid_summary['sensitivity_surprise_std'], dtype=float) <= 1e-8
        ).astype(float)
        if len(calib_df) > 0:
            calib_df = calib_df.merge(sid_summary, on='scenario_id', how='left')

            calib_df['sensitivity_scan_rows'] = float(len(sensitivity_df))
            calib_df['sensitivity_scan_scenarios'] = float(sid_summary['scenario_id'].nunique())
            calib_df['sensitivity_scan_surprise_std'] = float(
                np.nanstd(np.asarray(sensitivity_df['surprise_pd'], dtype=float))
            )
            calib_df['sensitivity_scan_nonzero_fraction'] = float(
                np.nanmean(np.asarray(sensitivity_df['surprise_pd'], dtype=float) > 1e-9)
            )
            calib_df['sensitivity_scan_flat_scenario_fraction'] = float(
                np.nanmean(np.asarray(sid_summary['sensitivity_flat'], dtype=float))
            )
    elif len(calib_df) > 0:
        calib_df['sensitivity_scan_rows'] = np.nan
        calib_df['sensitivity_scan_scenarios'] = np.nan
        calib_df['sensitivity_scan_surprise_std'] = np.nan
        calib_df['sensitivity_scan_nonzero_fraction'] = np.nan
        calib_df['sensitivity_scan_flat_scenario_fraction'] = np.nan

    usable_mask = np.isfinite(calib_df['base_risk_sks']) & np.isfinite(calib_df['surprise_pd'])
    if 'proposal_effect_valid' in calib_df.columns:
        usable_mask = usable_mask & (np.asarray(calib_df['proposal_effect_valid'], dtype=float) > 0.5)
    usable = calib_df[usable_mask].copy()

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
    if 'proposal_effect_valid' in calib_df.columns:
        usable_mask = usable_mask & (np.asarray(calib_df['proposal_effect_valid'], dtype=float) > 0.5)
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
        'proposal_dist_source_model_ratio_mean': float(np.nanmean(usable['proposal_dist_source_model_ratio'])) if ('proposal_dist_source_model_ratio' in usable and len(usable) > 0) else np.nan,
        'proposal_dist_source_fallback_ratio_mean': float(np.nanmean(usable['proposal_dist_source_fallback_ratio'])) if ('proposal_dist_source_fallback_ratio' in usable and len(usable) > 0) else np.nan,
        'proposal_dist_source_proxy_ratio_mean': float(np.nanmean(usable['proposal_dist_source_proxy_ratio'])) if ('proposal_dist_source_proxy_ratio' in usable and len(usable) > 0) else np.nan,
        'proposal_effect_l2_mean': float(np.nanmean(usable['proposal_effect_l2_mean'])) if ('proposal_effect_l2_mean' in usable and len(usable) > 0) else np.nan,
        'proposal_effect_valid_fraction': float(np.nanmean(usable['proposal_effect_valid'])) if ('proposal_effect_valid' in usable and len(usable) > 0) else np.nan,
        'step_mean_l2_mean': float(np.nanmean(usable['step_mean_l2_mean'])) if ('step_mean_l2_mean' in usable and len(usable) > 0) else np.nan,
        'step_mean_l2_all_mean': float(np.nanmean(usable['step_mean_l2_all_mean'])) if ('step_mean_l2_all_mean' in usable and len(usable) > 0) else np.nan,
        'step_moment_kl_mean': float(np.nanmean(usable['step_moment_kl_mean'])) if ('step_moment_kl_mean' in usable and len(usable) > 0) else np.nan,
        'step_moment_kl_nonzero_ratio_mean': float(np.nanmean(usable['step_moment_kl_nonzero_ratio'])) if ('step_moment_kl_nonzero_ratio' in usable and len(usable) > 0) else np.nan,
        'step_moment_kl_all_mean': float(np.nanmean(usable['step_moment_kl_all_mean'])) if ('step_moment_kl_all_mean' in usable and len(usable) > 0) else np.nan,
        'step_moment_kl_all_nonzero_ratio_mean': float(np.nanmean(usable['step_moment_kl_all_nonzero_ratio'])) if ('step_moment_kl_all_nonzero_ratio' in usable and len(usable) > 0) else np.nan,
        'step_logit_l1_mean': float(np.nanmean(usable['step_logit_l1_mean'])) if ('step_logit_l1_mean' in usable and len(usable) > 0) else np.nan,
        'step_logit_l1_nonzero_ratio_mean': float(np.nanmean(usable['step_logit_l1_nonzero_ratio'])) if ('step_logit_l1_nonzero_ratio' in usable and len(usable) > 0) else np.nan,
        'step_logit_l1_all_mean': float(np.nanmean(usable['step_logit_l1_all_mean'])) if ('step_logit_l1_all_mean' in usable and len(usable) > 0) else np.nan,
        'step_logit_l1_all_nonzero_ratio_mean': float(np.nanmean(usable['step_logit_l1_all_nonzero_ratio'])) if ('step_logit_l1_all_nonzero_ratio' in usable and len(usable) > 0) else np.nan,
        'trace_pair_ratio_mean': float(np.nanmean(usable['trace_pair_ratio'])) if ('trace_pair_ratio' in usable and len(usable) > 0) else np.nan,
        'trace_pair_ratio_all_mean': float(np.nanmean(usable['trace_pair_ratio_all'])) if ('trace_pair_ratio_all' in usable and len(usable) > 0) else np.nan,
        'trace_fallback_pair_ratio_mean': float(np.nanmean(usable['trace_fallback_pair_ratio'])) if ('trace_fallback_pair_ratio' in usable and len(usable) > 0) else np.nan,
        'sensitivity_scan_surprise_std': float(np.nanmean(usable['sensitivity_scan_surprise_std'])) if ('sensitivity_scan_surprise_std' in usable and len(usable) > 0) else np.nan,
        'sensitivity_scan_nonzero_fraction': float(np.nanmean(usable['sensitivity_scan_nonzero_fraction'])) if ('sensitivity_scan_nonzero_fraction' in usable and len(usable) > 0) else np.nan,
        'sensitivity_scan_flat_scenario_fraction': float(np.nanmean(usable['sensitivity_scan_flat_scenario_fraction'])) if ('sensitivity_scan_flat_scenario_fraction' in usable and len(usable) > 0) else np.nan,
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
    surprise_min_std: float = 1e-4,
    surprise_min_nonzero_fraction: float = 0.10,
    surprise_fallback_hard_stop_ratio: float = 0.95,
    surprise_min_trace_pair_ratio: float = 0.10,
    surprise_min_effect_l2_mean: float = 0.05,
    surprise_min_logit_l1_mean: float = 1e-3,
    surprise_max_flat_sensitivity_fraction: float = 0.80,
    surprise_min_model_source_ratio: float = 0.01,
):
    raw_calib = closedloop_calib_df.copy() if isinstance(closedloop_calib_df, pd.DataFrame) else pd.DataFrame()

    raw_finite_mask = (
        np.isfinite(raw_calib['base_risk_sks']) & np.isfinite(raw_calib['surprise_pd'])
    ) if len(raw_calib) > 0 else np.asarray([], dtype=bool)
    raw_finite_rows = int(np.sum(raw_finite_mask)) if raw_finite_mask.size > 0 else 0
    raw_effect_valid_rows = (
        int(np.sum(np.asarray(raw_calib['proposal_effect_valid'], dtype=float) > 0.5))
        if (len(raw_calib) > 0 and 'proposal_effect_valid' in raw_calib.columns)
        else np.nan
    )

    usable_mask = (
        np.isfinite(closedloop_calib_df['base_risk_sks']) & np.isfinite(closedloop_calib_df['surprise_pd'])
    ) if isinstance(closedloop_calib_df, pd.DataFrame) and len(closedloop_calib_df) > 0 else np.asarray([], dtype=bool)
    if isinstance(closedloop_calib_df, pd.DataFrame) and ('proposal_effect_valid' in closedloop_calib_df.columns):
        usable_mask = usable_mask & (np.asarray(closedloop_calib_df['proposal_effect_valid'], dtype=float) > 0.5)
    usable_calib = closedloop_calib_df[usable_mask].copy() if isinstance(closedloop_calib_df, pd.DataFrame) and len(closedloop_calib_df) > 0 else pd.DataFrame()

    if len(usable_calib) == 0:
        gate_summary = pd.DataFrame([{
            'usable_calibration_rows': 0,
            'raw_calibration_rows': int(len(raw_calib)),
            'raw_finite_surprise_rows': int(raw_finite_rows),
            'raw_effect_valid_rows': raw_effect_valid_rows,
            'surprise_std': np.nan,
            'nonzero_surprise_fraction': np.nan,
            'fallback_usage_rate': np.nan,
            'actor_fallback_usage_rate': np.nan,
            'model_source_ratio': np.nan,
            'proxy_source_ratio': np.nan,
            'proposal_effect_l2_mean': np.nan,
            'proposal_effect_valid_fraction': np.nan,
            'trace_pair_ratio_mean': np.nan,
            'trace_pair_ratio_all_mean': np.nan,
            'trace_fallback_pair_ratio_mean': np.nan,
            'step_logit_l1_mean': np.nan,
            'step_logit_l1_all_mean': np.nan,
            'sensitivity_flat_scenario_fraction': np.nan,
        }])
        dist_change_summary = pd.DataFrame([{
            'trace_pair_ratio_mean': np.nan,
            'trace_pair_ratio_all_mean': np.nan,
            'trace_fallback_pair_ratio_mean': np.nan,
            'step_mean_l2_mean': np.nan,
            'step_mean_l2_all_mean': np.nan,
            'step_std_l2_mean': np.nan,
            'step_std_l2_all_mean': np.nan,
            'step_mean_l2_p50': np.nan,
            'step_mean_l2_p95': np.nan,
            'step_moment_kl_mean': np.nan,
            'step_moment_kl_all_mean': np.nan,
            'step_moment_kl_p50': np.nan,
            'step_moment_kl_p95': np.nan,
            'step_moment_kl_nonzero_ratio_mean': np.nan,
            'step_moment_kl_all_nonzero_ratio_mean': np.nan,
            'step_logit_l1_mean': np.nan,
            'step_logit_l1_all_mean': np.nan,
            'step_logit_l1_p50': np.nan,
            'step_logit_l1_p95': np.nan,
            'step_logit_l1_nonzero_ratio_mean': np.nan,
            'step_logit_l1_all_nonzero_ratio_mean': np.nan,
            'proposal_dist_source_model_ratio_mean': np.nan,
            'proposal_dist_source_fallback_ratio_mean': np.nan,
            'proposal_dist_source_proxy_ratio_mean': np.nan,
        }])
        reason = (
            'No usable closed-loop calibration rows. '
            f'raw_rows={len(raw_calib)}, raw_finite_surprise_rows={raw_finite_rows}, '
            f'raw_effect_valid_rows={raw_effect_valid_rows}.'
        )
        if surprise_gate_enabled:
            raise RuntimeError(reason)
        print(f'[gate] warning: {reason}')
        return gate_summary, dist_change_summary

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
    proposal_effect_l2_mean = _col_mean(usable_calib, 'proposal_effect_l2_mean')
    proposal_effect_valid_fraction = _col_mean(usable_calib, 'proposal_effect_valid')
    trace_pair_ratio_mean = _col_mean(usable_calib, 'trace_pair_ratio')
    trace_pair_ratio_all_mean = _col_mean(usable_calib, 'trace_pair_ratio_all')
    trace_fallback_pair_ratio_mean = _col_mean(usable_calib, 'trace_fallback_pair_ratio')
    step_logit_l1_mean = _col_mean(usable_calib, 'step_logit_l1_mean')
    model_source_ratio = _col_mean(usable_calib, 'proposal_dist_source_model_ratio')
    sensitivity_flat_scenario_fraction = _col_mean(usable_calib, 'sensitivity_scan_flat_scenario_fraction')

    dist_change_summary = pd.DataFrame([{
        'trace_pair_ratio_mean': _col_mean(usable_calib, 'trace_pair_ratio'),
        'trace_pair_ratio_all_mean': _col_mean(usable_calib, 'trace_pair_ratio_all'),
        'trace_fallback_pair_ratio_mean': _col_mean(usable_calib, 'trace_fallback_pair_ratio'),
        'step_mean_l2_mean': _col_mean(usable_calib, 'step_mean_l2_mean'),
        'step_mean_l2_all_mean': _col_mean(usable_calib, 'step_mean_l2_all_mean'),
        'step_std_l2_mean': _col_mean(usable_calib, 'step_std_l2_mean'),
        'step_std_l2_all_mean': _col_mean(usable_calib, 'step_std_l2_all_mean'),
        'step_mean_l2_p50': _col_q(usable_calib, 'step_mean_l2_mean', 0.50),
        'step_mean_l2_p95': _col_q(usable_calib, 'step_mean_l2_mean', 0.95),
        'step_moment_kl_mean': _col_mean(usable_calib, 'step_moment_kl_mean'),
        'step_moment_kl_all_mean': _col_mean(usable_calib, 'step_moment_kl_all_mean'),
        'step_moment_kl_p50': _col_q(usable_calib, 'step_moment_kl_mean', 0.50),
        'step_moment_kl_p95': _col_q(usable_calib, 'step_moment_kl_mean', 0.95),
        'step_moment_kl_nonzero_ratio_mean': _col_mean(usable_calib, 'step_moment_kl_nonzero_ratio'),
        'step_moment_kl_all_nonzero_ratio_mean': _col_mean(usable_calib, 'step_moment_kl_all_nonzero_ratio'),
        'step_logit_l1_mean': _col_mean(usable_calib, 'step_logit_l1_mean'),
        'step_logit_l1_all_mean': _col_mean(usable_calib, 'step_logit_l1_all_mean'),
        'step_logit_l1_p50': _col_q(usable_calib, 'step_logit_l1_mean', 0.50),
        'step_logit_l1_p95': _col_q(usable_calib, 'step_logit_l1_mean', 0.95),
        'step_logit_l1_nonzero_ratio_mean': _col_mean(usable_calib, 'step_logit_l1_nonzero_ratio'),
        'step_logit_l1_all_nonzero_ratio_mean': _col_mean(usable_calib, 'step_logit_l1_all_nonzero_ratio'),
        'proposal_dist_source_model_ratio_mean': _col_mean(usable_calib, 'proposal_dist_source_model_ratio'),
        'proposal_dist_source_fallback_ratio_mean': _col_mean(usable_calib, 'proposal_dist_source_fallback_ratio'),
        'proposal_dist_source_proxy_ratio_mean': _col_mean(usable_calib, 'proposal_dist_source_proxy_ratio'),
    }])

    gate_summary = pd.DataFrame([{
        'usable_calibration_rows': int(len(usable_calib)),
        'raw_calibration_rows': int(len(raw_calib)),
        'raw_finite_surprise_rows': int(raw_finite_rows),
        'raw_effect_valid_rows': raw_effect_valid_rows,
        'surprise_std': surprise_std,
        'nonzero_surprise_fraction': nonzero_surprise_fraction,
        'fallback_usage_rate': fallback_usage_rate,
        'actor_fallback_usage_rate': actor_fallback_usage_rate,
        'model_source_ratio': _col_mean(usable_calib, 'proposal_dist_source_model_ratio'),
        'proxy_source_ratio': _col_mean(usable_calib, 'proposal_dist_source_proxy_ratio'),
        'proposal_effect_l2_mean': proposal_effect_l2_mean,
        'proposal_effect_valid_fraction': proposal_effect_valid_fraction,
        'trace_pair_ratio_mean': trace_pair_ratio_mean,
        'trace_pair_ratio_all_mean': trace_pair_ratio_all_mean,
        'trace_fallback_pair_ratio_mean': trace_fallback_pair_ratio_mean,
        'step_logit_l1_mean': step_logit_l1_mean,
        'step_logit_l1_all_mean': _col_mean(usable_calib, 'step_logit_l1_all_mean'),
        'sensitivity_flat_scenario_fraction': sensitivity_flat_scenario_fraction,
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
    if np.isfinite(proposal_effect_l2_mean) and proposal_effect_l2_mean < float(surprise_min_effect_l2_mean):
        reasons.append(
            f'proposal_effect_l2_mean too low: {proposal_effect_l2_mean:.4f} < {surprise_min_effect_l2_mean:.4f}'
        )
    if np.isfinite(step_logit_l1_mean) and step_logit_l1_mean < float(surprise_min_logit_l1_mean):
        reasons.append(
            f'step_logit_l1_mean too low: {step_logit_l1_mean:.4e} < {surprise_min_logit_l1_mean:.4e}'
        )
    if np.isfinite(model_source_ratio) and model_source_ratio < float(surprise_min_model_source_ratio):
        reasons.append(
            f'model_source_ratio too low: {model_source_ratio:.4f} < {surprise_min_model_source_ratio:.4f}'
        )
    if (
        np.isfinite(sensitivity_flat_scenario_fraction)
        and sensitivity_flat_scenario_fraction > float(surprise_max_flat_sensitivity_fraction)
    ):
        reasons.append(
            f'sensitivity_flat_scenario_fraction too high: {sensitivity_flat_scenario_fraction:.4f} > {surprise_max_flat_sensitivity_fraction:.4f}'
        )

    divergence_channels = np.asarray([
        _col_mean(usable_calib, 'step_mean_l2_mean'),
        _col_mean(usable_calib, 'step_mean_l2_all_mean'),
        _col_mean(usable_calib, 'step_std_l2_mean'),
        _col_mean(usable_calib, 'step_std_l2_all_mean'),
        _col_mean(usable_calib, 'step_moment_kl_mean'),
        _col_mean(usable_calib, 'step_moment_kl_all_mean'),
        _col_mean(usable_calib, 'step_logit_l1_mean'),
        _col_mean(usable_calib, 'step_logit_l1_all_mean'),
    ], dtype=float)
    divergence_channels = np.abs(divergence_channels[np.isfinite(divergence_channels)])
    if divergence_channels.size > 0 and float(np.max(divergence_channels)) <= 1e-10:
        reasons.append(
            'all divergence channels collapsed to ~0 (mean_l2/std_l2/moment_kl/logit_l1); surprise signal is not informative.'
        )

    if surprise_gate_enabled and len(reasons) > 0:
        raise RuntimeError('Surprise diagnostics gate FAILED:\n- ' + '\n- '.join(reasons))

    return gate_summary, dist_change_summary


def diagnose_surprise_root_cause(
    preflight_df: pd.DataFrame,
    closedloop_calib_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def _safe_mean(df: pd.DataFrame, col: str) -> float:
        if (not isinstance(df, pd.DataFrame)) or (col not in df.columns) or len(df) == 0:
            return float('nan')
        return float(np.nanmean(df[col].to_numpy(dtype=float)))

    def _safe_rate(df: pd.DataFrame, col: str, thr: float = 1e-9) -> float:
        if (not isinstance(df, pd.DataFrame)) or (col not in df.columns) or len(df) == 0:
            return float('nan')
        arr = df[col].to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float('nan')
        return float(np.mean(arr > float(thr)))

    raw = closedloop_calib_df.copy() if isinstance(closedloop_calib_df, pd.DataFrame) else pd.DataFrame()
    usable = raw.copy()
    if len(usable) > 0 and 'proposal_effect_valid' in usable.columns:
        usable = usable[np.asarray(usable['proposal_effect_valid'], dtype=float) > 0.5].copy()

    raw_finite_surprise_rows = (
        int(np.sum(np.isfinite(raw['surprise_pd'].to_numpy(dtype=float))))
        if (len(raw) > 0 and 'surprise_pd' in raw.columns)
        else 0
    )
    raw_effect_valid_rows = (
        int(np.sum(np.asarray(raw['proposal_effect_valid'], dtype=float) > 0.5))
        if (len(raw) > 0 and 'proposal_effect_valid' in raw.columns)
        else 0
    )
    raw_surprise_nonzero_fraction = _safe_rate(raw, 'surprise_pd', thr=1e-9)
    raw_fallback_ratio = _safe_mean(raw, 'proposal_dist_fallback_ratio')
    raw_trace_pair_ratio = _safe_mean(raw, 'trace_pair_ratio')

    surprise_nonzero_fraction = _safe_rate(usable, 'surprise_pd', thr=1e-9)
    fallback_ratio = _safe_mean(usable, 'proposal_dist_fallback_ratio')
    actor_fallback_ratio = _safe_mean(usable, 'proposal_dist_actor_fallback_ratio')
    model_source_ratio = _safe_mean(usable, 'proposal_dist_source_model_ratio')
    proxy_source_ratio = _safe_mean(usable, 'proposal_dist_source_proxy_ratio')
    trace_pair_ratio = _safe_mean(usable, 'trace_pair_ratio')
    trace_pair_ratio_all = _safe_mean(usable, 'trace_pair_ratio_all')
    step_l2_mean = _safe_mean(usable, 'step_mean_l2_mean')
    step_l2_all_mean = _safe_mean(usable, 'step_mean_l2_all_mean')
    step_logit_mean = _safe_mean(usable, 'step_logit_l1_mean')
    step_logit_all_mean = _safe_mean(usable, 'step_logit_l1_all_mean')
    proposal_effect_l2_mean = _safe_mean(usable, 'proposal_effect_l2_mean')

    preflight_failed = pd.DataFrame()
    if isinstance(preflight_df, pd.DataFrame) and ('pass' in preflight_df.columns):
        preflight_failed = preflight_df[~preflight_df['pass']].copy()
    failed_checks = (
        ', '.join(preflight_failed['check'].astype(str).tolist())
        if len(preflight_failed) > 0 and ('check' in preflight_failed.columns)
        else ''
    )
    forward_fail_detail = ''
    if len(preflight_failed) > 0 and 'check' in preflight_failed.columns and 'detail' in preflight_failed.columns:
        forward_rows = preflight_failed[preflight_failed['check'] == 'latentdriver_forward_route_ok']
        if len(forward_rows) > 0:
            forward_fail_detail = str(forward_rows.iloc[0].get('detail', ''))

    summary = pd.DataFrame([{
        'preflight_failed_count': int(len(preflight_failed)),
        'preflight_failed_checks': failed_checks,
        'forward_fail_detail': forward_fail_detail,
        'calibration_rows': int(len(closedloop_calib_df)) if isinstance(closedloop_calib_df, pd.DataFrame) else 0,
        'raw_finite_surprise_rows': int(raw_finite_surprise_rows),
        'raw_effect_valid_rows': int(raw_effect_valid_rows),
        'raw_surprise_nonzero_fraction': raw_surprise_nonzero_fraction,
        'raw_fallback_ratio': raw_fallback_ratio,
        'raw_trace_pair_ratio': raw_trace_pair_ratio,
        'usable_rows': int(len(usable)),
        'surprise_nonzero_fraction': surprise_nonzero_fraction,
        'proposal_fallback_ratio': fallback_ratio,
        'proposal_actor_fallback_ratio': actor_fallback_ratio,
        'proposal_model_source_ratio': model_source_ratio,
        'proposal_proxy_source_ratio': proxy_source_ratio,
        'trace_pair_ratio': trace_pair_ratio,
        'trace_pair_ratio_all': trace_pair_ratio_all,
        'step_mean_l2_mean': step_l2_mean,
        'step_mean_l2_all_mean': step_l2_all_mean,
        'step_logit_l1_mean': step_logit_mean,
        'step_logit_l1_all_mean': step_logit_all_mean,
        'proposal_effect_l2_mean': proposal_effect_l2_mean,
    }])

    findings: List[Dict[str, Any]] = []

    def add_finding(issue: str, severity: str, evidence: str, action: str) -> None:
        findings.append({
            'issue': str(issue),
            'severity': str(severity),
            'evidence': str(evidence),
            'suggested_action': str(action),
        })

    if len(preflight_failed) > 0:
        add_finding(
            'preflight_failures_present',
            'high',
            failed_checks if len(failed_checks) > 0 else 'preflight contains failed checks',
            'Fix preflight failures before relying on calibration/gate outputs.',
        )
    if len(raw) > 0 and len(usable) == 0:
        add_finding(
            'all_rows_filtered_before_gate',
            'high',
            f'raw_rows={len(raw)}, raw_finite_surprise_rows={raw_finite_surprise_rows}, raw_effect_valid_rows={raw_effect_valid_rows}',
            'Check proposal_effect_valid filtering and surprise NaN generation in calibration rows.',
        )
    if len(raw) > 0 and raw_finite_surprise_rows == 0:
        add_finding(
            'surprise_nan_for_all_calibration_rows',
            'high',
            f'raw_finite_surprise_rows={raw_finite_surprise_rows}, raw_trace_pair_ratio={raw_trace_pair_ratio:.4f}, raw_fallback_ratio={raw_fallback_ratio:.4f}',
            'Inspect surprise_source distribution and trace_pair_ratio; predictive distribution comparisons are not yielding valid surprise.',
        )
    if len(raw) > 0 and ('proposal_effect_valid' in raw.columns) and raw_effect_valid_rows == 0:
        add_finding(
            'proposal_effect_filter_eliminated_all_rows',
            'high',
            f'raw_effect_valid_rows={raw_effect_valid_rows}, mean_effect={_safe_mean(raw, "proposal_effect_l2_mean"):.4f}',
            'Lower cfg.surprise_min_effect_l2_mean or increase proposal perturbation scale/coverage.',
        )
    if np.isfinite(fallback_ratio) and fallback_ratio >= 0.95:
        add_finding(
            'distribution_fallback_dominant',
            'high',
            f'proposal_dist_fallback_ratio={fallback_ratio:.4f}',
            'Focus on LatentDriver forward path (token alignment, route errors) before tuning gate thresholds.',
        )
    if np.isfinite(model_source_ratio) and model_source_ratio <= 0.05:
        add_finding(
            'model_distribution_source_near_zero',
            'high',
            f'proposal_dist_source_model_ratio={model_source_ratio:.4f}',
            'Model-produced distributions are rarely used; inspect forward-route errors and model input shape alignment.',
        )
    if np.isfinite(trace_pair_ratio_all) and np.isfinite(trace_pair_ratio):
        if trace_pair_ratio_all > 0.50 and trace_pair_ratio <= 0.01:
            add_finding(
                'nonfallback_pairs_missing',
                'high',
                f'trace_pair_ratio_all={trace_pair_ratio_all:.4f}, trace_pair_ratio={trace_pair_ratio:.4f}',
                'Most pairs are fallback pairs; try diagnostic run with cfg.predictive_kl_skip_fallback_steps=False.',
            )
    if np.isfinite(step_l2_all_mean) and np.isfinite(step_l2_mean):
        if step_l2_all_mean > 1e-4 and (not np.isfinite(step_l2_mean) or step_l2_mean <= 1e-8):
            add_finding(
                'changes_exist_only_in_fallback_pairs',
                'medium',
                f'step_mean_l2_all_mean={step_l2_all_mean:.4e}, step_mean_l2_mean={step_l2_mean:.4e}',
                'Distribution change exists but only where fallback is active; reduce fallback usage.',
            )
    if np.isfinite(proposal_effect_l2_mean) and proposal_effect_l2_mean < 0.05:
        add_finding(
            'proposal_effect_too_small',
            'medium',
            f'proposal_effect_l2_mean={proposal_effect_l2_mean:.4f}',
            'Increase perturbation coverage (scales/angles/proposals) before calibrating thresholds.',
        )
    if np.isfinite(surprise_nonzero_fraction) and surprise_nonzero_fraction <= 0.01:
        add_finding(
            'surprise_almost_always_zero',
            'high',
            f'surprise_nonzero_fraction={surprise_nonzero_fraction:.4f}',
            'Treat as upstream planner/distribution issue first, not a threshold tuning issue.',
        )

    detail_df = pd.DataFrame(findings)
    if detail_df.empty:
        detail_df = pd.DataFrame([{
            'issue': 'no_critical_root_cause_flags',
            'severity': 'info',
            'evidence': 'No high-confidence collapse signature detected from current inputs.',
            'suggested_action': 'Proceed with gate and verify signal usefulness report after simulation.',
        }])

    return summary, detail_df
