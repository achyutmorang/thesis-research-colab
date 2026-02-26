from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .config import SearchConfig, ClosedLoopConfig
from .latentdriver import (
    closed_loop_rollout_selected,
    dist_trace_change_stats,
    dist_trace_diagnostics,
    predictive_kl_from_dist_traces,
    project_delta_vec,
)
from .metrics import compute_risk_metrics, planner_action_surprise_kl, risk_kwargs_from_cfg

def evaluate_delta_closed_loop(
    rec: Dict[str, Any],
    planner_bundle: Dict[str, Any],
    target_idx: int,
    delta_xy: np.ndarray,
    cfg: ClosedLoopConfig,
    search_cfg: SearchConfig,
    thresholds: Dict[str, float],
    base_metrics: Dict[str, Any],
    w_r: float,
    w_s: float,
    reg_lambda: float,
    seed: int,
) -> Dict[str, float]:
    delta_proj = project_delta_vec(delta_xy, search_cfg.delta_clip, search_cfg.delta_l2_budget)

    xy, valid, actions, action_valid, dist_trace, rollout_feasible, rollout_note = closed_loop_rollout_selected(
        base_state=rec['state'],
        selected_idx=np.asarray(rec['selected_indices'], dtype=np.int32),
        target_obj_idx=target_idx,
        delta_xy=delta_proj,
        cfg=cfg,
        planner_bundle=planner_bundle,
        seed=seed,
    )

    risk = compute_risk_metrics(xy, valid, **risk_kwargs_from_cfg(cfg))

    if planner_bundle['planner_type'] == 'latentdriver':
        surprise_pd = predictive_kl_from_dist_traces(
            dist_trace,
            base_metrics['base_dist_trace'],
            estimator=cfg.predictive_kl_estimator,
            n_mc_samples=cfg.predictive_kl_mc_samples,
            seed=int(cfg.predictive_kl_mc_seed + int(seed)),
            eps=float(cfg.predictive_kl_eps),
            symmetric=bool(cfg.predictive_kl_symmetric),
            skip_fallback_steps=bool(cfg.predictive_kl_skip_fallback_steps),
        )
        dist_diag = dist_trace_diagnostics(dist_trace)
        surprise_source = 'predictive_kl'
        if (not np.isfinite(surprise_pd)) or (float(surprise_pd) <= 1e-12):
            trace_change_diag = dist_trace_change_stats(dist_trace, base_metrics['base_dist_trace'])
            action_surprise = planner_action_surprise_kl(
                actions,
                action_valid,
                base_metrics['base_actions'],
                base_metrics['base_action_valid'],
                sigma=0.25,
            )
            if np.isfinite(action_surprise) and float(action_surprise) > 1e-12:
                surprise_pd = float(action_surprise)
                if float(trace_change_diag.get('trace_pair_ratio', 0.0)) > 0.0:
                    surprise_source = 'action_kl_fallback'
                else:
                    surprise_source = 'action_kl_no_dist_pairs'
    else:
        surprise_pd = planner_action_surprise_kl(
            actions,
            action_valid,
            base_metrics['base_actions'],
            base_metrics['base_action_valid'],
            sigma=0.25,
        )
        dist_diag = {
            'dist_non_null_steps': np.nan,
            'dist_non_null_ratio': np.nan,
            'dist_mean_components': np.nan,
            'dist_min_weight': np.nan,
            'dist_min_std': np.nan,
            'dist_max_std': np.nan,
            'dist_finite_ratio': np.nan,
        }
        surprise_source = 'action_kl'

    delta_risk = float(risk['risk_sks'] - base_metrics['base_risk'])
    delta_surprise = float(surprise_pd - base_metrics['base_surprise'])

    norm_delta_risk = delta_risk / max(float(thresholds['risk_scale']), search_cfg.min_scale)
    norm_delta_surprise = delta_surprise / max(float(thresholds['surprise_scale']), search_cfg.min_scale)

    reg = reg_lambda * float(np.sum(delta_proj ** 2))
    objective = float(w_r * norm_delta_risk + w_s * norm_delta_surprise - reg)

    l2_norm = float(np.linalg.norm(delta_proj))
    max_abs = float(np.max(np.abs(delta_proj)))
    finite_delta = bool(np.isfinite(delta_proj).all())
    clip_ok = bool(max_abs <= (search_cfg.delta_clip + 1e-8))
    l2_ok = bool(l2_norm <= (search_cfg.delta_l2_budget + 1e-8))
    feasible = int(bool(finite_delta and clip_ok and l2_ok and rollout_feasible))

    q1_hit = int((risk['risk_sks'] >= thresholds['risk_high_threshold']) and (surprise_pd >= thresholds['surprise_high_threshold']))
    q4_hit = int((risk['risk_sks'] <= thresholds['risk_low_threshold']) and (surprise_pd >= thresholds['surprise_high_threshold']))
    blind_spot_proxy_hit = int((risk['failure_extended_proxy'] > 0.0) and (surprise_pd >= thresholds['surprise_high_threshold']))

    return {
        'objective': objective,
        'risk_sks': float(risk['risk_sks']),
        'surprise_pd': float(surprise_pd),
        'surprise_kl': float(surprise_pd),
        'surprise_source': str(surprise_source),
        'surprise_metric': cfg.planner_surprise_name,
        'delta_risk': float(delta_risk),
        'delta_surprise': float(delta_surprise),
        'failure_proxy': float(risk['failure_extended_proxy']),
        'failure_strict_proxy': float(risk['failure_strict_proxy']),
        'collision': float(risk['collision']),
        'min_dist': float(risk['min_dist']),
        'min_ttc': float(risk['min_ttc']),
        'max_acc': float(risk['max_acc']),
        'max_jerk': float(risk['max_jerk']),
        'planner_action_valid_rate': float(np.mean(action_valid.astype(float))) if action_valid.size > 0 else 0.0,
        'rollout_feasible': int(rollout_feasible),
        'feasible': feasible,
        'feasibility_violation': float(1 - feasible),
        'rollout_note': rollout_note,
        'max_abs_delta': max_abs,
        'delta_l2': l2_norm,
        'delta_x': float(delta_proj[0]),
        'delta_y': float(delta_proj[1]),
        'q1_hit': q1_hit,
        'q4_hit': q4_hit,
        'blind_spot_proxy_hit': blind_spot_proxy_hit,
        'surprise_finite': int(np.isfinite(surprise_pd)),
        **dist_diag,
    }

def method_weights(method: str, cfg_opt: SearchConfig) -> Tuple[float, float]:
    if method == 'risk_only':
        return cfg_opt.w_risk_only
    if method == 'surprise_only':
        return cfg_opt.w_surprise_only
    if method == 'joint':
        return cfg_opt.w_joint
    if method == 'random':
        return cfg_opt.w_joint
    raise ValueError(f'Unknown method: {method}')

def optimize_method_closed_loop(
    method: str,
    rec: Dict[str, Any],
    planner_bundle: Dict[str, Any],
    target_idx: int,
    cfg: ClosedLoopConfig,
    search_cfg: SearchConfig,
    thresholds: Dict[str, float],
    scenario_seed: int,
    proposal_bank: np.ndarray,
    rollout_seed_schedule: List[int],
) -> Dict[str, Any]:
    w_r, w_s = method_weights(method, search_cfg)

    expected_props = max(0, int(search_cfg.budget_evals) - 1)
    if int(proposal_bank.shape[0]) < expected_props:
        raise ValueError(f'proposal_bank too small: have {proposal_bank.shape[0]}, need {expected_props}.')
    if len(rollout_seed_schedule) < int(search_cfg.budget_evals) + 1:
        raise ValueError('rollout_seed_schedule too short for budget_evals.')

    base_xy, base_valid, base_actions, base_action_valid, base_dist_trace, base_feasible, base_note = closed_loop_rollout_selected(
        base_state=rec['state'],
        selected_idx=np.asarray(rec['selected_indices'], dtype=np.int32),
        target_obj_idx=target_idx,
        delta_xy=np.zeros((2,), dtype=float),
        cfg=cfg,
        planner_bundle=planner_bundle,
        seed=int(rollout_seed_schedule[0]),
    )
    base_risk = compute_risk_metrics(base_xy, base_valid, **risk_kwargs_from_cfg(cfg))

    base_metrics = {
        'base_risk': float(base_risk['risk_sks']),
        'base_surprise': 0.0,
        'base_actions': np.asarray(base_actions, dtype=np.float32),
        'base_action_valid': np.asarray(base_action_valid, dtype=bool),
        'base_dist_trace': base_dist_trace,
    }

    current_delta = np.zeros((2,), dtype=float)
    current = evaluate_delta_closed_loop(
        rec=rec,
        planner_bundle=planner_bundle,
        target_idx=target_idx,
        delta_xy=current_delta,
        cfg=cfg,
        search_cfg=search_cfg,
        thresholds=thresholds,
        base_metrics=base_metrics,
        w_r=w_r,
        w_s=w_s,
        reg_lambda=search_cfg.reg_lambda,
        seed=int(rollout_seed_schedule[0]),
    )
    initial = dict(current)

    best_delta = current_delta.copy()
    best = dict(current)

    evals_used = 1
    improve_count = 0
    best_eval_index = 0
    eval_trace: List[Dict[str, Any]] = []

    def _append_eval_trace(
        eval_index: int,
        stage: str,
        delta_vec: np.ndarray,
        stats: Dict[str, Any],
        accepted: int,
        is_best: int,
        step_scale: float,
    ) -> None:
        eval_trace.append({
            'eval_index': int(eval_index),
            'stage': stage,
            'proposal_seed': int(rollout_seed_schedule[min(eval_index, len(rollout_seed_schedule) - 1)]),
            'delta_x': float(delta_vec[0]),
            'delta_y': float(delta_vec[1]),
            'delta_l2': float(np.linalg.norm(delta_vec)),
            'objective': float(stats.get('objective', np.nan)),
            'risk_sks': float(stats.get('risk_sks', np.nan)),
            'surprise_pd': float(stats.get('surprise_pd', np.nan)),
            'failure_proxy': float(stats.get('failure_proxy', np.nan)),
            'feasible': int(stats.get('feasible', 0)),
            'rollout_feasible': int(stats.get('rollout_feasible', 0)),
            'surprise_finite': int(stats.get('surprise_finite', 0)),
            'dist_non_null_ratio': float(stats.get('dist_non_null_ratio', np.nan)),
            'dist_mean_components': float(stats.get('dist_mean_components', np.nan)),
            'dist_min_weight': float(stats.get('dist_min_weight', np.nan)),
            'dist_min_std': float(stats.get('dist_min_std', np.nan)),
            'dist_max_std': float(stats.get('dist_max_std', np.nan)),
            'accepted': int(accepted),
            'is_best_so_far': int(is_best),
            'step_scale': float(step_scale),
        })

    _append_eval_trace(
        eval_index=0,
        stage='initial',
        delta_vec=current_delta,
        stats=current,
        accepted=1,
        is_best=1,
        step_scale=0.0,
    )

    if method == 'random':
        while evals_used < search_cfg.budget_evals:
            proposal_vec = np.asarray(proposal_bank[evals_used - 1], dtype=float)
            prop = project_delta_vec(search_cfg.random_scale * proposal_vec, search_cfg.delta_clip, search_cfg.delta_l2_budget)

            trial = evaluate_delta_closed_loop(
                rec=rec,
                planner_bundle=planner_bundle,
                target_idx=target_idx,
                delta_xy=prop,
                cfg=cfg,
                thresholds=thresholds,
                base_metrics=base_metrics,
                w_r=w_r,
                w_s=w_s,
                reg_lambda=search_cfg.reg_lambda,
                seed=int(rollout_seed_schedule[evals_used]),
            )
            eval_idx = int(evals_used)
            evals_used += 1

            improved = bool(trial['objective'] > best['objective'])
            if improved:
                best = dict(trial)
                best_delta = prop.copy()
                improve_count += 1
                best_eval_index = eval_idx

            _append_eval_trace(
                eval_index=eval_idx,
                stage='random_proposal',
                delta_vec=prop,
                stats=trial,
                accepted=int(improved),
                is_best=int(improved),
                step_scale=0.0,
            )

    else:
        step_scale = float(search_cfg.step_scale_init)
        while evals_used < search_cfg.budget_evals:
            proposal_vec = np.asarray(proposal_bank[evals_used - 1], dtype=float)
            norm = float(np.linalg.norm(proposal_vec))
            if norm < 1e-12:
                proposal_vec = np.array([1.0, 0.0], dtype=float)
                norm = 1.0
            direction = proposal_vec / norm

            prop = current_delta + step_scale * direction
            prop = project_delta_vec(prop, search_cfg.delta_clip, search_cfg.delta_l2_budget)

            trial = evaluate_delta_closed_loop(
                rec=rec,
                planner_bundle=planner_bundle,
                target_idx=target_idx,
                delta_xy=prop,
                cfg=cfg,
                thresholds=thresholds,
                base_metrics=base_metrics,
                w_r=w_r,
                w_s=w_s,
                reg_lambda=search_cfg.reg_lambda,
                seed=int(rollout_seed_schedule[evals_used]),
            )
            eval_idx = int(evals_used)
            evals_used += 1

            accepted = bool(trial['objective'] >= current['objective'])
            improved_best = False
            if accepted:
                current = dict(trial)
                current_delta = prop.copy()
                improve_count += 1
                if trial['objective'] > best['objective']:
                    best = dict(trial)
                    best_delta = prop.copy()
                    improved_best = True
                    best_eval_index = eval_idx
            _append_eval_trace(
                eval_index=eval_idx,
                stage='guided_proposal',
                delta_vec=prop,
                stats=trial,
                accepted=int(accepted),
                is_best=int(improved_best),
                step_scale=float(step_scale),
            )
            step_scale *= search_cfg.step_scale_decay
            step_scale = max(step_scale, 0.02)

    out = {
        **best,
        'objective_start': float(initial['objective']),
        'objective_gain': float(best['objective'] - initial['objective']),
        'delta_risk_start': float(initial['delta_risk']),
        'delta_surprise_start': float(initial['delta_surprise']),
        'base_rollout_feasible': int(base_feasible),
        'base_rollout_note': base_note,
        'optimizer_used': 'random_search' if method == 'random' else 'stochastic_hillclimb',
        'budget_units_used': int(evals_used),
        'accepted_improvements': int(improve_count),
        'method': method,
        'planner_used': planner_bundle['planner_used'],
        'common_random_numbers_used': 1,
        'scenario_seed_base': int(scenario_seed),
        'best_eval_index': int(best_eval_index),
        'surprise_estimator': cfg.predictive_kl_estimator,
        'surprise_mc_samples': int(cfg.predictive_kl_mc_samples),
        'surprise_eps': float(cfg.predictive_kl_eps),
        'eval_trace': eval_trace,
    }
    out['delta_x'] = float(best_delta[0])
    out['delta_y'] = float(best_delta[1])
    out['delta_l2'] = float(np.linalg.norm(best_delta))
    return out
