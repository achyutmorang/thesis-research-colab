from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from src.closedloop.metrics import compute_finite_differences, compute_risk_metrics, risk_kwargs_from_cfg


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _clip_prob_vector(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size == 0:
        return np.asarray([1.0], dtype=float)
    w = np.maximum(w, 1e-12)
    s = float(np.sum(w))
    if s <= 0.0 or (not np.isfinite(s)):
        return np.ones_like(w) / max(1, w.size)
    return w / s


def dist_entropy_from_weights(weights: Any) -> float:
    w = _clip_prob_vector(np.asarray(weights, dtype=float))
    return float(-np.sum(w * np.log(w)))


def extract_dist_step_features(dist_step: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not isinstance(dist_step, dict) or len(dist_step) == 0:
        return {
            'dist_entropy': 0.0,
            'dist_top1_weight': 1.0,
            'dist_num_components': 0.0,
            'dist_std_mean': 0.0,
            'dist_std_max': 0.0,
            'belief_kl_current': 0.0,
            'belief_kl_available': 0.0,
        }

    weights = _clip_prob_vector(np.asarray(dist_step.get('weights', [1.0]), dtype=float))
    stds = np.asarray(dist_step.get('stds', np.zeros((weights.size, 1))), dtype=float)
    if stds.ndim == 1:
        stds = stds.reshape(-1, 1)
    stds = np.maximum(stds, 0.0)
    belief_kl = _safe_float(dist_step.get('belief_kl_step', np.nan), default=0.0)
    belief_avail = 1.0 if _safe_float(dist_step.get('belief_kl_available', 0.0), default=0.0) > 0.5 else 0.0

    return {
        'dist_entropy': dist_entropy_from_weights(weights),
        'dist_top1_weight': float(np.max(weights)) if weights.size > 0 else 1.0,
        'dist_num_components': float(weights.size),
        'dist_std_mean': float(np.mean(stds)) if stds.size > 0 else 0.0,
        'dist_std_max': float(np.max(stds)) if stds.size > 0 else 0.0,
        'belief_kl_current': belief_kl,
        'belief_kl_available': belief_avail,
    }


def extract_dist_trace_features(
    dist_trace: Optional[Iterable[Optional[Dict[str, Any]]]],
    predictive_seq_kl_nominal: float = np.nan,
    predictive_seq_w2_nominal: float = np.nan,
) -> Dict[str, float]:
    if dist_trace is None:
        return {
            'belief_kl_rolling_mean': 0.0,
            'predictive_seq_kl_nominal': _safe_float(predictive_seq_kl_nominal, default=0.0),
            'predictive_seq_w2_nominal': _safe_float(predictive_seq_w2_nominal, default=0.0),
        }

    belief_vals = []
    ent_vals = []
    for d in dist_trace:
        if not isinstance(d, dict):
            continue
        ent_vals.append(extract_dist_step_features(d)['dist_entropy'])
        cur = _safe_float(d.get('belief_kl_step', np.nan), default=np.nan)
        if np.isfinite(cur):
            belief_vals.append(cur)

    return {
        'belief_kl_rolling_mean': float(np.mean(belief_vals)) if len(belief_vals) > 0 else 0.0,
        'predictive_seq_kl_nominal': _safe_float(predictive_seq_kl_nominal, default=0.0),
        'predictive_seq_w2_nominal': _safe_float(predictive_seq_w2_nominal, default=0.0),
        'dist_entropy_trace_mean': float(np.mean(ent_vals)) if len(ent_vals) > 0 else 0.0,
    }


def extract_rollout_summary_features(
    xy: np.ndarray,
    valid: np.ndarray,
    cfg: Any,
    horizon_steps: Optional[int] = None,
    target_interaction_score: float = np.nan,
    route_deviation_h: float = np.nan,
) -> Dict[str, float]:
    xy = np.asarray(xy, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if xy.ndim != 3 or valid.ndim != 2:
        raise ValueError('Expected xy shape (N,T,2) and valid shape (N,T).')

    n_h = int(horizon_steps or xy.shape[1])
    n_h = max(1, min(n_h, xy.shape[1], valid.shape[1]))
    xy_h = xy[:, :n_h, :]
    valid_h = valid[:, :n_h]

    risk = compute_risk_metrics(xy_h, valid_h, **risk_kwargs_from_cfg(cfg))
    vel, acc, jerk = compute_finite_differences(xy_h)
    ego_vel = np.linalg.norm(vel[0], axis=-1) if xy_h.shape[0] > 0 else np.asarray([0.0], dtype=float)
    ego_acc = np.linalg.norm(acc[0], axis=-1) if xy_h.shape[0] > 0 else np.asarray([0.0], dtype=float)
    ego_jerk = np.linalg.norm(jerk[0], axis=-1) if xy_h.shape[0] > 0 else np.asarray([0.0], dtype=float)

    if xy_h.shape[0] > 0 and xy_h.shape[1] > 0:
        progress = float(np.linalg.norm(xy_h[0, -1, :2] - xy_h[0, 0, :2]))
    else:
        progress = 0.0

    return {
        f'ego_speed_h{n_h}': float(np.nanmean(ego_vel)) if ego_vel.size > 0 else 0.0,
        f'progress_h{n_h}': progress,
        f'min_ttc_h{n_h}': float(risk['min_ttc']),
        f'min_distance_h{n_h}': float(risk['min_dist']),
        f'max_abs_acc_h{n_h}': float(np.nanmax(ego_acc)) if ego_acc.size > 0 else 0.0,
        f'max_abs_jerk_h{n_h}': float(np.nanmax(ego_jerk)) if ego_jerk.size > 0 else 0.0,
        f'route_deviation_h{n_h}': _safe_float(route_deviation_h, default=0.0),
        f'target_interaction_score': _safe_float(target_interaction_score, default=0.0),
        f'risk_sks_short_h{n_h}': float(risk['risk_sks']),
        f'collision_h{n_h}_proxy': float(risk['collision']),
        f'failure_proxy_h{n_h}_proxy': float(risk['failure_extended_proxy']),
    }


def extract_candidate_risk_features(
    *,
    dist_step: Optional[Dict[str, Any]],
    dist_trace: Optional[Iterable[Optional[Dict[str, Any]]]],
    xy: np.ndarray,
    valid: np.ndarray,
    cfg: Any,
    control_horizon_steps: int,
    target_interaction_score: float = np.nan,
    predictive_seq_kl_nominal: float = np.nan,
    predictive_seq_w2_nominal: float = np.nan,
    route_deviation_h: float = np.nan,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out.update(extract_dist_step_features(dist_step))
    out.update(
        extract_dist_trace_features(
            dist_trace,
            predictive_seq_kl_nominal=predictive_seq_kl_nominal,
            predictive_seq_w2_nominal=predictive_seq_w2_nominal,
        )
    )
    out.update(
        extract_rollout_summary_features(
            xy,
            valid,
            cfg=cfg,
            horizon_steps=control_horizon_steps,
            target_interaction_score=target_interaction_score,
            route_deviation_h=route_deviation_h,
        )
    )
    return out
