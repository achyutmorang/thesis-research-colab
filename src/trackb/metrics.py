from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

def compute_finite_differences(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vel = np.diff(xy, axis=1, prepend=xy[:, :1, :])
    acc = np.diff(vel, axis=1, prepend=vel[:, :1, :])
    jerk = np.diff(acc, axis=1, prepend=acc[:, :1, :])
    return vel, acc, jerk

def min_pairwise_distance(xy: np.ndarray, valid: np.ndarray) -> float:
    best = np.inf
    T = xy.shape[1]
    for t in range(T):
        pts = xy[:, t, :][valid[:, t]]
        if pts.shape[0] < 2:
            continue
        dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        dmat += np.eye(dmat.shape[0]) * 1e6
        best = min(best, float(dmat.min()))
    return float(best)

def min_ttc(xy: np.ndarray, vel: np.ndarray, valid: np.ndarray, collision_radius: float = 1.5) -> float:
    best = np.inf
    T = xy.shape[1]
    for t in range(T):
        pts = xy[:, t, :]
        vels = vel[:, t, :]
        idx = np.flatnonzero(valid[:, t])
        for ii in range(len(idx)):
            i = idx[ii]
            for jj in range(ii + 1, len(idx)):
                j = idx[jj]
                r = pts[i] - pts[j]
                v = vels[i] - vels[j]
                vv = float(v @ v)
                if vv <= 1e-12:
                    continue
                tau = -float(r @ v) / vv
                if tau <= 0:
                    continue
                d_tau = np.linalg.norm(r + tau * v)
                if d_tau < collision_radius:
                    best = min(best, tau)
    return float(best)

def compute_risk_metrics(
    xy: np.ndarray,
    valid: np.ndarray,
    collision_distance: float,
    w_dist: float,
    w_ttc: float,
    w_sks: float,
    ttc_fail_seconds: float = 2.0,
    no_hazard_ttc_seconds: float = 3.0,
    no_hazard_dist_m: float = 8.0,
    hard_brake_mps2: float = 6.0,
    hard_jerk_mps3: float = 8.0,
    enable_intervention_proxy: bool = True,
) -> Dict[str, float]:
    vel, acc, jerk = compute_finite_differences(xy)
    min_dist = min_pairwise_distance(xy, valid)
    ttc = min_ttc(xy, vel, valid, collision_radius=collision_distance)

    vel_n = np.linalg.norm(vel, axis=-1)
    acc_n = np.linalg.norm(acc, axis=-1)
    jerk_n = np.linalg.norm(jerk, axis=-1)
    valid_f = valid.astype(float)

    sks = float(np.sum((0.5 * vel_n + 0.3 * acc_n + 0.2 * jerk_n) * valid_f))
    denom = float(max(valid_f.sum(), 1.0))

    ttc_safe = 30.0 if (not np.isfinite(ttc)) else max(ttc, 1e-3)
    dist_term = 1.0 / max(min_dist, 1e-3)
    ttc_term = 1.0 / ttc_safe
    sks_term = sks / denom

    risk_sks = float(w_dist * dist_term + w_ttc * ttc_term + w_sks * sks_term)

    collision = float(min_dist < collision_distance)
    ttc_fail_proxy = float(np.isfinite(ttc) and (ttc < ttc_fail_seconds))

    max_acc = float(np.max(acc_n[valid])) if np.any(valid) else 0.0
    max_jerk = float(np.max(jerk_n[valid])) if np.any(valid) else 0.0
    no_hazard = ((not np.isfinite(ttc)) or (ttc >= no_hazard_ttc_seconds)) and (min_dist >= no_hazard_dist_m)
    unsafe_intervention_proxy = float(
        bool(enable_intervention_proxy)
        and no_hazard
        and (max_acc >= hard_brake_mps2)
        and (max_jerk >= hard_jerk_mps3)
    )

    failure_strict_proxy = float(collision > 0.0)
    failure_extended_proxy = float((collision > 0.0) or (ttc_fail_proxy > 0.0) or (unsafe_intervention_proxy > 0.0))

    return {
        'min_dist': float(min_dist),
        'min_ttc': float(ttc),
        'sks': float(sks),
        'risk_sks': float(risk_sks),
        'collision': collision,
        'ttc_fail_proxy': ttc_fail_proxy,
        'unsafe_intervention_proxy': unsafe_intervention_proxy,
        'failure_strict_proxy': failure_strict_proxy,
        'failure_extended_proxy': failure_extended_proxy,
        'max_acc': max_acc,
        'max_jerk': max_jerk,
    }

def risk_kwargs_from_cfg(cfg: Any) -> Dict[str, Any]:
    return {
        'collision_distance': cfg.collision_distance,
        'w_dist': cfg.risk_w_dist,
        'w_ttc': cfg.risk_w_ttc,
        'w_sks': cfg.risk_w_sks,
        'ttc_fail_seconds': cfg.ttc_fail_seconds,
        'no_hazard_ttc_seconds': cfg.no_hazard_ttc_seconds,
        'no_hazard_dist_m': cfg.no_hazard_dist_m,
        'hard_brake_mps2': cfg.hard_brake_mps2,
        'hard_jerk_mps3': cfg.hard_jerk_mps3,
        'enable_intervention_proxy': cfg.enable_intervention_proxy,
    }

def robust_scale(values: np.ndarray, min_scale: float = 1e-6) -> float:
    x = np.asarray(values, dtype=float)
    q75, q25 = np.quantile(x, [0.75, 0.25])
    iqr = float(q75 - q25)
    std = float(np.std(x, ddof=0))
    scale = iqr if iqr > min_scale else std
    return max(scale, min_scale)

def planner_action_surprise_kl(
    actions: np.ndarray,
    action_valid: np.ndarray,
    base_actions: np.ndarray,
    base_action_valid: np.ndarray,
    sigma: float,
) -> float:
    sigma2 = float(max(sigma, 1e-4) ** 2)

    t = int(min(actions.shape[0], base_actions.shape[0]))
    a = int(min(actions.shape[1], base_actions.shape[1]))
    if t == 0 or a == 0:
        return 0.0

    cur = np.asarray(actions[:t, :a], dtype=float)
    ref = np.asarray(base_actions[:t, :a], dtype=float)
    vcur = np.asarray(action_valid[:t], dtype=bool)
    vref = np.asarray(base_action_valid[:t], dtype=bool)
    mask = (vcur & vref).astype(float)[:, None]

    denom = float(mask.sum() * a)
    if denom <= 0.0:
        return 0.0

    diff2 = (cur - ref) ** 2
    kl = 0.5 * float(np.sum((diff2 / sigma2) * mask) / denom)
    return float(max(0.0, kl))
