from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .config import ClosedLoopConfig


def _softmax_np(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float).reshape(-1)
    if z.size <= 0:
        return np.asarray([1.0], dtype=np.float32)
    z = z - float(np.max(z))
    e = np.exp(z)
    s = float(np.sum(e))
    if (not np.isfinite(s)) or (s <= 0.0):
        return np.ones((z.size,), dtype=np.float32) / float(max(1, z.size))
    return (e / s).astype(np.float32)


def _safe_scalar(x: Any, default: float = 0.0) -> float:
    try:
        a = np.asarray(x)
    except Exception:
        return float(default)
    if a.size <= 0:
        return float(default)
    flat = a.reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size > 0:
        return float(finite[0])
    return float(flat[0])


def _extract_xy_valid_speed_yaw(state: Any) -> Dict[str, np.ndarray]:
    traj = getattr(state, 'current_sim_trajectory', getattr(state, 'log_trajectory', None))
    if traj is None:
        return {
            'xy': np.zeros((0, 2), dtype=np.float32),
            'valid': np.zeros((0,), dtype=bool),
            'speed': np.zeros((0,), dtype=np.float32),
            'yaw': np.zeros((0,), dtype=np.float32),
        }

    xy = np.asarray(getattr(traj, 'xy', np.zeros((0, 2), dtype=np.float32)))
    valid = np.asarray(getattr(traj, 'valid', np.zeros((0,), dtype=bool))).astype(bool)
    speed = np.asarray(getattr(traj, 'speed', np.zeros((0,), dtype=np.float32)))
    yaw = np.asarray(getattr(traj, 'yaw', np.zeros((0,), dtype=np.float32)))

    try:
        if xy.ndim >= 3:
            xy = np.asarray(xy[..., 0, :], dtype=np.float32)
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)
        if xy.ndim > 2:
            xy = xy.reshape(-1, xy.shape[-1])
        if xy.shape[1] < 2:
            xy = np.pad(xy, ((0, 0), (0, 2 - xy.shape[1])), mode='constant')
        xy = xy[:, :2]
    except Exception:
        xy = np.zeros((0, 2), dtype=np.float32)

    try:
        if valid.ndim >= 2:
            valid = np.asarray(valid[..., 0], dtype=bool).reshape(-1)
        else:
            valid = np.asarray(valid, dtype=bool).reshape(-1)
    except Exception:
        valid = np.zeros((xy.shape[0],), dtype=bool)

    def _squeeze_feature(a: np.ndarray) -> np.ndarray:
        out = np.asarray(a)
        while out.ndim > 1 and out.shape[0] == 1:
            out = np.squeeze(out, axis=0)
        if out.ndim >= 2 and out.shape[-1] == 1:
            out = np.squeeze(out, axis=-1)
        return np.asarray(out).reshape(-1)

    speed = _squeeze_feature(speed).astype(np.float32)
    yaw = _squeeze_feature(yaw).astype(np.float32)

    n = int(xy.shape[0])
    if valid.shape[0] < n:
        valid = np.concatenate([valid, np.zeros((n - valid.shape[0],), dtype=bool)], axis=0)
    elif valid.shape[0] > n:
        valid = valid[:n]
    if speed.shape[0] < n:
        speed = np.concatenate([speed, np.zeros((n - speed.shape[0],), dtype=np.float32)], axis=0)
    elif speed.shape[0] > n:
        speed = speed[:n]
    if yaw.shape[0] < n:
        yaw = np.concatenate([yaw, np.zeros((n - yaw.shape[0],), dtype=np.float32)], axis=0)
    elif yaw.shape[0] > n:
        yaw = yaw[:n]

    return {
        'xy': xy.astype(np.float32),
        'valid': valid.astype(bool),
        'speed': speed.astype(np.float32),
        'yaw': yaw.astype(np.float32),
    }


def smart_observation_contract() -> Dict[str, Any]:
    return {
        'distribution_family': 'diag_gmm',
        'distribution_dims': 3,
        'mode': 'proxy',
        'notes': (
            'SMART backend currently exposes a robust proxy predictive distribution in this runtime. '
            'Use strict mode only when full SMART model wiring is available.'
        ),
    }


class SmartPredictiveAdapter:
    def __init__(self, cfg: ClosedLoopConfig) -> None:
        self.cfg = cfg
        self.last_obs_info: Dict[str, Any] = {}
        self.last_route: str = 'proxy'
        self.last_error: str = ''

    def _proxy_distribution(
        self,
        state: Any,
        sdc_idx: int,
        action_hint: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        obs = _extract_xy_valid_speed_yaw(state)
        xy = np.asarray(obs['xy'], dtype=float)
        valid = np.asarray(obs['valid'], dtype=bool)
        speed = np.asarray(obs['speed'], dtype=float)
        yaw = np.asarray(obs['yaw'], dtype=float)

        n_obj = int(xy.shape[0])
        if (n_obj <= 0) or (sdc_idx < 0) or (sdc_idx >= n_obj):
            self.last_obs_info = {
                'n_obj': int(n_obj),
                'valid_obj': int(np.sum(valid.astype(int))),
                'finite': False,
                'reason': 'invalid_sdc_or_empty_obs',
            }
            return {
                'weights': np.asarray([1.0], dtype=np.float32),
                'means': np.zeros((1, 3), dtype=np.float32),
                'stds': np.asarray([[0.6, 0.6, 0.2]], dtype=np.float32),
                'fallback': np.asarray(1, dtype=np.int32),
                'source': 'fallback:smart_proxy_empty_obs',
                'actor_fallback': np.asarray(0, dtype=np.int32),
            }

        ego_xy = np.asarray(xy[int(sdc_idx), :2], dtype=float)
        ego_speed = float(speed[int(sdc_idx)]) if int(sdc_idx) < speed.shape[0] else 0.0
        ego_yaw = float(yaw[int(sdc_idx)]) if int(sdc_idx) < yaw.shape[0] else 0.0
        ego_fwd = np.asarray([np.cos(ego_yaw), np.sin(ego_yaw)], dtype=float)
        ego_left = np.asarray([-ego_fwd[1], ego_fwd[0]], dtype=float)

        rel = xy - ego_xy[None, :]
        rel_dist = np.linalg.norm(rel, axis=1)
        non_ego = np.where((np.arange(n_obj) != int(sdc_idx)) & valid)[0]
        nearest_idx = int(non_ego[np.argmin(rel_dist[non_ego])]) if non_ego.size > 0 else -1
        nearest_dist = float(rel_dist[nearest_idx]) if nearest_idx >= 0 else float('inf')
        dist_scale = float(max(1.0, self.cfg.smart_interaction_dist_scale_m))
        interaction = float(np.exp(-nearest_dist / dist_scale)) if np.isfinite(nearest_dist) else 0.0

        closing = 0.0
        heading_conflict = 0.0
        if nearest_idx >= 0:
            nbr_speed = float(speed[nearest_idx]) if nearest_idx < speed.shape[0] else 0.0
            nbr_yaw = float(yaw[nearest_idx]) if nearest_idx < yaw.shape[0] else 0.0
            nbr_vel = np.asarray([np.cos(nbr_yaw), np.sin(nbr_yaw)], dtype=float) * nbr_speed
            ego_vel = ego_fwd * ego_speed
            rel_vec = np.asarray(rel[nearest_idx], dtype=float)
            rel_norm = float(np.linalg.norm(rel_vec))
            rel_hat = rel_vec / rel_norm if rel_norm > 1e-6 else np.asarray([1.0, 0.0], dtype=float)
            closing = float(max(0.0, -np.dot(rel_hat, nbr_vel - ego_vel)))
            yaw_diff = float(np.arctan2(np.sin(nbr_yaw - ego_yaw), np.cos(nbr_yaw - ego_yaw)))
            heading_conflict = float(abs(np.sin(yaw_diff)))

        closing_scale = float(max(0.5, self.cfg.smart_interaction_closing_speed_scale_mps))
        closing_norm = float(np.clip(closing / closing_scale, 0.0, 1.5))

        dt = float(max(1e-3, self.cfg.smart_action_dt_seconds))
        forward_step = float(np.clip(max(ego_speed, 0.0) * dt, 0.2, 2.0))
        lateral_step = float(0.18 + 0.65 * interaction)
        yaw_step = float(0.05 + 0.22 * interaction)

        means = np.zeros((4, 3), dtype=np.float32)
        # continue
        means[0, :2] = (forward_step * ego_fwd).astype(np.float32)
        means[0, 2] = 0.0
        # brake / cautious
        means[1, :2] = (0.45 * forward_step * ego_fwd).astype(np.float32)
        means[1, 2] = 0.0
        # slight left
        means[2, :2] = (forward_step * ego_fwd + lateral_step * ego_left).astype(np.float32)
        means[2, 2] = float(yaw_step)
        # slight right
        means[3, :2] = (forward_step * ego_fwd - lateral_step * ego_left).astype(np.float32)
        means[3, 2] = float(-yaw_step)

        logits = np.asarray([
            1.0 - 1.15 * interaction,                                 # continue
            -0.2 + 1.00 * interaction + 0.65 * closing_norm,         # cautious/brake
            -0.1 + 0.85 * interaction * (0.5 + 0.5 * heading_conflict),  # left
            -0.1 + 0.85 * interaction * (0.5 + 0.5 * heading_conflict),  # right
        ], dtype=float)

        # If an action hint is available from the control actor, align weights
        # to the nearest component without collapsing diversity.
        if action_hint is not None:
            a = np.asarray(action_hint, dtype=float).reshape(-1)
            if a.size > 0:
                ah = np.zeros((3,), dtype=float)
                ah[: int(min(3, a.size))] = a[: int(min(3, a.size))]
                d2 = np.sum((means.astype(float) - ah[None, :]) ** 2, axis=1)
                logits += np.exp(-d2 / (2.0 * 0.35 ** 2))

        weights = _softmax_np(logits)
        std_xy = float(max(1e-3, self.cfg.smart_base_std_xy)) * (1.0 + 0.45 * interaction)
        std_yaw = float(max(1e-3, self.cfg.smart_base_std_yaw)) * (1.0 + 0.30 * interaction)
        stds = np.asarray(
            [[std_xy, std_xy, std_yaw]] * int(weights.shape[0]),
            dtype=np.float32,
        )

        self.last_obs_info = {
            'n_obj': int(n_obj),
            'valid_obj': int(np.sum(valid.astype(int))),
            'nearest_dist': float(nearest_dist) if np.isfinite(nearest_dist) else np.nan,
            'interaction': float(interaction),
            'closing_speed': float(closing),
            'heading_conflict': float(heading_conflict),
            'finite': bool(np.isfinite(means).all() and np.isfinite(stds).all() and np.isfinite(weights).all()),
        }
        return {
            'weights': np.asarray(weights, dtype=np.float32),
            'means': np.asarray(means, dtype=np.float32),
            'stds': np.asarray(stds, dtype=np.float32),
            'fallback': np.asarray(0, dtype=np.int32),
            'source': 'model:smart_proxy',
            'actor_fallback': np.asarray(0, dtype=np.int32),
        }

    def predict_distribution(
        self,
        state: Any,
        sdc_idx: int,
        action_hint: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        mode = str(getattr(self.cfg, 'smart_mode', 'proxy')).strip().lower()
        try:
            if mode == 'strict':
                # Strict mode is reserved for full SMART model wiring.
                # Keep an explicit diagnostic instead of silently falling back.
                raise RuntimeError(
                    'smart_mode=strict requires full SMART model runtime wiring, '
                    'which is not enabled in this notebook runtime. '
                    'Set cfg.smart_mode="proxy" to run SMART-style predictive distributions now.'
                )

            self.last_route = 'proxy'
            self.last_error = ''
            return self._proxy_distribution(state=state, sdc_idx=int(sdc_idx), action_hint=action_hint)
        except Exception as e:
            self.last_route = 'failed'
            self.last_error = str(e)
            return {
                'weights': np.asarray([1.0], dtype=np.float32),
                'means': np.zeros((1, 3), dtype=np.float32),
                'stds': np.asarray([[0.6, 0.6, 0.2]], dtype=np.float32),
                'fallback': np.asarray(1, dtype=np.int32),
                'source': 'fallback:smart_proxy_error',
                'actor_fallback': np.asarray(0, dtype=np.int32),
            }


_SMART_ADAPTER: Optional[SmartPredictiveAdapter] = None


def get_smart_adapter(cfg: ClosedLoopConfig) -> SmartPredictiveAdapter:
    global _SMART_ADAPTER
    if (_SMART_ADAPTER is None) or (_SMART_ADAPTER.cfg is not cfg):
        _SMART_ADAPTER = SmartPredictiveAdapter(cfg)
    return _SMART_ADAPTER
