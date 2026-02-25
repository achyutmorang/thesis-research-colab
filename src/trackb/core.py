from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm.auto import tqdm

from waymax import agents as waymax_agents
from waymax import config as waymax_config
from waymax import dataloader as waymax_dataloader
from waymax import datatypes as waymax_datatypes
from waymax import dynamics as waymax_dynamics
from waymax import env as waymax_env

from .config import (
    SearchConfig,
    TrackBConfig,
    build_run_artifact_paths,
    required_total_scenarios,
    restore_artifacts_via_upload,
)

# JAX compatibility shim for libraries still calling removed top-level tree APIs.
if not hasattr(jax, 'tree_map'):
    jax.tree_map = jax.tree_util.tree_map
if not hasattr(jax, 'tree_leaves'):
    jax.tree_leaves = jax.tree_util.tree_leaves
if not hasattr(jax, 'tree_flatten'):
    jax.tree_flatten = jax.tree_util.tree_flatten
if not hasattr(jax, 'tree_unflatten'):
    jax.tree_unflatten = jax.tree_util.tree_unflatten
if not hasattr(jax, 'tree_structure'):
    jax.tree_structure = jax.tree_util.tree_structure

sns.set_theme(style='whitegrid')
# Colab GCS authentication helper for WOMD.
# This is required when reading gs://waymo_open_dataset_motion_v_1_1_0/... paths.

import subprocess


def ensure_womd_gcs_access(gcs_path: str) -> None:
    if not str(gcs_path).startswith('gs://'):
        print('[auth] Non-GCS path detected; skipping GCS authentication checks.')
        return

    bucket = 'gs://waymo_open_dataset_motion_v_1_1_0'
    probe = bucket + '/uncompressed/tf_example/training/'

    def _probe() -> bool:
        try:
            res = subprocess.run(
                ['gsutil', 'ls', probe],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            return res.returncode == 0
        except Exception:
            return False

    if _probe():
        print('[auth] GCS access already available for WOMD bucket.')
        return

    print('[auth] GCS access missing. Starting Colab auth flow...')
    try:
        from google.colab import auth
        auth.authenticate_user()
    except Exception as e:
        raise RuntimeError(
            'Colab authentication step failed. Ensure this is running in Colab and retry.'
        ) from e

    # Acquire ADC token (required by TF/Waymax GCS reads).
    adc = subprocess.run(
        ['gcloud', 'auth', 'application-default', 'print-access-token'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if adc.returncode != 0:
        raise RuntimeError(
            'Failed to acquire application-default credentials. '
            'Run: !gcloud auth application-default login and retry.'
        )

    if not _probe():
        raise RuntimeError(
            'GCS access still denied for WOMD bucket after authentication. '
            'Use an account with WOMD access (storage.objects.get) and accepted terms.'
        )

    print('[auth] GCS authentication successful for WOMD bucket.')


class WaymaxScenarioLoader:
    def __init__(
        self,
        config: TrackBConfig,
        data_iter: Optional[Iterable[Any]] = None,
        dataset_config: Optional[Any] = None,
    ):
        self.cfg = config
        self.dataset_config = dataset_config or waymax_config.DatasetConfig(
            path=self.cfg.waymax_path,
            data_format=waymax_config.DataFormat.TFRECORD,
            max_num_rg_points=self.cfg.waymax_max_rg_points,
            batch_dims=self.cfg.waymax_batch_dims,
        )
        self.data_iter = data_iter or waymax_dataloader.simulator_state_generator(self.dataset_config)

    @staticmethod
    def _field(state: Any, key: str) -> Any:
        if isinstance(state, dict):
            return state[key]
        return getattr(state, key)

    def _select_agents(self, xy: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid_count = valid.sum(axis=1)
        keep_idx = np.argsort(valid_count)[::-1][: self.cfg.n_agents]
        return xy[keep_idx], valid[keep_idx], keep_idx

    def _single(self, scenario_id: int, keep_state: bool = False) -> Dict[str, Any]:
        state = next(self.data_iter)

        log_traj = self._field(state, 'log_trajectory')
        xy_all = np.asarray(log_traj.xy)
        valid_all = np.asarray(log_traj.valid).astype(bool)

        obj_meta = getattr(state, 'object_metadata', None) if not isinstance(state, dict) else state.get('object_metadata', None)
        if obj_meta is not None and hasattr(obj_meta, 'is_sdc'):
            is_sdc_all = np.asarray(obj_meta.is_sdc).astype(bool)
        else:
            is_sdc_all = np.zeros(xy_all.shape[0], dtype=bool)

        if xy_all.ndim != 3:
            raise ValueError(f'Unexpected trajectory shape: {xy_all.shape}')

        xy, valid, keep_idx = self._select_agents(xy_all, valid_all)
        selected_is_sdc = is_sdc_all[keep_idx]

        required_steps = self.cfg.history_steps + self.cfg.future_steps
        if xy.shape[1] < required_steps:
            pad = required_steps - xy.shape[1]
            xy = np.pad(xy, ((0, 0), (0, pad), (0, 0)), mode='edge')
            valid = np.pad(valid, ((0, 0), (0, pad)), mode='constant', constant_values=False)
        else:
            xy = xy[:, :required_steps, :]
            valid = valid[:, :required_steps]

        H, F = self.cfg.history_steps, self.cfg.future_steps
        history = np.where(valid[:, :H, None], xy[:, :H, :], 0.0)
        future = np.where(valid[:, H:H+F, None], xy[:, H:H+F, :], 0.0)

        perturb_mask = valid[:, H:H+F] & (~selected_is_sdc[:, None])

        rec = {
            'scenario_id': int(scenario_id),
            'history': history.astype(np.float32),
            'future': future.astype(np.float32),
            'full': xy.astype(np.float32),
            'valid': valid.astype(bool),
            'selected_indices': keep_idx.astype(int),
            'is_sdc': selected_is_sdc.astype(bool),
            'perturb_mask': perturb_mask.astype(bool),
        }
        if keep_state and self.cfg.keep_raw_state:
            rec['state'] = state
        return rec

    def generate(
        self,
        n_scenarios: int,
        keep_state_ids: Optional[set] = None,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        keep_state_ids = keep_state_ids or set()

        iterator = range(n_scenarios)
        if show_progress:
            iterator = tqdm(iterator, desc='Loading WOMD scenarios', total=n_scenarios)

        for sid in iterator:
            try:
                keep_state = sid in keep_state_ids
                rows.append(self._single(sid, keep_state=keep_state))
                if sid > 0 and (sid % 500 == 0):
                    import gc
                    gc.collect()
            except StopIteration:
                print(f'Iterator exhausted at {sid} scenarios.')
                break
        return rows


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


class TrackBRunner:
    def __init__(
        self,
        config: TrackBConfig,
        data_iter: Optional[Iterable[Any]] = None,
        dataset_config: Optional[Any] = None,
    ):
        self.cfg = config
        self.loader = WaymaxScenarioLoader(config, data_iter=data_iter, dataset_config=dataset_config)

    def build_dataset(self):
        n_train = int(self.cfg.n_total_scenarios * self.cfg.train_fraction)
        n_eval_target = int(max(self.cfg.n_eval_scenarios, self.cfg.strict_min_eval))
        eval_start = n_train
        eval_end = min(self.cfg.n_total_scenarios, eval_start + n_eval_target)
        keep_state_ids = set(range(eval_start, eval_end))

        scenarios = self.loader.generate(
            self.cfg.n_total_scenarios,
            keep_state_ids=keep_state_ids,
            show_progress=True,
        )

        X, meta = [], []
        for rec in tqdm(scenarios, desc='Building feature table'):
            history = rec['history'].reshape(-1)
            risk = compute_risk_metrics(rec['full'], rec['valid'], **risk_kwargs_from_cfg(self.cfg))

            X.append(history)
            meta.append({'scenario_id': rec['scenario_id'], **risk})

        X = np.asarray(X, dtype=float)
        meta_df = pd.DataFrame(meta)

        n_train = int(len(X) * self.cfg.train_fraction)
        train_idx = np.arange(n_train)
        test_idx = np.arange(n_train, len(X))

        self.data = {
            'X': X,
            'meta': meta_df,
            'scenarios': scenarios,
            'train_idx': train_idx,
            'test_idx': test_idx,
        }
        return self.data

    def score_indices_openloop(self, idx: np.ndarray, label: str, show_progress: bool = True) -> pd.DataFrame:
        rows = []
        iterator = idx
        if show_progress:
            iterator = tqdm(idx, desc=f'Scoring open-loop {label}', total=len(idx))

        for sid in iterator:
            sid_int = int(sid)
            rec = self.data['scenarios'][sid_int]

            risk = compute_risk_metrics(rec['full'], rec['valid'], **risk_kwargs_from_cfg(self.cfg))

            rows.append({
                'scenario_id': sid_int,
                'split': label,
                'planner': self.cfg.planner_name,
                'risk_sks': float(risk['risk_sks']),
                'failure_proxy': float(risk['failure_extended_proxy']),
                'min_dist': float(risk['min_dist']),
                'min_ttc': float(risk['min_ttc']),
            })

        return pd.DataFrame(rows)


def resolve_env_class():
    if hasattr(waymax_env, 'BaseEnvironment'):
        return waymax_env.BaseEnvironment
    if hasattr(waymax_env, 'MultiAgentEnvironment'):
        return waymax_env.MultiAgentEnvironment
    from waymax.env import base_environment as _base_env
    if hasattr(_base_env, 'BaseEnvironment'):
        return _base_env.BaseEnvironment
    return _base_env.MultiAgentEnvironment


ENV_CLASS = resolve_env_class()


def squeeze_xy_valid_from_current(traj: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
    xy = np.asarray(traj.xy)
    valid = np.asarray(traj.valid)

    # Remove leading singleton batch dims if present.
    while xy.ndim > 3 and xy.shape[0] == 1:
        xy = np.squeeze(xy, axis=0)
    while valid.ndim > 2 and valid.shape[0] == 1:
        valid = np.squeeze(valid, axis=0)

    # Remove common singleton dims around time/channel.
    if xy.ndim >= 3 and xy.shape[-2] == 1:
        xy = np.squeeze(xy, axis=-2)
    if valid.ndim >= 2 and valid.shape[-1] == 1:
        valid = np.squeeze(valid, axis=-1)

    # Canonicalize to [num_obj, 2].
    if xy.ndim == 1:
        if xy.size % 2 == 0:
            xy = xy.reshape(-1, 2)
        else:
            xy = np.pad(xy.reshape(-1, 1), ((0, 0), (0, 1)), mode='constant')
    elif xy.ndim >= 3 and xy.shape[-1] >= 2:
        xy = xy.reshape(xy.shape[0], -1, xy.shape[-1])[:, 0, :2]
    elif xy.ndim == 2 and xy.shape[1] < 2:
        xy = np.pad(xy, ((0, 0), (0, 2 - xy.shape[1])), mode='constant')
    elif xy.ndim != 2:
        xy = np.zeros((0, 2), dtype=np.float32)

    # Canonicalize valid to [num_obj].
    if valid.ndim == 0:
        valid = np.asarray([bool(valid)], dtype=bool)
    elif valid.ndim >= 2:
        valid = valid.reshape(valid.shape[0], -1)[:, 0]
    valid = valid.astype(bool).reshape(-1)

    n_obj = int(xy.shape[0]) if xy.ndim == 2 else 0
    if valid.shape[0] < n_obj:
        pad = np.zeros((n_obj - valid.shape[0],), dtype=bool)
        valid = np.concatenate([valid, pad], axis=0)
    elif valid.shape[0] > n_obj:
        valid = valid[:n_obj]

    return jnp.asarray(xy, dtype=jnp.float32), jnp.asarray(valid, dtype=jnp.bool_)


def _squeeze_feature(arr: Any) -> np.ndarray:
    a = np.asarray(arr)
    while a.ndim > 1 and a.shape[0] == 1:
        a = np.squeeze(a, axis=0)
    if a.ndim >= 2 and a.shape[-1] == 1:
        a = np.squeeze(a, axis=-1)
    return a


def _safe_scalar(x: Any, default: float = 0.0) -> float:
    a = np.asarray(x)
    if a.size == 0:
        return float(default)
    flat = a.reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size > 0:
        return float(finite[0])
    return float(flat[0])


def _feature_value_for_object(arr: Any, obj_idx: int, default: float = 0.0) -> float:
    a = _squeeze_feature(arr)
    if a.size == 0:
        return float(default)
    try:
        obj = a[int(obj_idx)]
    except Exception:
        return _safe_scalar(a, default)
    return _safe_scalar(obj, default)


def _xy_for_object(xy: Any, obj_idx: int) -> Tuple[float, float]:
    a = np.asarray(xy)
    if a.size == 0:
        return 0.0, 0.0

    while a.ndim > 2 and a.shape[0] == 1:
        a = np.squeeze(a, axis=0)

    if a.ndim == 2 and a.shape[1] >= 2 and obj_idx < a.shape[0]:
        return _safe_scalar(a[obj_idx, 0]), _safe_scalar(a[obj_idx, 1])

    if a.ndim >= 3 and a.shape[-1] >= 2:
        try:
            obj = a[int(obj_idx)]
        except Exception:
            obj = a
        obj2 = np.asarray(obj).reshape(-1, a.shape[-1])
        return _safe_scalar(obj2[0, 0]), _safe_scalar(obj2[0, 1])

    flat = a.reshape(-1)
    if flat.size >= 2:
        return _safe_scalar(flat[0]), _safe_scalar(flat[1])
    return _safe_scalar(flat, 0.0), 0.0


def project_delta_vec(delta: np.ndarray, clip: float, l2_budget: float) -> np.ndarray:
    d = np.clip(np.asarray(delta, dtype=float), -clip, clip)
    norm = float(np.linalg.norm(d))
    if norm > l2_budget and norm > 1e-12:
        d = d * (l2_budget / norm)
    return d


def _choose_target_non_ego(base_state: Any, selected_idx: np.ndarray) -> int:
    is_sdc_all = np.asarray(base_state.object_metadata.is_sdc).astype(bool)
    valid_raw = np.asarray(base_state.log_trajectory.valid)
    while valid_raw.ndim > 1 and valid_raw.shape[0] == 1:
        valid_raw = np.squeeze(valid_raw, axis=0)
    while valid_raw.ndim > 1 and valid_raw.shape[-1] == 1:
        valid_raw = np.squeeze(valid_raw, axis=-1)
    if valid_raw.ndim == 1:
        valid_t0_all = valid_raw.astype(bool)
    else:
        valid_t0_all = np.asarray(valid_raw[:, 0]).astype(bool).reshape(-1)
    candidates = [int(i) for i in selected_idx.tolist() if (not is_sdc_all[i]) and valid_t0_all[i]]
    if len(candidates) == 0:
        raise ValueError('No valid non-ego object found for perturbation.')
    return int(candidates[0])


def _replace_obj(obj: Any, **kwargs: Any) -> Any:
    try:
        return dataclasses.replace(obj, **kwargs)
    except Exception:
        pass
    if hasattr(obj, 'replace'):
        try:
            return obj.replace(**kwargs)
        except Exception:
            pass
    if hasattr(obj, '_replace'):
        try:
            return obj._replace(**kwargs)
        except Exception:
            pass
    raise RuntimeError(f'Unable to replace object fields: {list(kwargs.keys())}')


def _shift_traj_t0_xy(traj: Any, target_obj_idx: int, delta_xy_j: jnp.ndarray) -> Any:
    # Waymax trajectory schemas vary across versions.
    # In many versions, xy is a derived property while canonical fields are x/y.
    if hasattr(traj, 'x') and hasattr(traj, 'y'):
        try:
            x_new = traj.x.at[target_obj_idx, 0].add(delta_xy_j[0])
            y_new = traj.y.at[target_obj_idx, 0].add(delta_xy_j[1])
            return _replace_obj(traj, x=x_new, y=y_new)
        except Exception:
            pass

    if hasattr(traj, 'xy'):
        try:
            xy_new = traj.xy.at[target_obj_idx, 0, :].add(delta_xy_j)
            return _replace_obj(traj, xy=xy_new)
        except Exception:
            pass

    raise RuntimeError(
        'Unable to apply perturbation: trajectory replacement with (x,y) and xy both failed.'
    )


def perturb_initial_state(base_state: Any, target_obj_idx: int, delta_xy: np.ndarray) -> Any:
    delta_xy_j = jnp.asarray(delta_xy, dtype=jnp.float32)
    log_traj_new = _shift_traj_t0_xy(base_state.log_trajectory, target_obj_idx, delta_xy_j)
    sim_traj_new = _shift_traj_t0_xy(base_state.sim_trajectory, target_obj_idx, delta_xy_j)
    return _replace_obj(base_state, log_trajectory=log_traj_new, sim_trajectory=sim_traj_new)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def _replace_action(output: Any, new_action: Any) -> Any:
    try:
        return dataclasses.replace(output, action=new_action)
    except Exception:
        pass
    if hasattr(output, '_replace'):
        try:
            return output._replace(action=new_action)
        except Exception:
            pass
    try:
        output.action = new_action
        return output
    except Exception:
        raise RuntimeError('Unable to replace actor action output.')


def _latentdriver_fallback_dist(action_dim: int = 3) -> Dict[str, np.ndarray]:
    d = int(max(1, action_dim))
    return {
        'weights': np.asarray([1.0], dtype=np.float32),
        'means': np.zeros((1, d), dtype=np.float32),
        'stds': np.ones((1, d), dtype=np.float32) * 0.5,
        'fallback': np.asarray(1, dtype=np.int32),
        'source': 'fallback',
    }


def latentdriver_observation_contract() -> Dict[str, Any]:
    return {
        'feature_dim': 7,
        'feature_order': ['type_id', 'x', 'y', 'width', 'length', 'yaw_deg', 'speed_mps'],
        'expected_tensor_rank_for_model': 4,  # [batch, time, tokens, feature_dim]
        'notes': 'Full LatentDriver pipeline usually concatenates route, vehicle, and roadgraph tokens.',
    }


def _moment_match_diag_gmm(dist: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    w = np.asarray(dist['weights'], dtype=float)
    w = np.maximum(w, 1e-8)
    w = w / w.sum()

    means = np.asarray(dist['means'], dtype=float)
    stds = np.asarray(dist['stds'], dtype=float)

    mu = np.sum(w[:, None] * means, axis=0)
    d = means.shape[1]
    cov = np.zeros((d, d), dtype=float)

    for k in range(means.shape[0]):
        diff = (means[k] - mu).reshape(-1, 1)
        diag_cov = np.diag(np.maximum(stds[k], 1e-4) ** 2)
        cov += w[k] * (diag_cov + diff @ diff.T)

    cov += 1e-6 * np.eye(d)
    return mu, cov


def _gaussian_kl(mu_p: np.ndarray, cov_p: np.ndarray, mu_q: np.ndarray, cov_q: np.ndarray) -> float:
    d = mu_p.shape[0]
    inv_q = np.linalg.inv(cov_q)
    diff = (mu_q - mu_p).reshape(-1, 1)
    term_trace = float(np.trace(inv_q @ cov_p))
    term_quad = float(diff.T @ inv_q @ diff)
    sign_p, logdet_p = np.linalg.slogdet(cov_p)
    sign_q, logdet_q = np.linalg.slogdet(cov_q)
    if sign_p <= 0 or sign_q <= 0:
        return 0.0
    kl = 0.5 * (term_trace + term_quad - d + (logdet_q - logdet_p))
    return float(max(0.0, kl))


def _sanitize_diag_gmm(dist: Dict[str, np.ndarray], eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.asarray(dist['weights'], dtype=float).reshape(-1)
    means = np.asarray(dist['means'], dtype=float)
    stds = np.asarray(dist['stds'], dtype=float)

    if means.ndim != 2:
        means = means.reshape(means.shape[0], -1)
    if stds.ndim != 2:
        stds = stds.reshape(stds.shape[0], -1)

    k = min(w.shape[0], means.shape[0], stds.shape[0])
    if k <= 0:
        return np.asarray([1.0], dtype=float), np.zeros((1, 1), dtype=float), np.ones((1, 1), dtype=float)

    w = w[:k]
    means = means[:k]
    stds = stds[:k]

    w = np.maximum(w, eps)
    w = w / np.sum(w)
    stds = np.maximum(stds, eps)
    return w, means, stds


def _logsumexp_np(a: np.ndarray, axis: int = -1) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _log_prob_diag_gmm(samples: np.ndarray, dist: Dict[str, np.ndarray], eps: float = 1e-8) -> np.ndarray:
    w, means, stds = _sanitize_diag_gmm(dist, eps=max(eps, 1e-8))
    x = np.asarray(samples, dtype=float)
    if x.ndim == 1:
        x = x[None, :]

    d = means.shape[1]
    if x.shape[1] != d:
        if x.shape[1] < d:
            pad = np.zeros((x.shape[0], d - x.shape[1]), dtype=float)
            x = np.concatenate([x, pad], axis=1)
        else:
            x = x[:, :d]

    log_w = np.log(np.maximum(w, eps))[None, :]
    diff = x[:, None, :] - means[None, :, :]
    var = np.maximum(stds, eps) ** 2
    log_det = np.sum(np.log(2.0 * np.pi * var), axis=1)[None, :]
    quad = np.sum((diff ** 2) / var[None, :, :], axis=2)
    comp_log = log_w - 0.5 * (log_det + quad)
    return _logsumexp_np(comp_log, axis=1)


def _sample_from_diag_gmm(dist: Dict[str, np.ndarray], n: int, rng: np.random.Generator, eps: float = 1e-8) -> np.ndarray:
    w, means, stds = _sanitize_diag_gmm(dist, eps=max(eps, 1e-8))
    n = int(max(1, n))
    comp_idx = rng.choice(len(w), size=n, p=w)
    loc = means[comp_idx]
    scale = stds[comp_idx]
    z = rng.normal(size=loc.shape)
    return loc + scale * z


def _mc_kl_diag_gmm(
    dist_p: Dict[str, np.ndarray],
    dist_q: Dict[str, np.ndarray],
    n_samples: int,
    rng: np.random.Generator,
    eps: float = 1e-8,
) -> float:
    samples = _sample_from_diag_gmm(dist_p, n=int(max(1, n_samples)), rng=rng, eps=eps)
    log_p = _log_prob_diag_gmm(samples, dist_p, eps=eps)
    log_q = _log_prob_diag_gmm(samples, dist_q, eps=eps)
    val = float(np.mean(log_p - log_q))
    return float(max(0.0, val)) if np.isfinite(val) else 0.0


def _trace_step_is_fallback(dist: Optional[Dict[str, np.ndarray]]) -> bool:
    if dist is None:
        return True
    try:
        f0 = float(np.asarray(dist.get('fallback', 0)).reshape(-1)[0])
    except Exception:
        f0 = 0.0
    try:
        f1 = float(np.asarray(dist.get('actor_fallback', 0)).reshape(-1)[0])
    except Exception:
        f1 = 0.0
    return bool((f0 > 0.5) or (f1 > 0.5))


def predictive_kl_from_dist_traces(
    trace_p: List[Optional[Dict[str, np.ndarray]]],
    trace_q: List[Optional[Dict[str, np.ndarray]]],
    estimator: str = 'mixture_mc',
    n_mc_samples: int = 96,
    seed: int = 12345,
    eps: float = 1e-8,
    symmetric: bool = True,
    skip_fallback_steps: bool = True,
) -> float:
    vals: List[float] = []
    n = min(len(trace_p), len(trace_q))
    base_rng = np.random.default_rng(int(seed))
    for i in range(n):
        dp = trace_p[i]
        dq = trace_q[i]
        if dp is None or dq is None:
            continue
        if bool(skip_fallback_steps) and (_trace_step_is_fallback(dp) or _trace_step_is_fallback(dq)):
            continue
        if estimator == 'moment_match':
            mu_p, cov_p = _moment_match_diag_gmm(dp)
            mu_q, cov_q = _moment_match_diag_gmm(dq)
            kl_pq = _gaussian_kl(mu_p, cov_p, mu_q, cov_q)
            if bool(symmetric):
                kl_qp = _gaussian_kl(mu_q, cov_q, mu_p, cov_p)
                vals.append(0.5 * float(kl_pq + kl_qp))
            else:
                vals.append(float(kl_pq))
        else:
            step_seed = int(base_rng.integers(0, 2**31 - 1))
            rng_pq = np.random.default_rng(step_seed + int(i))
            kl_pq = _mc_kl_diag_gmm(
                dist_p=dp,
                dist_q=dq,
                n_samples=n_mc_samples,
                rng=rng_pq,
                eps=eps,
            )
            if bool(symmetric):
                rng_qp = np.random.default_rng(step_seed + int(i) + 104729)
                kl_qp = _mc_kl_diag_gmm(
                    dist_p=dq,
                    dist_q=dp,
                    n_samples=n_mc_samples,
                    rng=rng_qp,
                    eps=eps,
                )
                vals.append(0.5 * float(kl_pq + kl_qp))
            else:
                vals.append(float(kl_pq))
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


class LatentDriverPredictiveKLAdapter:
    def __init__(self, cfg: TrackBConfig):
        self.cfg = cfg
        self.repo_path = Path(cfg.latentdriver_repo_path)
        self.context_len = int(max(1, cfg.latentdriver_context_len))
        self.device = None
        self.model = None
        self.act_dim = 3 if cfg.latentdriver_action_type == 'waypoint' else 2
        self.control_actor_factory = None
        self.last_obs_info: Dict[str, Any] = {}

        self._setup()

    def _setup(self):
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f'LatentDriver repo not found at {self.repo_path}. '
                'Clone it in Colab (e.g. /content/LatentDriver) or update cfg.latentdriver_repo_path.'
            )

        repo_str = str(self.repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        import torch
        from omegaconf import OmegaConf
        from src.policy import build_model
        from simulator.actor import create_control_actor

        self.torch = torch
        self.OmegaConf = OmegaConf
        self.build_model = build_model
        self.control_actor_factory = create_control_actor

        method_cfg = OmegaConf.load(self.repo_path / 'configs' / 'method' / 'latentdriver.yaml')

        # Hydra defaults are not auto-composed by OmegaConf.load in this script path.
        # Compose encoder/world configs explicitly so build_model gets a full method config.
        defaults = list(method_cfg.get('defaults', [])) if 'defaults' in method_cfg else []

        def _default_name(section: str, fallback: str) -> str:
            for d in defaults:
                if isinstance(d, dict) and (section in d):
                    return str(d[section])
            return fallback

        if ('encoder' not in method_cfg) or (method_cfg.encoder is None):
            enc_name = _default_name('encoder', 'bert')
            enc_path = self.repo_path / 'configs' / 'method' / 'encoder' / f'{enc_name}.yaml'
            if enc_path.exists():
                method_cfg.encoder = OmegaConf.load(enc_path)

        if ('world' not in method_cfg) or (method_cfg.world is None):
            world_name = _default_name('world', 'latent_world_model')
            world_path = self.repo_path / 'configs' / 'method' / 'world' / f'{world_name}.yaml'
            if world_path.exists():
                method_cfg.world = OmegaConf.load(world_path)

        method_cfg.model_name = self.cfg.latentdriver_method_name
        method_cfg.max_length = int(self.context_len)
        method_cfg.eval_context_length = int(self.context_len)

        if self.cfg.latentdriver_action_type == 'waypoint':
            method_cfg.action_space.dynamic_type = 'waypoint'
        else:
            method_cfg.action_space.dynamic_type = 'bicycle'

        method_cfg.ckpt_path = self.cfg.latentdriver_ckpt_path
        OmegaConf.resolve(method_cfg)

        if ('world' not in method_cfg) or (method_cfg.world is None) or ('act_dim' not in method_cfg.world):
            raise RuntimeError(
                'LatentDriver method config missing world.act_dim after composition. '
                'Check /content/LatentDriver/configs/method/world/*.yaml availability.'
            )
        composite_cfg = OmegaConf.create({'method': method_cfg})

        model = build_model(composite_cfg)
        ckpt_path = Path(self.cfg.latentdriver_ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f'LatentDriver checkpoint not found at {ckpt_path}. Set cfg.latentdriver_ckpt_path.'
            )

        ckpt = torch.load(str(ckpt_path), map_location='cpu')
        state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(self.device)
        model.eval()
        self.model = model

        self.act_dim = int(getattr(model, 'act_dim', self.act_dim))

        print(f'[LatentDriver] loaded ckpt: {ckpt_path}')
        print(f'[LatentDriver] device: {self.device}, act_dim={self.act_dim}, context_len={self.context_len}')

    def build_control_actor(self, is_controlled_func):
        return self.control_actor_factory(is_controlled_func=is_controlled_func)

    def encode_tokens(self, state: Any, selected_idx: np.ndarray) -> np.ndarray:
        cur = state.current_sim_trajectory
        xy_j, valid_j = squeeze_xy_valid_from_current(cur)
        xy = np.asarray(xy_j)
        valid = np.asarray(valid_j).astype(bool).reshape(-1)
        speed = cur.speed
        yaw = cur.yaw
        width = cur.width
        length = cur.length

        is_sdc_raw = np.asarray(state.object_metadata.is_sdc).astype(bool).reshape(-1)
        n_obj = int(xy.shape[0]) if xy.ndim == 2 else valid.shape[0]
        if is_sdc_raw.shape[0] < n_obj:
            is_sdc = np.concatenate(
                [is_sdc_raw, np.zeros((n_obj - is_sdc_raw.shape[0],), dtype=bool)],
                axis=0,
            )
        else:
            is_sdc = is_sdc_raw[:n_obj]

        tokens = []
        for obj_idx in selected_idx.tolist():
            obj_idx = int(obj_idx)
            v = bool(obj_idx < valid.shape[0] and valid[obj_idx])
            x, y = _xy_for_object(xy, obj_idx) if v else (0.0, 0.0)
            w = _feature_value_for_object(width, obj_idx, 0.0) if v else 0.0
            l = _feature_value_for_object(length, obj_idx, 0.0) if v else 0.0
            ya = _feature_value_for_object(yaw, obj_idx, 0.0) if v else 0.0
            sp = _feature_value_for_object(speed, obj_idx, 0.0) if v else 0.0

            type_id = 4.0 if (obj_idx < is_sdc.shape[0] and bool(is_sdc[obj_idx])) else 2.0
            if not v:
                type_id = 0.0
            tokens.append([type_id, x, y, w, l, ya, sp])
        tok = np.asarray(tokens, dtype=np.float32)
        self.last_obs_info = {
            'tokens_shape': tuple(tok.shape),
            'feature_dim': int(tok.shape[-1]) if tok.ndim >= 2 else -1,
            'finite': bool(np.isfinite(tok).all()) if tok.size > 0 else True,
            'type_id_min': float(tok[:, 0].min()) if tok.ndim == 2 and tok.shape[0] > 0 else np.nan,
            'type_id_max': float(tok[:, 0].max()) if tok.ndim == 2 and tok.shape[0] > 0 else np.nan,
        }
        return tok

    def _predict_distribution(self, state_hist: List[np.ndarray], action_hist: List[np.ndarray]) -> Dict[str, np.ndarray]:
        import torch

        states = np.stack(state_hist[-self.context_len:], axis=0)
        t = states.shape[0]

        if len(action_hist) >= t:
            actions = np.stack(action_hist[-t:], axis=0)
        else:
            pad = np.zeros((t - len(action_hist), self.act_dim), dtype=np.float32)
            if len(action_hist) > 0:
                actions = np.concatenate([pad, np.stack(action_hist, axis=0)], axis=0)
            else:
                actions = pad

        timesteps = np.arange(t, dtype=np.int64)

        states_t = torch.tensor(states[None, ...], dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions[None, ...], dtype=torch.float32, device=self.device)
        timesteps_t = torch.tensor(timesteps[None, ...], dtype=torch.long, device=self.device)

        try:
            with torch.no_grad():
                actions_layers, _, _ = self.model.forward(states_t, actions_t, timesteps_t, padding_mask=None)
            action_dis = actions_layers[-1][0, -1].detach().cpu().numpy().astype(np.float32)  # [M,7]
        except Exception:
            # Keep rollout alive when model forward path is brittle under specific package/runtime combos.
            return _latentdriver_fallback_dist(action_dim=self.act_dim)

        logits = action_dis[:, 0]
        out_model = action_dis[:, 1:6]
        yaw = action_dis[:, 6:7]

        weights = _softmax_np(logits)
        means = np.concatenate([out_model[:, 0:2], yaw], axis=-1)

        lo, hi = self.cfg.latentdriver_log_std_clip
        log_std_xy = np.clip(out_model[:, 2:4], lo, hi)
        std_xy = np.exp(log_std_xy)
        std_yaw = np.ones((std_xy.shape[0], 1), dtype=np.float32) * float(self.cfg.latentdriver_yaw_sigma)
        stds = np.concatenate([std_xy, std_yaw], axis=-1)

        return {
            'weights': weights.astype(np.float32),
            'means': means.astype(np.float32),
            'stds': stds.astype(np.float32),
            'fallback': np.asarray(0, dtype=np.int32),
            'source': 'model',
        }

    def _deterministic_action(self, dist: Dict[str, np.ndarray]) -> np.ndarray:
        weights = dist['weights']
        means = dist['means']
        idx = int(np.argmax(weights))
        action = np.asarray(means[idx], dtype=np.float32)

        clip = np.asarray(self.cfg.latentdriver_action_clip, dtype=np.float32)
        dim = min(action.shape[0], clip.shape[0])
        action[:dim] = np.clip(action[:dim], -clip[:dim], clip[:dim])
        return action

    def predict_action_and_dist(self, state_hist: List[np.ndarray], action_hist: List[np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        dist = self._predict_distribution(state_hist, action_hist)
        action = self._deterministic_action(dist)
        return action, dist


_LATENTDRIVER_ADAPTER: Optional[LatentDriverPredictiveKLAdapter] = None


def get_latentdriver_adapter(cfg: TrackBConfig) -> LatentDriverPredictiveKLAdapter:
    global _LATENTDRIVER_ADAPTER
    if _LATENTDRIVER_ADAPTER is None:
        _LATENTDRIVER_ADAPTER = LatentDriverPredictiveKLAdapter(cfg)
    return _LATENTDRIVER_ADAPTER


def make_closed_loop_components(base_state: Any, planner_kind: str, planner_name: str, cfg: TrackBConfig):
    num_objects = int(base_state.log_trajectory.xy.shape[0])
    is_sdc_all = np.asarray(base_state.object_metadata.is_sdc).astype(bool)
    if not np.any(is_sdc_all):
        raise ValueError('No SDC object found in state metadata.')
    sdc_idx = int(np.argmax(is_sdc_all))

    env_config = dataclasses.replace(
        waymax_config.EnvironmentConfig(),
        max_num_objects=num_objects,
        controlled_object=waymax_config.ObjectType.VALID,
    )

    env_dynamics = waymax_dynamics.StateDynamics()
    env = ENV_CLASS(dynamics_model=env_dynamics, config=env_config)

    obj_idx = jnp.arange(num_objects, dtype=jnp.int32)
    planner_mask_fn = lambda state: obj_idx == sdc_idx
    others_mask_fn = lambda state: obj_idx != sdc_idx

    non_sdc_actor = waymax_agents.create_expert_actor(
        dynamics_model=env_dynamics,
        is_controlled_func=others_mask_fn,
    )

    planner_bundle = {
        'env': env,
        'env_dynamics': env_dynamics,
        'non_sdc_actor': non_sdc_actor,
        'num_objects': num_objects,
        'sdc_idx': sdc_idx,
    }

    if planner_kind == 'latentdriver':
        adapter = get_latentdriver_adapter(cfg)
        planner_actor = adapter.build_control_actor(is_controlled_func=planner_mask_fn)
        sdc_fallback_actor = waymax_agents.create_expert_actor(
            dynamics_model=env_dynamics,
            is_controlled_func=planner_mask_fn,
        )

        if cfg.latentdriver_action_type == 'waypoint':
            control_dynamics_model = waymax_dynamics.DeltaLocal()
        elif cfg.latentdriver_action_type == 'bicycle':
            control_dynamics_model = waymax_dynamics.InvertibleBicycleModel()
        else:
            raise ValueError(f'Unknown latentdriver_action_type: {cfg.latentdriver_action_type}')

        planner_bundle.update({
            'planner_type': 'latentdriver',
            'planner_actor': planner_actor,
            'sdc_fallback_actor': sdc_fallback_actor,
            'control_dynamics_model': control_dynamics_model,
            'ld_adapter': adapter,
            'planner_used': 'LatentDriverPredictiveKL',
        })

    elif planner_kind == 'idm_route':
        planner_actor = None
        errors = []

        try:
            planner_actor = waymax_agents.IDMRoutePolicy(is_controlled_func=planner_mask_fn)
            planner_used = 'IDMRoutePolicy'
        except Exception as e:
            errors.append(f'IDMRoutePolicy failed: {e}')

        if planner_actor is None:
            try:
                planner_actor = waymax_agents.create_expert_actor(
                    dynamics_model=env_dynamics,
                    is_controlled_func=planner_mask_fn,
                )
                planner_used = 'ExpertActorFallback'
            except Exception as e:
                errors.append(f'Expert fallback failed: {e}')
                raise RuntimeError('Planner actor construction failed. ' + ' | '.join(errors))

        planner_bundle.update({
            'planner_type': 'actor',
            'planner_actor': planner_actor,
            'planner_used': planner_used,
        })

    else:
        raise ValueError(f'Unsupported planner_kind: {planner_kind}')

    if planner_bundle['planner_used'] != planner_name:
        print(f"[planner notice] requested={planner_name}, used={planner_bundle['planner_used']}")

    return planner_bundle


def _extract_sdc_action_vector(planner_out: Any, sdc_idx: int, fallback_dim: int = 3) -> Tuple[np.ndarray, bool]:
    action = planner_out.action if hasattr(planner_out, 'action') else planner_out
    data = np.asarray(action.data)
    valid = np.asarray(action.valid).astype(bool)

    if data.ndim >= 1 and valid.ndim >= 1 and data.shape[0] == valid.shape[0] and sdc_idx < data.shape[0]:
        vec = np.asarray(data[sdc_idx]).reshape(-1)
        val = bool(np.asarray(valid[sdc_idx]).all())
    elif data.ndim >= 2 and sdc_idx < data.shape[0]:
        vec = np.asarray(data[sdc_idx]).reshape(-1)
        if valid.ndim >= 1 and sdc_idx < valid.shape[0]:
            val = bool(np.asarray(valid[sdc_idx]).all())
        else:
            val = bool(np.asarray(valid).all())
    else:
        vec = np.asarray(data).reshape(-1)
        val = bool(np.asarray(valid).all())

    if vec.size == 0:
        vec = np.zeros((fallback_dim,), dtype=np.float32)
        val = False

    return vec.astype(np.float32), val


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


def closed_loop_rollout_selected(
    base_state: Any,
    selected_idx: np.ndarray,
    target_obj_idx: int,
    delta_xy: np.ndarray,
    cfg: TrackBConfig,
    planner_bundle: Dict[str, Any],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Optional[Dict[str, np.ndarray]]], bool, str]:
    env = planner_bundle['env']
    non_sdc_actor = planner_bundle['non_sdc_actor']
    planner_type = planner_bundle['planner_type']
    sdc_idx = int(planner_bundle['sdc_idx'])

    try:
        perturbed = perturb_initial_state(base_state, target_obj_idx, delta_xy)
        state = env.reset(perturbed)

        rng = jax.random.PRNGKey(int(seed))
        k1, step_key = jax.random.split(rng, 2)

        actor_state_non_sdc = non_sdc_actor.init(k1, state)

        xy_seq = []
        valid_seq = []
        action_seq = []
        action_valid_seq = []
        dist_trace: List[Optional[Dict[str, np.ndarray]]] = []

        selected_idx_j = jnp.asarray(selected_idx, dtype=jnp.int32)
        action_dim = None

        ld_state_hist: List[np.ndarray] = []
        ld_action_hist: List[np.ndarray] = []

        if planner_type == 'actor':
            planner_actor = planner_bundle['planner_actor']
            k0, step_key = jax.random.split(step_key)
            actor_state_planner = planner_actor.init(k0, state)
        else:
            planner_actor = planner_bundle['planner_actor']
            sdc_fallback_actor = planner_bundle['sdc_fallback_actor']
            control_dynamics_model = planner_bundle['control_dynamics_model']
            ld_adapter = planner_bundle['ld_adapter']
            actor_state_planner = None
            kfb, step_key = jax.random.split(step_key)
            actor_state_sdc_fallback = sdc_fallback_actor.init(kfb, state)

        for t in range(int(cfg.future_steps)):
            cur = state.current_sim_trajectory
            xy_t_all, valid_t_all = squeeze_xy_valid_from_current(cur)

            xy_t = xy_t_all[selected_idx_j, :]
            valid_t = valid_t_all[selected_idx_j]
            xy_seq.append(xy_t)
            valid_seq.append(valid_t)

            step_key, sub = jax.random.split(step_key)
            subkeys = jax.random.split(sub, 2)

            if planner_type == 'actor':
                out_p = planner_actor.select_action({}, state, actor_state_planner, subkeys[0])
                action_vec, action_ok = _extract_sdc_action_vector(out_p, sdc_idx=sdc_idx, fallback_dim=3)
                dist_trace.append(None)
                actor_state_planner = out_p.actor_state

            else:
                tokens = ld_adapter.encode_tokens(state, selected_idx)
                ld_state_hist.append(tokens)

                try:
                    action_vec, dist_step = ld_adapter.predict_action_and_dist(ld_state_hist, ld_action_hist)
                except Exception:
                    action_vec = np.zeros((3,), dtype=np.float32)
                    dist_step = _latentdriver_fallback_dist(action_dim=np.asarray(action_vec).size)
                if isinstance(dist_step, dict):
                    dist_step.setdefault('fallback', np.asarray(0, dtype=np.int32))
                    dist_step.setdefault('source', 'model')
                    dist_step.setdefault('actor_fallback', np.asarray(0, dtype=np.int32))
                ld_action_hist.append(np.asarray(action_vec, dtype=np.float32))
                dist_trace.append(dist_step)

                try:
                    out_p_raw = planner_actor.select_action({'actions': action_vec.tolist()}, state, None, None)

                    cur_timestep = int(np.asarray(state.timestep).reshape(-1)[0])
                    traj = waymax_datatypes.dynamic_slice(
                        inputs=state.sim_trajectory,
                        start_index=cur_timestep,
                        slice_size=1,
                        axis=-1,
                    )
                    action_transformed = control_dynamics_model.compute_update(out_p_raw.action, traj).as_action()
                    out_p = _replace_action(out_p_raw, action_transformed)
                    action_ok = True
                except Exception:
                    # Fallback keeps simulation running if LatentDriver actor path breaks.
                    out_p = sdc_fallback_actor.select_action({}, state, actor_state_sdc_fallback, subkeys[0])
                    actor_state_sdc_fallback = out_p.actor_state
                    action_vec, action_ok = _extract_sdc_action_vector(out_p, sdc_idx=sdc_idx, fallback_dim=3)
                    if isinstance(dist_step, dict):
                        dist_step['actor_fallback'] = np.asarray(1, dtype=np.int32)

            if action_dim is None:
                action_dim = int(np.asarray(action_vec).size)
            if np.asarray(action_vec).size != action_dim:
                vec = np.zeros((action_dim,), dtype=np.float32)
                n = min(action_dim, np.asarray(action_vec).size)
                vec[:n] = np.asarray(action_vec).reshape(-1)[:n]
                action_vec = vec

            action_seq.append(np.asarray(action_vec, dtype=np.float32))
            action_valid_seq.append(bool(action_ok))

            out_n = non_sdc_actor.select_action({}, state, actor_state_non_sdc, subkeys[1])
            actor_state_non_sdc = out_n.actor_state

            merged = waymax_agents.merge_actions([out_p, out_n])
            state = env.step(state, merged)

        xy = np.asarray(jnp.stack(xy_seq, axis=1), dtype=np.float32)
        valid = np.asarray(jnp.stack(valid_seq, axis=1), dtype=bool)

        if len(action_seq) > 0:
            planner_actions = np.asarray(np.stack(action_seq, axis=0), dtype=np.float32)
            planner_action_valid = np.asarray(action_valid_seq, dtype=bool)
        else:
            planner_actions = np.zeros((cfg.future_steps, 3), dtype=np.float32)
            planner_action_valid = np.zeros((cfg.future_steps,), dtype=bool)

        finite_ok = bool(np.isfinite(xy).all()) and bool(np.isfinite(planner_actions).all())
        any_valid = bool(valid.any()) and bool(planner_action_valid.any())
        rollout_feasible = bool(finite_ok and any_valid)
        note = '' if rollout_feasible else 'non_finite_or_all_invalid_rollout_or_action'
        return xy, valid, planner_actions, planner_action_valid, dist_trace, rollout_feasible, note

    except Exception as e:
        n_obj = int(selected_idx.shape[0])
        xy = np.zeros((n_obj, cfg.future_steps, 2), dtype=np.float32)
        valid = np.zeros((n_obj, cfg.future_steps), dtype=bool)
        planner_actions = np.zeros((cfg.future_steps, 3), dtype=np.float32)
        planner_action_valid = np.zeros((cfg.future_steps,), dtype=bool)
        dist_trace = [None for _ in range(int(cfg.future_steps))]
        return xy, valid, planner_actions, planner_action_valid, dist_trace, False, f'rollout_exception: {e}'



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
            obs_info = planner_bundle['ld_adapter'].last_obs_info
            feat_dim_ok = int(obs_info.get('feature_dim', -1)) == int(latentdriver_observation_contract()['feature_dim'])
            add('latentdriver_obs_feature_dim_ok', feat_dim_ok, str(obs_info))
            add('latentdriver_obs_finite', bool(obs_info.get('finite', False)), str(obs_info))

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


def dist_trace_diagnostics(dist_trace: List[Optional[Dict[str, np.ndarray]]]) -> Dict[str, float]:
    total_steps = int(len(dist_trace))
    non_null = [d for d in dist_trace if d is not None]
    non_null_steps = int(len(non_null))
    if non_null_steps == 0:
        return {
            'dist_non_null_steps': 0.0,
            'dist_non_null_ratio': 0.0,
            'dist_mean_components': np.nan,
            'dist_min_weight': np.nan,
            'dist_min_std': np.nan,
            'dist_max_std': np.nan,
            'dist_finite_ratio': 0.0,
            'dist_fallback_steps': 0.0,
            'dist_fallback_ratio': 0.0,
            'dist_actor_fallback_steps': 0.0,
            'dist_actor_fallback_ratio': 0.0,
        }

    comp_counts = []
    min_weights = []
    min_stds = []
    max_stds = []
    finite_flags = []
    fallback_flags = []
    actor_fallback_flags = []
    for d in non_null:
        w = np.asarray(d.get('weights', []), dtype=float).reshape(-1)
        s = np.asarray(d.get('stds', []), dtype=float)
        comp_counts.append(float(w.shape[0]))
        min_weights.append(float(np.min(w)) if w.size > 0 else np.nan)
        min_stds.append(float(np.min(s)) if s.size > 0 else np.nan)
        max_stds.append(float(np.max(s)) if s.size > 0 else np.nan)
        finite_flags.append(float(np.isfinite(w).all() and np.isfinite(s).all()))
        fallback_flags.append(float(np.asarray(d.get('fallback', 0)).reshape(-1)[0]))
        actor_fallback_flags.append(float(np.asarray(d.get('actor_fallback', 0)).reshape(-1)[0]))

    return {
        'dist_non_null_steps': float(non_null_steps),
        'dist_non_null_ratio': float(non_null_steps / max(total_steps, 1)),
        'dist_mean_components': float(np.nanmean(comp_counts)) if len(comp_counts) else np.nan,
        'dist_min_weight': float(np.nanmin(min_weights)) if len(min_weights) else np.nan,
        'dist_min_std': float(np.nanmin(min_stds)) if len(min_stds) else np.nan,
        'dist_max_std': float(np.nanmax(max_stds)) if len(max_stds) else np.nan,
        'dist_finite_ratio': float(np.nanmean(finite_flags)) if len(finite_flags) else 0.0,
        'dist_fallback_steps': float(np.nansum(fallback_flags)) if len(fallback_flags) else 0.0,
        'dist_fallback_ratio': float(np.nanmean(fallback_flags)) if len(fallback_flags) else 0.0,
        'dist_actor_fallback_steps': float(np.nansum(actor_fallback_flags)) if len(actor_fallback_flags) else 0.0,
        'dist_actor_fallback_ratio': float(np.nanmean(actor_fallback_flags)) if len(actor_fallback_flags) else 0.0,
    }


def dist_trace_change_stats(
    trace_p: List[Optional[Dict[str, np.ndarray]]],
    trace_q: List[Optional[Dict[str, np.ndarray]]],
) -> Dict[str, float]:
    n = int(min(len(trace_p), len(trace_q)))
    if n <= 0:
        return {
            'trace_pair_steps': 0.0,
            'trace_pair_ratio': 0.0,
            'step_mean_l2_mean': np.nan,
            'step_mean_l2_p50': np.nan,
            'step_mean_l2_p95': np.nan,
            'step_std_l2_mean': np.nan,
            'step_moment_kl_mean': np.nan,
            'step_moment_kl_p50': np.nan,
            'step_moment_kl_p95': np.nan,
            'step_moment_kl_nonzero_ratio': 0.0,
        }

    mean_l2_vals: List[float] = []
    std_l2_vals: List[float] = []
    kl_vals: List[float] = []

    for i in range(n):
        dp = trace_p[i]
        dq = trace_q[i]
        if dp is None or dq is None:
            continue

        mu_p, cov_p = _moment_match_diag_gmm(dp)
        mu_q, cov_q = _moment_match_diag_gmm(dq)

        d = int(min(mu_p.shape[0], mu_q.shape[0]))
        if d <= 0:
            continue

        mean_l2_vals.append(float(np.linalg.norm(mu_p[:d] - mu_q[:d])))

        std_p = np.sqrt(np.maximum(np.diag(cov_p)[:d], 0.0))
        std_q = np.sqrt(np.maximum(np.diag(cov_q)[:d], 0.0))
        std_l2_vals.append(float(np.linalg.norm(std_p - std_q)))

        kl_vals.append(float(_gaussian_kl(mu_p[:d], cov_p[:d, :d], mu_q[:d], cov_q[:d, :d])))

    pair_steps = int(len(kl_vals))
    if pair_steps == 0:
        return {
            'trace_pair_steps': 0.0,
            'trace_pair_ratio': 0.0,
            'step_mean_l2_mean': np.nan,
            'step_mean_l2_p50': np.nan,
            'step_mean_l2_p95': np.nan,
            'step_std_l2_mean': np.nan,
            'step_moment_kl_mean': np.nan,
            'step_moment_kl_p50': np.nan,
            'step_moment_kl_p95': np.nan,
            'step_moment_kl_nonzero_ratio': 0.0,
        }

    kl_arr = np.asarray(kl_vals, dtype=float)
    mean_l2_arr = np.asarray(mean_l2_vals, dtype=float)
    std_l2_arr = np.asarray(std_l2_vals, dtype=float)

    return {
        'trace_pair_steps': float(pair_steps),
        'trace_pair_ratio': float(pair_steps / max(n, 1)),
        'step_mean_l2_mean': float(np.mean(mean_l2_arr)),
        'step_mean_l2_p50': float(np.quantile(mean_l2_arr, 0.50)),
        'step_mean_l2_p95': float(np.quantile(mean_l2_arr, 0.95)),
        'step_std_l2_mean': float(np.mean(std_l2_arr)),
        'step_moment_kl_mean': float(np.mean(kl_arr)),
        'step_moment_kl_p50': float(np.quantile(kl_arr, 0.50)),
        'step_moment_kl_p95': float(np.quantile(kl_arr, 0.95)),
        'step_moment_kl_nonzero_ratio': float(np.mean(kl_arr > 1e-9)),
    }


def evaluate_delta_closed_loop(
    rec: Dict[str, Any],
    planner_bundle: Dict[str, Any],
    target_idx: int,
    delta_xy: np.ndarray,
    cfg: TrackBConfig,
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
    if method == 'prism_joint':
        return cfg_opt.w_prism_joint
    if method == 'random':
        return cfg_opt.w_prism_joint
    raise ValueError(f'Unknown method: {method}')


def optimize_method_closed_loop(
    method: str,
    rec: Dict[str, Any],
    planner_bundle: Dict[str, Any],
    target_idx: int,
    cfg: TrackBConfig,
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


def _load_existing_results(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
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
    methods = ['random', 'risk_only', 'surprise_only', 'prism_joint']
    usable = df[df['method'].isin(methods)].copy()

    if len(usable) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    quick_summary = (
        usable.groupby('method', as_index=False)
        .agg(
            n=('scenario_id', 'size'),
            mean_risk=('risk_sks', 'mean'),
            mean_surprise=('surprise_pd', 'mean'),
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
    for method in ['risk_only', 'surprise_only', 'prism_joint']:
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
    cfg: TrackBConfig,
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
    ]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(per_scenario_path, index=False)
    if bool(cfg.save_per_eval_trace) and isinstance(trace_df, pd.DataFrame):
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
            results_df['method'].isin(['random', 'risk_only', 'surprise_only', 'prism_joint'])
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


def run_trackb_closed_loop(
    runner: TrackBRunner,
    eval_idx: np.ndarray,
    cfg: TrackBConfig,
    search_cfg: SearchConfig,
    thresholds: Dict[str, float],
    run_prefix: Optional[str] = None,
    static_frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    methods = ['random', 'risk_only', 'surprise_only', 'prism_joint']

    run_prefix = run_prefix or cfg.run_prefix
    checkpoint_path = f'{run_prefix}_per_scenario_results.csv'
    trace_checkpoint_path = f'{run_prefix}_per_eval_trace.csv'

    existing_df = _load_existing_results(checkpoint_path) if bool(cfg.resume_from_existing) else pd.DataFrame()
    existing_trace_df = (
        _load_existing_results(trace_checkpoint_path)
        if bool(cfg.resume_from_existing) and bool(cfg.save_per_eval_trace)
        else pd.DataFrame()
    )
    completed = _completed_scenarios(existing_df, methods)

    pending = [int(sid) for sid in eval_idx if int(sid) not in completed]
    print(f'[run] completed={len(completed)}, pending={len(pending)}, total_eval={len(eval_idx)}')

    rows_buffer: List[Dict[str, Any]] = []
    trace_buffer: List[Dict[str, Any]] = []
    skipped = 0
    processed = 0

    chunk_size = int(max(1, cfg.run_chunk_size))
    for chunk_id, start in enumerate(range(0, len(pending), chunk_size), start=1):
        chunk = pending[start:start + chunk_size]
        iterator = tqdm(chunk, desc=f'Track B chunk {chunk_id} ({len(chunk)} scenarios)', total=len(chunk))

        for sid in iterator:
            rec = runner.data['scenarios'][sid]

            if 'state' not in rec:
                skipped += 1
                continue

            try:
                selected_idx = np.asarray(rec['selected_indices'], dtype=np.int32)
                target_idx = _choose_target_non_ego(rec['state'], selected_idx)
                planner_bundle = make_closed_loop_components(rec['state'], cfg.planner_kind, cfg.planner_name, cfg)

                # Common random numbers across methods:
                # same proposal bank + same rollout seed schedule for all methods in this scenario.
                scenario_seed = int(cfg.global_seed + sid * 7919)
                scenario_rng = np.random.default_rng(scenario_seed)
                n_props = int(max(0, search_cfg.budget_evals - 1))
                proposal_bank = scenario_rng.normal(size=(n_props, 2)).astype(np.float32) if n_props > 0 else np.zeros((0, 2), dtype=np.float32)
                rollout_seed_schedule = [
                    int(cfg.global_seed + sid * cfg.rollout_seed_stride + k)
                    for k in range(int(search_cfg.budget_evals) + 1)
                ]

                for method in methods:
                    stats = optimize_method_closed_loop(
                        method=method,
                        rec=rec,
                        planner_bundle=planner_bundle,
                        target_idx=target_idx,
                        cfg=cfg,
                        search_cfg=search_cfg,
                        thresholds=thresholds,
                        scenario_seed=scenario_seed,
                        proposal_bank=proposal_bank,
                        rollout_seed_schedule=rollout_seed_schedule,
                    )
                    eval_trace = stats.pop('eval_trace', [])

                    rows_buffer.append({
                        'scenario_id': sid,
                        'method': method,
                        'seed_used': scenario_seed,
                        'target_obj_idx': int(target_idx),
                        'planner': cfg.planner_name,
                        **stats,
                    })
                    if bool(cfg.save_per_eval_trace) and len(eval_trace) > 0:
                        for tr in eval_trace:
                            trace_buffer.append({
                                'scenario_id': int(sid),
                                'method': method,
                                'seed_used': int(scenario_seed),
                                'target_obj_idx': int(target_idx),
                                'planner': cfg.planner_name,
                                **tr,
                            })

            except Exception as e:
                skipped += 1
                rows_buffer.append({
                    'scenario_id': sid,
                    'method': 'scenario_error',
                    'seed_used': int(cfg.global_seed + sid),
                    'target_obj_idx': -1,
                    'planner': cfg.planner_name,
                    'objective': np.nan,
                    'risk_sks': np.nan,
                    'surprise_pd': np.nan,
                    'surprise_kl': np.nan,
                    'failure_proxy': np.nan,
                    'failure_strict_proxy': np.nan,
                    'collision': np.nan,
                    'min_dist': np.nan,
                    'min_ttc': np.nan,
                    'max_acc': np.nan,
                    'max_jerk': np.nan,
                    'delta_risk': np.nan,
                    'delta_surprise': np.nan,
                    'objective_start': np.nan,
                    'objective_gain': np.nan,
                    'delta_risk_start': np.nan,
                    'delta_surprise_start': np.nan,
                    'delta_x': np.nan,
                    'delta_y': np.nan,
                    'delta_l2': np.nan,
                    'max_abs_delta': np.nan,
                    'rollout_feasible': 0,
                    'feasible': 0,
                    'feasibility_violation': 1.0,
                    'q1_hit': 0,
                    'q4_hit': 0,
                    'blind_spot_proxy_hit': 0,
                    'optimizer_used': 'error',
                    'budget_units_used': 0,
                    'accepted_improvements': 0,
                    'planner_used': 'error',
                    'rollout_note': f'scenario_exception: {e}',
                })

            processed += 1
            if int(cfg.checkpoint_every_scenarios) > 0 and (processed % int(cfg.checkpoint_every_scenarios) == 0):
                existing_df = _flush_checkpoint(rows_buffer, existing_df, checkpoint_path)
                rows_buffer = []
                if bool(cfg.save_per_eval_trace):
                    existing_trace_df = _flush_checkpoint(
                        trace_buffer,
                        existing_trace_df,
                        trace_checkpoint_path,
                        dedup_cols=['scenario_id', 'method', 'eval_index'],
                    )
                    trace_buffer = []
                _write_progress_artifacts(
                    run_prefix=run_prefix,
                    results_df=existing_df,
                    trace_df=existing_trace_df,
                    cfg=cfg,
                    search_cfg=search_cfg,
                    thresholds=thresholds,
                    static_frames=static_frames,
                )

        existing_df = _flush_checkpoint(rows_buffer, existing_df, checkpoint_path)
        rows_buffer = []
        if bool(cfg.save_per_eval_trace):
            existing_trace_df = _flush_checkpoint(
                trace_buffer,
                existing_trace_df,
                trace_checkpoint_path,
                dedup_cols=['scenario_id', 'method', 'eval_index'],
            )
            trace_buffer = []
        _write_progress_artifacts(
            run_prefix=run_prefix,
            results_df=existing_df,
            trace_df=existing_trace_df,
            cfg=cfg,
            search_cfg=search_cfg,
            thresholds=thresholds,
            static_frames=static_frames,
        )
        completed_now = _completed_scenarios(existing_df, methods)
        print(
            f'[progress] chunk {chunk_id} saved. '
            f'completed={len(completed_now)}/{len(eval_idx)} scenarios.'
        )

    final_df = existing_df if not existing_df.empty else pd.DataFrame(rows_buffer)
    final_trace_df = existing_trace_df if not existing_trace_df.empty else pd.DataFrame(trace_buffer)
    print(f'Closed-loop run complete. rows={len(final_df)}, skipped={skipped}, saved={checkpoint_path}')
    if bool(cfg.save_per_eval_trace):
        print(f'Per-eval trace rows={len(final_trace_df)}, saved={trace_checkpoint_path}')
    return final_df, final_trace_df


def make_waymax_data_iter(cfg: TrackBConfig):
    """Build dataset config and iterator (with GCS auth precheck for gs:// paths)."""
    ensure_womd_gcs_access(cfg.waymax_path)
    dataset_config = waymax_config.DatasetConfig(
        path=cfg.waymax_path,
        data_format=waymax_config.DataFormat.TFRECORD,
        max_num_rg_points=cfg.waymax_max_rg_points,
        batch_dims=cfg.waymax_batch_dims,
    )
    data_iter = waymax_dataloader.simulator_state_generator(dataset_config)
    return dataset_config, data_iter


def build_trackb_runner_and_splits(
    cfg: TrackBConfig,
    data_iter: Optional[Iterable[Any]],
    dataset_config: Optional[Any],
    n_shards: int,
    shard_id: int,
):
    runner = TrackBRunner(cfg, data_iter=data_iter, dataset_config=dataset_config)
    data = runner.build_dataset()

    train_idx = data['train_idx']
    test_idx = data['test_idx']

    required_eval = int(max(cfg.n_eval_scenarios, cfg.strict_min_eval))
    if len(test_idx) < required_eval:
        required_total = required_total_scenarios(required_eval, cfg.train_fraction)
        raise ValueError(
            f'Not enough test scenarios for strict evaluation: have {len(test_idx)}, need {required_eval}. '
            f'Current n_total_scenarios={cfg.n_total_scenarios}, train_fraction={cfg.train_fraction}. '
            f'Set n_total_scenarios >= {required_total} (or reduce strict_min_eval / n_eval_scenarios).'
        )

    eval_idx_all = test_idx[:cfg.n_eval_scenarios]
    if int(max(1, n_shards)) > 1:
        eval_idx = eval_idx_all[int(shard_id)::int(n_shards)]
    else:
        eval_idx = eval_idx_all

    if len(eval_idx) == 0:
        raise ValueError(f'Empty shard eval set for shard_id={shard_id}, n_shards={n_shards}.')

    reference_df = runner.score_indices_openloop(train_idx, label='reference_openloop', show_progress=True)
    base_eval_openloop_df = runner.score_indices_openloop(eval_idx, label='base_eval_openloop', show_progress=True)

    return runner, data, train_idx, test_idx, eval_idx_all, eval_idx, reference_df, base_eval_openloop_df


def run_preflight_and_calibration(
    runner: TrackBRunner,
    cfg: TrackBConfig,
    search_cfg: SearchConfig,
    eval_idx: np.ndarray,
    reference_df: pd.DataFrame,
    restore_from_upload: bool = False,
):
    if bool(restore_from_upload):
        _ = restore_artifacts_via_upload(
            cfg.run_prefix,
            required_keys=['per_scenario_results', 'per_eval_trace', 'thresholds', 'closedloop_calibration'],
        )

    artifact_paths = build_run_artifact_paths(cfg.run_prefix)
    thresholds_path_resume = artifact_paths['thresholds']
    closedloop_calib_path_resume = artifact_paths['closedloop_calibration']

    preflight_df = run_trackb_preflight_checks(runner, cfg, eval_idx)
    if bool(cfg.require_preflight_pass) and (not preflight_df.empty) and (not bool(preflight_df['pass'].all())):
        failed = preflight_df[~preflight_df['pass']]
        raise RuntimeError(
            'Track B preflight failed. Fix these checks before running calibration/main loop:\n'
            + failed.to_string(index=False)
        )

    if Path(thresholds_path_resume).exists():
        with open(thresholds_path_resume, 'r') as f:
            trackb_thresholds = json.load(f)
        if Path(closedloop_calib_path_resume).exists():
            closedloop_calib_df = pd.read_csv(closedloop_calib_path_resume)
        else:
            closedloop_calib_df = pd.DataFrame()
        calib_diag_df, calib_quant_df = build_calibration_diagnostics(closedloop_calib_df, trackb_thresholds)

        loaded_surprise_scale = float(trackb_thresholds.get('surprise_scale', np.nan))
        loaded_surprise_thr = float(trackb_thresholds.get('surprise_high_threshold', np.nan))
        force_recalib = (
            (not np.isfinite(loaded_surprise_scale))
            or (loaded_surprise_scale <= float(search_cfg.min_scale) * 1.01)
            or (np.isfinite(loaded_surprise_thr) and loaded_surprise_thr <= 0.0)
        )
        if force_recalib:
            print('[resume] existing thresholds look degenerate; recalibrating closed-loop surprise.')
            closedloop_calib_df, trackb_thresholds = calibrate_closed_loop_thresholds(
                runner,
                eval_idx,
                cfg,
                search_cfg,
                reference_df=reference_df,
            )
            calib_diag_df, calib_quant_df = build_calibration_diagnostics(closedloop_calib_df, trackb_thresholds)
    else:
        closedloop_calib_df, trackb_thresholds = calibrate_closed_loop_thresholds(
            runner,
            eval_idx,
            cfg,
            search_cfg,
            reference_df=reference_df,
        )
        calib_diag_df, calib_quant_df = build_calibration_diagnostics(closedloop_calib_df, trackb_thresholds)

    return preflight_df, closedloop_calib_df, trackb_thresholds, calib_diag_df, calib_quant_df


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

    dist_change_summary = pd.DataFrame([{
        'trace_pair_ratio_mean': _col_mean(usable_calib, 'trace_pair_ratio'),
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


def summarize_method_outputs(trackb_results_df: pd.DataFrame, trackb_trace_df: pd.DataFrame):
    usable_df = trackb_results_df[
        trackb_results_df['method'].isin(['random', 'risk_only', 'surprise_only', 'prism_joint'])
    ].copy()

    quick_summary_df = (
        usable_df.groupby('method', as_index=False)
        .agg(
            n=('scenario_id', 'size'),
            mean_risk=('risk_sks', 'mean'),
            mean_surprise=('surprise_pd', 'mean'),
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
    for method in ['risk_only', 'surprise_only', 'prism_joint']:
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

    if isinstance(trackb_trace_df, pd.DataFrame) and len(trackb_trace_df) > 0:
        trace_diag_df = (
            trackb_trace_df.groupby('method', as_index=False)
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


def export_trackb_artifacts(
    cfg: TrackBConfig,
    search_cfg: SearchConfig,
    eval_idx: np.ndarray,
    trackb_results_df: pd.DataFrame,
    trackb_trace_df: pd.DataFrame,
    base_eval_openloop_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    closedloop_calib_df: pd.DataFrame,
    preflight_df: pd.DataFrame,
    calib_diag_df: pd.DataFrame,
    calib_quant_df: pd.DataFrame,
    trackb_thresholds: Dict[str, Any],
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
    runtime_manifest_path = f'{run_prefix}_runtime_manifest.json'

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
        'runtime_manifest': runtime_manifest_path,
    }

    for p in all_paths.values():
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    trackb_results_df.to_csv(per_scenario_path, index=False)
    if isinstance(trackb_trace_df, pd.DataFrame):
        trackb_trace_df.to_csv(per_eval_trace_path, index=False)
    base_eval_openloop_df.to_csv(base_eval_openloop_path, index=False)
    reference_df.to_csv(reference_openloop_path, index=False)
    closedloop_calib_df.to_csv(closedloop_calib_path, index=False)
    preflight_df.to_csv(preflight_path, index=False)
    calib_diag_df.to_csv(calib_diag_path, index=False)
    calib_quant_df.to_csv(calib_quant_path, index=False)
    quick_summary_df.to_csv(quick_summary_path, index=False)
    sanity_df.to_csv(sanity_path, index=False)
    fairness_checks_df.to_csv(fairness_path, index=False)
    if isinstance(trace_diag_df, pd.DataFrame):
        trace_diag_df.to_csv(trace_diag_path, index=False)

    seed_map_df = trackb_results_df[
        trackb_results_df['method'].isin(['random', 'risk_only', 'surprise_only', 'prism_joint'])
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
                for k, v in trackb_thresholds.items()
            },
            f,
            indent=2,
        )

    carry_forward_config = {
        'experiment_track': 'B_closed_loop_simulation_only',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'planner': {
            'planner_kind': cfg.planner_kind,
            'planner_name_config': cfg.planner_name,
        },
        'surprise_definition': {
            'name': cfg.planner_surprise_name,
            'type': 'planner_dependent_predictive_kl',
            'formula': 'KL( p_pi(a|state_delta) || p_pi(a|state_base) ) estimated by Monte Carlo on LatentDriver diagonal-GMM action distributions',
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
            'train_fraction': float(cfg.train_fraction),
            'eval_scenario_ids': [int(x) for x in eval_idx],
        },
        'optimization': dataclasses.asdict(search_cfg),
        'thresholds': trackb_thresholds,
        'risk_failure_thresholds': {
            'collision_distance': float(cfg.collision_distance),
            'ttc_fail_seconds': float(cfg.ttc_fail_seconds),
            'no_hazard_ttc_seconds': float(cfg.no_hazard_ttc_seconds),
            'no_hazard_dist_m': float(cfg.no_hazard_dist_m),
            'hard_brake_mps2': float(cfg.hard_brake_mps2),
            'hard_jerk_mps3': float(cfg.hard_jerk_mps3),
        },
        'method_labels': ['random', 'risk_only', 'surprise_only', 'prism_joint'],
    }
    with open(carry_path, 'w') as f:
        json.dump(carry_forward_config, f, indent=2)

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
        'jax_backend': str(jax.default_backend()),
        'jax_devices': [str(d) for d in jax.devices()],
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
