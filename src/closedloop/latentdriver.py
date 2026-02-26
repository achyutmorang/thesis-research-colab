from __future__ import annotations

import dataclasses
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from waymax import agents as waymax_agents
from waymax import config as waymax_config
from waymax import datatypes as waymax_datatypes
from waymax import dynamics as waymax_dynamics
from waymax import env as waymax_env

from .config import ClosedLoopConfig

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

def resolve_env_class():
    if hasattr(waymax_env, 'BaseEnvironment'):
        return waymax_env.BaseEnvironment
    if hasattr(waymax_env, 'MultiAgentEnvironment'):
        return waymax_env.MultiAgentEnvironment
    from waymax.env import base_environment as _base_env
    if hasattr(_base_env, 'BaseEnvironment'):
        return _base_env.BaseEnvironment
    return _base_env.MultiAgentEnvironment

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
    def __init__(self, cfg: ClosedLoopConfig):
        self.cfg = cfg
        self.repo_path = Path(cfg.latentdriver_repo_path)
        self.context_len = int(max(1, cfg.latentdriver_context_len))
        self.device = None
        self.model = None
        self.act_dim = 3 if cfg.latentdriver_action_type == 'waypoint' else 2
        self.control_actor_factory = None
        self.last_obs_info: Dict[str, Any] = {}
        self._forward_error_count = 0
        self._last_forward_route = 'uninitialized'
        self._last_forward_error = ''

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

    def _forward_kwargs_from_signature(
        self,
        states_t: Any,
        actions_t: Any,
        timesteps_t: Any,
        padding_mask_t: Optional[Any] = None,
    ) -> Dict[str, Any]:
        try:
            sig = inspect.signature(self.model.forward)
        except Exception:
            return {}

        params = sig.parameters
        has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        kwargs: Dict[str, Any] = {}

        def _bind_first(names: List[str], value: Any) -> bool:
            for name in names:
                if has_var_kwargs or (name in params):
                    kwargs[name] = value
                    return True
            return False

        _bind_first(['states', 'state', 'obs', 'observations', 'inputs'], states_t)
        _bind_first(['actions', 'action', 'acts'], actions_t)
        _bind_first(['timesteps', 'timestep', 'time_steps', 'times', 'time'], timesteps_t)

        if has_var_kwargs or ('padding_mask' in params):
            kwargs.setdefault('padding_mask', padding_mask_t)
        elif 'attention_mask' in params:
            kwargs.setdefault('attention_mask', padding_mask_t)

        return kwargs

    def _call_model_forward_flexible(
        self,
        states_t: Any,
        actions_t: Any,
        timesteps_t: Any,
        padding_mask_t: Optional[Any] = None,
    ) -> Any:
        attempts: List[Tuple[str, Any]] = [
            (
                'positional+padding_mask',
                lambda: self.model.forward(states_t, actions_t, timesteps_t, padding_mask=padding_mask_t),
            ),
            (
                'positional',
                lambda: self.model.forward(states_t, actions_t, timesteps_t),
            ),
            (
                'positional_no_timestep+padding_mask',
                lambda: self.model.forward(states_t, actions_t, padding_mask=None),
            ),
            (
                'positional_no_timestep',
                lambda: self.model.forward(states_t, actions_t),
            ),
        ]

        kw = self._forward_kwargs_from_signature(
            states_t=states_t,
            actions_t=actions_t,
            timesteps_t=timesteps_t,
            padding_mask_t=padding_mask_t,
        )
        if len(kw) > 0:
            attempts.append(('keyword', lambda kw=kw: self.model.forward(**kw)))
            kw_no_mask = {
                k: v for k, v in kw.items() if k not in {'padding_mask', 'attention_mask'}
            }
            if len(kw_no_mask) > 0 and kw_no_mask != kw:
                attempts.append(
                    ('keyword_no_mask', lambda kw=kw_no_mask: self.model.forward(**kw))
                )

        errors: List[str] = []
        for label, fn in attempts:
            try:
                out = fn()
                self._last_forward_route = label
                self._last_forward_error = ''
                return out
            except Exception as e:
                errors.append(f'{label}: {type(e).__name__}: {e}')

        self._last_forward_route = 'failed'
        self._last_forward_error = ' | '.join(errors[:6])
        raise RuntimeError(
            f'LatentDriver forward failed across {len(attempts)} call variants. '
            f'{self._last_forward_error}'
        )

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
        # Some LatentDriver world-model versions require a non-None padding mask.
        padding_mask_t = torch.ones(
            (states_t.shape[0], states_t.shape[1]),
            dtype=torch.long,
            device=self.device,
        )

        try:
            with torch.no_grad():
                model_out = self._call_model_forward_flexible(
                    states_t=states_t,
                    actions_t=actions_t,
                    timesteps_t=timesteps_t,
                    padding_mask_t=padding_mask_t,
                )
            action_dis = self._extract_action_distribution(model_out)  # [M,7]
        except Exception as e:
            self._maybe_log_forward_error(e, states_t=states_t, actions_t=actions_t, timesteps_t=timesteps_t)
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
            'source': f'model:{self._last_forward_route}',
        }

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if hasattr(x, 'detach'):
            return np.asarray(x.detach().cpu().numpy())
        return np.asarray(x)

    def _iter_action_distribution_candidates(self, model_out: Any):
        priority_names = [
            'actions_layers',
            'action_layers',
            'actions',
            'action_distributions',
            'action_dis',
            'pred_actions',
            'action',
        ]
        queue: List[Any] = [model_out]
        seen: set[int] = set()

        while len(queue) > 0 and len(seen) < 128:
            obj = queue.pop(0)
            oid = id(obj)
            if oid in seen:
                continue
            seen.add(oid)

            if isinstance(obj, dict):
                for key in priority_names:
                    if key in obj:
                        queue.insert(0, obj[key])
                for value in obj.values():
                    queue.append(value)
                continue

            if isinstance(obj, (list, tuple)):
                for item in reversed(list(obj)):
                    queue.insert(0, item)
                continue

            for name in priority_names:
                if hasattr(obj, name):
                    try:
                        queue.insert(0, getattr(obj, name))
                    except Exception:
                        pass

            yield obj

    def _coerce_distribution_array(self, step_obj: Any) -> np.ndarray:
        arr = self._to_numpy(step_obj)
        if arr.ndim == 4:
            # [batch, time, mixture, params]
            step = arr[0, -1]
        elif arr.ndim == 3:
            # commonly [batch, time, params] or [time, mixture, params]
            if arr.shape[0] == 1:
                step = arr[0, -1]
            else:
                step = arr[-1]
        elif arr.ndim == 2:
            step = arr
        elif arr.ndim == 1:
            if arr.size % 7 != 0:
                raise ValueError(f'Unexpected action distribution shape={arr.shape}; cannot reshape to [:,7].')
            step = arr.reshape(-1, 7)
        else:
            raise ValueError(f'Unexpected action distribution tensor rank={arr.ndim}, shape={arr.shape}.')

        step = np.asarray(step, dtype=np.float32)
        if step.ndim == 1:
            step = step.reshape(1, -1)
        if step.shape[1] < 7:
            raise ValueError(f'Action distribution has too few params: shape={step.shape}, expected >=7.')
        return step[:, :7]

    def _extract_action_distribution(self, model_out: Any) -> np.ndarray:
        errors: List[str] = []
        for i, candidate in enumerate(self._iter_action_distribution_candidates(model_out)):
            try:
                return self._coerce_distribution_array(candidate)
            except Exception as e:
                if len(errors) < 8:
                    errors.append(f'candidate[{i}]={type(candidate).__name__}: {type(e).__name__}: {e}')
                continue

        detail = '; '.join(errors) if len(errors) > 0 else f'output_type={type(model_out).__name__}'
        raise ValueError(f'Unable to extract action distribution from model output. {detail}')

    def _maybe_log_forward_error(
        self,
        exc: Exception,
        states_t: Any,
        actions_t: Any,
        timesteps_t: Any,
    ) -> None:
        if not bool(getattr(self.cfg, 'latentdriver_log_forward_errors', False)):
            return
        max_logs = int(max(1, getattr(self.cfg, 'latentdriver_log_forward_errors_max', 5)))
        if self._forward_error_count >= max_logs:
            return
        self._forward_error_count += 1
        print(
            f'[LatentDriver warning] forward fallback #{self._forward_error_count}/{max_logs}: '
            f'{type(exc).__name__}: {exc}'
        )
        if self._last_forward_error:
            print('[LatentDriver warning] forward attempt errors:', self._last_forward_error)
        print('[LatentDriver warning] forward route:', self._last_forward_route)
        try:
            print(
                '[LatentDriver warning] input shapes:',
                'states', tuple(states_t.shape),
                'actions', tuple(actions_t.shape),
                'timesteps', tuple(timesteps_t.shape),
            )
        except Exception:
            pass

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

def get_latentdriver_adapter(cfg: ClosedLoopConfig) -> LatentDriverPredictiveKLAdapter:
    global _LATENTDRIVER_ADAPTER
    if _LATENTDRIVER_ADAPTER is None:
        _LATENTDRIVER_ADAPTER = LatentDriverPredictiveKLAdapter(cfg)
    return _LATENTDRIVER_ADAPTER

def make_closed_loop_components(base_state: Any, planner_kind: str, planner_name: str, cfg: ClosedLoopConfig):
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

def closed_loop_rollout_selected(
    base_state: Any,
    selected_idx: np.ndarray,
    target_obj_idx: int,
    delta_xy: np.ndarray,
    cfg: ClosedLoopConfig,
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
            'trace_pair_steps_all': 0.0,
            'trace_pair_ratio_all': 0.0,
            'trace_fallback_pair_ratio': 0.0,
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
    pair_steps_all = 0

    for i in range(n):
        dp = trace_p[i]
        dq = trace_q[i]
        if dp is None or dq is None:
            continue
        pair_steps_all += 1
        if _trace_step_is_fallback(dp) or _trace_step_is_fallback(dq):
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

    pair_steps_non_fallback = int(len(kl_vals))
    if pair_steps_non_fallback == 0:
        return {
            'trace_pair_steps': 0.0,
            'trace_pair_ratio': 0.0,
            'trace_pair_steps_all': float(pair_steps_all),
            'trace_pair_ratio_all': float(pair_steps_all / max(n, 1)),
            'trace_fallback_pair_ratio': float(pair_steps_all / max(n, 1)),
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
        'trace_pair_steps': float(pair_steps_non_fallback),
        'trace_pair_ratio': float(pair_steps_non_fallback / max(n, 1)),
        'trace_pair_steps_all': float(pair_steps_all),
        'trace_pair_ratio_all': float(pair_steps_all / max(n, 1)),
        'trace_fallback_pair_ratio': float((pair_steps_all - pair_steps_non_fallback) / max(n, 1)),
        'step_mean_l2_mean': float(np.mean(mean_l2_arr)),
        'step_mean_l2_p50': float(np.quantile(mean_l2_arr, 0.50)),
        'step_mean_l2_p95': float(np.quantile(mean_l2_arr, 0.95)),
        'step_std_l2_mean': float(np.mean(std_l2_arr)),
        'step_moment_kl_mean': float(np.mean(kl_arr)),
        'step_moment_kl_p50': float(np.quantile(kl_arr, 0.50)),
        'step_moment_kl_p95': float(np.quantile(kl_arr, 0.95)),
        'step_moment_kl_nonzero_ratio': float(np.mean(kl_arr > 1e-9)),
    }

ENV_CLASS = resolve_env_class()

_LATENTDRIVER_ADAPTER: Optional[LatentDriverPredictiveKLAdapter] = None
