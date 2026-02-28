from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import risk_kwargs_from_cfg
from .planner_backends import (
    closed_loop_rollout_action_prefix,
    dist_entropy_from_step,
    latentdriver_current_action_and_dist,
)
from src.risk_model.features import extract_candidate_risk_features
from src.risk_model.labels import label_candidate_rollout_events


def _clip_action(action: np.ndarray, cfg: Any) -> np.ndarray:
    vec = np.asarray(action, dtype=float).reshape(-1)
    clip = np.asarray(getattr(cfg, 'latentdriver_action_clip', (6.0, 0.35, 0.35)), dtype=float).reshape(-1)
    if clip.size < vec.size:
        clip = np.pad(clip, (0, vec.size - clip.size), constant_values=float(np.max(clip) if clip.size > 0 else 1.0))
    return np.clip(vec, -clip[:vec.size], clip[:vec.size]).astype(np.float32)


def sample_action_candidates_from_dist(dist_step: Dict[str, Any], cfg: Any, seed: int) -> Tuple[np.ndarray, List[str]]:
    rng = np.random.default_rng(int(seed))
    weights = np.asarray(dist_step.get('weights', [1.0]), dtype=float).reshape(-1)
    means = np.asarray(dist_step.get('means', np.zeros((1, 3))), dtype=float)
    stds = np.asarray(dist_step.get('stds', np.ones_like(means) * 0.5), dtype=float)
    if means.ndim == 1:
        means = means.reshape(1, -1)
    if stds.ndim == 1:
        stds = stds.reshape(1, -1)
    k = int(min(weights.size, means.shape[0], stds.shape[0]))
    weights = np.maximum(weights[:k], 1e-12)
    weights = weights / np.sum(weights)
    means = means[:k]
    stds = np.maximum(stds[:k], 1e-3)

    top_modes = int(max(0, getattr(cfg, 'risk_control_candidate_top_modes', 2)))
    random_samples = int(max(0, getattr(cfg, 'risk_control_candidate_random_samples', 6)))
    candidates: List[np.ndarray] = []
    sources: List[str] = []

    order = np.argsort(-weights)
    for idx in order[:top_modes]:
        candidates.append(_clip_action(means[idx], cfg))
        sources.append(f'mode_{int(idx)}')

    for sample_idx in range(random_samples):
        comp = int(rng.choice(np.arange(k, dtype=int), p=weights)) if k > 0 else 0
        sample = rng.normal(loc=means[comp], scale=stds[comp])
        candidates.append(_clip_action(sample, cfg))
        sources.append(f'sample_{sample_idx}_comp_{comp}')

    if len(candidates) == 0:
        candidates.append(_clip_action(np.zeros((means.shape[1] if means.ndim == 2 else 3,), dtype=float), cfg))
        sources.append('fallback_zero')
    return np.asarray(candidates, dtype=np.float32), sources


def build_candidate_risk_dataset_rows(
    *,
    scenario_id: int,
    state: Any,
    selected_idx: np.ndarray,
    planner_bundle: Dict[str, Any],
    cfg: Any,
    seed: int,
    step_idx: int = 0,
    shift_suite: str = 'nominal_clean',
    target_interaction_score: float = np.nan,
) -> List[Dict[str, Any]]:
    action_vec, dist_step = latentdriver_current_action_and_dist(
        state=state,
        selected_idx=selected_idx,
        planner_bundle=planner_bundle,
        cfg=cfg,
    )
    candidates, sources = sample_action_candidates_from_dist(dist_step, cfg=cfg, seed=seed)
    rows: List[Dict[str, Any]] = []
    horizon = int(max(1, getattr(cfg, 'risk_dataset_control_horizon_steps', 6)))

    for cid, (candidate, source) in enumerate(zip(candidates, sources)):
        xy, valid, _, _, dist_trace, rollout_feasible, note = closed_loop_rollout_action_prefix(
            state=state,
            selected_idx=np.asarray(selected_idx, dtype=np.int32),
            action_prefix=[candidate],
            cfg=cfg,
            planner_bundle=planner_bundle,
            seed=int(seed + cid),
            horizon_steps=horizon,
        )
        feature_dict = extract_candidate_risk_features(
            dist_step=dist_step,
            dist_trace=dist_trace,
            xy=xy,
            valid=valid,
            cfg=cfg,
            control_horizon_steps=horizon,
            target_interaction_score=target_interaction_score,
        )
        label_dict = label_candidate_rollout_events(
            xy,
            valid,
            cfg=cfg,
            horizons=getattr(cfg, 'risk_dataset_label_horizons', (5, 10, 15)),
        )
        row = {
            'scenario_id': int(scenario_id),
            'eval_split': 'train',
            'shift_suite': str(shift_suite),
            'step_idx': int(step_idx),
            'candidate_id': int(cid),
            'candidate_source': str(source),
            'planner_backend': str(getattr(cfg, 'planner_kind', 'latentdriver')),
            'seed': int(seed),
            'action_0': float(candidate[0]) if candidate.size > 0 else np.nan,
            'action_1': float(candidate[1]) if candidate.size > 1 else np.nan,
            'action_2': float(candidate[2]) if candidate.size > 2 else np.nan,
            'planner_action_0': float(action_vec[0]) if np.asarray(action_vec).size > 0 else np.nan,
            'planner_action_1': float(action_vec[1]) if np.asarray(action_vec).size > 1 else np.nan,
            'planner_action_2': float(action_vec[2]) if np.asarray(action_vec).size > 2 else np.nan,
            'rollout_feasible': int(bool(rollout_feasible)),
            'rollout_note': str(note),
            'dist_entropy_step': float(dist_entropy_from_step(dist_step)),
        }
        row.update(feature_dict)
        row.update(label_dict)
        rows.append(row)
    return rows
