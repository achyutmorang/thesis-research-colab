from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from waymax import config as waymax_config
from waymax import dataloader as waymax_dataloader

from .calibration import (
    build_calibration_diagnostics,
    calibrate_closed_loop_thresholds,
    make_calibration_delta_proposal,
    proposal_realization_ok,
    run_closedloop_preflight_checks,
)
from .config import (
    SearchConfig,
    ClosedLoopConfig,
    build_run_artifact_paths,
    required_total_scenarios,
    restore_artifacts_via_upload,
)
from .planner_backends import (
    _choose_target_non_ego,
    closed_loop_rollout_selected,
    latent_belief_kl_from_dist_trace,
    perturb_initial_state,
    dist_trace_change_stats,
    dist_trace_diagnostics,
    make_closed_loop_components,
)
from .metrics import (
    compute_counterfactual_surprise_score,
    compute_risk_metrics,
    planner_action_surprise_kl,
    risk_kwargs_from_cfg,
)
from .resume_io import (
    RESULTS_REQUIRED_COLUMNS,
    TRACE_REQUIRED_COLUMNS,
    _completed_scenarios,
    _flush_checkpoint,
    _load_existing_results,
    _write_progress_artifacts,
    validate_artifact_schema_manifest,
)
from .search import optimize_method_closed_loop

SURPRISE_COL_CANDIDATES: Tuple[str, ...] = ('delta_surprise', 'delta_surprise_pd', 'surprise_pd')


def _resolve_surprise_col(df: pd.DataFrame) -> str:
    for col in SURPRISE_COL_CANDIDATES:
        if isinstance(df, pd.DataFrame) and (col in df.columns):
            return col
    return 'delta_surprise'


def _ensure_surprise_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or len(df.columns) == 0:
        return df
    if ('delta_surprise' not in df.columns):
        if 'delta_surprise_pd' in df.columns:
            df['delta_surprise'] = df['delta_surprise_pd']
        elif 'surprise_pd' in df.columns:
            df['delta_surprise'] = df['surprise_pd']
    if ('delta_surprise_pd' not in df.columns) and ('delta_surprise' in df.columns):
        df['delta_surprise_pd'] = df['delta_surprise']
    if ('surprise_pd' not in df.columns) and ('delta_surprise' in df.columns):
        df['surprise_pd'] = df['delta_surprise']
    return df


def _validate_runtime_prereqs() -> None:
    try:
        from numpy._core.umath import _center, _expandtabs  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Runtime dependency check failed: NumPy private symbols are unavailable. "
            "In Colab, re-run deterministic setup; if it reports dependency changes, restart runtime before simulation."
        ) from e


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
        config: ClosedLoopConfig,
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

class ClosedLoopRunner:
    def __init__(
        self,
        config: ClosedLoopConfig,
        data_iter: Optional[Iterable[Any]] = None,
        dataset_config: Optional[Any] = None,
    ):
        self.cfg = config
        self.loader = WaymaxScenarioLoader(config, data_iter=data_iter, dataset_config=dataset_config)

    def build_dataset(self):
        # "train_fraction" acts as a reference/evaluation partition in this pipeline.
        n_reference = int(self.cfg.n_total_scenarios * self.cfg.train_fraction)
        n_eval_target = int(max(self.cfg.n_eval_scenarios, self.cfg.strict_min_eval))
        eval_start = n_reference
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

        n_reference = int(len(X) * self.cfg.train_fraction)
        reference_idx = np.arange(n_reference)
        candidate_eval_idx = np.arange(n_reference, len(X))

        self.data = {
            'X': X,
            'meta': meta_df,
            'scenarios': scenarios,
            'reference_idx': reference_idx,
            'candidate_eval_idx': candidate_eval_idx,
            # Backward-compatible aliases for older notebooks/scripts.
            'train_idx': reference_idx,
            'test_idx': candidate_eval_idx,
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


def _trajectory_effect_l2_mean(
    base_xy: np.ndarray,
    base_valid: np.ndarray,
    prop_xy: np.ndarray,
    prop_valid: np.ndarray,
) -> float:
    bx = np.asarray(base_xy, dtype=float)
    px = np.asarray(prop_xy, dtype=float)
    bv = np.asarray(base_valid, dtype=bool)
    pv = np.asarray(prop_valid, dtype=bool)

    if bx.ndim != 3 or px.ndim != 3:
        return 0.0

    n_obj = int(min(bx.shape[0], px.shape[0]))
    n_steps = int(min(bx.shape[1], px.shape[1]))
    if n_obj <= 0 or n_steps <= 0:
        return 0.0

    bx = bx[:n_obj, :n_steps, :2]
    px = px[:n_obj, :n_steps, :2]
    bv = bv[:n_obj, :n_steps]
    pv = pv[:n_obj, :n_steps]
    valid = bv & pv
    if not bool(np.any(valid)):
        return 0.0

    step_l2 = np.linalg.norm(bx - px, axis=-1)
    vals = step_l2[valid]
    if vals.size == 0:
        return 0.0
    return float(np.nanmean(vals))


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    n = int(arr.size)
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(arr, kind='mergesort')
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and arr[order[j]] == arr[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    n = int(min(xx.size, yy.size))
    if n < 2:
        return float('nan')
    xx = xx[:n]
    yy = yy[:n]
    m = np.isfinite(xx) & np.isfinite(yy)
    if int(np.sum(m)) < 2:
        return float('nan')
    rx = _rankdata_average(xx[m])
    ry = _rankdata_average(yy[m])
    if rx.size < 2:
        return float('nan')
    sx = float(np.std(rx))
    sy = float(np.std(ry))
    if (sx <= 1e-12) or (sy <= 1e-12):
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def _topk_jaccard(proposal_ids: np.ndarray, x: np.ndarray, y: np.ndarray, k: int) -> float:
    ids = np.asarray(proposal_ids).reshape(-1)
    xx = np.asarray(x, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    n = int(min(ids.size, xx.size, yy.size))
    if n <= 0:
        return float('nan')
    ids = ids[:n]
    xx = xx[:n]
    yy = yy[:n]
    m = np.isfinite(xx) & np.isfinite(yy)
    if int(np.sum(m)) <= 0:
        return float('nan')
    ids = ids[m]
    xx = xx[m]
    yy = yy[m]
    kk = int(max(1, min(int(k), ids.size)))
    top_x = set(ids[np.argsort(xx)[-kk:]].tolist())
    top_y = set(ids[np.argsort(yy)[-kk:]].tolist())
    union = top_x.union(top_y)
    if len(union) == 0:
        return float('nan')
    return float(len(top_x.intersection(top_y)) / len(union))


def _probe_ranking_stability(probe_df: pd.DataFrame, topk: int) -> Dict[str, float]:
    if probe_df.empty:
        return {
            'ranking_scenarios_count': 0.0,
            'rank_spearman_mean': np.nan,
            'rank_spearman_min': np.nan,
            'topk_jaccard_mean': np.nan,
            'topk_jaccard_min': np.nan,
        }
    surprise_col = _resolve_surprise_col(probe_df)
    required = {'scenario_id', 'proposal_id', 'repeat_id', surprise_col}
    if not required.issubset(set(probe_df.columns)):
        return {
            'ranking_scenarios_count': 0.0,
            'rank_spearman_mean': np.nan,
            'rank_spearman_min': np.nan,
            'topk_jaccard_mean': np.nan,
            'topk_jaccard_min': np.nan,
        }

    spearman_vals: List[float] = []
    jaccard_vals: List[float] = []
    scenario_count = 0

    base = probe_df[np.isfinite(probe_df[surprise_col].to_numpy(dtype=float))].copy()
    base = base[base['proposal_id'].astype(int) >= 0]
    for sid, grp in base.groupby('scenario_id'):
        piv = grp.pivot_table(
            index='proposal_id',
            columns='repeat_id',
            values=surprise_col,
            aggfunc='mean',
        )
        if (0 not in piv.columns) or (piv.shape[0] < 2) or (piv.shape[1] < 2):
            continue
        proposal_ids = np.asarray(piv.index.to_numpy(), dtype=int)
        x0 = piv[0].to_numpy(dtype=float)
        scenario_has_metric = False
        for rid in piv.columns:
            if int(rid) == 0:
                continue
            xr = piv[rid].to_numpy(dtype=float)
            rho = _spearman_corr(x0, xr)
            if np.isfinite(rho):
                spearman_vals.append(float(rho))
                scenario_has_metric = True
            jac = _topk_jaccard(proposal_ids, x0, xr, k=topk)
            if np.isfinite(jac):
                jaccard_vals.append(float(jac))
                scenario_has_metric = True
        if scenario_has_metric:
            scenario_count += 1

    return {
        'ranking_scenarios_count': float(scenario_count),
        'rank_spearman_mean': float(np.mean(spearman_vals)) if len(spearman_vals) > 0 else np.nan,
        'rank_spearman_min': float(np.min(spearman_vals)) if len(spearman_vals) > 0 else np.nan,
        'topk_jaccard_mean': float(np.mean(jaccard_vals)) if len(jaccard_vals) > 0 else np.nan,
        'topk_jaccard_min': float(np.min(jaccard_vals)) if len(jaccard_vals) > 0 else np.nan,
    }


def run_quick_surprise_probe(
    cfg: ClosedLoopConfig,
    search_cfg: SearchConfig,
    n_probe_scenarios: int = 8,
    proposals_per_scenario: int = 4,
    data_iter: Optional[Iterable[Any]] = None,
    dataset_config: Optional[Any] = None,
    repeat_seeds: Optional[int] = None,
    max_resample_attempts: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_probe_scenarios = int(max(1, n_probe_scenarios))
    proposals_per_scenario = int(max(1, proposals_per_scenario))
    repeat_seeds = int(max(1, repeat_seeds if repeat_seeds is not None else getattr(cfg, 'quick_probe_repeat_seeds', 3)))
    max_resample_attempts = int(max(1, max_resample_attempts if max_resample_attempts is not None else getattr(cfg, 'surprise_proposal_max_resample_attempts', 4)))
    topk = int(max(1, getattr(cfg, 'quick_probe_stability_topk', 3)))
    scenario_oversample_factor = int(max(1, getattr(cfg, 'quick_probe_scenario_oversample_factor', 6)))
    n_candidate_scenarios = int(max(n_probe_scenarios, n_probe_scenarios * scenario_oversample_factor))

    loader = WaymaxScenarioLoader(cfg, data_iter=data_iter, dataset_config=dataset_config)
    keep_state_ids = set(range(n_candidate_scenarios))
    probe_scenarios = loader.generate(
        n_scenarios=n_candidate_scenarios,
        keep_state_ids=keep_state_ids,
        show_progress=True,
    )

    rows: List[Dict[str, Any]] = []
    n_scenarios_used = 0
    n_scenarios_skipped_no_state = 0
    n_scenarios_skipped_base_infeasible = 0
    skipped_base_infeasible_notes: List[str] = []
    iterator = tqdm(probe_scenarios, desc='Quick surprise probe', total=len(probe_scenarios))
    for rec in iterator:
        if n_scenarios_used >= n_probe_scenarios:
            break

        sid = int(rec.get('scenario_id', -1))
        if 'state' not in rec:
            n_scenarios_skipped_no_state += 1
            continue

        try:
            selected_idx = np.asarray(rec['selected_indices'], dtype=np.int32)
            target_idx = _choose_target_non_ego(rec['state'], selected_idx, cfg=cfg)
            planner_bundle = make_closed_loop_components(rec['state'], cfg.planner_kind, cfg.planner_name, cfg)

            base_rollouts: List[Dict[str, Any]] = []
            for r in range(repeat_seeds):
                base_seed = int(cfg.global_seed + sid + r * 100003)
                base_xy, base_valid, base_actions, base_action_valid, base_dist_trace, base_feasible, base_note = closed_loop_rollout_selected(
                    base_state=rec['state'],
                    selected_idx=selected_idx,
                    target_obj_idx=target_idx,
                    delta_xy=np.zeros((2,), dtype=float),
                    cfg=cfg,
                    planner_bundle=planner_bundle,
                    seed=base_seed,
                )
                if planner_bundle['planner_type'] == 'latentdriver':
                    base_dist_diag = dist_trace_diagnostics(base_dist_trace)
                    base_surprise_abs_raw, _ = latent_belief_kl_from_dist_trace(
                        trace=base_dist_trace,
                        skip_fallback_steps=bool(cfg.predictive_kl_skip_fallback_steps),
                    )
                    if np.isfinite(base_surprise_abs_raw):
                        base_surprise_abs = float(base_surprise_abs_raw)
                        base_surprise_source = 'latent_belief_kl'
                    else:
                        base_surprise_abs = 0.0
                        base_surprise_source = 'base_no_belief_pairs_zero'
                else:
                    base_dist_diag = {
                        'dist_fallback_ratio': np.nan,
                        'dist_actor_fallback_ratio': np.nan,
                        'dist_source_model_ratio': np.nan,
                        'dist_source_proxy_ratio': np.nan,
                    }
                    base_surprise_abs = 0.0
                    base_surprise_source = 'base_zero_non_latentdriver'
                base_rollouts.append({
                    'base_xy': base_xy,
                    'base_valid': base_valid,
                    'base_actions': base_actions,
                    'base_action_valid': base_action_valid,
                    'base_dist_trace': base_dist_trace,
                    'base_feasible': base_feasible,
                    'base_note': str(base_note),
                    'base_dist_diag': base_dist_diag,
                    'base_surprise_abs': float(base_surprise_abs),
                    'base_surprise_source': str(base_surprise_source),
                })

            feasible_repeat_ids = [rid for rid, info in enumerate(base_rollouts) if bool(info['base_feasible'])]
            if len(feasible_repeat_ids) == 0:
                n_scenarios_skipped_base_infeasible += 1
                for info in base_rollouts:
                    note = str(info.get('base_note', '')).strip()
                    if note:
                        skipped_base_infeasible_notes.append(note)
                continue
            anchor_repeat_id = int(feasible_repeat_ids[0])
            n_scenarios_used += 1

            def _compute_surprise_for_repeat(
                p_actions: np.ndarray,
                p_action_valid: np.ndarray,
                p_dist_trace: List[Optional[Dict[str, np.ndarray]]],
                repeat_id: int,
                seed_offset: int,
            ) -> Tuple[float, float, Dict[str, float], Dict[str, float], str]:
                base_info = base_rollouts[int(repeat_id)]
                action_surprise = np.nan
                if planner_bundle['planner_type'] == 'latentdriver':
                    p_surprise_abs, belief_diag = latent_belief_kl_from_dist_trace(
                        trace=p_dist_trace,
                        skip_fallback_steps=bool(cfg.predictive_kl_skip_fallback_steps),
                    )
                    p_dist_diag = dist_trace_diagnostics(p_dist_trace)
                    p_dist_diag.update(belief_diag)
                    trace_diag = dist_trace_change_stats(p_dist_trace, base_info['base_dist_trace'])
                    surprise_source = 'latent_belief_kl'

                    if (not np.isfinite(p_surprise_abs)) or (float(p_surprise_abs) <= 1e-12):
                        action_surprise = planner_action_surprise_kl(
                            p_actions,
                            p_action_valid,
                            base_info['base_actions'],
                            base_info['base_action_valid'],
                            sigma=0.25,
                        )
                        if np.isfinite(action_surprise) and float(action_surprise) > 1e-12:
                            p_surprise_abs = float(action_surprise)
                            if float(belief_diag.get('belief_kl_step_count', 0.0)) > 0.0:
                                surprise_source = 'action_kl_fallback'
                            else:
                                surprise_source = 'action_kl_no_belief_pairs'
                        elif float(belief_diag.get('belief_kl_step_count', 0.0)) <= 0.0:
                            p_surprise_abs = np.nan
                else:
                    p_surprise_abs = planner_action_surprise_kl(
                        p_actions,
                        p_action_valid,
                        base_info['base_actions'],
                        base_info['base_action_valid'],
                        sigma=0.25,
                    )
                    action_surprise = float(p_surprise_abs) if np.isfinite(p_surprise_abs) else np.nan
                    p_dist_diag = {
                        'dist_fallback_ratio': np.nan,
                        'dist_actor_fallback_ratio': np.nan,
                        'dist_source_model_ratio': np.nan,
                        'dist_source_proxy_ratio': np.nan,
                    }
                    trace_diag = {
                        'trace_pair_ratio': np.nan,
                        'trace_pair_ratio_all': np.nan,
                        'step_mean_l2_mean': np.nan,
                        'step_mean_l2_all_mean': np.nan,
                        'step_w2_mean': np.nan,
                        'step_w2_all_mean': np.nan,
                        'step_logit_l1_mean': np.nan,
                        'step_logit_l1_all_mean': np.nan,
                    }
                    surprise_source = 'action_kl'
                return (
                    float(p_surprise_abs) if np.isfinite(p_surprise_abs) else np.nan,
                    float(action_surprise) if np.isfinite(action_surprise) else np.nan,
                    p_dist_diag,
                    trace_diag,
                    str(surprise_source),
                )

            rng = np.random.default_rng(int(cfg.global_seed + sid * 10007 + 77))
            for k in range(proposals_per_scenario):
                chosen_prop = None
                chosen_attempts = 0
                chosen_meta: Dict[str, Any] = {}
                for attempt in range(max_resample_attempts):
                    proposal_key = int(k + attempt * proposals_per_scenario)
                    prop_try, prop_meta_try = make_calibration_delta_proposal(
                        rng=rng,
                        k=proposal_key,
                        search_cfg=search_cfg,
                        base_state=rec['state'],
                        target_obj_idx=target_idx,
                        cfg=cfg,
                        return_meta=True,
                    )
                    seed_try = int(cfg.global_seed + sid + 1000 + k * 101 + attempt)
                    p_xy_try, p_valid_try, p_actions_try, p_action_valid_try, p_dist_trace_try, p_try_feasible, _ = closed_loop_rollout_selected(
                        base_state=rec['state'],
                        selected_idx=selected_idx,
                        target_obj_idx=target_idx,
                        delta_xy=prop_try,
                        cfg=cfg,
                        planner_bundle=planner_bundle,
                        seed=seed_try,
                    )
                    _, _, _, trace_diag_try, _ = _compute_surprise_for_repeat(
                        p_actions=p_actions_try,
                        p_action_valid=p_action_valid_try,
                        p_dist_trace=p_dist_trace_try,
                        repeat_id=anchor_repeat_id,
                        seed_offset=1000 + k * 101 + attempt,
                    )
                    effect_try = float(_trajectory_effect_l2_mean(
                        base_rollouts[anchor_repeat_id]['base_xy'],
                        base_rollouts[anchor_repeat_id]['base_valid'],
                        p_xy_try,
                        p_valid_try,
                    ))
                    realized_ok, realize_reason = proposal_realization_ok(
                        cfg=cfg,
                        effect_l2_mean=effect_try,
                        trace_change_diag=trace_diag_try,
                    )
                    if not bool(p_try_feasible):
                        realized_ok = False
                        realize_reason = 'rollout_infeasible'
                    chosen_prop = np.asarray(prop_try, dtype=float)
                    chosen_attempts = int(attempt + 1)
                    chosen_meta = dict(prop_meta_try) if isinstance(prop_meta_try, dict) else {}
                    if realized_ok:
                        break

                if chosen_prop is None:
                    chosen_prop = np.zeros((2,), dtype=float)
                    chosen_attempts = 0
                    chosen_meta = {}

                token_shift_l2 = np.nan
                if planner_bundle['planner_type'] == 'latentdriver':
                    try:
                        ld_adapter = planner_bundle['ld_adapter']
                        base_tok = np.asarray(ld_adapter.encode_tokens(rec['state'], selected_idx), dtype=float)
                        pert_state = perturb_initial_state(
                            base_state=rec['state'],
                            target_obj_idx=int(target_idx),
                            delta_xy=np.asarray(chosen_prop, dtype=float),
                            cfg=cfg,
                        )
                        prop_tok = np.asarray(ld_adapter.encode_tokens(pert_state, selected_idx), dtype=float)
                        if base_tok.shape == prop_tok.shape and base_tok.size > 0:
                            token_shift_l2 = float(np.linalg.norm(prop_tok - base_tok))
                    except Exception:
                        token_shift_l2 = np.nan

                for repeat_id in range(repeat_seeds):
                    seed_offset = int(1000 + k * 101 + repeat_id * 10007)
                    p_xy, p_valid, p_actions, p_action_valid, p_dist_trace, p_feasible, p_note = closed_loop_rollout_selected(
                        base_state=rec['state'],
                        selected_idx=selected_idx,
                        target_obj_idx=target_idx,
                        delta_xy=chosen_prop,
                        cfg=cfg,
                        planner_bundle=planner_bundle,
                        seed=int(cfg.global_seed + sid + seed_offset),
                    )
                    p_surprise_abs, p_action_divergence, p_dist_diag, trace_diag, surprise_source_raw = _compute_surprise_for_repeat(
                        p_actions=p_actions,
                        p_action_valid=p_action_valid,
                        p_dist_trace=p_dist_trace,
                        repeat_id=repeat_id,
                        seed_offset=seed_offset,
                    )
                    effect_l2_mean = float(_trajectory_effect_l2_mean(
                        base_rollouts[repeat_id]['base_xy'],
                        base_rollouts[repeat_id]['base_valid'],
                        p_xy,
                        p_valid,
                    ))
                    if (not bool(base_rollouts[repeat_id]['base_feasible'])) or (not bool(p_feasible)):
                        realized_ok = False
                        realize_reason = 'rollout_infeasible'
                        p_surprise = np.nan
                        p_score_diag = {}
                    else:
                        realized_ok, realize_reason = proposal_realization_ok(
                            cfg=cfg,
                            effect_l2_mean=effect_l2_mean,
                            trace_change_diag=trace_diag,
                        )
                        p_surprise, p_score_diag = compute_counterfactual_surprise_score(
                            trace_change_diag=trace_diag,
                            effect_l2_mean=float(effect_l2_mean),
                            effect_l2_budget=float(search_cfg.delta_l2_budget),
                            base_surprise_abs=float(base_rollouts[repeat_id].get('base_surprise_abs', 0.0)),
                            proposal_surprise_abs=float(p_surprise_abs) if np.isfinite(p_surprise_abs) else np.nan,
                            action_divergence=float(p_action_divergence) if np.isfinite(p_action_divergence) else np.nan,
                            metric_hint=str(getattr(cfg, 'planner_surprise_name', 'predictive_seq_w2')),
                            proposal_delta_l2=float(np.linalg.norm(chosen_prop)),
                            perturb_floor_weight=float(getattr(cfg, 'surprise_counterfactual_floor_weight', 0.02)),
                            response_floor_weight=float(getattr(cfg, 'surprise_counterfactual_response_weight', 0.35)),
                            use_additive_score=bool(getattr(cfg, 'surprise_counterfactual_use_additive_score', True)),
                        )
                    surprise_source = (
                        'counterfactual_composite:'
                        f"{p_score_diag.get('surprise_belief_source', 'none')}"
                        f"+{p_score_diag.get('surprise_policy_source', 'none')}"
                    ) if isinstance(p_score_diag, dict) else str(surprise_source_raw)

                    base_dist_diag = base_rollouts[repeat_id]['base_dist_diag']
                    rows.append({
                        'scenario_id': int(sid),
                        'proposal_id': int(k),
                        'repeat_id': int(repeat_id),
                        'base_rollout_feasible': int(base_rollouts[repeat_id]['base_feasible']),
                        'base_rollout_note': str(base_rollouts[repeat_id].get('base_note', '')),
                        'proposal_rollout_feasible': int(p_feasible),
                        'proposal_rollout_note': str(p_note),
                        'delta_x': float(chosen_prop[0]),
                        'delta_y': float(chosen_prop[1]),
                        'delta_l2': float(np.linalg.norm(chosen_prop)),
                        'delta_surprise': float(p_surprise),
                        'delta_surprise_pd': float(p_surprise),
                        'surprise_pd': float(p_surprise),
                        'base_surprise_pd': float(base_rollouts[repeat_id].get('base_surprise_abs', 0.0)),
                        'proposal_surprise_pd': float(p_surprise_abs) if np.isfinite(p_surprise_abs) else np.nan,
                        'surprise_source': str(surprise_source),
                        'base_surprise_source': str(base_rollouts[repeat_id].get('base_surprise_source', 'unknown')),
                        'proposal_effect_l2_mean': float(effect_l2_mean),
                        'proposal_realized': int(realized_ok),
                        'surprise_belief_shift': float(p_score_diag.get('surprise_belief_shift', np.nan)) if isinstance(p_score_diag, dict) else np.nan,
                        'surprise_belief_shift_raw': float(p_score_diag.get('surprise_belief_shift_raw', np.nan)) if isinstance(p_score_diag, dict) else np.nan,
                        'surprise_policy_shift': float(p_score_diag.get('surprise_policy_shift', np.nan)) if isinstance(p_score_diag, dict) else np.nan,
                        'surprise_policy_shift_raw': float(p_score_diag.get('surprise_policy_shift_raw', np.nan)) if isinstance(p_score_diag, dict) else np.nan,
                        'surprise_realization_ratio': float(p_score_diag.get('surprise_realization_ratio', np.nan)) if isinstance(p_score_diag, dict) else np.nan,
                        'surprise_realization_ratio_raw': float(p_score_diag.get('surprise_realization_ratio_raw', np.nan)) if isinstance(p_score_diag, dict) else np.nan,
                        'surprise_response_ratio': float(p_score_diag.get('surprise_response_ratio', np.nan)) if isinstance(p_score_diag, dict) else np.nan,
                        'surprise_signal_floor': float(p_score_diag.get('surprise_signal_floor', np.nan)) if isinstance(p_score_diag, dict) else np.nan,
                        'surprise_response_floor': float(p_score_diag.get('surprise_response_floor', np.nan)) if isinstance(p_score_diag, dict) else np.nan,
                        'surprise_score_mode': str(p_score_diag.get('surprise_score_mode', '')) if isinstance(p_score_diag, dict) else '',
                        'surprise_belief_term': float(p_score_diag.get('surprise_belief_term', np.nan)) if isinstance(p_score_diag, dict) else np.nan,
                        'surprise_policy_term': float(p_score_diag.get('surprise_policy_term', np.nan)) if isinstance(p_score_diag, dict) else np.nan,
                        'surprise_action_divergence': float(p_score_diag.get('surprise_action_divergence', p_action_divergence)) if isinstance(p_score_diag, dict) else float(p_action_divergence) if np.isfinite(p_action_divergence) else np.nan,
                        'proposal_attempts': int(chosen_attempts),
                        'proposal_realization_reason': str(realize_reason),
                        'proposal_kind': str(chosen_meta.get('proposal_kind', 'unknown')),
                        'proposal_primitive': str(chosen_meta.get('primitive', '')),
                        'proposal_primitive_scale': float(chosen_meta.get('primitive_scale', np.nan)),
                        'proposal_primitive_gain': float(chosen_meta.get('primitive_gain', np.nan)),
                        'token_shift_l2': float(token_shift_l2) if np.isfinite(token_shift_l2) else np.nan,
                        'base_dist_fallback_ratio': float(base_dist_diag.get('dist_fallback_ratio', np.nan)),
                        'base_dist_actor_fallback_ratio': float(base_dist_diag.get('dist_actor_fallback_ratio', np.nan)),
                        'base_dist_source_model_ratio': float(base_dist_diag.get('dist_source_model_ratio', np.nan)),
                        'base_dist_source_proxy_ratio': float(base_dist_diag.get('dist_source_proxy_ratio', np.nan)),
                        'proposal_dist_fallback_ratio': float(p_dist_diag.get('dist_fallback_ratio', np.nan)),
                        'proposal_dist_actor_fallback_ratio': float(p_dist_diag.get('dist_actor_fallback_ratio', np.nan)),
                        'proposal_dist_source_model_ratio': float(p_dist_diag.get('dist_source_model_ratio', np.nan)),
                        'proposal_dist_source_proxy_ratio': float(p_dist_diag.get('dist_source_proxy_ratio', np.nan)),
                        'trace_pair_ratio': float(trace_diag.get('trace_pair_ratio', np.nan)),
                        'trace_pair_ratio_all': float(trace_diag.get('trace_pair_ratio_all', np.nan)),
                        'step_mean_l2_mean': float(trace_diag.get('step_mean_l2_mean', np.nan)),
                        'step_mean_l2_all_mean': float(trace_diag.get('step_mean_l2_all_mean', np.nan)),
                        'step_w2_mean': float(trace_diag.get('step_w2_mean', np.nan)),
                        'step_w2_all_mean': float(trace_diag.get('step_w2_all_mean', np.nan)),
                        'step_logit_l1_mean': float(trace_diag.get('step_logit_l1_mean', np.nan)),
                        'step_logit_l1_all_mean': float(trace_diag.get('step_logit_l1_all_mean', np.nan)),
                        'probe_error': '',
                    })
        except Exception as e:
            rows.append({
                'scenario_id': int(sid),
                'proposal_id': -1,
                'repeat_id': 0,
                'delta_surprise': np.nan,
                'delta_surprise_pd': np.nan,
                'surprise_pd': np.nan,
                'base_surprise_pd': np.nan,
                'proposal_surprise_pd': np.nan,
                'surprise_source': 'probe_exception',
                'base_surprise_source': 'probe_exception',
                'proposal_effect_l2_mean': np.nan,
                'proposal_realized': 0,
                'surprise_belief_shift': np.nan,
                'surprise_belief_shift_raw': np.nan,
                'surprise_policy_shift': np.nan,
                'surprise_policy_shift_raw': np.nan,
                'surprise_realization_ratio': np.nan,
                'surprise_realization_ratio_raw': np.nan,
                'surprise_response_ratio': np.nan,
                'surprise_signal_floor': np.nan,
                'surprise_response_floor': np.nan,
                'surprise_score_mode': '',
                'surprise_belief_term': np.nan,
                'surprise_policy_term': np.nan,
                'surprise_action_divergence': np.nan,
                'proposal_attempts': 0,
                'proposal_realization_reason': 'probe_exception',
                'proposal_kind': 'probe_exception',
                'proposal_primitive': '',
                'proposal_primitive_scale': np.nan,
                'proposal_primitive_gain': np.nan,
                'token_shift_l2': np.nan,
                'proposal_dist_fallback_ratio': np.nan,
                'proposal_dist_actor_fallback_ratio': np.nan,
                'proposal_dist_source_model_ratio': np.nan,
                'proposal_dist_source_proxy_ratio': np.nan,
                'trace_pair_ratio': np.nan,
                'trace_pair_ratio_all': np.nan,
                'step_mean_l2_mean': np.nan,
                'step_mean_l2_all_mean': np.nan,
                'step_w2_mean': np.nan,
                'step_w2_all_mean': np.nan,
                'step_logit_l1_mean': np.nan,
                'step_logit_l1_all_mean': np.nan,
                'proposal_rollout_feasible': 0,
                'probe_error': str(e),
            })

    if n_scenarios_used < n_probe_scenarios:
        print(
            f"[probe] used {n_scenarios_used}/{n_probe_scenarios} scenarios with feasible base rollouts. "
            f"candidates_loaded={len(probe_scenarios)}, "
            f"skipped_no_state={n_scenarios_skipped_no_state}, "
            f"skipped_base_infeasible={n_scenarios_skipped_base_infeasible}."
        )

    probe_df = _ensure_surprise_alias_columns(pd.DataFrame(rows))

    skipped_base_note_counts = (
        pd.Series(skipped_base_infeasible_notes, dtype='object')
        .fillna('')
        .astype(str)
        .str.strip()
    )
    skipped_base_note_counts = skipped_base_note_counts[skipped_base_note_counts.str.len() > 0]
    skipped_base_note_counts_dict = skipped_base_note_counts.value_counts().to_dict() if len(skipped_base_note_counts) else {}

    def _reason_counts(df: pd.DataFrame, feasible_col: str, note_col: str) -> Dict[str, int]:
        if feasible_col not in df.columns or note_col not in df.columns or df.empty:
            return {}
        mask = (df[feasible_col].fillna(0).astype(int) == 0)
        notes = df.loc[mask, note_col].fillna('').astype(str).str.strip()
        notes = notes[notes.str.len() > 0]
        if len(notes) == 0:
            return {}
        return notes.value_counts().to_dict()

    def _safe_col_nanmean(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns:
            return np.nan
        arr = df[col].to_numpy(dtype=float)
        if arr.size == 0:
            return np.nan
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        return float(np.mean(arr))

    if probe_df.empty:
        summary_df = pd.DataFrame([{
            'n_rows': 0,
            'n_scenarios': 0,
            'n_candidate_scenarios_loaded': int(len(probe_scenarios)),
            'n_scenarios_used': int(n_scenarios_used),
            'n_scenarios_skipped_no_state': int(n_scenarios_skipped_no_state),
            'n_scenarios_skipped_base_infeasible': int(n_scenarios_skipped_base_infeasible),
            'skipped_base_infeasible_reason_counts': json.dumps(skipped_base_note_counts_dict),
            'n_finite_surprise': 0,
            'finite_surprise_rate': 0.0,
            'nonzero_surprise_fraction': 0.0,
            'surprise_std': np.nan,
            'proposal_effect_l2_mean': np.nan,
            'proposal_fallback_ratio_mean': np.nan,
            'proposal_actor_fallback_ratio_mean': np.nan,
            'proposal_model_source_ratio_mean': np.nan,
            'proposal_proxy_source_ratio_mean': np.nan,
            'trace_pair_ratio_mean': np.nan,
            'trace_pair_ratio_all_mean': np.nan,
            'step_mean_l2_all_mean': np.nan,
            'step_w2_all_mean': np.nan,
            'step_logit_l1_all_mean': np.nan,
            'surprise_belief_shift_mean': np.nan,
            'surprise_policy_shift_mean': np.nan,
            'surprise_realization_ratio_mean': np.nan,
            'token_shift_l2_mean': np.nan,
            'token_shift_nonzero_fraction': np.nan,
            'proposal_realized_fraction': np.nan,
            'proposal_attempts_mean': np.nan,
            'base_infeasible_reason_counts': '{}',
            'proposal_infeasible_reason_counts': '{}',
            'ranking_scenarios_count': 0.0,
            'rank_spearman_mean': np.nan,
            'rank_spearman_min': np.nan,
            'topk_jaccard_mean': np.nan,
            'topk_jaccard_min': np.nan,
            'n_probe_errors': 0,
            'surprise_source_counts': '{}',
        }])
        return probe_df, summary_df

    surprise_col = _resolve_surprise_col(probe_df)
    finite_mask = np.isfinite(probe_df[surprise_col].to_numpy(dtype=float))
    finite_df = probe_df[finite_mask].copy()
    source_counts = (
        probe_df['surprise_source'].fillna('unknown').astype(str).value_counts(dropna=False).to_dict()
        if 'surprise_source' in probe_df.columns
        else {}
    )
    stability = _probe_ranking_stability(probe_df=probe_df, topk=topk)
    summary_df = pd.DataFrame([{
        'n_rows': int(len(probe_df)),
        'n_scenarios': int(probe_df['scenario_id'].nunique()) if 'scenario_id' in probe_df else 0,
        'n_candidate_scenarios_loaded': int(len(probe_scenarios)),
        'n_scenarios_used': int(n_scenarios_used),
        'n_scenarios_skipped_no_state': int(n_scenarios_skipped_no_state),
        'n_scenarios_skipped_base_infeasible': int(n_scenarios_skipped_base_infeasible),
        'skipped_base_infeasible_reason_counts': json.dumps(skipped_base_note_counts_dict),
        'n_finite_surprise': int(np.sum(finite_mask)),
        'finite_surprise_rate': float(np.mean(finite_mask)),
        'nonzero_surprise_fraction': float(np.mean(finite_df[surprise_col].to_numpy(dtype=float) > 1e-9)) if len(finite_df) > 0 else 0.0,
        'surprise_std': float(np.nanstd(finite_df[surprise_col].to_numpy(dtype=float))) if len(finite_df) > 1 else np.nan,
        'proposal_effect_l2_mean': _safe_col_nanmean(probe_df, 'proposal_effect_l2_mean'),
        'proposal_fallback_ratio_mean': _safe_col_nanmean(probe_df, 'proposal_dist_fallback_ratio'),
        'proposal_actor_fallback_ratio_mean': _safe_col_nanmean(probe_df, 'proposal_dist_actor_fallback_ratio'),
        'proposal_model_source_ratio_mean': _safe_col_nanmean(probe_df, 'proposal_dist_source_model_ratio'),
        'proposal_proxy_source_ratio_mean': _safe_col_nanmean(probe_df, 'proposal_dist_source_proxy_ratio'),
        'trace_pair_ratio_mean': _safe_col_nanmean(probe_df, 'trace_pair_ratio'),
        'trace_pair_ratio_all_mean': _safe_col_nanmean(probe_df, 'trace_pair_ratio_all'),
        'step_mean_l2_all_mean': _safe_col_nanmean(probe_df, 'step_mean_l2_all_mean'),
        'step_w2_all_mean': _safe_col_nanmean(probe_df, 'step_w2_all_mean'),
        'step_logit_l1_all_mean': _safe_col_nanmean(probe_df, 'step_logit_l1_all_mean'),
        'surprise_belief_shift_mean': _safe_col_nanmean(probe_df, 'surprise_belief_shift'),
        'surprise_policy_shift_mean': _safe_col_nanmean(probe_df, 'surprise_policy_shift'),
        'surprise_realization_ratio_mean': _safe_col_nanmean(probe_df, 'surprise_realization_ratio'),
        'token_shift_l2_mean': _safe_col_nanmean(probe_df, 'token_shift_l2'),
        'token_shift_nonzero_fraction': float(np.nanmean(np.asarray(probe_df.get('token_shift_l2', np.nan), dtype=float) > 1e-8)) if len(probe_df) > 0 else np.nan,
        'proposal_realized_fraction': _safe_col_nanmean(probe_df, 'proposal_realized'),
        'proposal_attempts_mean': _safe_col_nanmean(probe_df, 'proposal_attempts'),
        'base_infeasible_reason_counts': json.dumps(_reason_counts(probe_df, 'base_rollout_feasible', 'base_rollout_note')),
        'proposal_infeasible_reason_counts': json.dumps(_reason_counts(probe_df, 'proposal_rollout_feasible', 'proposal_rollout_note')),
        'ranking_scenarios_count': float(stability.get('ranking_scenarios_count', 0.0)),
        'rank_spearman_mean': float(stability.get('rank_spearman_mean', np.nan)),
        'rank_spearman_min': float(stability.get('rank_spearman_min', np.nan)),
        'topk_jaccard_mean': float(stability.get('topk_jaccard_mean', np.nan)),
        'topk_jaccard_min': float(stability.get('topk_jaccard_min', np.nan)),
        'n_probe_errors': int(np.sum(probe_df.get('probe_error', '').astype(str).str.len() > 0)),
        'surprise_source_counts': json.dumps(source_counts),
    }])
    try:
        token_shift_nonzero = float(summary_df['token_shift_nonzero_fraction'].iloc[0])
        step_w2_mean = float(summary_df['step_w2_all_mean'].iloc[0])
        if np.isfinite(token_shift_nonzero) and token_shift_nonzero > 0.0:
            if (not np.isfinite(step_w2_mean)) or (step_w2_mean <= 1e-8):
                print(
                    '[probe] warning: planner input changed, but predictive divergence remained ~0. '
                    'This suggests planner invariance under current perturbation family.'
                )
    except Exception:
        pass
    return probe_df, summary_df

def run_closed_loop(
    runner: ClosedLoopRunner,
    eval_idx: np.ndarray,
    cfg: ClosedLoopConfig,
    search_cfg: SearchConfig,
    thresholds: Dict[str, float],
    run_prefix: Optional[str] = None,
    static_frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    methods = ['random', 'risk_only', 'surprise_only', 'joint']

    run_prefix = run_prefix or cfg.run_prefix
    checkpoint_path = f'{run_prefix}_per_scenario_results.csv'
    trace_checkpoint_path = f'{run_prefix}_per_eval_trace.csv'

    if bool(cfg.resume_from_existing):
        validate_artifact_schema_manifest(run_prefix, strict=True)

    existing_df = (
        _load_existing_results(
            checkpoint_path,
            required_cols=RESULTS_REQUIRED_COLUMNS,
            artifact_name='per_scenario_results',
        )
        if bool(cfg.resume_from_existing)
        else pd.DataFrame()
    )
    existing_trace_df = (
        _load_existing_results(
            trace_checkpoint_path,
            required_cols=TRACE_REQUIRED_COLUMNS,
            artifact_name='per_eval_trace',
        )
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
        iterator = tqdm(chunk, desc=f'Closed-Loop chunk {chunk_id} ({len(chunk)} scenarios)', total=len(chunk))

        for sid in iterator:
            rec = runner.data['scenarios'][sid]

            if 'state' not in rec:
                skipped += 1
                continue

            try:
                selected_idx = np.asarray(rec['selected_indices'], dtype=np.int32)
                target_idx = _choose_target_non_ego(rec['state'], selected_idx, cfg=cfg)
                planner_bundle = make_closed_loop_components(rec['state'], cfg.planner_kind, cfg.planner_name, cfg)

                # Common random numbers across methods:
                # same proposal bank + same rollout seed schedule for all methods in this scenario.
                scenario_seed = int(cfg.global_seed + sid * 7919)
                scenario_rng = np.random.default_rng(scenario_seed)
                n_props = int(max(0, search_cfg.budget_evals - 1))
                if n_props > 0:
                    proposal_dirs: List[np.ndarray] = []
                    proposal_meta_bank: List[Dict[str, Any]] = []
                    for k in range(n_props):
                        prop_raw, prop_meta = make_calibration_delta_proposal(
                                rng=scenario_rng,
                                k=k,
                                search_cfg=search_cfg,
                                base_state=rec['state'],
                                target_obj_idx=target_idx,
                                cfg=cfg,
                                return_meta=True,
                            )
                        prop = np.asarray(prop_raw, dtype=float).reshape(-1)
                        if prop.size < 2:
                            prop = np.asarray([1.0, 0.0], dtype=float)
                        norm = float(np.linalg.norm(prop[:2]))
                        if norm < 1e-12:
                            direction = np.asarray([1.0, 0.0], dtype=np.float32)
                        else:
                            direction = (prop[:2] / norm).astype(np.float32)
                        proposal_dirs.append(direction)
                        meta = dict(prop_meta) if isinstance(prop_meta, dict) else {}
                        meta.setdefault('counterfactual_family', str(getattr(cfg, 'counterfactual_family', '')))
                        proposal_meta_bank.append(meta)
                    proposal_bank = np.asarray(proposal_dirs, dtype=np.float32)
                else:
                    proposal_bank = np.zeros((0, 2), dtype=np.float32)
                    proposal_meta_bank = []
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
                        proposal_meta_bank=proposal_meta_bank,
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
                    'delta_surprise': np.nan,
                    'delta_surprise_pd': np.nan,
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

def make_waymax_data_iter(cfg: ClosedLoopConfig):
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

def build_closedloop_runner_and_splits(
    cfg: ClosedLoopConfig,
    data_iter: Optional[Iterable[Any]],
    dataset_config: Optional[Any],
    n_shards: int,
    shard_id: int,
):
    runner = ClosedLoopRunner(cfg, data_iter=data_iter, dataset_config=dataset_config)
    data = runner.build_dataset()

    reference_idx = data.get('reference_idx', data['train_idx'])
    candidate_eval_idx = data.get('candidate_eval_idx', data['test_idx'])

    required_eval = int(max(cfg.n_eval_scenarios, cfg.strict_min_eval))
    if len(candidate_eval_idx) < required_eval:
        required_total = required_total_scenarios(required_eval, cfg.train_fraction)
        raise ValueError(
            f'Not enough evaluation candidates for strict evaluation: have {len(candidate_eval_idx)}, need {required_eval}. '
            f'Current n_total_scenarios={cfg.n_total_scenarios}, reference_fraction(train_fraction)={cfg.train_fraction}. '
            f'Set n_total_scenarios >= {required_total} (or reduce strict_min_eval / n_eval_scenarios).'
        )

    eval_idx_all = candidate_eval_idx[:cfg.n_eval_scenarios]
    if int(max(1, n_shards)) > 1:
        eval_idx = eval_idx_all[int(shard_id)::int(n_shards)]
    else:
        eval_idx = eval_idx_all

    if len(eval_idx) == 0:
        raise ValueError(f'Empty shard eval set for shard_id={shard_id}, n_shards={n_shards}.')

    reference_df = runner.score_indices_openloop(reference_idx, label='reference_openloop', show_progress=True)
    base_eval_openloop_df = runner.score_indices_openloop(eval_idx, label='base_eval_openloop', show_progress=True)

    return runner, data, reference_idx, candidate_eval_idx, eval_idx_all, eval_idx, reference_df, base_eval_openloop_df

def run_preflight_and_calibration(
    runner: ClosedLoopRunner,
    cfg: ClosedLoopConfig,
    search_cfg: SearchConfig,
    eval_idx: np.ndarray,
    reference_df: pd.DataFrame,
    restore_from_upload: bool = False,
):
    _validate_runtime_prereqs()

    if bool(restore_from_upload):
        _ = restore_artifacts_via_upload(
            cfg.run_prefix,
            required_keys=['per_scenario_results', 'per_eval_trace', 'thresholds', 'closedloop_calibration'],
        )

    artifact_paths = build_run_artifact_paths(cfg.run_prefix)
    thresholds_path_resume = artifact_paths['thresholds']
    closedloop_calib_path_resume = artifact_paths['closedloop_calibration']

    preflight_df = run_closedloop_preflight_checks(runner, cfg, eval_idx)
    if bool(cfg.require_preflight_pass) and (not preflight_df.empty) and (not bool(preflight_df['pass'].all())):
        failed = preflight_df[~preflight_df['pass']]
        raise RuntimeError(
            'Closed-Loop preflight failed. Fix these checks before running calibration/main loop:\n'
            + failed.to_string(index=False)
        )

    if Path(thresholds_path_resume).exists():
        with open(thresholds_path_resume, 'r') as f:
            closedloop_thresholds = json.load(f)
        if Path(closedloop_calib_path_resume).exists():
            closedloop_calib_df = pd.read_csv(closedloop_calib_path_resume)
        else:
            closedloop_calib_df = pd.DataFrame()
        calib_diag_df, calib_quant_df = build_calibration_diagnostics(closedloop_calib_df, closedloop_thresholds)

        loaded_surprise_scale = float(closedloop_thresholds.get('surprise_scale', np.nan))
        loaded_surprise_thr = float(closedloop_thresholds.get('surprise_high_threshold', np.nan))
        force_recalib = (
            (not np.isfinite(loaded_surprise_scale))
            or (loaded_surprise_scale <= float(search_cfg.min_scale) * 1.01)
            or (np.isfinite(loaded_surprise_thr) and loaded_surprise_thr <= 0.0)
        )
        if force_recalib:
            print('[resume] existing thresholds look degenerate; recalibrating closed-loop surprise.')
            closedloop_calib_df, closedloop_thresholds = calibrate_closed_loop_thresholds(
                runner,
                eval_idx,
                cfg,
                search_cfg,
                reference_df=reference_df,
            )
            calib_diag_df, calib_quant_df = build_calibration_diagnostics(closedloop_calib_df, closedloop_thresholds)
    else:
        closedloop_calib_df, closedloop_thresholds = calibrate_closed_loop_thresholds(
            runner,
            eval_idx,
            cfg,
            search_cfg,
            reference_df=reference_df,
        )
        calib_diag_df, calib_quant_df = build_calibration_diagnostics(closedloop_calib_df, closedloop_thresholds)

    return preflight_df, closedloop_calib_df, closedloop_thresholds, calib_diag_df, calib_quant_df
