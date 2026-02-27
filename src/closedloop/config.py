from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ClosedLoopConfig:
    global_seed: int = 17

    # Dataset scale (pilot defaults; scale up after first successful run)
    n_total_scenarios: int = 2400
    n_eval_scenarios: int = 200
    strict_min_eval: int = 200
    # Historical field name retained for compatibility; semantically this is reference_fraction.
    train_fraction: float = 0.75

    # Trajectory slicing
    n_agents: int = 8
    history_steps: int = 10
    future_steps: int = 15

    # WOMD / Waymax source
    waymax_path: str = (
        'gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/'
        'tf_example/training/training_tfexample.tfrecord@1000'
    )
    waymax_max_rg_points: int = 20000
    waymax_batch_dims: Tuple[int, ...] = ()

    # Risk metric controls
    collision_distance: float = 1.5
    risk_w_dist: float = 1.0
    risk_w_ttc: float = 0.5
    risk_w_sks: float = 0.01

    # Failure proxy thresholds
    ttc_fail_seconds: float = 2.0
    no_hazard_ttc_seconds: float = 3.0
    no_hazard_dist_m: float = 8.0
    hard_brake_mps2: float = 6.0
    hard_jerk_mps3: float = 8.0
    enable_intervention_proxy: bool = True

    # Closed-loop planner settings
    planner_kind: str = 'latentdriver'  # 'latentdriver' | 'smart' | 'idm_route'
    planner_name: str = 'latentdriver_waypoint_sdc'

    # Planner-dependent surprise settings
    planner_surprise_name: str = 'predictive_kl'  # predictive_kl | predictive_w2 | action_kl
    predictive_kl_estimator: str = 'mixture_mc'  # 'mixture_mc' or 'moment_match'
    predictive_kl_mc_samples: int = 192
    predictive_kl_mc_seed: int = 12345
    predictive_kl_eps: float = 1e-6
    predictive_kl_symmetric: bool = True
    predictive_kl_skip_fallback_steps: bool = True

    # LatentDriver integration
    latentdriver_repo_path: str = '/content/LatentDriver'
    latentdriver_method_name: str = 'latentdriver'
    latentdriver_ckpt_path: str = '/content/checkpoints/lantentdriver_t2_J3.ckpt'
    latentdriver_context_len: int = 2
    latentdriver_action_type: str = 'waypoint'  # waypoint (dx,dy,dyaw) or bicycle
    latentdriver_action_clip: Tuple[float, float, float] = (6.0, 0.35, 0.35)
    latentdriver_yaw_sigma: float = 0.15
    latentdriver_log_std_clip: Tuple[float, float] = (-1.609, 5.0)
    latentdriver_log_forward_errors: bool = True
    latentdriver_log_forward_errors_max: int = 5
    latentdriver_auto_align_token_count: bool = True
    latentdriver_expected_token_count: int = 0
    latentdriver_preflight_max_fallback_ratio: float = 0.95
    latentdriver_use_all_vehicle_tokens: bool = True
    latentdriver_vehicle_token_cap: int = 128
    latentdriver_encode_in_ego_frame: bool = True
    latentdriver_encode_yaw_degrees: bool = True

    # SMART integration (predictive belief backend)
    # Current implementation supports a robust proxy mode in the same runtime.
    # Use strict mode only when full SMART model runtime wiring is available.
    smart_mode: str = 'proxy'  # proxy | strict
    smart_repo_path: str = '/content/SMART'
    smart_ckpt_path: str = ''
    smart_control_actor: str = 'idm_route'  # idm_route | expert
    smart_action_dt_seconds: float = 0.1
    smart_base_std_xy: float = 0.35
    smart_base_std_yaw: float = 0.12
    smart_interaction_dist_scale_m: float = 8.0
    smart_interaction_closing_speed_scale_mps: float = 6.0
    # Perturbation injection controls:
    # apply at current simulator timestep and optionally persist for a short window.
    perturb_from_current_timestep: bool = True
    perturb_persist_steps: int = 3
    # Target-selection policy for perturbations.
    perturb_target_selection_mode: str = 'highest_interaction'  # highest_interaction | nearest | first_valid
    perturb_target_top_k: int = 2
    perturb_interaction_ttc_horizon_s: float = 6.0
    perturb_interaction_w_proximity: float = 1.0
    perturb_interaction_w_ttc: float = 1.25
    perturb_interaction_w_closing_speed: float = 0.35
    perturb_interaction_w_heading_conflict: float = 0.35
    # Proposal generation policy.
    perturb_use_behavioral_proposals: bool = True
    perturb_behavioral_primitive_cycle: Tuple[str, ...] = (
        'toward_ego',
        'away_from_ego',
        'target_brake',
        'target_accel',
        'lateral_left',
        'lateral_right',
        'diag_toward_left',
        'diag_toward_right',
    )
    perturb_behavioral_longitudinal_gain: float = 1.05
    perturb_behavioral_lateral_gain: float = 1.20
    perturb_behavioral_interaction_gain: float = 1.25
    perturb_behavioral_toward_ego_blend: float = 0.65

    # Calibration from closed-loop base rollouts
    n_closedloop_calib: int = 120
    n_surprise_calib_proposals: int = 6
    surprise_min_effect_l2_mean: float = 0.05
    surprise_realization_min_logit_l1_all_mean: float = 1e-3
    surprise_realization_min_mean_l2_all_mean: float = 1e-2
    surprise_realization_min_moment_kl_all_mean: float = 1e-4
    surprise_proposal_max_resample_attempts: int = 4
    sensitivity_scan_max_scenarios: int = 20
    sensitivity_scan_num_angles: int = 8
    sensitivity_scan_scales: Tuple[float, float, float] = (0.45, 0.9, 1.2)
    quick_probe_scenario_oversample_factor: int = 6
    quick_probe_repeat_seeds: int = 3
    quick_probe_stability_topk: int = 3
    high_quantile: float = 0.80

    # Run controls: fairness, chunking, resume
    run_prefix: str = 'closedloop_run'
    run_chunk_size: int = 200
    checkpoint_every_scenarios: int = 25
    resume_from_existing: bool = True
    save_per_eval_trace: bool = True
    rollout_seed_stride: int = 10000
    require_preflight_pass: bool = True

    # Keep raw simulator states for eval scenarios
    keep_raw_state: bool = True


@dataclass
class SearchConfig:
    # Fair query budget per method per scenario (includes initial evaluation)
    budget_evals: int = 15

    # Stochastic hill-climb hyperparameters
    step_scale_init: float = 0.35
    step_scale_decay: float = 0.75

    # Feasibility projection on 2D delta (meters)
    delta_clip: float = 1.2
    delta_l2_budget: float = 1.5

    # Objective regularization
    reg_lambda: float = 1e-3

    # Random baseline proposal scale
    random_scale: float = 0.35
    proposal_scale_ladder: Tuple[float, float, float, float] = (0.45, 0.75, 1.05, 1.35)
    proposal_jitter_sigma: float = 0.12

    # Objective scales floor
    min_scale: float = 1e-6

    # Method weights
    w_risk_only: Tuple[float, float] = (1.0, 0.0)
    w_surprise_only: Tuple[float, float] = (0.0, 1.0)
    w_joint: Tuple[float, float] = (1.0, 1.0)


def required_total_scenarios(min_eval: int, train_fraction: float) -> int:
    eval_fraction = max(1e-9, 1.0 - float(train_fraction))
    return int(math.ceil(float(min_eval) / eval_fraction))


def align_dataset_scale(cfg: ClosedLoopConfig) -> ClosedLoopConfig:
    required_eval = int(max(cfg.n_eval_scenarios, cfg.strict_min_eval))
    required_total = required_total_scenarios(required_eval, cfg.train_fraction)
    if cfg.n_total_scenarios < required_total:
        old = cfg.n_total_scenarios
        cfg.n_total_scenarios = required_total
        print(
            f"[config auto-fix] n_total_scenarios increased from {old} to {cfg.n_total_scenarios} "
            f"to support eval target={required_eval} with reference_fraction(train_fraction)={cfg.train_fraction:.2f}."
        )
    return cfg


def scan_latentdriver_checkpoints(search_roots: Optional[List[str]] = None) -> pd.DataFrame:
    roots = search_roots or [
        '/content/checkpoints',
        '/content/LatentDriver/checkpoints',
        '/content',
    ]
    seen = set()
    rows: List[Dict[str, Any]] = []

    preferred_names = {
        'lantentdriver_t2_j3.ckpt': 100,
        'lantentdriver_t2_j4.ckpt': 95,
        'latentdriver_t2_j3.ckpt': 90,
        'latentdriver_t2_j4.ckpt': 85,
        'plant.ckpt': 50,
    }

    for root_rank, root in enumerate(roots):
        rp = Path(root)
        if not rp.exists():
            continue
        for fp in rp.rglob('*'):
            if not fp.is_file():
                continue
            name_l = fp.name.lower()
            if not (name_l.endswith('.ckpt') or name_l.endswith('.pth.tar')):
                continue
            key = str(fp.resolve())
            if key in seen:
                continue
            seen.add(key)

            st = fp.stat()
            rows.append({
                'path': key,
                'name': fp.name,
                'size_mb': float(st.st_size / (1024 ** 2)),
                'mtime_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(st.st_mtime)),
                'root': root,
                'root_rank': int(root_rank),
                'name_score': int(preferred_names.get(name_l, 0)),
            })

    if len(rows) == 0:
        return pd.DataFrame(columns=['path', 'name', 'size_mb', 'mtime_utc', 'root', 'root_rank', 'name_score', 'score'])

    df = pd.DataFrame(rows)
    df['score'] = (df['name_score'] * 1000.0) + (100.0 - df['root_rank']) + np.clip(df['size_mb'], 0.0, 5000.0) / 10000.0
    df = df.sort_values(['score', 'size_mb'], ascending=[False, False]).reset_index(drop=True)
    return df


def resolve_latentdriver_checkpoint(cfg: ClosedLoopConfig) -> Tuple[ClosedLoopConfig, pd.DataFrame]:
    if cfg.planner_kind != 'latentdriver':
        return cfg, pd.DataFrame()

    configured = Path(cfg.latentdriver_ckpt_path)
    scan_df = scan_latentdriver_checkpoints()

    if configured.exists():
        print(f'[ckpt] using configured checkpoint: {configured}')
        return cfg, scan_df

    if len(scan_df) == 0:
        print('[ckpt] no checkpoint found under /content.')
        print(f'[ckpt] expected path missing: {cfg.latentdriver_ckpt_path}')
        return cfg, scan_df

    best = str(scan_df.iloc[0]['path'])
    old = cfg.latentdriver_ckpt_path
    cfg.latentdriver_ckpt_path = best
    print(f'[ckpt] configured checkpoint missing: {old}')
    print(f'[ckpt] auto-selected checkpoint: {best}')
    return cfg, scan_df


def _normalize_planner_kind(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    kind = str(value).strip().lower()
    if kind in {'', 'auto'}:
        return None
    if kind not in {'latentdriver', 'smart', 'idm_route'}:
        raise ValueError(f'Unsupported planner_kind={value!r}. Use one of: latentdriver, smart, idm_route.')
    return kind


def initialize_configs(planner_kind_override: Optional[str] = None) -> Tuple[ClosedLoopConfig, SearchConfig, pd.DataFrame]:
    cfg = align_dataset_scale(ClosedLoopConfig())
    planner_kind = _normalize_planner_kind(planner_kind_override)
    if planner_kind is not None:
        cfg.planner_kind = planner_kind
        if planner_kind == 'smart':
            cfg.planner_name = 'smart_predictive_proxy_sdc'
        elif planner_kind == 'idm_route':
            cfg.planner_name = 'idm_route'
        else:
            cfg.planner_name = 'latentdriver_waypoint_sdc'

    search_cfg = SearchConfig()
    cfg, scan_df = resolve_latentdriver_checkpoint(cfg)

    np.random.seed(cfg.global_seed)
    random.seed(cfg.global_seed)

    if cfg.planner_kind == 'latentdriver':
        print('[ckpt] final cfg.latentdriver_ckpt_path =', cfg.latentdriver_ckpt_path)
    else:
        print(f'[planner] planner_kind={cfg.planner_kind}; latentdriver checkpoint scan skipped.')
    return cfg, search_cfg, scan_df


def configure_persistent_run_prefix(
    cfg: ClosedLoopConfig,
    run_tag: str,
    persist_root: str,
    shard_id: int,
    n_shards: int,
) -> str:
    n_shards = int(max(1, n_shards))
    shard_id = int(shard_id)
    if shard_id < 0 or shard_id >= n_shards:
        raise ValueError(f'Invalid shard_id={shard_id} for n_shards={n_shards}.')

    run_name = str(run_tag)
    if n_shards > 1:
        run_name = f'{run_name}_shard{shard_id:02d}of{n_shards:02d}'

    root = Path(str(persist_root)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    cfg.run_prefix = str(root / run_name)
    return cfg.run_prefix


def _shard_run_name(run_tag: str, shard_id: int, n_shards: int) -> str:
    n_shards = int(max(1, n_shards))
    shard_id = int(shard_id)
    if shard_id < 0 or shard_id >= n_shards:
        raise ValueError(f'Invalid shard_id={shard_id} for n_shards={n_shards}.')
    if n_shards == 1:
        return str(run_tag)
    return f'{run_tag}_shard{shard_id:02d}of{n_shards:02d}'


def shard_run_prefix(run_tag: str, persist_root: str, shard_id: int, n_shards: int) -> str:
    root = Path(str(persist_root)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return str(root / _shard_run_name(run_tag, shard_id, n_shards))


def _completed_scenarios_from_results(path: Path, methods: List[str]) -> Tuple[int, int, int]:
    if not path.exists():
        return 0, 0, 0
    try:
        df = pd.read_csv(path)
    except Exception:
        return 0, 0, 0
    if df.empty:
        return 0, 0, 0
    if ('scenario_id' not in df.columns) or ('method' not in df.columns):
        return int(len(df)), 0, 0

    sub = df[df['method'].isin(methods)].copy()
    if sub.empty:
        return int(len(df)), 0, int(df['scenario_id'].nunique())

    counts = sub.groupby('scenario_id')['method'].nunique()
    completed = int((counts >= len(methods)).sum())
    touched = int(sub['scenario_id'].nunique())
    return int(len(df)), completed, touched


def inspect_shard_progress(
    run_tag: str,
    persist_root: str,
    n_shards: int,
    methods: Optional[List[str]] = None,
) -> pd.DataFrame:
    methods = methods or ['random', 'risk_only', 'surprise_only', 'joint']
    n_shards = int(max(1, n_shards))
    rows: List[Dict[str, Any]] = []
    for sid in range(n_shards):
        run_prefix = shard_run_prefix(run_tag, persist_root, sid, n_shards)
        results_path = Path(f'{run_prefix}_per_scenario_results.csv')
        n_rows, n_completed, n_touched = _completed_scenarios_from_results(results_path, methods)
        if not results_path.exists():
            status = 'missing'
        elif n_completed == 0 and n_touched == 0:
            status = 'empty_or_invalid'
        else:
            status = 'in_progress_or_completed'
        rows.append({
            'shard_id': int(sid),
            'run_prefix': run_prefix,
            'results_exists': int(results_path.exists()),
            'n_rows': int(n_rows),
            'n_touched_scenarios': int(n_touched),
            'n_completed_scenarios': int(n_completed),
            'status': status,
        })

    return pd.DataFrame(rows).sort_values('shard_id').reset_index(drop=True)


def auto_select_shard_id(
    run_tag: str,
    persist_root: str,
    n_shards: int,
    methods: Optional[List[str]] = None,
) -> int:
    progress = inspect_shard_progress(
        run_tag=run_tag,
        persist_root=persist_root,
        n_shards=n_shards,
        methods=methods,
    )
    if progress.empty:
        return 0

    missing = progress[progress['results_exists'] == 0]
    if len(missing) > 0:
        return int(missing.iloc[0]['shard_id'])

    # Resume the least-complete shard first when all shard files already exist.
    ranked = progress.sort_values(
        ['n_completed_scenarios', 'n_touched_scenarios', 'n_rows', 'shard_id'],
        ascending=[True, True, True, True],
    )
    return int(ranked.iloc[0]['shard_id'])


def build_run_artifact_paths(run_prefix: str) -> Dict[str, str]:
    return {
        'per_scenario_results': f'{run_prefix}_per_scenario_results.csv',
        'per_eval_trace': f'{run_prefix}_per_eval_trace.csv',
        'thresholds': f'{run_prefix}_thresholds.json',
        'closedloop_calibration': f'{run_prefix}_closedloop_calibration.csv',
        'calibration_diagnostics': f'{run_prefix}_calibration_diagnostics.csv',
        'calibration_quantiles': f'{run_prefix}_calibration_quantiles.csv',
    }


def restore_artifacts_via_upload(run_prefix: str, required_keys: Optional[List[str]] = None) -> Dict[str, str]:
    paths = build_run_artifact_paths(run_prefix)
    keys = required_keys or list(paths.keys())
    missing = [k for k in keys if not Path(paths[k]).exists()]
    if len(missing) == 0:
        print('[resume-upload] all requested artifacts already exist.')
        return paths

    try:
        from google.colab import files
    except Exception:
        print('[resume-upload] google.colab.files not available in this environment.')
        print('[resume-upload] missing:', missing)
        return paths

    import shutil

    expected = {Path(paths[k]).name: paths[k] for k in missing}
    print('[resume-upload] upload any of these exact filenames:')
    for name in expected:
        print(' -', name)

    uploaded = files.upload()
    for src_name in list(uploaded.keys()):
        src = Path(src_name)
        matched_dst = None
        if src.name in expected:
            matched_dst = expected[src.name]
        else:
            for exp_name, dst_path in expected.items():
                if src.name.endswith(exp_name):
                    matched_dst = dst_path
                    break

        if matched_dst is not None:
            dst = Path(matched_dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f'[resume-upload] restored: {dst}')
        else:
            print(f'[resume-upload] ignored unmatched file: {src.name}')

    still_missing = [k for k in keys if not Path(paths[k]).exists()]
    if still_missing:
        print('[resume-upload] still missing:', still_missing)
    else:
        print('[resume-upload] all requested artifacts restored.')
    return paths
