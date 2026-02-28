from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.closedloop.calibration import (
    diagnose_surprise_root_cause,
    run_closedloop_preflight_checks,
    run_surprise_quality_gate,
)
from src.closedloop.config import (
    ClosedLoopConfig,
    SearchConfig,
    auto_select_shard_id,
    configure_persistent_run_prefix,
    initialize_configs,
    inspect_shard_progress,
)
from src.closedloop.core import (
    build_closedloop_runner_and_splits,
    make_waymax_data_iter,
    run_closed_loop,
    run_preflight_and_calibration,
    run_quick_surprise_probe,
)
from src.closedloop.resume_io import export_closedloop_artifacts, summarize_method_outputs
from src.closedloop.signal_analysis import analyze_surprise_signal_usefulness


def _float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _normalize_planner_backend(value: Any) -> str:
    backend = str(value).strip().lower()
    if backend != "latentdriver":
        raise ValueError(f"Unsupported planner backend={value!r}. Only latentdriver is supported.")
    return backend


def configure_experiment_profile(
    cfg: ClosedLoopConfig,
    planner_backend: str = "latentdriver",
    planner_surprise_name: str = "latent_belief_kl",
) -> pd.DataFrame:
    backend = _normalize_planner_backend(planner_backend)
    cfg.planner_kind = backend
    cfg.planner_name = "latentdriver_waypoint_sdc"
    cfg.latentdriver_use_all_vehicle_tokens = True
    cfg.latentdriver_vehicle_token_cap = 128
    cfg.latentdriver_encode_in_ego_frame = True
    cfg.latentdriver_encode_yaw_degrees = True

    # Shared perturbation defaults used across notebook experiments.
    cfg.perturb_use_behavioral_proposals = True
    cfg.perturb_target_selection_mode = "highest_interaction"
    cfg.perturb_target_top_k = 2
    cfg.perturb_interaction_ttc_horizon_s = 6.0
    cfg.perturb_interaction_w_proximity = 1.0
    cfg.perturb_interaction_w_ttc = 1.25
    cfg.perturb_interaction_w_closing_speed = 0.35
    cfg.perturb_interaction_w_heading_conflict = 0.35
    cfg.perturb_behavioral_primitive_cycle = (
        "toward_ego",
        "away_from_ego",
        "target_brake",
        "target_accel",
        "lateral_left",
        "lateral_right",
        "diag_toward_left",
        "diag_toward_right",
    )
    cfg.perturb_behavioral_longitudinal_gain = 1.05
    cfg.perturb_behavioral_lateral_gain = 1.20
    cfg.perturb_behavioral_interaction_gain = 1.25
    cfg.perturb_behavioral_toward_ego_blend = 0.65
    cfg.planner_surprise_name = str(planner_surprise_name).strip().lower()

    rows = [
        {"group": "planner", "key": "backend", "value": cfg.planner_kind},
        {"group": "planner", "key": "planner_name", "value": cfg.planner_name},
        {"group": "planner", "key": "surprise_metric", "value": cfg.planner_surprise_name},
        {"group": "perturb", "key": "target_selection_mode", "value": cfg.perturb_target_selection_mode},
        {"group": "perturb", "key": "behavioral_primitives", "value": ",".join(cfg.perturb_behavioral_primitive_cycle)},
    ]
    if cfg.planner_kind == "latentdriver":
        rows.extend([
            {"group": "latentdriver", "key": "use_all_vehicle_tokens", "value": bool(cfg.latentdriver_use_all_vehicle_tokens)},
            {"group": "latentdriver", "key": "vehicle_token_cap", "value": int(cfg.latentdriver_vehicle_token_cap)},
            {"group": "latentdriver", "key": "encode_in_ego_frame", "value": bool(cfg.latentdriver_encode_in_ego_frame)},
            {"group": "latentdriver", "key": "encode_yaw_degrees", "value": bool(cfg.latentdriver_encode_yaw_degrees)},
        ])
    return pd.DataFrame(rows)


@dataclass
class QuickProbeBundle:
    cfg: ClosedLoopConfig
    search_cfg: SearchConfig
    quick_probe_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    quick_probe_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    quick_probe_attempts_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    quick_probe_metric_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    quick_probe_feasibility_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    final_collapsed: bool = True
    signal_feasible: bool = False
    signal_failure_reasons: str = ""
    selected_surprise_metric: str = ""
    selected_belief_source_mode: str = "auto"
    applied_tuning: bool = False
    dataset_config: Any = None
    data_iter: Optional[Iterable[Any]] = None
    runner: Any = None
    data: Any = None
    reference_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    candidate_eval_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    eval_idx_all: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    eval_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    reference_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    base_eval_openloop_df: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class RunContextBundle:
    cfg: ClosedLoopConfig
    search_cfg: SearchConfig
    ckpt_scan_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    shard_progress_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    run_tag_candidates_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    run_plan_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    config_drift_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    run_prefix: str = ""
    requested_run_tag: str = ""
    run_tag: str = ""
    run_tag_selection_source: str = "provided"
    run_mode_requested: str = "auto"
    run_mode_inferred: str = "fresh"
    run_mode_applied: str = "fresh"
    auto_generated_run_tag: bool = False
    adopted_existing_run_tag: bool = False
    has_existing_progress: bool = False
    existing_results_files: int = 0
    total_touched_scenarios: int = 0
    total_completed_scenarios: int = 0
    persist_root: str = ""
    n_shards: int = 1
    shard_id: int = 0
    auto_run_main_loop_when_ready: bool = True
    run_main_loop_override: Optional[bool] = None
    planner_backend: str = ""
    planner_profile_df: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class SimulationContextBundle:
    dataset_config: Any = None
    data_iter: Optional[Iterable[Any]] = None
    runner: Any = None
    data: Any = None
    reference_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    candidate_eval_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    eval_idx_all: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    eval_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    reference_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    base_eval_openloop_df: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class PreflightBundle:
    ready_for_main_loop: bool = False
    preflight_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    closedloop_calib_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    closedloop_thresholds: Dict[str, Any] = field(default_factory=dict)
    calib_diag_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    calib_quant_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    root_cause_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    root_cause_findings_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    preflight_error: str = ""
    root_cause_diag_error: str = ""


@dataclass
class SurpriseGateBundle:
    ready_for_optimization: bool = False
    gate_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    dist_change_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    gate_error: str = ""
    gate_skipped_reason: str = ""


@dataclass
class MainLoopBundle:
    should_run_main_loop: bool = False
    policy: str = ""
    closedloop_results_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    closedloop_trace_df: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class ExportBundle:
    quick_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    sanity_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    fairness_checks_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    trace_diag_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    artifact_paths: Dict[str, str] = field(default_factory=dict)


@dataclass
class SignalBundle:
    signal_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    signal_method_corr_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    signal_bin_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    signal_topk_lift_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    signal_within_scenario_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def _sanitize_run_tag_prefix(prefix: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(prefix).strip().lower())
    out = out.strip("_")
    return out or "closedloop"


def _auto_generate_run_tag(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{_sanitize_run_tag_prefix(prefix)}_{stamp}"


def _format_utc_from_epoch(ts: float) -> str:
    try:
        if not np.isfinite(float(ts)):
            return ""
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""


def _discover_run_tag_candidates(
    persist_root: str,
    run_tag_prefix: str,
    n_shards: int,
) -> pd.DataFrame:
    root = Path(str(persist_root)).expanduser()
    if not root.exists():
        return pd.DataFrame()

    prefix = _sanitize_run_tag_prefix(run_tag_prefix)
    pattern = re.compile(r"^(?P<tag>.+)_shard(?P<sid>\d{2})of(?P<n>\d{2})$")
    suffix = "_per_scenario_results.csv"
    tags = set()

    for path in root.glob(f"*{suffix}"):
        name = path.name
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        m = pattern.match(stem)
        if m:
            try:
                n = int(m.group("n"))
            except Exception:
                continue
            if int(n) != int(max(1, n_shards)):
                continue
            tag = str(m.group("tag"))
        else:
            if int(max(1, n_shards)) != 1:
                continue
            tag = str(stem)

        if not tag:
            continue
        if not (tag == prefix or tag.startswith(prefix + "_")):
            continue
        tags.add(tag)

    rows = []
    for tag in sorted(tags):
        progress = inspect_shard_progress(
            run_tag=str(tag),
            persist_root=str(persist_root),
            n_shards=int(max(1, n_shards)),
        )
        if progress.empty:
            continue

        mtimes = []
        for run_prefix in progress.get("run_prefix", pd.Series(dtype=object)).astype(str).tolist():
            p = Path(f"{run_prefix}_per_scenario_results.csv")
            if p.exists():
                try:
                    mtimes.append(float(p.stat().st_mtime))
                except Exception:
                    pass
        latest_mtime = max(mtimes) if len(mtimes) > 0 else float("nan")
        rows.append({
            "run_tag": str(tag),
            "existing_results_files": int(progress.get("results_exists", pd.Series(dtype=int)).sum()),
            "total_touched_scenarios": int(progress.get("n_touched_scenarios", pd.Series(dtype=int)).sum()),
            "total_completed_scenarios": int(progress.get("n_completed_scenarios", pd.Series(dtype=int)).sum()),
            "total_rows": int(progress.get("n_rows", pd.Series(dtype=int)).sum()),
            "latest_results_mtime_epoch": float(latest_mtime) if np.isfinite(latest_mtime) else np.nan,
            "latest_results_mtime_utc": _format_utc_from_epoch(latest_mtime),
        })

    if len(rows) == 0:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(
        ["latest_results_mtime_epoch", "total_rows", "run_tag"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return out


def _normalize_resume_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in {"auto", "fresh", "resume"}:
        raise ValueError(f"Invalid resume_mode={value!r}. Expected one of: auto, fresh, resume.")
    return mode


def _normalize_value_for_compare(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_value_for_compare(v) for v in value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if not np.isfinite(v):
            return str(v)
        return float(v)
    return value


def _values_differ(current: Any, previous: Any, atol: float = 1e-9) -> bool:
    c = _normalize_value_for_compare(current)
    p = _normalize_value_for_compare(previous)

    if isinstance(c, float) and isinstance(p, float):
        return bool(abs(c - p) > float(atol))
    return bool(c != p)


def _safe_load_json(path: Path) -> Dict[str, Any]:
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _build_config_drift_df(cfg: ClosedLoopConfig, search_cfg: SearchConfig, run_prefix: str) -> pd.DataFrame:
    carry_path = Path(f"{run_prefix}_carry_forward_config.json")
    if not carry_path.exists():
        return pd.DataFrame()

    carry_cfg = _safe_load_json(carry_path)
    if not carry_cfg:
        return pd.DataFrame([{
            "field": "carry_forward_config",
            "current_value": "<unavailable>",
            "previous_value": str(carry_path),
            "severity": "info",
            "note": "Existing carry-forward config is unreadable; drift check skipped.",
        }])

    planner_prev = carry_cfg.get("planner", {}) if isinstance(carry_cfg.get("planner"), dict) else {}
    surprise_prev = carry_cfg.get("surprise_definition", {}) if isinstance(carry_cfg.get("surprise_definition"), dict) else {}
    run_prev = carry_cfg.get("run_controls", {}) if isinstance(carry_cfg.get("run_controls"), dict) else {}
    split_prev = carry_cfg.get("data_split", {}) if isinstance(carry_cfg.get("data_split"), dict) else {}
    opt_prev = carry_cfg.get("optimization", {}) if isinstance(carry_cfg.get("optimization"), dict) else {}

    checks = [
        ("planner_kind", cfg.planner_kind, planner_prev.get("planner_kind")),
        ("planner_name", cfg.planner_name, planner_prev.get("planner_name_config")),
        ("planner_surprise_name", cfg.planner_surprise_name, surprise_prev.get("name")),
        ("predictive_kl_estimator", cfg.predictive_kl_estimator, surprise_prev.get("predictive_kl_estimator")),
        ("predictive_kl_mc_samples", int(cfg.predictive_kl_mc_samples), surprise_prev.get("predictive_kl_mc_samples")),
        ("n_total_scenarios", int(cfg.n_total_scenarios), split_prev.get("n_total_scenarios")),
        ("run_chunk_size", int(cfg.run_chunk_size), run_prev.get("run_chunk_size")),
        ("checkpoint_every_scenarios", int(cfg.checkpoint_every_scenarios), run_prev.get("checkpoint_every_scenarios")),
        ("search.budget_evals", int(search_cfg.budget_evals), opt_prev.get("budget_evals")),
        ("search.random_scale", float(search_cfg.random_scale), opt_prev.get("random_scale")),
        ("search.delta_l2_budget", float(search_cfg.delta_l2_budget), opt_prev.get("delta_l2_budget")),
        ("search.delta_clip", float(search_cfg.delta_clip), opt_prev.get("delta_clip")),
        (
            "search.proposal_scale_ladder",
            tuple(float(x) for x in tuple(search_cfg.proposal_scale_ladder)),
            tuple(float(x) for x in opt_prev.get("proposal_scale_ladder", [])) if opt_prev.get("proposal_scale_ladder") is not None else None,
        ),
    ]

    rows = []
    for field_name, current_value, previous_value in checks:
        if previous_value is None:
            continue
        if _values_differ(current_value, previous_value):
            rows.append({
                "field": str(field_name),
                "current_value": str(current_value),
                "previous_value": str(previous_value),
                "severity": "warn",
                "note": "Config differs from existing carry-forward settings for this run prefix.",
            })
    return pd.DataFrame(rows)


def initialize_run_context(
    run_tag: str,
    persist_root: str,
    n_shards: int,
    shard_id: int | str,
    auto_run_main_loop_when_ready: bool = True,
    run_main_loop_override: Optional[bool] = None,
    n_eval_scenarios: int = 100,
    strict_min_eval: int = 100,
    n_total_scenarios_floor: int = 400,
    latentdriver_auto_align_token_count: bool = True,
    latentdriver_log_forward_errors: bool = True,
    latentdriver_log_forward_errors_max: int = 10,
    run_tag_prefix: str = "closedloop",
    planner_backend: str = "latentdriver",
    planner_surprise_name: str = "latent_belief_kl",
    auto_generate_run_tag_if_empty: bool = True,
    resume_mode: str = "auto",
    warn_on_config_drift: bool = True,
) -> RunContextBundle:
    planner_backend = _normalize_planner_backend(planner_backend)
    cfg, search_cfg, ckpt_scan_df = initialize_configs(planner_kind_override=planner_backend)

    # Fast iteration defaults for Colab loops.
    cfg.n_eval_scenarios = int(max(1, n_eval_scenarios))
    cfg.strict_min_eval = int(max(1, strict_min_eval))
    cfg.n_total_scenarios = int(max(cfg.n_total_scenarios, n_total_scenarios_floor))

    # LatentDriver stability defaults for notebook runs.
    cfg.latentdriver_auto_align_token_count = bool(latentdriver_auto_align_token_count)
    cfg.latentdriver_log_forward_errors = bool(latentdriver_log_forward_errors)
    cfg.latentdriver_log_forward_errors_max = int(max(1, latentdriver_log_forward_errors_max))
    planner_profile_df = configure_experiment_profile(
        cfg=cfg,
        planner_backend=planner_backend,
        planner_surprise_name=planner_surprise_name,
    )

    run_mode_requested = _normalize_resume_mode(resume_mode)
    requested_run_tag = str(run_tag).strip()
    run_tag_candidates_df = pd.DataFrame()
    run_tag_selection_source = "provided"
    auto_generated_run_tag = False
    adopted_existing_run_tag = False
    if not requested_run_tag:
        if run_mode_requested in {"auto", "resume"}:
            run_tag_candidates_df = _discover_run_tag_candidates(
                persist_root=str(persist_root),
                run_tag_prefix=str(run_tag_prefix),
                n_shards=int(max(1, n_shards)),
            )
            if len(run_tag_candidates_df) > 0:
                requested_run_tag = str(run_tag_candidates_df.iloc[0]["run_tag"])
                adopted_existing_run_tag = True
                run_tag_selection_source = "adopt_existing"

        if not requested_run_tag:
            if not bool(auto_generate_run_tag_if_empty):
                raise ValueError("RUN_TAG is empty. Set RUN_TAG or enable auto_generate_run_tag_if_empty.")
            requested_run_tag = _auto_generate_run_tag(prefix=run_tag_prefix)
            auto_generated_run_tag = True
            run_tag_selection_source = "auto_generated"

    n_shards = int(max(1, n_shards))
    shard_progress_df = inspect_shard_progress(
        run_tag=str(requested_run_tag),
        persist_root=str(persist_root),
        n_shards=n_shards,
    )
    existing_results_files = int(shard_progress_df.get("results_exists", pd.Series(dtype=int)).sum()) if len(shard_progress_df) else 0
    total_touched_scenarios = int(shard_progress_df.get("n_touched_scenarios", pd.Series(dtype=int)).sum()) if len(shard_progress_df) else 0
    total_completed_scenarios = int(shard_progress_df.get("n_completed_scenarios", pd.Series(dtype=int)).sum()) if len(shard_progress_df) else 0
    has_existing_progress = bool(
        (existing_results_files > 0)
        or (total_touched_scenarios > 0)
        or (total_completed_scenarios > 0)
    )
    run_mode_inferred = "resume" if has_existing_progress else "fresh"
    run_mode_applied = run_mode_inferred if run_mode_requested == "auto" else run_mode_requested
    cfg.resume_from_existing = bool(run_mode_applied == "resume")

    if isinstance(shard_id, str) and shard_id.strip().lower() == "auto":
        shard_id_int = auto_select_shard_id(
            run_tag=str(requested_run_tag),
            persist_root=str(persist_root),
            n_shards=n_shards,
        )
    else:
        shard_id_int = int(shard_id)

    # Carry run identity into cfg so downstream exporters/manifests can include
    # run-level metadata without changing every call signature.
    setattr(cfg, 'run_tag', str(requested_run_tag))
    setattr(cfg, 'run_tag_prefix', str(run_tag_prefix))
    setattr(cfg, 'persist_root', str(persist_root))
    setattr(cfg, 'n_shards', int(n_shards))
    setattr(cfg, 'shard_id', int(shard_id_int))
    setattr(cfg, 'run_mode_applied', str(run_mode_applied))

    run_prefix = configure_persistent_run_prefix(
        cfg=cfg,
        run_tag=str(requested_run_tag),
        persist_root=str(persist_root),
        shard_id=int(shard_id_int),
        n_shards=n_shards,
    )

    config_drift_df = pd.DataFrame()
    if bool(warn_on_config_drift) and bool(run_mode_applied == "resume"):
        config_drift_df = _build_config_drift_df(
            cfg=cfg,
            search_cfg=search_cfg,
            run_prefix=str(run_prefix),
        )

    run_plan_df = pd.DataFrame([{
        "run_tag": str(requested_run_tag),
        "requested_run_tag": str(run_tag),
        "run_tag_selection_source": str(run_tag_selection_source),
        "auto_generated_run_tag": int(bool(auto_generated_run_tag)),
        "adopted_existing_run_tag": int(bool(adopted_existing_run_tag)),
        "run_mode_requested": str(run_mode_requested),
        "run_mode_inferred": str(run_mode_inferred),
        "run_mode_applied": str(run_mode_applied),
        "resume_from_existing": int(bool(cfg.resume_from_existing)),
        "n_shards": int(n_shards),
        "shard_id": int(shard_id_int),
        "existing_results_files": int(existing_results_files),
        "total_touched_scenarios": int(total_touched_scenarios),
        "total_completed_scenarios": int(total_completed_scenarios),
        "has_existing_progress": int(bool(has_existing_progress)),
    }])

    return RunContextBundle(
        cfg=cfg,
        search_cfg=search_cfg,
        ckpt_scan_df=ckpt_scan_df,
        shard_progress_df=shard_progress_df,
        run_tag_candidates_df=run_tag_candidates_df,
        run_plan_df=run_plan_df,
        config_drift_df=config_drift_df,
        run_prefix=str(run_prefix),
        requested_run_tag=str(run_tag),
        run_tag=str(requested_run_tag),
        run_tag_selection_source=str(run_tag_selection_source),
        run_mode_requested=str(run_mode_requested),
        run_mode_inferred=str(run_mode_inferred),
        run_mode_applied=str(run_mode_applied),
        auto_generated_run_tag=bool(auto_generated_run_tag),
        adopted_existing_run_tag=bool(adopted_existing_run_tag),
        has_existing_progress=bool(has_existing_progress),
        existing_results_files=int(existing_results_files),
        total_touched_scenarios=int(total_touched_scenarios),
        total_completed_scenarios=int(total_completed_scenarios),
        persist_root=str(persist_root),
        n_shards=int(n_shards),
        shard_id=int(shard_id_int),
        auto_run_main_loop_when_ready=bool(auto_run_main_loop_when_ready),
        run_main_loop_override=run_main_loop_override,
        planner_backend=str(planner_backend),
        planner_profile_df=planner_profile_df,
    )


def _probe_collapse_stats(
    probe_summary_df: pd.DataFrame,
    min_nonzero_surprise_fraction: float = 0.01,
    min_realized_fraction: float = 0.10,
    min_effect_l2_mean: float = 0.05,
    min_belief_shift_mean: float = 1e-8,
    min_policy_shift_mean: float = 1e-8,
    min_realization_ratio_mean: float = 0.05,
    require_belief_and_policy: bool = True,
    min_raw_belief_shift_mean: float = 1e-8,
    min_raw_policy_shift_mean: float = 1e-8,
    min_raw_fraction_of_total: float = 0.10,
    require_raw_signal: bool = True,
) -> Tuple[bool, Dict[str, float], List[str], pd.DataFrame]:
    if len(probe_summary_df) == 0:
        metrics = {
            "n_finite_surprise": 0.0,
            "nonzero_surprise_fraction": 0.0,
            "proposal_realized_fraction": 0.0,
            "proposal_effect_l2_mean": 0.0,
            "surprise_belief_shift_mean": 0.0,
            "surprise_belief_shift_raw_mean": 0.0,
            "surprise_policy_shift_mean": 0.0,
            "surprise_policy_shift_raw_mean": 0.0,
            "surprise_realization_ratio_mean": 0.0,
            "surprise_realization_ratio_raw_mean": 0.0,
            "surprise_belief_raw_fraction": 0.0,
            "surprise_policy_raw_fraction": 0.0,
        }
        reasons = ["quick_probe_summary_empty"]
    else:
        row = probe_summary_df.iloc[0]
        metrics = {
            "n_finite_surprise": float(max(0.0, _float_or_default(row.get("n_finite_surprise"), 0.0))),
            "nonzero_surprise_fraction": _float_or_default(row.get("nonzero_surprise_fraction"), 0.0),
            "proposal_realized_fraction": _float_or_default(row.get("proposal_realized_fraction"), 0.0),
            "proposal_effect_l2_mean": _float_or_default(row.get("proposal_effect_l2_mean"), 0.0),
            "surprise_belief_shift_mean": _float_or_default(row.get("surprise_belief_shift_mean"), 0.0),
            "surprise_belief_shift_raw_mean": _float_or_default(row.get("surprise_belief_shift_raw_mean"), 0.0),
            "surprise_policy_shift_mean": _float_or_default(row.get("surprise_policy_shift_mean"), 0.0),
            "surprise_policy_shift_raw_mean": _float_or_default(row.get("surprise_policy_shift_raw_mean"), 0.0),
            "surprise_realization_ratio_mean": _float_or_default(row.get("surprise_realization_ratio_mean"), 0.0),
            "surprise_realization_ratio_raw_mean": _float_or_default(row.get("surprise_realization_ratio_raw_mean"), 0.0),
        }
        belief_total = max(1e-12, float(metrics["surprise_belief_shift_mean"]))
        policy_total = max(1e-12, float(metrics["surprise_policy_shift_mean"]))
        metrics["surprise_belief_raw_fraction"] = float(np.clip(float(metrics["surprise_belief_shift_raw_mean"]) / belief_total, 0.0, 1.0))
        metrics["surprise_policy_raw_fraction"] = float(np.clip(float(metrics["surprise_policy_shift_raw_mean"]) / policy_total, 0.0, 1.0))
        reasons = []

    checks = [
        ("finite_surprise_rows", metrics["n_finite_surprise"] > 0.0, "n_finite_surprise<=0"),
        (
            "nonzero_surprise_fraction",
            metrics["nonzero_surprise_fraction"] >= float(min_nonzero_surprise_fraction),
            f"nonzero_surprise_fraction<{float(min_nonzero_surprise_fraction):.4f}",
        ),
        (
            "proposal_realized_fraction",
            metrics["proposal_realized_fraction"] >= float(min_realized_fraction),
            f"proposal_realized_fraction<{float(min_realized_fraction):.4f}",
        ),
        (
            "proposal_effect_l2_mean",
            metrics["proposal_effect_l2_mean"] >= float(min_effect_l2_mean),
            f"proposal_effect_l2_mean<{float(min_effect_l2_mean):.4f}",
        ),
        (
            "surprise_realization_ratio_mean",
            metrics["surprise_realization_ratio_mean"] >= float(min_realization_ratio_mean),
            f"surprise_realization_ratio_mean<{float(min_realization_ratio_mean):.4f}",
        ),
    ]

    belief_ok = bool(metrics["surprise_belief_shift_mean"] >= float(min_belief_shift_mean))
    policy_ok = bool(metrics["surprise_policy_shift_mean"] >= float(min_policy_shift_mean))
    if bool(require_belief_and_policy):
        bp_ok = bool(belief_ok and policy_ok)
        bp_reason = (
            f"belief_or_policy_shift_too_small(belief<{float(min_belief_shift_mean):.2e},"
            f"policy<{float(min_policy_shift_mean):.2e})"
        )
    else:
        bp_ok = bool(belief_ok or policy_ok)
        bp_reason = (
            f"belief_and_policy_shift_too_small(belief<{float(min_belief_shift_mean):.2e},"
            f"policy<{float(min_policy_shift_mean):.2e})"
        )
    checks.append(("belief_policy_shift", bp_ok, bp_reason))

    raw_belief_ok = bool(metrics["surprise_belief_shift_raw_mean"] >= float(min_raw_belief_shift_mean))
    raw_policy_ok = bool(metrics["surprise_policy_shift_raw_mean"] >= float(min_raw_policy_shift_mean))
    raw_belief_frac_ok = bool(metrics["surprise_belief_raw_fraction"] >= float(min_raw_fraction_of_total))
    raw_policy_frac_ok = bool(metrics["surprise_policy_raw_fraction"] >= float(min_raw_fraction_of_total))
    if bool(require_belief_and_policy):
        raw_mag_ok = bool(raw_belief_ok and raw_policy_ok)
        raw_frac_ok = bool(raw_belief_frac_ok and raw_policy_frac_ok)
        raw_mag_reason = (
            f"raw_belief_or_policy_shift_too_small(raw_belief<{float(min_raw_belief_shift_mean):.2e},"
            f"raw_policy<{float(min_raw_policy_shift_mean):.2e})"
        )
        raw_frac_reason = (
            f"raw_signal_floor_dominated(raw_belief_frac<{float(min_raw_fraction_of_total):.2f},"
            f"raw_policy_frac<{float(min_raw_fraction_of_total):.2f})"
        )
    else:
        raw_mag_ok = bool(raw_belief_ok or raw_policy_ok)
        raw_frac_ok = bool(raw_belief_frac_ok or raw_policy_frac_ok)
        raw_mag_reason = (
            f"raw_belief_and_policy_shift_too_small(raw_belief<{float(min_raw_belief_shift_mean):.2e},"
            f"raw_policy<{float(min_raw_policy_shift_mean):.2e})"
        )
        raw_frac_reason = (
            f"raw_signal_floor_dominated_both(raw_belief_frac<{float(min_raw_fraction_of_total):.2f},"
            f"raw_policy_frac<{float(min_raw_fraction_of_total):.2f})"
        )
    if bool(require_raw_signal):
        checks.append(("raw_belief_policy_shift", raw_mag_ok, raw_mag_reason))
        checks.append(("raw_signal_fraction", raw_frac_ok, raw_frac_reason))

    for _, ok, reason in checks:
        if not bool(ok):
            reasons.append(str(reason))

    collapsed = bool(len(reasons) > 0)
    feasibility_df = pd.DataFrame(
        [
            {"check": str(name), "pass": bool(ok), "detail": str(reason_if_fail)}
            for name, ok, reason_if_fail in checks
        ]
    )
    return collapsed, metrics, reasons, feasibility_df


def _normalize_probe_metrics(metrics: Optional[Sequence[str]], fallback: str) -> List[str]:
    raw = list(metrics) if metrics is not None else [fallback]
    out: List[str] = []
    seen = set()
    for m in raw:
        key = str(m).strip().lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    if len(out) <= 0:
        out = [str(fallback).strip().lower() or "predictive_seq_w2"]
    return out


def _normalize_probe_belief_modes(modes: Optional[Sequence[str]], fallback: str) -> List[str]:
    raw = list(modes) if modes is not None else [fallback]
    out: List[str] = []
    seen = set()
    for m in raw:
        key = str(m).strip().lower()
        if key in {"", "default"}:
            key = "auto"
        alias = {
            "step_moment_kl_all_mean": "b1",
            "step_moment_kl_mean": "b2",
            "rollout_belief_delta": "b3",
            "b1_only": "b1",
            "b2_only": "b2",
            "b3_only": "b3",
        }
        key = alias.get(key, key)
        if key not in {"auto", "b1", "b2", "b3"}:
            key = "auto"
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    if len(out) <= 0:
        out = ["auto"]
    return out


def _probe_feasibility_score(metrics: Dict[str, float]) -> float:
    nonzero = max(0.0, float(metrics.get("nonzero_surprise_fraction", 0.0)))
    realized = max(0.0, float(metrics.get("proposal_realized_fraction", 0.0)))
    effect = max(0.0, float(metrics.get("proposal_effect_l2_mean", 0.0)))
    belief = max(0.0, float(metrics.get("surprise_belief_shift_mean", 0.0)))
    policy = max(0.0, float(metrics.get("surprise_policy_shift_mean", 0.0)))
    ratio = max(0.0, float(metrics.get("surprise_realization_ratio_mean", 0.0)))
    raw_belief = max(0.0, float(metrics.get("surprise_belief_shift_raw_mean", 0.0)))
    raw_policy = max(0.0, float(metrics.get("surprise_policy_shift_raw_mean", 0.0)))
    belief_total = max(1e-12, belief)
    policy_total = max(1e-12, policy)
    raw_frac = 0.5 * (
        np.clip(raw_belief / belief_total, 0.0, 1.0)
        + np.clip(raw_policy / policy_total, 0.0, 1.0)
    )
    raw_mag = np.log1p(raw_belief) + np.log1p(raw_policy)
    return float(
        nonzero
        * realized
        * np.log1p(effect)
        * np.log1p(belief)
        * np.log1p(policy)
        * max(ratio, 1e-12)
        * max(raw_frac, 1e-12)
        * max(raw_mag, 1e-12)
    )


def build_full_simulation_context(
    cfg: ClosedLoopConfig,
    n_shards: int,
    shard_id: int,
) -> SimulationContextBundle:
    dataset_config, data_iter = make_waymax_data_iter(cfg)
    (
        runner,
        data,
        reference_idx,
        candidate_eval_idx,
        eval_idx_all,
        eval_idx,
        reference_df,
        base_eval_openloop_df,
    ) = build_closedloop_runner_and_splits(
        cfg=cfg,
        data_iter=data_iter,
        dataset_config=dataset_config,
        n_shards=int(n_shards),
        shard_id=int(shard_id),
    )
    return SimulationContextBundle(
        dataset_config=dataset_config,
        data_iter=data_iter,
        runner=runner,
        data=data,
        reference_idx=np.asarray(reference_idx),
        candidate_eval_idx=np.asarray(candidate_eval_idx),
        eval_idx_all=np.asarray(eval_idx_all),
        eval_idx=np.asarray(eval_idx),
        reference_df=reference_df,
        base_eval_openloop_df=base_eval_openloop_df,
    )


def run_quick_probe_with_auto_escalation(
    cfg: ClosedLoopConfig,
    search_cfg: SearchConfig,
    n_shards: int,
    shard_id: int,
    run_quick_surprise_probe_enabled: bool = True,
    quick_probe_scenarios: int = 1,
    quick_probe_proposals_per_scenario: int = 4,
    stop_if_quick_probe_collapsed: bool = False,
    auto_escalate_quick_probe: bool = True,
    max_probe_escalations: int = 3,
    probe_scale_multipliers: Sequence[float] = (1.0, 1.35, 1.8),
    probe_delta_l2_multipliers: Sequence[float] = (1.0, 1.2, 1.4),
    probe_delta_clip_multipliers: Sequence[float] = (1.0, 1.1, 1.2),
    probe_persist_step_multipliers: Sequence[float] = (1.0, 2.0, 3.0),
    probe_interaction_gain_multipliers: Sequence[float] = (1.0, 1.5, 2.0),
    probe_budget_bump_per_escalation: int = 2,
    probe_force_behavioral_preset: bool = True,
    probe_target_top_k: int = 3,
    probe_hist_prim_selector_mode: str = "interaction_band",
    probe_surprise_metrics: Optional[Sequence[str]] = None,
    probe_belief_source_modes: Optional[Sequence[str]] = None,
    probe_metric_selection_policy: str = "first_feasible",  # first_feasible | best_feasible_score
    probe_min_nonzero_surprise_fraction: float = 0.01,
    probe_min_realized_fraction: float = 0.10,
    probe_min_effect_l2_mean: float = 0.05,
    probe_min_belief_shift_mean: float = 1e-8,
    probe_min_policy_shift_mean: float = 1e-8,
    probe_min_realization_ratio_mean: float = 0.05,
    probe_min_raw_belief_shift_mean: float = 1e-8,
    probe_min_raw_policy_shift_mean: float = 1e-8,
    probe_min_raw_fraction_of_total: float = 0.10,
    probe_require_belief_and_policy: bool = True,
    probe_require_raw_signal: bool = True,
    apply_successful_probe_tuning: bool = True,
    build_simulation_context: bool = True,
) -> QuickProbeBundle:
    scale_mults = tuple(float(x) for x in probe_scale_multipliers) or (1.0,)
    l2_mults = tuple(float(x) for x in probe_delta_l2_multipliers) or (1.0,)
    clip_mults = tuple(float(x) for x in probe_delta_clip_multipliers) or (1.0,)
    persist_mults = tuple(float(x) for x in probe_persist_step_multipliers) or (1.0,)
    interaction_mults = tuple(float(x) for x in probe_interaction_gain_multipliers) or (1.0,)

    selected_cfg = cfg
    selected_search_cfg = search_cfg
    selected_probe_df = pd.DataFrame()
    selected_probe_summary_df = pd.DataFrame()
    selected_probe_metric_summary_df = pd.DataFrame()
    selected_probe_feasibility_df = pd.DataFrame()
    selected_failure_reasons: List[str] = []
    selected_surprise_metric = str(getattr(cfg, "planner_surprise_name", "predictive_seq_w2")).strip().lower()
    selected_belief_source_mode = str(getattr(cfg, "surprise_belief_source_mode", "auto")).strip().lower() or "auto"
    attempt_rows = []
    metric_summary_rows = []
    final_collapsed = False
    signal_feasible = False
    applied_tuning = False

    if bool(run_quick_surprise_probe_enabled):
        max_attempts = int(max_probe_escalations) if bool(auto_escalate_quick_probe) else 1
        max_attempts = max(1, max_attempts)

        selected_cfg = None
        selected_search_cfg = None

        metric_order = _normalize_probe_metrics(
            probe_surprise_metrics,
            fallback=str(getattr(cfg, "planner_surprise_name", "predictive_seq_w2")),
        )
        belief_mode_order = _normalize_probe_belief_modes(
            probe_belief_source_modes,
            fallback=str(getattr(cfg, "surprise_belief_source_mode", "auto")),
        )
        selection_policy = str(probe_metric_selection_policy).strip().lower()
        if selection_policy not in {"first_feasible", "best_feasible_score"}:
            selection_policy = "first_feasible"

        best_candidate: Optional[Dict[str, Any]] = None
        all_candidates: List[Dict[str, Any]] = []
        selected_candidate: Optional[Dict[str, Any]] = None
        stop_metric_sweep = False

        for metric_name in metric_order:
            metric_any_feasible = False
            for belief_mode in belief_mode_order:
                metric_mode_best_row: Optional[Dict[str, Any]] = None
                metric_mode_attempts = 0
                metric_mode_feasible = False

                for attempt in range(max_attempts):
                    metric_mode_attempts += 1
                    scale_mult = float(scale_mults[min(attempt, len(scale_mults) - 1)])
                    l2_mult = float(l2_mults[min(attempt, len(l2_mults) - 1)])
                    clip_mult = float(clip_mults[min(attempt, len(clip_mults) - 1)])
                    persist_mult = float(persist_mults[min(attempt, len(persist_mults) - 1)])
                    interaction_mult = float(interaction_mults[min(attempt, len(interaction_mults) - 1)])

                    cfg_trial = copy.deepcopy(cfg)
                    cfg_trial.planner_surprise_name = str(metric_name)
                    cfg_trial.surprise_belief_source_mode = str(belief_mode)
                    if bool(probe_force_behavioral_preset):
                        cfg_trial.perturb_use_behavioral_proposals = True
                        cfg_trial.counterfactual_family = "hist_prim"
                        cfg_trial.perturb_target_selection_mode = "highest_interaction"
                        cfg_trial.perturb_target_top_k = int(max(int(getattr(cfg_trial, "perturb_target_top_k", 1)), int(probe_target_top_k)))
                        cfg_trial.perturb_hist_prim_selector_mode = str(probe_hist_prim_selector_mode).strip().lower() or "interaction_band"
                        base_persist_steps = int(max(1, int(getattr(cfg, "perturb_persist_steps", 3))))
                        cfg_trial.perturb_persist_steps = int(max(1, round(float(base_persist_steps) * persist_mult)))
                        base_interaction_gain = float(max(0.1, float(getattr(cfg, "perturb_behavioral_interaction_gain", 1.25))))
                        base_longitudinal_gain = float(max(0.1, float(getattr(cfg, "perturb_behavioral_longitudinal_gain", 1.05))))
                        base_lateral_gain = float(max(0.1, float(getattr(cfg, "perturb_behavioral_lateral_gain", 1.20))))
                        cfg_trial.perturb_behavioral_interaction_gain = float(base_interaction_gain * interaction_mult)
                        cfg_trial.perturb_behavioral_longitudinal_gain = float(base_longitudinal_gain * interaction_mult)
                        cfg_trial.perturb_behavioral_lateral_gain = float(base_lateral_gain * interaction_mult)
                    search_trial = copy.deepcopy(search_cfg)

                    search_trial.proposal_scale_ladder = tuple(float(x * scale_mult) for x in tuple(search_cfg.proposal_scale_ladder))
                    search_trial.random_scale = float(search_cfg.random_scale * scale_mult)
                    search_trial.delta_l2_budget = float(search_cfg.delta_l2_budget * l2_mult)
                    search_trial.delta_clip = float(search_cfg.delta_clip * clip_mult)
                    search_trial.budget_evals = int(search_cfg.budget_evals + attempt * int(probe_budget_bump_per_escalation))

                    probe_dataset_config, probe_data_iter = make_waymax_data_iter(cfg_trial)
                    probe_df, probe_summary_df = run_quick_surprise_probe(
                        cfg=cfg_trial,
                        search_cfg=search_trial,
                        n_probe_scenarios=int(quick_probe_scenarios),
                        proposals_per_scenario=int(quick_probe_proposals_per_scenario),
                        data_iter=probe_data_iter,
                        dataset_config=probe_dataset_config,
                    )
                    collapsed, probe_metrics, failure_reasons, feasibility_df = _probe_collapse_stats(
                        probe_summary_df=probe_summary_df,
                        min_nonzero_surprise_fraction=float(probe_min_nonzero_surprise_fraction),
                        min_realized_fraction=float(probe_min_realized_fraction),
                        min_effect_l2_mean=float(probe_min_effect_l2_mean),
                        min_belief_shift_mean=float(probe_min_belief_shift_mean),
                        min_policy_shift_mean=float(probe_min_policy_shift_mean),
                        min_realization_ratio_mean=float(probe_min_realization_ratio_mean),
                        require_belief_and_policy=bool(probe_require_belief_and_policy),
                        min_raw_belief_shift_mean=float(probe_min_raw_belief_shift_mean),
                        min_raw_policy_shift_mean=float(probe_min_raw_policy_shift_mean),
                        min_raw_fraction_of_total=float(probe_min_raw_fraction_of_total),
                        require_raw_signal=bool(probe_require_raw_signal),
                    )
                    feasibility_score = _probe_feasibility_score(probe_metrics)

                    row = {
                        "surprise_metric": str(metric_name),
                        "belief_source_mode": str(belief_mode),
                        "attempt": int(attempt + 1),
                        "collapsed": int(collapsed),
                        "scale_mult": float(scale_mult),
                        "delta_l2_budget": float(search_trial.delta_l2_budget),
                        "delta_clip": float(search_trial.delta_clip),
                        "budget_evals": int(search_trial.budget_evals),
                        "probe_persist_steps": int(getattr(cfg_trial, "perturb_persist_steps", 0)),
                        "probe_interaction_gain": float(getattr(cfg_trial, "perturb_behavioral_interaction_gain", np.nan)),
                        "probe_hist_prim_selector_mode": str(getattr(cfg_trial, "perturb_hist_prim_selector_mode", "")),
                        "n_finite_surprise": int(probe_metrics.get("n_finite_surprise", 0.0)),
                        "nonzero_surprise_fraction": float(probe_metrics.get("nonzero_surprise_fraction", 0.0)),
                        "proposal_realized_fraction": float(probe_metrics.get("proposal_realized_fraction", 0.0)),
                        "proposal_effect_l2_mean": float(probe_metrics.get("proposal_effect_l2_mean", 0.0)),
                        "surprise_belief_shift_mean": float(probe_metrics.get("surprise_belief_shift_mean", 0.0)),
                        "surprise_belief_shift_raw_mean": float(probe_metrics.get("surprise_belief_shift_raw_mean", 0.0)),
                        "surprise_policy_shift_mean": float(probe_metrics.get("surprise_policy_shift_mean", 0.0)),
                        "surprise_policy_shift_raw_mean": float(probe_metrics.get("surprise_policy_shift_raw_mean", 0.0)),
                        "surprise_belief_raw_fraction": float(probe_metrics.get("surprise_belief_raw_fraction", 0.0)),
                        "surprise_policy_raw_fraction": float(probe_metrics.get("surprise_policy_raw_fraction", 0.0)),
                        "surprise_realization_ratio_mean": float(probe_metrics.get("surprise_realization_ratio_mean", 0.0)),
                        "surprise_realization_ratio_raw_mean": float(probe_metrics.get("surprise_realization_ratio_raw_mean", 0.0)),
                        "feasibility_score": float(feasibility_score),
                        "failure_reasons": "|".join(str(x) for x in failure_reasons),
                    }
                    attempt_rows.append(row)

                    candidate = {
                        "row": row,
                        "cfg": cfg_trial,
                        "search_cfg": search_trial,
                        "probe_df": probe_df.copy(),
                        "probe_summary_df": probe_summary_df.copy(),
                        "probe_feasibility_df": feasibility_df.copy(),
                        "failure_reasons": list(failure_reasons),
                        "collapsed": bool(collapsed),
                        "score": float(feasibility_score),
                        "metric": str(metric_name),
                        "belief_mode": str(belief_mode),
                    }
                    all_candidates.append(candidate)

                    if (metric_mode_best_row is None) or (float(row["feasibility_score"]) > float(metric_mode_best_row["feasibility_score"])):
                        metric_mode_best_row = dict(row)
                    if (best_candidate is None) or (float(candidate["score"]) > float(best_candidate["score"])):
                        best_candidate = candidate

                    if not bool(collapsed):
                        metric_mode_feasible = True
                        metric_any_feasible = True
                        if selection_policy == "first_feasible":
                            selected_candidate = candidate
                            stop_metric_sweep = True
                            break

                if metric_mode_best_row is not None:
                    metric_summary_rows.append(
                        {
                            "surprise_metric": str(metric_name),
                            "belief_source_mode": str(belief_mode),
                            "metric_feasible": int(metric_mode_feasible),
                            "attempts_tried": int(metric_mode_attempts),
                            **metric_mode_best_row,
                        }
                    )

                if stop_metric_sweep:
                    break

            if stop_metric_sweep:
                break

        if (selected_candidate is None) and (selection_policy == "best_feasible_score"):
            feasible_candidates = [c for c in all_candidates if not bool(c.get("collapsed", True))]
            if len(feasible_candidates) > 0:
                feasible_candidates = sorted(feasible_candidates, key=lambda c: float(c.get("score", 0.0)), reverse=True)
                selected_candidate = feasible_candidates[0]

        if (selected_candidate is None) and (best_candidate is not None):
            selected_candidate = best_candidate

        if selected_candidate is not None:
            selected_cfg = selected_candidate["cfg"]
            selected_search_cfg = selected_candidate["search_cfg"]
            selected_probe_df = selected_candidate["probe_df"]
            selected_probe_summary_df = selected_candidate["probe_summary_df"]
            selected_probe_feasibility_df = selected_candidate["probe_feasibility_df"]
            selected_failure_reasons = list(selected_candidate["failure_reasons"])
            selected_surprise_metric = str(selected_candidate["metric"])
            selected_belief_source_mode = str(selected_candidate.get("belief_mode", "auto"))
            final_collapsed = bool(selected_candidate["collapsed"])
            signal_feasible = bool(not final_collapsed)
        else:
            final_collapsed = True
            signal_feasible = False
            selected_failure_reasons = ["quick_probe_metric_sweep_empty"]

        if (not final_collapsed) and bool(apply_successful_probe_tuning) and (selected_cfg is not None) and (selected_search_cfg is not None):
            cfg = selected_cfg
            search_cfg = selected_search_cfg
            applied_tuning = True

        if final_collapsed and bool(stop_if_quick_probe_collapsed):
            reasons = ", ".join(selected_failure_reasons) if selected_failure_reasons else "unknown"
            raise RuntimeError(
                "Quick surprise probe collapsed after escalation attempts. "
                f"Composite signal checks failed: {reasons}"
            )
    else:
        selected_failure_reasons = ["quick_probe_skipped"]

    quick_probe_attempts_df = pd.DataFrame(attempt_rows)
    quick_probe_metric_summary_df = pd.DataFrame(metric_summary_rows)
    if selected_cfg is None:
        selected_cfg = cfg
    if selected_search_cfg is None:
        selected_search_cfg = search_cfg

    context_bundle = SimulationContextBundle()
    if bool(build_simulation_context):
        context_bundle = build_full_simulation_context(
            cfg=selected_cfg,
            n_shards=int(n_shards),
            shard_id=int(shard_id),
        )

    return QuickProbeBundle(
        cfg=selected_cfg,
        search_cfg=selected_search_cfg,
        quick_probe_df=selected_probe_df,
        quick_probe_summary_df=selected_probe_summary_df,
        quick_probe_attempts_df=quick_probe_attempts_df,
        quick_probe_metric_summary_df=quick_probe_metric_summary_df,
        quick_probe_feasibility_df=selected_probe_feasibility_df,
        final_collapsed=bool(final_collapsed),
        signal_feasible=bool(signal_feasible),
        signal_failure_reasons=", ".join(selected_failure_reasons),
        selected_surprise_metric=str(selected_surprise_metric),
        selected_belief_source_mode=str(selected_belief_source_mode),
        applied_tuning=bool(applied_tuning),
        dataset_config=context_bundle.dataset_config,
        data_iter=context_bundle.data_iter,
        runner=context_bundle.runner,
        data=context_bundle.data,
        reference_idx=context_bundle.reference_idx,
        candidate_eval_idx=context_bundle.candidate_eval_idx,
        eval_idx_all=context_bundle.eval_idx_all,
        eval_idx=context_bundle.eval_idx,
        reference_df=context_bundle.reference_df,
        base_eval_openloop_df=context_bundle.base_eval_openloop_df,
    )


def run_preflight_bundle(
    runner: Any,
    cfg: ClosedLoopConfig,
    search_cfg: SearchConfig,
    eval_idx: np.ndarray,
    reference_df: pd.DataFrame,
    restore_from_upload: bool = False,
    stop_on_preflight_fail: bool = False,
) -> PreflightBundle:
    bundle = PreflightBundle()
    try:
        (
            bundle.preflight_df,
            bundle.closedloop_calib_df,
            bundle.closedloop_thresholds,
            bundle.calib_diag_df,
            bundle.calib_quant_df,
        ) = run_preflight_and_calibration(
            runner=runner,
            cfg=cfg,
            search_cfg=search_cfg,
            eval_idx=eval_idx,
            reference_df=reference_df,
            restore_from_upload=restore_from_upload,
        )
        bundle.ready_for_main_loop = True
    except RuntimeError as exc:
        bundle.preflight_error = str(exc)
        if "preflight failed" in bundle.preflight_error.lower():
            try:
                bundle.preflight_df = run_closedloop_preflight_checks(runner=runner, cfg=cfg, eval_idx=eval_idx)
            except Exception as inner_exc:
                if bundle.preflight_error:
                    bundle.preflight_error += f" | detailed_preflight_recompute_failed={inner_exc}"
                else:
                    bundle.preflight_error = f"detailed_preflight_recompute_failed={inner_exc}"
            if bool(stop_on_preflight_fail):
                raise
        else:
            raise

    try:
        bundle.root_cause_summary_df, bundle.root_cause_findings_df = diagnose_surprise_root_cause(
            preflight_df=bundle.preflight_df,
            closedloop_calib_df=bundle.closedloop_calib_df,
        )
    except Exception as exc:
        bundle.root_cause_diag_error = str(exc)

    return bundle


def report_run_context(bundle: RunContextBundle, display_fn: Optional[Any] = None) -> None:
    if bundle.auto_generated_run_tag:
        print(f"[run-tag] auto-generated RUN_TAG={bundle.run_tag}")
    if bundle.adopted_existing_run_tag:
        print(f"[run-tag] auto-adopted existing RUN_TAG={bundle.run_tag} from persist root history.")
    print(
        "[run-mode] "
        f"requested={bundle.run_mode_requested}, inferred={bundle.run_mode_inferred}, "
        f"applied={bundle.run_mode_applied}, resume_from_existing={bool(bundle.cfg.resume_from_existing)}"
    )
    print("run_prefix =", bundle.run_prefix)
    print(f"[shard] running {bundle.shard_id + 1}/{max(1, bundle.n_shards)}")
    print(
        "[run-policy] "
        f"AUTO_RUN_MAIN_LOOP_WHEN_READY={bundle.auto_run_main_loop_when_ready}, "
        f"RUN_MAIN_LOOP_OVERRIDE={bundle.run_main_loop_override}"
    )
    print(
        "[planner] "
        f"backend={bundle.planner_backend}, planner_name={bundle.cfg.planner_name}, "
        f"surprise_metric={bundle.cfg.planner_surprise_name}"
    )
    if bundle.run_mode_applied == "fresh" and bundle.has_existing_progress:
        print(
            "[run-mode warning] fresh mode requested but existing progress was detected for this RUN_TAG. "
            "Outputs for this run prefix will be recomputed."
        )
    if len(bundle.config_drift_df):
        print("[run-plan warning] config drift detected vs existing carry-forward config for this run prefix.")
    if display_fn is not None:
        if len(bundle.run_tag_candidates_df):
            display_fn(bundle.run_tag_candidates_df.head(10))
        if len(bundle.run_plan_df):
            display_fn(bundle.run_plan_df)
        if len(bundle.shard_progress_df):
            display_fn(bundle.shard_progress_df)
        if len(bundle.config_drift_df):
            display_fn(bundle.config_drift_df)
        if len(bundle.ckpt_scan_df):
            display_fn(bundle.ckpt_scan_df.head(10))
        if len(bundle.planner_profile_df):
            display_fn(bundle.planner_profile_df)


def report_quick_probe_bundle(
    bundle: QuickProbeBundle,
    search_cfg: SearchConfig,
    display_fn: Optional[Any] = None,
    probe_preview_rows: int = 20,
) -> None:
    if len(bundle.quick_probe_summary_df):
        row = bundle.quick_probe_summary_df.iloc[0]
        used_raw = row.get("n_scenarios_used", row.get("n_scenarios", 0.0))
        used = int(max(0, _float_or_default(used_raw, 0.0)))
        skipped_infeasible = int(max(0, _float_or_default(row.get("n_scenarios_skipped_base_infeasible"), 0.0)))
        skipped_reason_counts = str(row.get("skipped_base_infeasible_reason_counts", "")).strip()
        if used <= 0:
            print(
                "[probe] no scenario produced a feasible base rollout; "
                "inspect preflight/planner forward path before trusting probe calibration."
            )
            if skipped_reason_counts and skipped_reason_counts not in {"{}", "nan"}:
                print(f"[probe] base infeasibility reasons: {skipped_reason_counts}")
        elif skipped_infeasible > 0:
            print(
                f"[probe] skipped {skipped_infeasible} candidate scenarios with infeasible base rollouts "
                "before collecting probe rows."
            )
        belief_total = _float_or_default(row.get("surprise_belief_shift_mean"), 0.0)
        policy_total = _float_or_default(row.get("surprise_policy_shift_mean"), 0.0)
        belief_raw = _float_or_default(row.get("surprise_belief_shift_raw_mean"), 0.0)
        policy_raw = _float_or_default(row.get("surprise_policy_shift_raw_mean"), 0.0)
        belief_raw_frac = 0.0 if belief_total <= 0.0 else float(np.clip(belief_raw / max(belief_total, 1e-12), 0.0, 1.0))
        policy_raw_frac = 0.0 if policy_total <= 0.0 else float(np.clip(policy_raw / max(policy_total, 1e-12), 0.0, 1.0))
        if (belief_total > 0.0 or policy_total > 0.0) and (max(belief_raw_frac, policy_raw_frac) < 0.10):
            print(
                "[probe] warning: surprise appears floor-dominated "
                f"(raw belief frac={belief_raw_frac:.3f}, raw policy frac={policy_raw_frac:.3f})."
            )

    if display_fn is not None:
        if len(bundle.quick_probe_metric_summary_df):
            display_fn(bundle.quick_probe_metric_summary_df)
        if len(bundle.quick_probe_attempts_df):
            display_fn(bundle.quick_probe_attempts_df)
        if len(bundle.quick_probe_summary_df):
            display_fn(bundle.quick_probe_summary_df)
        if len(bundle.quick_probe_feasibility_df):
            display_fn(bundle.quick_probe_feasibility_df)
        if len(bundle.quick_probe_df):
            display_fn(bundle.quick_probe_df.head(int(max(1, probe_preview_rows))))

    if str(bundle.selected_surprise_metric).strip():
        print(f"[probe] selected surprise instantiation = {bundle.selected_surprise_metric}")
    if str(bundle.selected_belief_source_mode).strip():
        print(f"[probe] selected belief source mode = {bundle.selected_belief_source_mode}")
    print(f"[probe] composite-signal feasibility = {bool(bundle.signal_feasible)}")
    if (not bool(bundle.signal_feasible)) and str(bundle.signal_failure_reasons).strip():
        print(f"[probe] failed checks: {bundle.signal_failure_reasons}")

    if bundle.applied_tuning:
        print("[probe] applied tuned search settings from successful attempt.")
        print("        proposal_scale_ladder=", search_cfg.proposal_scale_ladder)
        print("        delta_l2_budget=", search_cfg.delta_l2_budget)
        print("        delta_clip=", search_cfg.delta_clip)
        print("        budget_evals=", search_cfg.budget_evals)
        print("        perturb_persist_steps=", bundle.cfg.perturb_persist_steps)
        print("        perturb_behavioral_interaction_gain=", bundle.cfg.perturb_behavioral_interaction_gain)
        print("        perturb_hist_prim_selector_mode=", bundle.cfg.perturb_hist_prim_selector_mode)

    if bundle.final_collapsed:
        print("[probe] warning: quick probe remained collapsed after escalation attempts.")


def report_preflight_bundle(bundle: PreflightBundle, display_fn: Optional[Any] = None) -> None:
    if bundle.preflight_error and (not bundle.ready_for_main_loop):
        print("[preflight] failed; calibration/main loop will be skipped until checks pass.")
        print(bundle.preflight_error)
    if bundle.root_cause_diag_error:
        print(f"[diagnose] root-cause analysis failed: {bundle.root_cause_diag_error}")

    print("READY_FOR_MAIN_LOOP =", bundle.ready_for_main_loop)
    if display_fn is not None:
        if len(bundle.preflight_df):
            display_fn(bundle.preflight_df)
        if len(bundle.root_cause_summary_df):
            display_fn(bundle.root_cause_summary_df)
        if len(bundle.root_cause_findings_df):
            display_fn(bundle.root_cause_findings_df)
        if bundle.ready_for_main_loop:
            print("Calibration thresholds:", bundle.closedloop_thresholds)
            display_fn(bundle.calib_diag_df)
            display_fn(bundle.calib_quant_df)
        else:
            print("[preflight] Fix failing checks, then rerun this cell.")


def report_surprise_gate_bundle(bundle: SurpriseGateBundle, display_fn: Optional[Any] = None) -> None:
    if bundle.gate_skipped_reason:
        print(f"[gate] skipped: {bundle.gate_skipped_reason}")
    if bundle.gate_error:
        print("[gate] failed:", bundle.gate_error)

    print("READY_FOR_OPTIMIZATION =", bundle.ready_for_optimization)
    if display_fn is not None:
        if len(bundle.gate_summary_df):
            display_fn(bundle.gate_summary_df)
        if len(bundle.dist_change_summary_df):
            display_fn(bundle.dist_change_summary_df)


def report_main_loop_bundle(bundle: MainLoopBundle, ready_for_optimization: bool) -> None:
    print(f"[run-policy] {bundle.policy} -> should_run_main_loop={bundle.should_run_main_loop}")
    if not bundle.should_run_main_loop:
        if not bool(ready_for_optimization):
            print("[run] skipped: diagnostics not ready (READY_FOR_OPTIMIZATION=False).")
        else:
            print("[run] skipped by policy override/auto-run setting.")
    print("Result rows:", len(bundle.closedloop_results_df))
    print("Trace rows:", len(bundle.closedloop_trace_df))


def report_export_bundle(
    bundle: ExportBundle,
    closedloop_results_df: pd.DataFrame,
    display_fn: Optional[Any] = None,
) -> None:
    if closedloop_results_df.empty:
        print("[export] skipped: closedloop_results_df is empty (main loop was skipped or produced no rows).")

    if display_fn is not None:
        if len(bundle.quick_summary_df):
            display_fn(bundle.quick_summary_df)
        if len(bundle.sanity_df):
            display_fn(bundle.sanity_df)
        if len(bundle.fairness_checks_df):
            display_fn(bundle.fairness_checks_df)
        if len(bundle.trace_diag_df):
            display_fn(bundle.trace_diag_df)

    if bundle.artifact_paths:
        print("Artifacts:")
        for key, value in bundle.artifact_paths.items():
            print(f" - {key}: {value}")


def report_signal_bundle(
    bundle: SignalBundle,
    closedloop_results_df: pd.DataFrame,
    display_fn: Optional[Any] = None,
    preview_rows: int = 20,
) -> None:
    if closedloop_results_df.empty:
        print("[signal] skipped: closedloop_results_df is empty.")

    if display_fn is not None:
        if len(bundle.signal_summary_df):
            display_fn(bundle.signal_summary_df)
        if len(bundle.signal_method_corr_df):
            display_fn(bundle.signal_method_corr_df)
        if len(bundle.signal_bin_df):
            display_fn(bundle.signal_bin_df)
        if len(bundle.signal_topk_lift_df):
            display_fn(bundle.signal_topk_lift_df)
        if len(bundle.signal_within_scenario_df):
            display_fn(bundle.signal_within_scenario_df.head(int(max(1, preview_rows))))
            if len(bundle.signal_within_scenario_df) > int(max(1, preview_rows)):
                print(
                    f"[signal] showing first {int(max(1, preview_rows))} rows of "
                    f"{len(bundle.signal_within_scenario_df)} within-scenario rows."
                )


def run_surprise_gate_with_policy(
    closedloop_calib_df: pd.DataFrame,
    ready_for_main_loop: bool,
    surprise_gate_enabled: bool = True,
    stop_on_gate_fail: bool = False,
    allow_main_loop_when_gate_fails: bool = False,
) -> SurpriseGateBundle:
    bundle = SurpriseGateBundle()
    if not bool(ready_for_main_loop):
        bundle.gate_skipped_reason = "READY_FOR_MAIN_LOOP=False"
        return bundle

    try:
        bundle.gate_summary_df, bundle.dist_change_summary_df = run_surprise_quality_gate(
            closedloop_calib_df=closedloop_calib_df,
            surprise_gate_enabled=bool(surprise_gate_enabled),
        )
        bundle.ready_for_optimization = True
    except RuntimeError as exc:
        bundle.gate_error = str(exc)
        bundle.gate_summary_df, bundle.dist_change_summary_df = run_surprise_quality_gate(
            closedloop_calib_df=closedloop_calib_df,
            surprise_gate_enabled=False,
        )
        bundle.ready_for_optimization = bool(allow_main_loop_when_gate_fails)
        if bool(stop_on_gate_fail):
            raise

    return bundle


def resolve_main_loop_policy(
    ready_for_optimization: bool,
    auto_run_main_loop_when_ready: bool = True,
    run_main_loop_override: Optional[bool] = None,
) -> Tuple[bool, str]:
    if run_main_loop_override is None:
        should_run_main_loop = bool(auto_run_main_loop_when_ready and ready_for_optimization)
        policy = f"auto(auto_run={bool(auto_run_main_loop_when_ready)}, ready={bool(ready_for_optimization)})"
    elif bool(run_main_loop_override):
        should_run_main_loop = True
        policy = "override=True"
    else:
        should_run_main_loop = False
        policy = "override=False"
    return bool(should_run_main_loop), str(policy)


def run_main_loop_with_policy(
    runner: Any,
    eval_idx: np.ndarray,
    cfg: ClosedLoopConfig,
    search_cfg: SearchConfig,
    closedloop_thresholds: Dict[str, Any],
    base_eval_openloop_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    closedloop_calib_df: pd.DataFrame,
    preflight_df: pd.DataFrame,
    calib_diag_df: pd.DataFrame,
    calib_quant_df: pd.DataFrame,
    ready_for_optimization: bool,
    auto_run_main_loop_when_ready: bool = True,
    run_main_loop_override: Optional[bool] = None,
) -> MainLoopBundle:
    should_run_main_loop, policy = resolve_main_loop_policy(
        ready_for_optimization=bool(ready_for_optimization),
        auto_run_main_loop_when_ready=bool(auto_run_main_loop_when_ready),
        run_main_loop_override=run_main_loop_override,
    )
    if not should_run_main_loop:
        return MainLoopBundle(
            should_run_main_loop=False,
            policy=policy,
            closedloop_results_df=pd.DataFrame(),
            closedloop_trace_df=pd.DataFrame(),
        )

    closedloop_results_df, closedloop_trace_df = run_closed_loop(
        runner=runner,
        eval_idx=eval_idx,
        cfg=cfg,
        search_cfg=search_cfg,
        thresholds=closedloop_thresholds,
        run_prefix=cfg.run_prefix,
        static_frames={
            "base_eval_openloop_df": base_eval_openloop_df,
            "reference_df": reference_df,
            "closedloop_calib_df": closedloop_calib_df,
            "preflight_df": preflight_df,
            "calib_diag_df": calib_diag_df,
            "calib_quant_df": calib_quant_df,
        },
    )
    return MainLoopBundle(
        should_run_main_loop=True,
        policy=policy,
        closedloop_results_df=closedloop_results_df,
        closedloop_trace_df=closedloop_trace_df,
    )


def summarize_and_export_if_available(
    cfg: ClosedLoopConfig,
    search_cfg: SearchConfig,
    eval_idx: np.ndarray,
    closedloop_results_df: pd.DataFrame,
    closedloop_trace_df: pd.DataFrame,
    base_eval_openloop_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    closedloop_calib_df: pd.DataFrame,
    preflight_df: pd.DataFrame,
    calib_diag_df: pd.DataFrame,
    calib_quant_df: pd.DataFrame,
    closedloop_thresholds: Dict[str, Any],
) -> ExportBundle:
    bundle = ExportBundle()
    if closedloop_results_df.empty:
        return bundle

    (
        bundle.quick_summary_df,
        bundle.sanity_df,
        bundle.fairness_checks_df,
        bundle.trace_diag_df,
    ) = summarize_method_outputs(
        closedloop_results_df=closedloop_results_df,
        closedloop_trace_df=closedloop_trace_df,
    )

    bundle.artifact_paths = export_closedloop_artifacts(
        cfg=cfg,
        search_cfg=search_cfg,
        eval_idx=eval_idx,
        closedloop_results_df=closedloop_results_df,
        closedloop_trace_df=closedloop_trace_df,
        base_eval_openloop_df=base_eval_openloop_df,
        reference_df=reference_df,
        closedloop_calib_df=closedloop_calib_df,
        preflight_df=preflight_df,
        calib_diag_df=calib_diag_df,
        calib_quant_df=calib_quant_df,
        closedloop_thresholds=closedloop_thresholds,
        quick_summary_df=bundle.quick_summary_df,
        sanity_df=bundle.sanity_df,
        fairness_checks_df=bundle.fairness_checks_df,
        trace_diag_df=bundle.trace_diag_df,
    )
    return bundle


def analyze_signal_if_available(
    closedloop_results_df: pd.DataFrame,
    n_bins: int = 10,
    top_fracs: Sequence[float] = (0.10, 0.20),
    scenario_min_points: int = 3,
) -> SignalBundle:
    bundle = SignalBundle()
    if closedloop_results_df.empty:
        return bundle

    (
        bundle.signal_summary_df,
        bundle.signal_method_corr_df,
        bundle.signal_bin_df,
        bundle.signal_topk_lift_df,
        bundle.signal_within_scenario_df,
    ) = analyze_surprise_signal_usefulness(
        closedloop_results_df=closedloop_results_df,
        n_bins=int(n_bins),
        top_fracs=tuple(float(x) for x in top_fracs),
        scenario_min_points=int(scenario_min_points),
    )
    return bundle
