from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

from .closedloop_flow import (
    ExportBundle,
    MainLoopBundle,
    PreflightBundle,
    QuickProbeBundle,
    SignalBundle,
    SimulationContextBundle,
    SurpriseGateBundle,
    analyze_signal_if_available,
    build_full_simulation_context,
    initialize_run_context,
    run_main_loop_with_policy,
    run_preflight_bundle,
    run_quick_probe_with_auto_escalation,
    run_surprise_gate_with_policy,
    summarize_and_export_if_available,
)


def _slugify(value: str) -> str:
    token = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in str(value).strip().lower())
    token = token.strip("_")
    return token or "metric"


_PAPER_COUNTERFACTUAL_FAMILIES = (
    'fut_none',
    'fut_gt',
    'hist_rmv',
    'fut_cvm',
    'fut_cvm_l',
    'fut_pred',
    'hist_prim',
    'fut_prim',
)

_PAPER_FAMILY_ALIAS = {
    'fut-none': 'fut_none',
    'fut_none': 'fut_none',
    'fut-gt': 'fut_gt',
    'fut_gt': 'fut_gt',
    'hist-rmv': 'hist_rmv',
    'hist_rmv': 'hist_rmv',
    'fut-cvm': 'fut_cvm',
    'fut_cvm': 'fut_cvm',
    'fut-cvm-l': 'fut_cvm_l',
    'fut_cvm_l': 'fut_cvm_l',
    'fut-pred': 'fut_pred',
    'fut_pred': 'fut_pred',
    'hist-prim': 'hist_prim',
    'hist_prim': 'hist_prim',
    'fut-prim': 'fut_prim',
    'fut_prim': 'fut_prim',
}

_PAPER_FAMILY_PRIMITIVE_CYCLE = {
    'fut_none': (
        'target_accel',
        'target_brake',
        'lateral_left',
        'lateral_right',
    ),
    'fut_gt': (
        'target_accel',
        'target_brake',
        'diag_toward_left',
        'diag_toward_right',
    ),
    'hist_rmv': (
        'away_from_ego',
    ),
    'fut_cvm': (
        'target_accel',
        'target_brake',
        'away_from_ego',
    ),
    'fut_cvm_l': (
        'target_accel',
        'target_brake',
        'lateral_left',
        'lateral_right',
    ),
    'fut_pred': (
        'toward_ego',
        'away_from_ego',
        'diag_toward_left',
        'diag_toward_right',
    ),
    'hist_prim': (
        'toward_ego',
        'away_from_ego',
        'target_brake',
        'target_accel',
        'lateral_left',
        'lateral_right',
        'diag_toward_left',
        'diag_toward_right',
    ),
    'fut_prim': (
        'target_accel',
        'target_brake',
        'lateral_left',
        'lateral_right',
        'diag_toward_left',
        'diag_toward_right',
    ),
}


def normalize_paper_counterfactual_family(value: str) -> str:
    key = str(value).strip().lower()
    key = _PAPER_FAMILY_ALIAS.get(key, _PAPER_FAMILY_ALIAS.get(key.replace('_', '-'), key))
    if key not in _PAPER_COUNTERFACTUAL_FAMILIES:
        raise ValueError(
            f"Unsupported counterfactual_family={value!r}. "
            f"Supported: {', '.join(_PAPER_COUNTERFACTUAL_FAMILIES)}."
        )
    return key


def apply_paper_counterfactual_family_profile(cfg: Any, family: str) -> str:
    family_key = normalize_paper_counterfactual_family(family)
    cfg.counterfactual_family = str(family_key)
    cfg.perturb_use_behavioral_proposals = True
    cfg.perturb_behavioral_primitive_cycle = tuple(_PAPER_FAMILY_PRIMITIVE_CYCLE[family_key])

    if family_key == 'hist_prim':
        cfg.perturb_hist_prim_selector_mode = 'interaction_band'
        cfg.perturb_target_selection_mode = 'highest_interaction'
        cfg.perturb_target_top_k = max(2, int(getattr(cfg, 'perturb_target_top_k', 2)))
    elif family_key == 'hist_rmv':
        cfg.perturb_hist_prim_selector_mode = 'cyclic'
        cfg.perturb_target_selection_mode = 'nearest'
        cfg.perturb_target_top_k = 1
    else:
        cfg.perturb_hist_prim_selector_mode = 'cyclic'
        cfg.perturb_target_selection_mode = 'highest_interaction'
        cfg.perturb_target_top_k = max(1, int(getattr(cfg, 'perturb_target_top_k', 1)))

    return family_key


def _apply_overrides(target: Any, overrides: Optional[Mapping[str, Any]]) -> None:
    if not overrides:
        return
    for key, value in overrides.items():
        if not hasattr(target, key):
            raise AttributeError(f"{type(target).__name__} has no field {key!r}.")
        setattr(target, key, value)


def _to_simulation_context(probe_bundle: QuickProbeBundle) -> Optional[SimulationContextBundle]:
    if probe_bundle.runner is None:
        return None
    return SimulationContextBundle(
        dataset_config=probe_bundle.dataset_config,
        data_iter=probe_bundle.data_iter,
        runner=probe_bundle.runner,
        data=probe_bundle.data,
        reference_idx=probe_bundle.reference_idx,
        candidate_eval_idx=probe_bundle.candidate_eval_idx,
        eval_idx_all=probe_bundle.eval_idx_all,
        eval_idx=probe_bundle.eval_idx,
        reference_df=probe_bundle.reference_df,
        base_eval_openloop_df=probe_bundle.base_eval_openloop_df,
    )


def _method_rollup(
    closedloop_results_df: pd.DataFrame,
    metric: str,
    run_tag: str,
    counterfactual_family: str,
) -> pd.DataFrame:
    if closedloop_results_df.empty or ("method" not in closedloop_results_df.columns):
        return pd.DataFrame()

    out = closedloop_results_df.copy()
    numeric_cols = []
    for col in ("risk_sks", "surprise_pd", "failure_extended_proxy", "q1_hit", "q4_hit", "blind_spot_proxy_hit"):
        if col in out.columns:
            numeric_cols.append(col)
    if not numeric_cols:
        return pd.DataFrame()

    grouped = out.groupby("method", as_index=False)[numeric_cols].mean(numeric_only=True)
    grouped.insert(0, "metric", str(metric))
    grouped.insert(1, "run_tag", str(run_tag))
    grouped.insert(2, "counterfactual_family", str(counterfactual_family))
    return grouped


def _artifact_rows(
    artifact_paths: Mapping[str, str],
    metric: str,
    run_tag: str,
    run_prefix: str,
    counterfactual_family: str,
) -> pd.DataFrame:
    rows = []
    for key, value in sorted(dict(artifact_paths).items()):
        rows.append(
            {
                "metric": str(metric),
                "run_tag": str(run_tag),
                "counterfactual_family": str(counterfactual_family),
                "run_prefix": str(run_prefix),
                "artifact_key": str(key),
                "artifact_path": str(value),
            }
        )
    return pd.DataFrame(rows)


def _single_run_summary(
    metric: str,
    run_tag: str,
    run_prefix: str,
    counterfactual_family: str,
    probe_bundle: QuickProbeBundle,
    preflight_bundle: PreflightBundle,
    gate_bundle: SurpriseGateBundle,
    main_loop_bundle: MainLoopBundle,
    export_bundle: ExportBundle,
) -> pd.DataFrame:
    result_rows = int(len(main_loop_bundle.closedloop_results_df))
    trace_rows = int(len(main_loop_bundle.closedloop_trace_df))
    summary = {
        "metric": str(metric),
        "run_tag": str(run_tag),
        "run_prefix": str(run_prefix),
        "counterfactual_family": str(counterfactual_family),
        "quick_probe_collapsed": int(bool(probe_bundle.final_collapsed)),
        "quick_probe_tuning_applied": int(bool(probe_bundle.applied_tuning)),
        "ready_for_main_loop": int(bool(preflight_bundle.ready_for_main_loop)),
        "ready_for_optimization": int(bool(gate_bundle.ready_for_optimization)),
        "main_loop_executed": int(bool(main_loop_bundle.should_run_main_loop)),
        "result_rows": result_rows,
        "trace_rows": trace_rows,
        "artifacts_exported": int(len(export_bundle.artifact_paths)),
        "preflight_error": str(preflight_bundle.preflight_error or ""),
        "gate_error": str(gate_bundle.gate_error or ""),
    }
    return pd.DataFrame([summary])


@dataclass
class SurprisePotentialRunBundle:
    metric: str
    counterfactual_family: str
    run_tag: str
    run_prefix: str
    probe_bundle: QuickProbeBundle
    simulation_context: SimulationContextBundle
    preflight_bundle: PreflightBundle
    gate_bundle: SurpriseGateBundle
    main_loop_bundle: MainLoopBundle
    export_bundle: ExportBundle
    signal_bundle: SignalBundle
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    method_rollup_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    artifacts_df: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class SurprisePotentialSweepBundle:
    metric_runs: Dict[str, SurprisePotentialRunBundle] = field(default_factory=dict)
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    method_rollup_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    artifacts_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_surprise_potential_single_metric(
    metric: str,
    run_tag: str,
    persist_root: str,
    counterfactual_family: str = "hist_prim",
    n_shards: int = 1,
    shard_id: int | str = "auto",
    run_mode: str = "auto",
    planner_backend: str = "unimm_style",
    auto_run_main_loop_when_ready: bool = True,
    run_main_loop_override: Optional[bool] = None,
    run_tag_prefix: str = "surprise_potential",
    method_labels: Optional[Sequence[str]] = None,
    n_eval_scenarios: int = 100,
    strict_min_eval: int = 100,
    n_total_scenarios_floor: int = 400,
    warn_on_config_drift: bool = True,
    quick_probe_enabled: bool = True,
    quick_probe_scenarios: int = 1,
    quick_probe_proposals_per_scenario: int = 4,
    quick_probe_settings: Optional[Mapping[str, Any]] = None,
    stop_if_quick_probe_collapsed: bool = False,
    restore_from_upload: bool = False,
    stop_on_preflight_fail: bool = False,
    surprise_gate_enabled: bool = True,
    stop_on_gate_fail: bool = False,
    allow_main_loop_when_gate_fails: bool = False,
    cfg_overrides: Optional[Mapping[str, Any]] = None,
    search_cfg_overrides: Optional[Mapping[str, Any]] = None,
) -> SurprisePotentialRunBundle:
    metric_key = str(metric).strip().lower()
    family_key = normalize_paper_counterfactual_family(counterfactual_family)
    if not bool(quick_probe_enabled):
        raise ValueError(
            "quick_probe_enabled=False is not allowed for this contract-aligned workflow. "
            "Run quick probe before main loop."
        )
    if not bool(surprise_gate_enabled):
        raise ValueError(
            "surprise_gate_enabled=False is not allowed for this contract-aligned workflow. "
            "Run quality gate before main loop."
        )
    run_context = initialize_run_context(
        run_tag=str(run_tag),
        persist_root=str(persist_root),
        n_shards=int(n_shards),
        shard_id=shard_id,
        auto_run_main_loop_when_ready=bool(auto_run_main_loop_when_ready),
        run_main_loop_override=run_main_loop_override,
        n_eval_scenarios=int(n_eval_scenarios),
        strict_min_eval=int(strict_min_eval),
        n_total_scenarios_floor=int(n_total_scenarios_floor),
        run_tag_prefix=str(run_tag_prefix),
        planner_backend=str(planner_backend),
        planner_surprise_name=metric_key,
        method_labels=method_labels,
        resume_mode=str(run_mode),
        warn_on_config_drift=bool(warn_on_config_drift),
    )

    cfg = run_context.cfg
    search_cfg = run_context.search_cfg
    family_key = apply_paper_counterfactual_family_profile(cfg=cfg, family=family_key)
    _apply_overrides(cfg, cfg_overrides)
    _apply_overrides(search_cfg, search_cfg_overrides)

    probe_kwargs = {
        "quick_probe_scenarios": int(max(1, quick_probe_scenarios)),
        "quick_probe_proposals_per_scenario": int(max(1, quick_probe_proposals_per_scenario)),
        "stop_if_quick_probe_collapsed": bool(stop_if_quick_probe_collapsed),
        "auto_escalate_quick_probe": True,
        "max_probe_escalations": 3,
        "probe_scale_multipliers": (1.0, 1.35, 1.8),
        "probe_delta_l2_multipliers": (1.0, 1.2, 1.4),
        "probe_delta_clip_multipliers": (1.0, 1.1, 1.2),
        "probe_budget_bump_per_escalation": 2,
        "apply_successful_probe_tuning": True,
        "build_simulation_context": True,
    }
    if quick_probe_settings:
        probe_kwargs.update(dict(quick_probe_settings))

    probe_bundle = run_quick_probe_with_auto_escalation(
        cfg=cfg,
        search_cfg=search_cfg,
        n_shards=int(n_shards),
        shard_id=int(run_context.shard_id),
        run_quick_surprise_probe_enabled=bool(quick_probe_enabled),
        **probe_kwargs,
    )
    cfg = probe_bundle.cfg
    search_cfg = probe_bundle.search_cfg

    simulation_context = _to_simulation_context(probe_bundle)
    if simulation_context is None:
        simulation_context = build_full_simulation_context(
            cfg=cfg,
            n_shards=int(n_shards),
            shard_id=int(run_context.shard_id),
        )

    preflight_bundle = run_preflight_bundle(
        runner=simulation_context.runner,
        cfg=cfg,
        search_cfg=search_cfg,
        eval_idx=simulation_context.eval_idx,
        reference_df=simulation_context.reference_df,
        restore_from_upload=bool(restore_from_upload),
        stop_on_preflight_fail=bool(stop_on_preflight_fail),
    )
    gate_bundle = run_surprise_gate_with_policy(
        closedloop_calib_df=preflight_bundle.closedloop_calib_df,
        ready_for_main_loop=bool(preflight_bundle.ready_for_main_loop),
        surprise_gate_enabled=bool(surprise_gate_enabled),
        stop_on_gate_fail=bool(stop_on_gate_fail),
        allow_main_loop_when_gate_fails=bool(allow_main_loop_when_gate_fails),
    )
    main_loop_bundle = run_main_loop_with_policy(
        runner=simulation_context.runner,
        eval_idx=simulation_context.eval_idx,
        cfg=cfg,
        search_cfg=search_cfg,
        closedloop_thresholds=preflight_bundle.closedloop_thresholds,
        base_eval_openloop_df=simulation_context.base_eval_openloop_df,
        reference_df=simulation_context.reference_df,
        closedloop_calib_df=preflight_bundle.closedloop_calib_df,
        preflight_df=preflight_bundle.preflight_df,
        calib_diag_df=preflight_bundle.calib_diag_df,
        calib_quant_df=preflight_bundle.calib_quant_df,
        ready_for_optimization=bool(gate_bundle.ready_for_optimization),
        auto_run_main_loop_when_ready=bool(auto_run_main_loop_when_ready),
        run_main_loop_override=run_main_loop_override,
    )
    export_bundle = summarize_and_export_if_available(
        cfg=cfg,
        search_cfg=search_cfg,
        eval_idx=simulation_context.eval_idx,
        closedloop_results_df=main_loop_bundle.closedloop_results_df,
        closedloop_trace_df=main_loop_bundle.closedloop_trace_df,
        base_eval_openloop_df=simulation_context.base_eval_openloop_df,
        reference_df=simulation_context.reference_df,
        closedloop_calib_df=preflight_bundle.closedloop_calib_df,
        preflight_df=preflight_bundle.preflight_df,
        calib_diag_df=preflight_bundle.calib_diag_df,
        calib_quant_df=preflight_bundle.calib_quant_df,
        closedloop_thresholds=preflight_bundle.closedloop_thresholds,
    )
    signal_bundle = analyze_signal_if_available(
        closedloop_results_df=main_loop_bundle.closedloop_results_df,
        n_bins=10,
        top_fracs=(0.10, 0.20),
        scenario_min_points=3,
    )

    summary_df = _single_run_summary(
        metric=metric_key,
        run_tag=str(run_context.run_tag),
        run_prefix=str(run_context.run_prefix),
        counterfactual_family=str(family_key),
        probe_bundle=probe_bundle,
        preflight_bundle=preflight_bundle,
        gate_bundle=gate_bundle,
        main_loop_bundle=main_loop_bundle,
        export_bundle=export_bundle,
    )
    method_rollup_df = _method_rollup(
        closedloop_results_df=main_loop_bundle.closedloop_results_df,
        metric=metric_key,
        run_tag=str(run_context.run_tag),
        counterfactual_family=str(family_key),
    )
    artifacts_df = _artifact_rows(
        artifact_paths=export_bundle.artifact_paths,
        metric=metric_key,
        run_tag=str(run_context.run_tag),
        run_prefix=str(run_context.run_prefix),
        counterfactual_family=str(family_key),
    )

    return SurprisePotentialRunBundle(
        metric=metric_key,
        counterfactual_family=str(family_key),
        run_tag=str(run_context.run_tag),
        run_prefix=str(run_context.run_prefix),
        probe_bundle=probe_bundle,
        simulation_context=simulation_context,
        preflight_bundle=preflight_bundle,
        gate_bundle=gate_bundle,
        main_loop_bundle=main_loop_bundle,
        export_bundle=export_bundle,
        signal_bundle=signal_bundle,
        summary_df=summary_df,
        method_rollup_df=method_rollup_df,
        artifacts_df=artifacts_df,
    )


def run_surprise_potential_metric_sweep(
    metrics: Sequence[str],
    persist_root: str,
    counterfactual_families: Optional[Sequence[str]] = None,
    run_tag_base: str = "",
    run_tag_prefix: str = "surprise_potential",
    method_labels: Optional[Sequence[str]] = None,
    n_shards: int = 1,
    shard_id: int | str = "auto",
    run_mode: str = "fresh",
    planner_backend: str = "unimm_style",
    continue_on_error: bool = False,
    auto_run_main_loop_when_ready: bool = True,
    run_main_loop_override: Optional[bool] = None,
    cfg_overrides: Optional[Mapping[str, Any]] = None,
    search_cfg_overrides: Optional[Mapping[str, Any]] = None,
    quick_probe_settings: Optional[Mapping[str, Any]] = None,
    **single_metric_kwargs: Any,
) -> SurprisePotentialSweepBundle:
    metric_runs: Dict[str, SurprisePotentialRunBundle] = {}
    summary_frames = []
    method_frames = []
    artifact_frames = []
    error_rows = []

    metrics_clean = [str(m).strip().lower() for m in metrics if str(m).strip()]
    if not metrics_clean:
        raise ValueError("metrics is empty. Provide at least one surprise metric.")
    if counterfactual_families is None:
        families_clean = ['hist_prim']
    else:
        families_clean = [normalize_paper_counterfactual_family(x) for x in counterfactual_families if str(x).strip()]
    if not families_clean:
        raise ValueError("counterfactual_families is empty. Provide at least one family label.")

    if str(run_tag_base).strip():
        base_tag = str(run_tag_base).strip()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_tag = f"{_slugify(run_tag_prefix)}_{stamp}"

    for family in families_clean:
        family_slug = _slugify(family)
        for metric in metrics_clean:
            metric_slug = _slugify(metric)
            metric_run_tag = f"{base_tag}_{family_slug}_{metric_slug}"
            try:
                run_bundle = run_surprise_potential_single_metric(
                    metric=metric,
                    counterfactual_family=family,
                    run_tag=metric_run_tag,
                    persist_root=str(persist_root),
                    n_shards=int(n_shards),
                    shard_id=shard_id,
                    run_mode=str(run_mode),
                    planner_backend=str(planner_backend),
                    auto_run_main_loop_when_ready=bool(auto_run_main_loop_when_ready),
                    run_main_loop_override=run_main_loop_override,
                    run_tag_prefix=str(run_tag_prefix),
                    method_labels=method_labels,
                    cfg_overrides=cfg_overrides,
                    search_cfg_overrides=search_cfg_overrides,
                    quick_probe_settings=quick_probe_settings,
                    **single_metric_kwargs,
                )
                run_key = f"{family}|{metric}"
                metric_runs[run_key] = run_bundle
                summary_frames.append(run_bundle.summary_df)
                if not run_bundle.method_rollup_df.empty:
                    method_frames.append(run_bundle.method_rollup_df)
                if not run_bundle.artifacts_df.empty:
                    artifact_frames.append(run_bundle.artifacts_df)
            except Exception as exc:
                row = {
                    "counterfactual_family": str(family),
                    "metric": str(metric),
                    "run_tag": str(metric_run_tag),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                error_rows.append(row)
                if not bool(continue_on_error):
                    raise

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    method_rollup_df = pd.concat(method_frames, ignore_index=True) if method_frames else pd.DataFrame()
    artifacts_df = pd.concat(artifact_frames, ignore_index=True) if artifact_frames else pd.DataFrame()
    errors_df = pd.DataFrame(error_rows)
    return SurprisePotentialSweepBundle(
        metric_runs=metric_runs,
        summary_df=summary_df,
        method_rollup_df=method_rollup_df,
        artifacts_df=artifacts_df,
        errors_df=errors_df,
    )


def report_surprise_potential_sweep(
    bundle: SurprisePotentialSweepBundle,
    display_fn: Optional[Any] = None,
    preview_rows: int = 20,
) -> None:
    print(f"[sweep] completed metrics={len(bundle.metric_runs)}")
    if not bundle.errors_df.empty:
        print(f"[sweep] errors={len(bundle.errors_df)}")
    if display_fn is None:
        return

    if not bundle.summary_df.empty:
        display_fn(bundle.summary_df)
    if not bundle.method_rollup_df.empty:
        display_fn(bundle.method_rollup_df.head(int(max(1, preview_rows))))
    if not bundle.errors_df.empty:
        display_fn(bundle.errors_df)
    if not bundle.artifacts_df.empty:
        display_fn(bundle.artifacts_df.head(int(max(1, preview_rows))))
