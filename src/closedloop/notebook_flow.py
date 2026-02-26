from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .calibration import (
    diagnose_surprise_root_cause,
    run_closedloop_preflight_checks,
    run_surprise_quality_gate,
)
from .config import ClosedLoopConfig, SearchConfig
from .core import (
    build_closedloop_runner_and_splits,
    make_waymax_data_iter,
    run_closed_loop,
    run_preflight_and_calibration,
    run_quick_surprise_probe,
)
from .resume_io import export_closedloop_artifacts, summarize_method_outputs
from .signal_analysis import analyze_surprise_signal_usefulness


def _float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


@dataclass
class QuickProbeBundle:
    cfg: ClosedLoopConfig
    search_cfg: SearchConfig
    quick_probe_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    quick_probe_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    quick_probe_attempts_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    final_collapsed: bool = True
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


def _probe_collapse_stats(probe_summary_df: pd.DataFrame) -> Tuple[bool, int, float, float, float]:
    if len(probe_summary_df) == 0:
        return True, 0, 0.0, 0.0, 0.0
    row = probe_summary_df.iloc[0]
    n_finite = int(max(0, _float_or_default(row.get("n_finite_surprise"), 0.0)))
    nonzero = _float_or_default(row.get("nonzero_surprise_fraction"), 0.0)
    realized = _float_or_default(row.get("proposal_realized_fraction"), 0.0)
    effect_l2 = _float_or_default(row.get("proposal_effect_l2_mean"), 0.0)
    collapsed = bool((n_finite <= 0) or (nonzero <= 0.0) or (realized <= 0.0) or (effect_l2 <= 1e-9))
    return collapsed, n_finite, nonzero, realized, effect_l2


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
    probe_budget_bump_per_escalation: int = 2,
    apply_successful_probe_tuning: bool = True,
) -> QuickProbeBundle:
    scale_mults = tuple(float(x) for x in probe_scale_multipliers) or (1.0,)
    l2_mults = tuple(float(x) for x in probe_delta_l2_multipliers) or (1.0,)
    clip_mults = tuple(float(x) for x in probe_delta_clip_multipliers) or (1.0,)

    selected_cfg = cfg
    selected_search_cfg = search_cfg
    selected_probe_df = pd.DataFrame()
    selected_probe_summary_df = pd.DataFrame()
    attempt_rows = []
    final_collapsed = False
    applied_tuning = False

    if bool(run_quick_surprise_probe_enabled):
        max_attempts = int(max_probe_escalations) if bool(auto_escalate_quick_probe) else 1
        max_attempts = max(1, max_attempts)

        selected_cfg = None
        selected_search_cfg = None

        for attempt in range(max_attempts):
            scale_mult = float(scale_mults[min(attempt, len(scale_mults) - 1)])
            l2_mult = float(l2_mults[min(attempt, len(l2_mults) - 1)])
            clip_mult = float(clip_mults[min(attempt, len(clip_mults) - 1)])

            cfg_trial = copy.deepcopy(cfg)
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
            collapsed, n_finite, nonzero, realized, effect_l2 = _probe_collapse_stats(probe_summary_df)

            attempt_rows.append(
                {
                    "attempt": int(attempt + 1),
                    "collapsed": int(collapsed),
                    "scale_mult": float(scale_mult),
                    "delta_l2_budget": float(search_trial.delta_l2_budget),
                    "delta_clip": float(search_trial.delta_clip),
                    "budget_evals": int(search_trial.budget_evals),
                    "n_finite_surprise": int(n_finite),
                    "nonzero_surprise_fraction": float(nonzero),
                    "proposal_realized_fraction": float(realized),
                    "proposal_effect_l2_mean": float(effect_l2),
                }
            )
            selected_probe_df = probe_df.copy()
            selected_probe_summary_df = probe_summary_df.copy()
            final_collapsed = bool(collapsed)

            if not collapsed:
                selected_cfg = cfg_trial
                selected_search_cfg = search_trial
                break

        if (not final_collapsed) and bool(apply_successful_probe_tuning) and (selected_cfg is not None) and (selected_search_cfg is not None):
            cfg = selected_cfg
            search_cfg = selected_search_cfg
            applied_tuning = True

        if final_collapsed and bool(stop_if_quick_probe_collapsed):
            raise RuntimeError("Quick surprise probe collapsed after escalation attempts. Stop before full run.")

    quick_probe_attempts_df = pd.DataFrame(attempt_rows)
    if selected_cfg is None:
        selected_cfg = cfg
    if selected_search_cfg is None:
        selected_search_cfg = search_cfg

    dataset_config, data_iter = make_waymax_data_iter(selected_cfg)
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
        cfg=selected_cfg,
        data_iter=data_iter,
        dataset_config=dataset_config,
        n_shards=int(n_shards),
        shard_id=int(shard_id),
    )

    return QuickProbeBundle(
        cfg=selected_cfg,
        search_cfg=selected_search_cfg,
        quick_probe_df=selected_probe_df,
        quick_probe_summary_df=selected_probe_summary_df,
        quick_probe_attempts_df=quick_probe_attempts_df,
        final_collapsed=bool(final_collapsed),
        applied_tuning=bool(applied_tuning),
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
