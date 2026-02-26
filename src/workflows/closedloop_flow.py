from __future__ import annotations

import copy
from dataclasses import dataclass, field
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
class RunContextBundle:
    cfg: ClosedLoopConfig
    search_cfg: SearchConfig
    ckpt_scan_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    shard_progress_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    run_prefix: str = ""
    run_tag: str = ""
    persist_root: str = ""
    n_shards: int = 1
    shard_id: int = 0
    auto_run_main_loop_when_ready: bool = True
    run_main_loop_override: Optional[bool] = None


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
) -> RunContextBundle:
    cfg, search_cfg, ckpt_scan_df = initialize_configs()

    # Fast iteration defaults for Colab loops.
    cfg.n_eval_scenarios = int(max(1, n_eval_scenarios))
    cfg.strict_min_eval = int(max(1, strict_min_eval))
    cfg.n_total_scenarios = int(max(cfg.n_total_scenarios, n_total_scenarios_floor))

    # LatentDriver stability defaults for notebook runs.
    cfg.latentdriver_auto_align_token_count = bool(latentdriver_auto_align_token_count)
    cfg.latentdriver_log_forward_errors = bool(latentdriver_log_forward_errors)
    cfg.latentdriver_log_forward_errors_max = int(max(1, latentdriver_log_forward_errors_max))

    n_shards = int(max(1, n_shards))
    shard_progress_df = inspect_shard_progress(
        run_tag=str(run_tag),
        persist_root=str(persist_root),
        n_shards=n_shards,
    )

    if isinstance(shard_id, str) and shard_id.strip().lower() == "auto":
        shard_id_int = auto_select_shard_id(
            run_tag=str(run_tag),
            persist_root=str(persist_root),
            n_shards=n_shards,
        )
    else:
        shard_id_int = int(shard_id)

    run_prefix = configure_persistent_run_prefix(
        cfg=cfg,
        run_tag=str(run_tag),
        persist_root=str(persist_root),
        shard_id=int(shard_id_int),
        n_shards=n_shards,
    )

    return RunContextBundle(
        cfg=cfg,
        search_cfg=search_cfg,
        ckpt_scan_df=ckpt_scan_df,
        shard_progress_df=shard_progress_df,
        run_prefix=str(run_prefix),
        run_tag=str(run_tag),
        persist_root=str(persist_root),
        n_shards=int(n_shards),
        shard_id=int(shard_id_int),
        auto_run_main_loop_when_ready=bool(auto_run_main_loop_when_ready),
        run_main_loop_override=run_main_loop_override,
    )


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
    probe_budget_bump_per_escalation: int = 2,
    apply_successful_probe_tuning: bool = True,
    build_simulation_context: bool = True,
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
        final_collapsed=bool(final_collapsed),
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
    print("run_prefix =", bundle.run_prefix)
    print(f"[shard] running {bundle.shard_id + 1}/{max(1, bundle.n_shards)}")
    print(
        "[run-policy] "
        f"AUTO_RUN_MAIN_LOOP_WHEN_READY={bundle.auto_run_main_loop_when_ready}, "
        f"RUN_MAIN_LOOP_OVERRIDE={bundle.run_main_loop_override}"
    )
    if display_fn is not None:
        if len(bundle.shard_progress_df):
            display_fn(bundle.shard_progress_df)
        if len(bundle.ckpt_scan_df):
            display_fn(bundle.ckpt_scan_df.head(10))


def report_quick_probe_bundle(
    bundle: QuickProbeBundle,
    search_cfg: SearchConfig,
    display_fn: Optional[Any] = None,
    probe_preview_rows: int = 20,
) -> None:
    if display_fn is not None:
        if len(bundle.quick_probe_attempts_df):
            display_fn(bundle.quick_probe_attempts_df)
        if len(bundle.quick_probe_summary_df):
            display_fn(bundle.quick_probe_summary_df)
        if len(bundle.quick_probe_df):
            display_fn(bundle.quick_probe_df.head(int(max(1, probe_preview_rows))))

    if bundle.applied_tuning:
        print("[probe] applied tuned search settings from successful attempt.")
        print("        proposal_scale_ladder=", search_cfg.proposal_scale_ladder)
        print("        delta_l2_budget=", search_cfg.delta_l2_budget)
        print("        delta_clip=", search_cfg.delta_clip)
        print("        budget_evals=", search_cfg.budget_evals)

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
