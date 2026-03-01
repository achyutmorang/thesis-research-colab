from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.risk_model.benchmark import BenchmarkBundle, run_uq_benchmark, summarize_controller_tradeoff
from src.risk_model.control import select_action_with_calibrated_risk
from src.risk_model.artifacts import load_risk_artifacts, save_risk_evaluation_artifacts
from src.risk_model.inference import predict_calibrated_risk


@dataclass
class UQBenchmarkFlowBundle:
    predictions_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_bundle: BenchmarkBundle = field(default_factory=BenchmarkBundle)
    controller_per_step_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    controller_per_scenario_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    controller_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    loaded_from_existing: bool = False


DEFAULT_LABEL_COLUMNS = (
    'collision_h5', 'collision_h10', 'collision_h15',
    'offroad_h5', 'offroad_h10', 'offroad_h15',
    'failure_proxy_h5', 'failure_proxy_h10', 'failure_proxy_h15',
)


def _normalize_resume_mode(value: Any) -> str:
    mode = str(value if value is not None else 'auto').strip().lower()
    if mode not in {'auto', 'fresh', 'resume'}:
        raise ValueError(f"Invalid resume_mode={value!r}. Expected one of: auto, fresh, resume.")
    return mode


def _read_frame_with_parquet_fallback(preferred_path: str) -> pd.DataFrame:
    p = Path(preferred_path)
    if p.exists():
        try:
            if p.suffix.lower() == '.parquet':
                return pd.read_parquet(p)
            return pd.read_csv(p)
        except (pd.errors.EmptyDataError, ValueError):
            return pd.DataFrame()
    if p.suffix.lower() == '.parquet':
        csv_path = p.with_suffix('.csv')
        if csv_path.exists():
            try:
                return pd.read_csv(csv_path)
            except (pd.errors.EmptyDataError, ValueError):
                return pd.DataFrame()
    return pd.DataFrame()


def _benchmark_artifact_paths(run_prefix: str) -> Dict[str, str]:
    return {
        'uq_benchmark_summary': f'{run_prefix}_uq_benchmark_summary.csv',
        'uq_benchmark_per_shift': f'{run_prefix}_uq_benchmark_per_shift.csv',
        'uq_reliability_bins': f'{run_prefix}_uq_reliability_bins.csv',
        'uq_selective_risk_curve': f'{run_prefix}_uq_selective_risk_curve.csv',
        'uq_shift_gap_summary': f'{run_prefix}_uq_shift_gap_summary.csv',
        'uq_predictions': f'{run_prefix}_uq_predictions.parquet',
        'risk_control_per_step_trace': f'{run_prefix}_risk_control_per_step_trace.csv',
        'risk_control_per_scenario_results': f'{run_prefix}_risk_control_per_scenario_results.csv',
        'risk_control_summary': f'{run_prefix}_risk_control_summary.csv',
    }


def has_existing_uq_benchmark_artifacts(run_prefix: str) -> bool:
    paths = _benchmark_artifact_paths(run_prefix)
    required = [
        paths['uq_benchmark_summary'],
        paths['uq_benchmark_per_shift'],
        paths['uq_reliability_bins'],
        paths['uq_selective_risk_curve'],
        paths['uq_shift_gap_summary'],
    ]
    return all(Path(p).exists() for p in required)


def load_existing_uq_benchmark_bundle(run_prefix: str) -> UQBenchmarkFlowBundle:
    paths = _benchmark_artifact_paths(run_prefix)
    summary_df = _read_frame_with_parquet_fallback(paths['uq_benchmark_summary'])
    per_shift_df = _read_frame_with_parquet_fallback(paths['uq_benchmark_per_shift'])
    reliability_df = _read_frame_with_parquet_fallback(paths['uq_reliability_bins'])
    selective_curve_df = _read_frame_with_parquet_fallback(paths['uq_selective_risk_curve'])
    shift_gap_df = _read_frame_with_parquet_fallback(paths['uq_shift_gap_summary'])
    pred_df = _read_frame_with_parquet_fallback(paths['uq_predictions'])
    ctrl_step_df = _read_frame_with_parquet_fallback(paths['risk_control_per_step_trace'])
    ctrl_scenario_df = _read_frame_with_parquet_fallback(paths['risk_control_per_scenario_results'])
    ctrl_df = _read_frame_with_parquet_fallback(paths['risk_control_summary'])
    bundle = BenchmarkBundle(
        summary_df=summary_df,
        per_shift_df=per_shift_df,
        reliability_df=reliability_df,
        selective_curve_df=selective_curve_df,
        shift_gap_df=shift_gap_df,
    )
    return UQBenchmarkFlowBundle(
        predictions_df=pred_df,
        benchmark_bundle=bundle,
        controller_per_step_df=ctrl_step_df,
        controller_per_scenario_df=ctrl_scenario_df,
        controller_summary_df=ctrl_df,
        artifact_paths=paths,
        loaded_from_existing=True,
    )


def _horizon_column(base: str, horizon: int) -> str:
    return f'{base}_h{int(horizon)}'


def _derive_base_and_controller_frames(
    pred_df: pd.DataFrame,
    cfg: Any,
    *,
    failure_label: str = 'failure_proxy_h15',
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if pred_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if ('scenario_id' not in pred_df.columns) or ('step_idx' not in pred_df.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    horizon = int(max(1, getattr(cfg, 'risk_dataset_control_horizon_steps', 6)))
    progress_col = _horizon_column('progress', horizon)
    fallback_progress_col = 'progress_h6'
    if progress_col not in pred_df.columns and fallback_progress_col in pred_df.columns:
        progress_col = fallback_progress_col
    fail_col = failure_label if failure_label in pred_df.columns else 'failure_proxy_h15'

    action_cols = [c for c in ['action_0', 'action_1', 'action_2'] if c in pred_df.columns]
    planner_action_cols = [c for c in ['planner_action_0', 'planner_action_1', 'planner_action_2'] if c in pred_df.columns]

    group_cols = ['scenario_id', 'step_idx']
    if 'shift_suite' in pred_df.columns:
        group_cols.append('shift_suite')

    step_rows = []
    base_rows = []
    ctrl_rows = []

    for _, grp in pred_df.groupby(group_cols, sort=False):
        work = grp.copy()
        if work.empty:
            continue

        # Approximate base planner choice with nearest candidate to planner action.
        if (len(action_cols) > 0) and (len(planner_action_cols) == len(action_cols)):
            dist = np.zeros((len(work),), dtype=float)
            for ac, pc in zip(action_cols, planner_action_cols):
                dist += np.square(pd.to_numeric(work[ac], errors='coerce').to_numpy(dtype=float) - pd.to_numeric(work[pc], errors='coerce').to_numpy(dtype=float))
            base_idx = int(np.nanargmin(np.where(np.isfinite(dist), dist, np.inf)))
        elif 'candidate_id' in work.columns:
            base_idx = int(np.argmin(pd.to_numeric(work['candidate_id'], errors='coerce').fillna(0).to_numpy(dtype=float)))
        else:
            base_idx = 0
        base_row = work.iloc[int(base_idx)]

        try:
            ctrl_sel = select_action_with_calibrated_risk(work, cfg=cfg, failure_label=str(fail_col))
            ctrl_row = pd.Series(ctrl_sel.selected_row)
        except Exception:
            ctrl_row = base_row.copy()
            ctrl_row['fallback_reason'] = 'controller_selection_exception'

        step = {
            'scenario_id': int(base_row.get('scenario_id', -1)),
            'step_idx': int(base_row.get('step_idx', -1)),
            'shift_suite': str(base_row.get('shift_suite', 'nominal_clean')),
            'base_candidate_id': int(base_row.get('candidate_id', -1)) if pd.notna(base_row.get('candidate_id', np.nan)) else -1,
            'ctrl_candidate_id': int(ctrl_row.get('candidate_id', -1)) if pd.notna(ctrl_row.get('candidate_id', np.nan)) else -1,
            'base_progress': float(base_row.get(progress_col, np.nan)),
            'ctrl_progress': float(ctrl_row.get(progress_col, np.nan)),
            'base_failure': float(base_row.get(fail_col, np.nan)),
            'ctrl_failure': float(ctrl_row.get(fail_col, np.nan)),
            'base_risk_cal_failure_proxy_h15': float(base_row.get('risk_cal_failure_proxy_h15', np.nan)),
            'ctrl_risk_cal_failure_proxy_h15': float(ctrl_row.get('risk_cal_failure_proxy_h15', np.nan)),
            'base_risk_epistemic_failure_proxy_h15': float(base_row.get('risk_epistemic_failure_proxy_h15', np.nan)),
            'ctrl_risk_epistemic_failure_proxy_h15': float(ctrl_row.get('risk_epistemic_failure_proxy_h15', np.nan)),
            'ctrl_budget_ok': int(float(ctrl_row.get('_budget_ok', np.nan)) > 0.5) if pd.notna(ctrl_row.get('_budget_ok', np.nan)) else 0,
            'ctrl_primary_score': float(ctrl_row.get('_primary_score', np.nan)),
            'ctrl_fallback_reason': str(ctrl_row.get('fallback_reason', '')),
        }
        step_rows.append(step)
        base_rows.append({
            'scenario_id': step['scenario_id'],
            'shift_suite': step['shift_suite'],
            progress_col: step['base_progress'],
            fail_col: step['base_failure'],
        })
        ctrl_rows.append({
            'scenario_id': step['scenario_id'],
            'shift_suite': step['shift_suite'],
            progress_col: step['ctrl_progress'],
            fail_col: step['ctrl_failure'],
        })

    step_df = pd.DataFrame(step_rows)
    base_df = pd.DataFrame(base_rows)
    controller_df = pd.DataFrame(ctrl_rows)
    if step_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Scenario-level aggregation for unbiased scenario comparisons.
    agg_map = {progress_col: 'mean', fail_col: 'mean'}
    base_scenario = (
        base_df.groupby(['scenario_id', 'shift_suite'], as_index=False)
        .agg(agg_map)
        .rename(columns={progress_col: 'base_progress', fail_col: 'base_failure'})
    )
    ctrl_scenario = (
        controller_df.groupby(['scenario_id', 'shift_suite'], as_index=False)
        .agg(agg_map)
        .rename(columns={progress_col: 'ctrl_progress', fail_col: 'ctrl_failure'})
    )
    scenario_df = base_scenario.merge(ctrl_scenario, on=['scenario_id', 'shift_suite'], how='inner')
    if not scenario_df.empty:
        scenario_df['relative_progress_change'] = (
            (scenario_df['ctrl_progress'] - scenario_df['base_progress'])
            / np.maximum(1e-6, np.abs(scenario_df['base_progress']))
        )
        scenario_df['failure_rate_delta'] = scenario_df['ctrl_failure'] - scenario_df['base_failure']
    base_scenario_df = pd.DataFrame(
        {
            'scenario_id': scenario_df['scenario_id'] if not scenario_df.empty else pd.Series(dtype=int),
            progress_col: scenario_df['base_progress'] if not scenario_df.empty else pd.Series(dtype=float),
            fail_col: scenario_df['base_failure'] if not scenario_df.empty else pd.Series(dtype=float),
        }
    )
    controller_scenario_df = pd.DataFrame(
        {
            'scenario_id': scenario_df['scenario_id'] if not scenario_df.empty else pd.Series(dtype=int),
            progress_col: scenario_df['ctrl_progress'] if not scenario_df.empty else pd.Series(dtype=float),
            fail_col: scenario_df['ctrl_failure'] if not scenario_df.empty else pd.Series(dtype=float),
        }
    )
    return step_df, scenario_df, base_scenario_df, controller_scenario_df


def run_uq_benchmark_flow(
    *,
    cfg: Any,
    dataset_df: pd.DataFrame,
    run_prefix: Optional[str] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    variants: Optional[Mapping[str, str]] = None,
    label_columns: Sequence[str] = ('failure_proxy_h15',),
    base_df: Optional[pd.DataFrame] = None,
    controller_df: Optional[pd.DataFrame] = None,
    resume_mode: str = 'auto',
    force_rerun: bool = False,
) -> UQBenchmarkFlowBundle:
    mode = _normalize_resume_mode(resume_mode)
    run_prefix = run_prefix or cfg.run_prefix
    if bool((mode in {'auto', 'resume'}) and (not force_rerun) and has_existing_uq_benchmark_artifacts(run_prefix)):
        existing = load_existing_uq_benchmark_bundle(run_prefix)
        return existing

    if dataset_df.empty:
        if mode == 'resume':
            raise ValueError('resume_mode=\"resume\" requested but dataset_df is empty and no benchmark artifacts were found.')
        return UQBenchmarkFlowBundle()

    if artifacts is None:
        artifacts = load_risk_artifacts(run_prefix)

    eval_df = dataset_df.copy()
    if 'eval_split' in eval_df.columns:
        eval_df = eval_df[eval_df['eval_split'].isin(['test', 'high_interaction_holdout'])].copy()
        if eval_df.empty:
            eval_df = dataset_df.copy()

    pred_df = predict_calibrated_risk(
        model=artifacts['model'],
        df=eval_df,
        feature_columns=artifacts['feature_columns'],
        label_columns=artifacts.get('label_columns', DEFAULT_LABEL_COLUMNS),
        feature_mean=np.asarray(artifacts['feature_mean'], dtype=float),
        feature_std=np.asarray(artifacts['feature_std'], dtype=float),
        scalers=artifacts.get('temperature_scalers', {}),
        conformal_thresholds=artifacts.get('conformal_thresholds', {}),
    )
    benchmark_bundle = run_uq_benchmark(
        pred_df,
        variants=variants or {'raw': 'risk_raw_failure_proxy_h15', 'cal': 'risk_cal_failure_proxy_h15'},
        label_columns=label_columns,
        n_bins=int(getattr(cfg, 'uq_eval_probability_bins', 15)),
    )

    controller_step_df = pd.DataFrame()
    controller_per_scenario_df = pd.DataFrame()
    controller_summary_df = pd.DataFrame()
    if isinstance(base_df, pd.DataFrame) and isinstance(controller_df, pd.DataFrame) and (not base_df.empty) and (not controller_df.empty):
        controller_summary_df = summarize_controller_tradeoff(base_df, controller_df)
    else:
        # Default path: derive an offline controller-vs-base comparison from candidate predictions.
        controller_step_df, controller_per_scenario_df, derived_base_df, derived_ctrl_df = _derive_base_and_controller_frames(
            pred_df,
            cfg=cfg,
            failure_label='failure_proxy_h15',
        )
        if (not derived_base_df.empty) and (not derived_ctrl_df.empty):
            controller_summary_df = summarize_controller_tradeoff(derived_base_df, derived_ctrl_df)

    artifact_paths = save_risk_evaluation_artifacts(
        run_prefix,
        {
            'uq_benchmark_summary': benchmark_bundle.summary_df,
            'uq_benchmark_per_shift': benchmark_bundle.per_shift_df,
            'uq_reliability_bins': benchmark_bundle.reliability_df,
            'uq_selective_risk_curve': benchmark_bundle.selective_curve_df,
            'uq_shift_gap_summary': benchmark_bundle.shift_gap_df,
            'uq_predictions': pred_df,
            'risk_control_per_step_trace': controller_step_df,
            'risk_control_per_scenario_results': controller_per_scenario_df,
            'risk_control_summary': controller_summary_df,
        },
    )
    return UQBenchmarkFlowBundle(
        predictions_df=pred_df,
        benchmark_bundle=benchmark_bundle,
        controller_per_step_df=controller_step_df,
        controller_per_scenario_df=controller_per_scenario_df,
        controller_summary_df=controller_summary_df,
        artifact_paths=artifact_paths,
        loaded_from_existing=False,
    )
