from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.closedloop.risk_benchmark import BenchmarkBundle, run_uq_benchmark, summarize_controller_tradeoff
from src.risk_model.artifacts import load_risk_artifacts, save_risk_evaluation_artifacts
from src.risk_model.inference import predict_calibrated_risk


@dataclass
class UQBenchmarkFlowBundle:
    predictions_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_bundle: BenchmarkBundle = field(default_factory=BenchmarkBundle)
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
        controller_summary_df=ctrl_df,
        artifact_paths=paths,
        loaded_from_existing=True,
    )


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
        return load_existing_uq_benchmark_bundle(run_prefix)

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

    controller_summary_df = pd.DataFrame()
    if isinstance(base_df, pd.DataFrame) and isinstance(controller_df, pd.DataFrame) and (not base_df.empty) and (not controller_df.empty):
        controller_summary_df = summarize_controller_tradeoff(base_df, controller_df)

    artifact_paths = save_risk_evaluation_artifacts(
        run_prefix,
        {
            'uq_benchmark_summary': benchmark_bundle.summary_df,
            'uq_benchmark_per_shift': benchmark_bundle.per_shift_df,
            'uq_reliability_bins': benchmark_bundle.reliability_df,
            'uq_selective_risk_curve': benchmark_bundle.selective_curve_df,
            'uq_shift_gap_summary': benchmark_bundle.shift_gap_df,
            'uq_predictions': pred_df,
            'risk_control_summary': controller_summary_df,
        },
    )
    return UQBenchmarkFlowBundle(
        predictions_df=pred_df,
        benchmark_bundle=benchmark_bundle,
        controller_summary_df=controller_summary_df,
        artifact_paths=artifact_paths,
        loaded_from_existing=False,
    )
