from __future__ import annotations

from dataclasses import dataclass, field
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


DEFAULT_LABEL_COLUMNS = (
    'collision_h5', 'collision_h10', 'collision_h15',
    'offroad_h5', 'offroad_h10', 'offroad_h15',
    'failure_proxy_h5', 'failure_proxy_h10', 'failure_proxy_h15',
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
) -> UQBenchmarkFlowBundle:
    if dataset_df.empty:
        return UQBenchmarkFlowBundle()
    run_prefix = run_prefix or cfg.run_prefix
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
    )
