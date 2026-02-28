from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.closedloop.risk_benchmark import BenchmarkBundle, run_uq_benchmark
from src.risk_model.artifacts import save_risk_evaluation_artifacts
from .risk_training_flow import build_risk_dataset_from_runner, load_existing_risk_dataset_artifact


@dataclass
class MiscalibrationProbeBundle:
    predictions_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_bundle: BenchmarkBundle = field(default_factory=BenchmarkBundle)
    threshold_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    loaded_from_existing: bool = False


DEFAULT_VARIANTS = {
    'planner_top1_proxy': 'planner_risk_top1_proxy',
    'planner_entropy_proxy': 'planner_risk_entropy_proxy',
    'planner_combo_proxy': 'planner_risk_combo_proxy',
}
DEFAULT_LABEL_COLUMNS = ('failure_proxy_h15',)


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


def _probe_artifact_paths(run_prefix: str) -> Dict[str, str]:
    return {
        'miscalibration_probe_summary': f'{run_prefix}_miscalibration_probe_summary.csv',
        'miscalibration_probe_per_shift': f'{run_prefix}_miscalibration_probe_per_shift.csv',
        'miscalibration_probe_reliability_bins': f'{run_prefix}_miscalibration_probe_reliability_bins.csv',
        'miscalibration_probe_selective_risk_curve': f'{run_prefix}_miscalibration_probe_selective_risk_curve.csv',
        'miscalibration_probe_shift_gap_summary': f'{run_prefix}_miscalibration_probe_shift_gap_summary.csv',
        'miscalibration_probe_threshold_diagnostics': f'{run_prefix}_miscalibration_probe_threshold_diagnostics.csv',
        'miscalibration_probe_predictions': f'{run_prefix}_miscalibration_probe_predictions.parquet',
    }


def has_existing_miscalibration_probe_artifacts(run_prefix: str) -> bool:
    paths = _probe_artifact_paths(run_prefix)
    required = [
        paths['miscalibration_probe_summary'],
        paths['miscalibration_probe_per_shift'],
        paths['miscalibration_probe_reliability_bins'],
        paths['miscalibration_probe_shift_gap_summary'],
        paths['miscalibration_probe_threshold_diagnostics'],
    ]
    return all(Path(p).exists() for p in required)


def load_existing_miscalibration_probe_bundle(run_prefix: str) -> MiscalibrationProbeBundle:
    paths = _probe_artifact_paths(run_prefix)
    summary_df = _read_frame_with_parquet_fallback(paths['miscalibration_probe_summary'])
    per_shift_df = _read_frame_with_parquet_fallback(paths['miscalibration_probe_per_shift'])
    reliability_df = _read_frame_with_parquet_fallback(paths['miscalibration_probe_reliability_bins'])
    selective_curve_df = _read_frame_with_parquet_fallback(paths['miscalibration_probe_selective_risk_curve'])
    shift_gap_df = _read_frame_with_parquet_fallback(paths['miscalibration_probe_shift_gap_summary'])
    threshold_df = _read_frame_with_parquet_fallback(paths['miscalibration_probe_threshold_diagnostics'])
    predictions_df = _read_frame_with_parquet_fallback(paths['miscalibration_probe_predictions'])
    benchmark_bundle = BenchmarkBundle(
        summary_df=summary_df,
        per_shift_df=per_shift_df,
        reliability_df=reliability_df,
        selective_curve_df=selective_curve_df,
        shift_gap_df=shift_gap_df,
    )
    return MiscalibrationProbeBundle(
        predictions_df=predictions_df,
        benchmark_bundle=benchmark_bundle,
        threshold_df=threshold_df,
        artifact_paths=paths,
        loaded_from_existing=True,
    )


def _safe_prob(values: Any, *, default: float = 0.5) -> np.ndarray:
    arr = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
    arr = np.where(np.isfinite(arr), arr, float(default))
    return np.clip(arr, 1e-6, 1.0 - 1e-6)


def _safe_float(values: Any, *, default: float = 0.0) -> np.ndarray:
    arr = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
    arr = np.where(np.isfinite(arr), arr, float(default))
    return arr


def _planner_proxy_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    top1 = _safe_prob(out.get('dist_top1_weight', 0.5), default=0.5)
    entropy = _safe_float(out.get('dist_entropy', 0.0), default=0.0)
    n_comp = _safe_float(out.get('dist_num_components', 2.0), default=2.0)
    n_comp = np.maximum(2.0, n_comp)
    entropy_max = np.log(n_comp)
    entropy_max = np.where(np.isfinite(entropy_max) & (entropy_max > 1e-8), entropy_max, 1.0)
    conf_entropy = 1.0 - np.clip(entropy / entropy_max, 0.0, 1.0)
    conf_combo = np.clip(0.70 * top1 + 0.30 * conf_entropy, 0.0, 1.0)

    out['planner_conf_top1_proxy'] = top1
    out['planner_conf_entropy_proxy'] = conf_entropy
    out['planner_conf_combo_proxy'] = conf_combo
    out['planner_risk_top1_proxy'] = 1.0 - top1
    out['planner_risk_entropy_proxy'] = 1.0 - conf_entropy
    out['planner_risk_combo_proxy'] = 1.0 - conf_combo
    return out


def _prepare_eval_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'shift_suite' not in out.columns:
        out['shift_suite'] = 'nominal_clean'
    else:
        out['shift_suite'] = out['shift_suite'].fillna('nominal_clean')

    if 'eval_split' in out.columns:
        mask_holdout = out['eval_split'].astype(str).eq('high_interaction_holdout')
        out.loc[mask_holdout, 'shift_suite'] = 'high_interaction_holdout'

        eval_mask = out['eval_split'].astype(str).isin(['test', 'high_interaction_holdout'])
        eval_df = out.loc[eval_mask].copy()
        if not eval_df.empty:
            return eval_df
    return out


def _threshold_diagnostics(
    df: pd.DataFrame,
    *,
    variants: Mapping[str, str],
    label_col: str,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    tau = float(np.clip(threshold, 0.0, 1.0))
    for shift_suite, grp in df.groupby('shift_suite', sort=True):
        labels = _safe_float(grp[label_col], default=0.0)
        labels = (labels > 0.5).astype(float)
        for variant, prob_col in variants.items():
            if prob_col not in grp.columns:
                continue
            probs = _safe_prob(grp[prob_col], default=0.5)
            accepted = probs <= tau
            rejected = ~accepted
            n_rows = int(len(grp))
            accepted_count = int(np.sum(accepted))
            empirical_failure_given_accepted = float(np.mean(labels[accepted])) if accepted_count > 0 else np.nan
            rows.append(
                {
                    'shift_suite': str(shift_suite),
                    'variant': str(variant),
                    'label': str(label_col),
                    'threshold': tau,
                    'n_rows': n_rows,
                    'accepted_count': accepted_count,
                    'accepted_rate': float(np.mean(accepted)) if n_rows > 0 else np.nan,
                    'mean_predicted_risk': float(np.mean(probs)) if n_rows > 0 else np.nan,
                    'empirical_failure_rate': float(np.mean(labels)) if n_rows > 0 else np.nan,
                    'risk_gap_empirical_minus_predicted': float(np.mean(labels) - np.mean(probs)) if n_rows > 0 else np.nan,
                    'empirical_failure_given_accepted': empirical_failure_given_accepted,
                    'false_safe_rate': float(np.mean(accepted & (labels > 0.5))) if n_rows > 0 else np.nan,
                    'safe_rejected_rate': float(np.mean(rejected & (labels <= 0.5))) if n_rows > 0 else np.nan,
                    'budget_violated': int(bool(np.isfinite(empirical_failure_given_accepted) and (empirical_failure_given_accepted > tau))),
                }
            )
    return pd.DataFrame(rows)


def run_miscalibration_probe_flow(
    *,
    cfg: Any,
    dataset_df: Optional[pd.DataFrame] = None,
    runner: Optional[Any] = None,
    eval_idx: Optional[Iterable[int]] = None,
    run_prefix: Optional[str] = None,
    variants: Optional[Mapping[str, str]] = None,
    label_columns: Sequence[str] = DEFAULT_LABEL_COLUMNS,
    threshold: Optional[float] = None,
    resume_mode: str = 'auto',
    force_rerun: bool = False,
    persist_dataset_if_built: bool = True,
) -> MiscalibrationProbeBundle:
    mode = _normalize_resume_mode(resume_mode)
    run_prefix = run_prefix or cfg.run_prefix

    if bool((mode in {'auto', 'resume'}) and (not force_rerun) and has_existing_miscalibration_probe_artifacts(run_prefix)):
        return load_existing_miscalibration_probe_bundle(run_prefix)

    source_df = dataset_df.copy() if isinstance(dataset_df, pd.DataFrame) else pd.DataFrame()
    if source_df.empty:
        source_df = load_existing_risk_dataset_artifact(run_prefix)

    if source_df.empty and (runner is not None):
        dataset_bundle = build_risk_dataset_from_runner(
            runner=runner,
            cfg=cfg,
            scenario_ids=eval_idx,
            shift_suite='nominal_clean',
            persist=bool(persist_dataset_if_built),
            run_prefix=run_prefix,
        )
        source_df = dataset_bundle.dataset_df

    if source_df.empty:
        if mode == 'resume':
            raise ValueError(
                'resume_mode="resume" requested but no dataset was provided/found and no existing probe artifacts were present.'
            )
        return MiscalibrationProbeBundle()

    eval_df = _prepare_eval_frame(source_df)
    pred_df = _planner_proxy_predictions(eval_df)

    variants_map = dict(variants or DEFAULT_VARIANTS)
    variants_map = {k: v for k, v in variants_map.items() if v in pred_df.columns}
    if len(variants_map) == 0:
        raise ValueError('No valid planner-proxy variant columns available for miscalibration probe.')

    labels = [str(c) for c in list(label_columns) if str(c) in pred_df.columns]
    if len(labels) == 0:
        raise ValueError(f'None of the requested label columns are present: {tuple(label_columns)!r}')

    benchmark_bundle = run_uq_benchmark(
        pred_df,
        variants=variants_map,
        label_columns=tuple(labels),
        n_bins=int(getattr(cfg, 'uq_eval_probability_bins', 15)),
    )

    diag_threshold = float(threshold if threshold is not None else getattr(cfg, 'risk_control_fail_budget', 0.20))
    threshold_df = _threshold_diagnostics(
        pred_df,
        variants=variants_map,
        label_col=labels[0],
        threshold=diag_threshold,
    )

    artifact_paths = save_risk_evaluation_artifacts(
        run_prefix,
        {
            'miscalibration_probe_summary': benchmark_bundle.summary_df,
            'miscalibration_probe_per_shift': benchmark_bundle.per_shift_df,
            'miscalibration_probe_reliability_bins': benchmark_bundle.reliability_df,
            'miscalibration_probe_selective_risk_curve': benchmark_bundle.selective_curve_df,
            'miscalibration_probe_shift_gap_summary': benchmark_bundle.shift_gap_df,
            'miscalibration_probe_threshold_diagnostics': threshold_df,
            'miscalibration_probe_predictions': pred_df,
        },
    )
    return MiscalibrationProbeBundle(
        predictions_df=pred_df,
        benchmark_bundle=benchmark_bundle,
        threshold_df=threshold_df,
        artifact_paths=artifact_paths,
        loaded_from_existing=False,
    )
