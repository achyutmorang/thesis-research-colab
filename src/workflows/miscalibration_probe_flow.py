from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.closedloop.risk_benchmark import BenchmarkBundle, binary_auroc, run_uq_benchmark
from src.risk_model.artifacts import save_risk_evaluation_artifacts
from .risk_training_flow import build_risk_dataset_from_runner, load_existing_risk_dataset_artifact
try:
    from .living_report import update_living_report_from_miscalibration_probe
except ImportError:  # pragma: no cover - supports direct module loading in tests
    from src.workflows.living_report import update_living_report_from_miscalibration_probe


@dataclass
class MiscalibrationProbeBundle:
    predictions_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_bundle: BenchmarkBundle = field(default_factory=BenchmarkBundle)
    threshold_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    leakage_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    shift_profile_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    class_balance_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    proxy_calibration_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    loaded_from_existing: bool = False


DEFAULT_VARIANTS = {
    'planner_top1_raw': 'planner_risk_top1_proxy',
    'planner_top1_platt': 'planner_risk_top1_platt',
    'planner_entropy_raw': 'planner_risk_entropy_proxy',
    'planner_entropy_platt': 'planner_risk_entropy_platt',
    'planner_combo_raw': 'planner_risk_combo_proxy',
    'planner_combo_platt': 'planner_risk_combo_platt',
}
DEFAULT_LABEL_COLUMNS = ('failure_proxy_h15',)
RAW_PROXY_COLUMNS = (
    'planner_risk_top1_proxy',
    'planner_risk_entropy_proxy',
    'planner_risk_combo_proxy',
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


def _probe_artifact_paths(run_prefix: str) -> Dict[str, str]:
    return {
        'miscalibration_probe_summary': f'{run_prefix}_miscalibration_probe_summary.csv',
        'miscalibration_probe_per_shift': f'{run_prefix}_miscalibration_probe_per_shift.csv',
        'miscalibration_probe_reliability_bins': f'{run_prefix}_miscalibration_probe_reliability_bins.csv',
        'miscalibration_probe_selective_risk_curve': f'{run_prefix}_miscalibration_probe_selective_risk_curve.csv',
        'miscalibration_probe_shift_gap_summary': f'{run_prefix}_miscalibration_probe_shift_gap_summary.csv',
        'miscalibration_probe_threshold_diagnostics': f'{run_prefix}_miscalibration_probe_threshold_diagnostics.csv',
        'miscalibration_probe_leakage_checks': f'{run_prefix}_miscalibration_probe_leakage_checks.csv',
        'miscalibration_probe_shift_profile': f'{run_prefix}_miscalibration_probe_shift_profile.csv',
        'miscalibration_probe_class_balance': f'{run_prefix}_miscalibration_probe_class_balance.csv',
        'miscalibration_probe_proxy_calibration': f'{run_prefix}_miscalibration_probe_proxy_calibration.csv',
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
    leakage_df = _read_frame_with_parquet_fallback(paths['miscalibration_probe_leakage_checks'])
    shift_profile_df = _read_frame_with_parquet_fallback(paths['miscalibration_probe_shift_profile'])
    class_balance_df = _read_frame_with_parquet_fallback(paths['miscalibration_probe_class_balance'])
    proxy_calibration_df = _read_frame_with_parquet_fallback(paths['miscalibration_probe_proxy_calibration'])
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
        leakage_df=leakage_df,
        shift_profile_df=shift_profile_df,
        class_balance_df=class_balance_df,
        proxy_calibration_df=proxy_calibration_df,
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


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


def _fit_binary_logistic_1d(score: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(score, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = (y[valid] > 0.5).astype(float)
    if x.size <= 2:
        prior = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6)) if y.size > 0 else 0.5
        return 0.0, float(np.log(prior / (1.0 - prior)))
    if np.all(y <= 0.0) or np.all(y >= 1.0):
        prior = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
        return 0.0, float(np.log(prior / (1.0 - prior)))

    x_mean = float(np.mean(x))
    x_std = float(np.std(x))
    x_std = x_std if x_std > 1e-8 else 1.0
    z = (x - x_mean) / x_std
    X = np.column_stack([z, np.ones_like(z)])
    w = np.zeros((2,), dtype=float)
    reg = 1e-4

    for _ in range(80):
        logits = X @ w
        p = _sigmoid(logits)
        grad = (X.T @ (p - y)) / float(max(1, y.size))
        grad[0] += reg * w[0]

        s = p * (1.0 - p)
        H = (X.T @ (X * s[:, None])) / float(max(1, y.size))
        H[0, 0] += reg
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        w = w - step
        if float(np.linalg.norm(step)) < 1e-7:
            break

    alpha = float(w[0] / x_std)
    beta = float(w[1] - (w[0] * x_mean / x_std))
    return alpha, beta


def _apply_binary_logistic_1d(score: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    x = np.asarray(score, dtype=float)
    return _sigmoid(alpha * x + beta)


def _fit_proxy_calibrators(df: pd.DataFrame, *, label_col: str) -> Tuple[Dict[str, Tuple[float, float]], pd.DataFrame]:
    work = df.copy()
    if 'eval_split' in work.columns:
        val_df = work[work['eval_split'].astype(str).eq('val')].copy()
    else:
        val_df = pd.DataFrame()
    if val_df.empty:
        val_df = work

    calibrators: Dict[str, Tuple[float, float]] = {}
    rows = []
    for col in RAW_PROXY_COLUMNS:
        if col not in work.columns:
            continue
        alpha, beta = _fit_binary_logistic_1d(
            score=_safe_float(val_df[col], default=0.5),
            labels=_safe_float(val_df[label_col], default=0.0),
        )
        calibrators[col] = (alpha, beta)
        p_val = _apply_binary_logistic_1d(_safe_float(val_df[col], default=0.5), alpha=alpha, beta=beta)
        y_val = (_safe_float(val_df[label_col], default=0.0) > 0.5).astype(float)
        rows.append(
            {
                'proxy_column': col,
                'alpha': float(alpha),
                'beta': float(beta),
                'n_calibration_rows': int(len(val_df)),
                'val_positive_rate': float(np.mean(y_val)) if len(y_val) > 0 else np.nan,
                'val_mean_raw_proxy': float(np.mean(_safe_float(val_df[col], default=0.5))) if len(val_df) > 0 else np.nan,
                'val_mean_platt_prob': float(np.mean(p_val)) if len(p_val) > 0 else np.nan,
            }
        )
    return calibrators, pd.DataFrame(rows)


def _apply_proxy_calibrators(df: pd.DataFrame, calibrators: Mapping[str, Tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for raw_col, (alpha, beta) in calibrators.items():
        if raw_col not in out.columns:
            continue
        suffix = raw_col.replace('_proxy', '_platt')
        out[suffix] = _apply_binary_logistic_1d(_safe_float(out[raw_col], default=0.5), alpha=float(alpha), beta=float(beta))
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


def _class_balance_summary(df: pd.DataFrame, label_columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    split_col = 'eval_split' if 'eval_split' in df.columns else None
    for label in label_columns:
        if label not in df.columns:
            continue
        for shift_suite, grp in df.groupby('shift_suite', sort=True):
            y = (_safe_float(grp[label], default=0.0) > 0.5).astype(float)
            row = {
                'label': str(label),
                'shift_suite': str(shift_suite),
                'n_rows': int(len(grp)),
                'positive_rate': float(np.mean(y)) if len(y) > 0 else np.nan,
            }
            if split_col is not None:
                split_vals = grp[split_col].astype(str).value_counts(dropna=False).to_dict()
                for k, v in split_vals.items():
                    row[f'split_count_{k}'] = int(v)
            rows.append(row)
    return pd.DataFrame(rows)


def _shift_profile_summary(df: pd.DataFrame, *, label_col: str) -> pd.DataFrame:
    features = [
        'dist_top1_weight',
        'dist_entropy',
        'dist_std_mean',
        'dist_std_max',
        'belief_kl_current',
        'belief_kl_rolling_mean',
        'predictive_seq_kl_nominal',
        'predictive_seq_w2_nominal',
        'progress_h6',
        'min_ttc_h6',
        'min_distance_h6',
        'max_abs_acc_h6',
        'max_abs_jerk_h6',
        'target_interaction_score',
    ]
    features = [f for f in features if f in df.columns]
    if len(features) == 0:
        return pd.DataFrame()

    rows = []
    nominal = df[df['shift_suite'].astype(str).eq('nominal_clean')]
    nominal_means = {}
    for feature in features:
        nominal_means[feature] = float(np.mean(_safe_float(nominal[feature], default=np.nan))) if not nominal.empty else np.nan

    for shift_suite, grp in df.groupby('shift_suite', sort=True):
        for feature in features:
            values = _safe_float(grp[feature], default=np.nan)
            mean_val = float(np.nanmean(values)) if values.size > 0 else np.nan
            std_val = float(np.nanstd(values)) if values.size > 0 else np.nan
            rows.append(
                {
                    'shift_suite': str(shift_suite),
                    'feature': str(feature),
                    'n_rows': int(len(grp)),
                    'mean': mean_val,
                    'std': std_val,
                    'nominal_mean': float(nominal_means.get(feature, np.nan)),
                    'delta_vs_nominal': float(mean_val - nominal_means.get(feature, np.nan)),
                }
            )

        if label_col in grp.columns:
            y = (_safe_float(grp[label_col], default=0.0) > 0.5).astype(float)
            rows.append(
                {
                    'shift_suite': str(shift_suite),
                    'feature': f'{label_col}_positive_rate',
                    'n_rows': int(len(grp)),
                    'mean': float(np.mean(y)) if len(y) > 0 else np.nan,
                    'std': np.nan,
                    'nominal_mean': np.nan,
                    'delta_vs_nominal': np.nan,
                }
            )

    return pd.DataFrame(rows)


def _extract_horizon(name: str) -> Optional[int]:
    m = re.search(r'_h(\d+)\b', str(name))
    if not m:
        return None
    return int(m.group(1))


def _leakage_checks(df: pd.DataFrame, *, label_col: str) -> pd.DataFrame:
    rows = []
    if label_col not in df.columns:
        return pd.DataFrame()

    # Candidate identity leakage proxy check: if within-(scenario,step) label shuffle
    # preserves AUROC, features may encode target too directly.
    score_col = 'planner_risk_combo_proxy' if 'planner_risk_combo_proxy' in df.columns else None
    if score_col is not None:
        y = (_safe_float(df[label_col], default=0.0) > 0.5).astype(float)
        s = _safe_prob(df[score_col], default=0.5)
        base_auc = float(binary_auroc(s, y))

        shuffled = y.copy()
        rng = np.random.default_rng(17)
        if ('scenario_id' in df.columns) and ('step_idx' in df.columns):
            for _, idx in df.groupby(['scenario_id', 'step_idx'], sort=False).groups.items():
                idx_arr = np.asarray(list(idx), dtype=int)
                if idx_arr.size > 1:
                    shuffled[idx_arr] = shuffled[rng.permutation(idx_arr)]
        else:
            shuffled = shuffled[rng.permutation(len(shuffled))]
        shuf_auc = float(binary_auroc(s, shuffled))
        auc_gap = float(base_auc - shuf_auc) if np.isfinite(base_auc) and np.isfinite(shuf_auc) else np.nan
        rows.append(
            {
                'check': 'candidate_identity_shuffle_auroc_gap',
                'pass': int(bool(np.isfinite(auc_gap) and (auc_gap > 0.02))),
                'base_auroc': base_auc,
                'shuffled_auroc': shuf_auc,
                'auroc_gap': auc_gap,
                'detail': 'expects noticeable drop after within-(scenario,step) label shuffle',
            }
        )

    # Temporal guard check: no feature horizon should exceed label horizon.
    target_h = _extract_horizon(label_col)
    if target_h is not None:
        excluded = set([label_col, *DEFAULT_LABEL_COLUMNS, *RAW_PROXY_COLUMNS])
        feature_horizons = []
        for col in df.columns:
            if col in excluded:
                continue
            h = _extract_horizon(col)
            if h is not None:
                feature_horizons.append((str(col), int(h)))
        violating = [c for c, h in feature_horizons if int(h) > int(target_h)]
        rows.append(
            {
                'check': 'temporal_horizon_guard',
                'pass': int(len(violating) == 0),
                'base_auroc': np.nan,
                'shuffled_auroc': np.nan,
                'auroc_gap': np.nan,
                'detail': 'no feature horizon should exceed label horizon',
                'target_horizon': int(target_h),
                'violating_feature_count': int(len(violating)),
                'violating_features': ';'.join(sorted(violating)[:20]),
            }
        )
    return pd.DataFrame(rows)


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
        existing = load_existing_miscalibration_probe_bundle(run_prefix)
        existing.artifact_paths.update(
            update_living_report_from_miscalibration_probe(
                cfg=cfg,
                run_prefix=run_prefix,
                summary_df=existing.benchmark_bundle.summary_df,
                per_shift_df=existing.benchmark_bundle.per_shift_df,
                threshold_df=existing.threshold_df,
                leakage_df=existing.leakage_df,
                class_balance_df=existing.class_balance_df,
                artifact_paths=existing.artifact_paths,
            )
        )
        return existing

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

    labels = [str(c) for c in list(label_columns) if str(c) in pred_df.columns]
    if len(labels) == 0:
        raise ValueError(f'None of the requested label columns are present: {tuple(label_columns)!r}')
    target_label = str(labels[0])

    proxy_calibrators, proxy_calibration_df = _fit_proxy_calibrators(pred_df, label_col=target_label)
    pred_df = _apply_proxy_calibrators(pred_df, proxy_calibrators)

    variants_map = dict(variants or DEFAULT_VARIANTS)
    variants_map = {k: v for k, v in variants_map.items() if v in pred_df.columns}
    if len(variants_map) == 0:
        raise ValueError('No valid planner-proxy variant columns available for miscalibration probe.')

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
        label_col=target_label,
        threshold=diag_threshold,
    )
    leakage_df = _leakage_checks(pred_df, label_col=target_label)
    shift_profile_df = _shift_profile_summary(pred_df, label_col=target_label)
    class_balance_df = _class_balance_summary(pred_df, label_columns=tuple(labels))

    artifact_paths = save_risk_evaluation_artifacts(
        run_prefix,
        {
            'miscalibration_probe_summary': benchmark_bundle.summary_df,
            'miscalibration_probe_per_shift': benchmark_bundle.per_shift_df,
            'miscalibration_probe_reliability_bins': benchmark_bundle.reliability_df,
            'miscalibration_probe_selective_risk_curve': benchmark_bundle.selective_curve_df,
            'miscalibration_probe_shift_gap_summary': benchmark_bundle.shift_gap_df,
            'miscalibration_probe_threshold_diagnostics': threshold_df,
            'miscalibration_probe_leakage_checks': leakage_df,
            'miscalibration_probe_shift_profile': shift_profile_df,
            'miscalibration_probe_class_balance': class_balance_df,
            'miscalibration_probe_proxy_calibration': proxy_calibration_df,
            'miscalibration_probe_predictions': pred_df,
        },
    )
    artifact_paths.update(
        update_living_report_from_miscalibration_probe(
            cfg=cfg,
            run_prefix=run_prefix,
            summary_df=benchmark_bundle.summary_df,
            per_shift_df=benchmark_bundle.per_shift_df,
            threshold_df=threshold_df,
            leakage_df=leakage_df,
            class_balance_df=class_balance_df,
            artifact_paths=artifact_paths,
        )
    )
    return MiscalibrationProbeBundle(
        predictions_df=pred_df,
        benchmark_bundle=benchmark_bundle,
        threshold_df=threshold_df,
        leakage_df=leakage_df,
        shift_profile_df=shift_profile_df,
        class_balance_df=class_balance_df,
        proxy_calibration_df=proxy_calibration_df,
        artifact_paths=artifact_paths,
        loaded_from_existing=False,
    )
