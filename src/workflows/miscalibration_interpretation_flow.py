from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .miscalibration_probe_flow import (
    DEFAULT_VARIANTS,
    MiscalibrationProbeBundle,
    has_existing_miscalibration_probe_artifacts,
    load_existing_miscalibration_probe_bundle,
)


@dataclass
class MiscalibrationInterpretationBundle:
    run_prefix: str
    focus_label: str
    threshold: float
    source_bundle: MiscalibrationProbeBundle = field(default_factory=MiscalibrationProbeBundle)
    per_shift_focus_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    reliability_focus_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    threshold_focus_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    metric_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    verdict_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    narrative: str = ''
    artifact_paths: Dict[str, str] = field(default_factory=dict)


def _safe_float(value: Any, default: float = float('nan')) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _status_row(claim: str, supported: Optional[bool], evidence: str) -> Dict[str, Any]:
    if supported is True:
        status = 'supported'
    elif supported is False:
        status = 'not_supported'
    else:
        status = 'inconclusive'
    return {'claim': str(claim), 'status': str(status), 'evidence': str(evidence)}


def _variant_group(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.lower()
    out = pd.Series(np.full((len(text),), 'other', dtype=object), index=text.index)
    out[text.str.contains('raw', na=False)] = 'raw'
    out[text.str.contains('platt|cal', na=False, regex=True)] = 'cal'
    return out


def _mean_metric(df: pd.DataFrame, metric: str, filters: Dict[str, Any]) -> float:
    if df.empty or (metric not in df.columns):
        return float('nan')
    mask = np.ones((len(df),), dtype=bool)
    for key, val in filters.items():
        if key not in df.columns:
            return float('nan')
        if isinstance(val, (list, tuple, set)):
            mask &= df[key].isin(list(val)).to_numpy(dtype=bool)
        else:
            mask &= df[key].astype(str).eq(str(val)).to_numpy(dtype=bool)
    values = pd.to_numeric(df.loc[mask, metric], errors='coerce').to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan')
    return float(np.mean(values))


def discover_probe_run_prefixes(
    persist_root: str | Path,
    *,
    limit: int = 50,
) -> pd.DataFrame:
    root = Path(str(persist_root)).expanduser()
    if not root.exists():
        return pd.DataFrame(columns=['run_prefix', 'summary_path', 'mtime_utc'])

    suffix = '_miscalibration_probe_summary.csv'
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob(f'*{suffix}')):
        prefix = str(path)[: -len(suffix)]
        try:
            mtime = float(path.stat().st_mtime)
        except Exception:
            mtime = float('nan')
        rows.append({'run_prefix': prefix, 'summary_path': str(path), 'mtime_epoch': mtime})
    if len(rows) == 0:
        return pd.DataFrame(columns=['run_prefix', 'summary_path', 'mtime_utc'])

    out = pd.DataFrame(rows).sort_values(['mtime_epoch'], ascending=False).head(int(max(1, limit)))
    out['mtime_utc'] = pd.to_datetime(out['mtime_epoch'], unit='s', utc=True).astype(str)
    return out[['run_prefix', 'summary_path', 'mtime_utc']].reset_index(drop=True)


def resolve_threshold_sweep_variants(
    predictions_df: pd.DataFrame,
    *,
    focus_label: str = 'failure_proxy_h15',
) -> Dict[str, str]:
    candidates = dict(DEFAULT_VARIANTS)

    focus_key = str(focus_label).strip().lower()
    if focus_key:
        suffix = focus_key
        candidates.update(
            {
                f'{focus_key}_model_raw': f'risk_raw_{suffix}',
                f'{focus_key}_model_cal': f'risk_cal_{suffix}',
            }
        )

    out: Dict[str, str] = {}
    for variant, prob_col in candidates.items():
        if str(prob_col) in predictions_df.columns:
            out[str(variant)] = str(prob_col)
    return out


def _safe_prob_array(values: Any, *, default: float = 0.5) -> np.ndarray:
    arr = np.asarray(pd.to_numeric(values, errors='coerce'), dtype=float)
    arr = np.where(np.isfinite(arr), arr, float(default))
    return np.clip(arr, 1e-6, 1.0 - 1e-6)


def _safe_label_array(values: Any) -> np.ndarray:
    arr = np.asarray(pd.to_numeric(values, errors='coerce'), dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return (arr > 0.5).astype(float)


def _decision_metrics_from_arrays(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    tau: float,
    step_keys: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    p = _safe_prob_array(probs, default=0.5)
    y = _safe_label_array(labels)
    n_rows = int(p.size)
    if n_rows <= 0:
        return {
            'n_rows': 0.0,
            'positive_rate': np.nan,
            'accept_count': 0.0,
            'reject_count': 0.0,
            'accept_rate': np.nan,
            'reject_rate': np.nan,
            'mean_predicted_risk': np.nan,
            'empirical_failure_rate': np.nan,
            'risk_gap_empirical_minus_predicted': np.nan,
            'false_safe_rate': np.nan,
            'safe_rejected_rate': np.nan,
            'false_safe_cond': np.nan,
            'safe_reject_cond': np.nan,
            'empirical_failure_given_accepted': np.nan,
            'budget_violated': np.nan,
            'feasible_set_rate': np.nan,
            'fallback_rate': np.nan,
        }

    tau_f = float(np.clip(float(tau), 0.0, 1.0))
    accepted = p <= tau_f
    rejected = ~accepted
    accept_count = int(np.sum(accepted))
    reject_count = int(np.sum(rejected))

    mean_pred = float(np.mean(p))
    empirical_failure = float(np.mean(y))
    false_safe_joint = float(np.mean(accepted & (y > 0.5)))
    safe_rejected_joint = float(np.mean(rejected & (y <= 0.5)))
    false_safe_cond = float(np.mean(y[accepted])) if accept_count > 0 else np.nan
    safe_reject_cond = float(np.mean((y <= 0.5)[rejected])) if reject_count > 0 else np.nan

    feasible_set_rate = np.nan
    fallback_rate = np.nan
    if step_keys is not None and len(step_keys) == n_rows:
        step_df = pd.DataFrame({'step_key': step_keys, 'accepted': accepted.astype(np.int32)})
        if not step_df.empty:
            feasible_set_rate = float(step_df.groupby('step_key', sort=False)['accepted'].max().mean())
            fallback_rate = float(1.0 - feasible_set_rate)

    return {
        'n_rows': float(n_rows),
        'positive_rate': empirical_failure,
        'accept_count': float(accept_count),
        'reject_count': float(reject_count),
        'accept_rate': float(accept_count / max(1, n_rows)),
        'reject_rate': float(reject_count / max(1, n_rows)),
        'mean_predicted_risk': mean_pred,
        'empirical_failure_rate': empirical_failure,
        'risk_gap_empirical_minus_predicted': float(empirical_failure - mean_pred),
        'false_safe_rate': false_safe_joint,
        'safe_rejected_rate': safe_rejected_joint,
        'false_safe_cond': false_safe_cond,
        'safe_reject_cond': safe_reject_cond,
        'empirical_failure_given_accepted': false_safe_cond,
        'budget_violated': float(
            bool(np.isfinite(false_safe_cond) and (false_safe_cond > tau_f))
        ),
        'feasible_set_rate': feasible_set_rate,
        'fallback_rate': fallback_rate,
    }


def _bootstrap_ci(values: List[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return float('nan'), float('nan')
    lo = float(np.quantile(arr, 0.025))
    hi = float(np.quantile(arr, 0.975))
    return lo, hi


def compute_threshold_sweep_diagnostics(
    predictions_df: pd.DataFrame,
    *,
    focus_label: str = 'failure_proxy_h15',
    tau_values: Optional[Sequence[float]] = None,
    variants: Optional[Dict[str, str]] = None,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 17,
) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame()

    label_col = str(focus_label)
    if label_col not in predictions_df.columns:
        raise ValueError(f'focus label column not found in predictions: {label_col!r}')

    work = predictions_df.copy()
    if 'shift_suite' not in work.columns:
        work['shift_suite'] = 'nominal_clean'
    else:
        work['shift_suite'] = work['shift_suite'].fillna('nominal_clean').astype(str)

    variant_map = dict(variants or resolve_threshold_sweep_variants(work, focus_label=label_col))
    variant_map = {str(k): str(v) for k, v in variant_map.items() if str(v) in work.columns}
    if len(variant_map) <= 0:
        raise ValueError('No valid probability columns found for tau sweep diagnostics.')

    if tau_values is None:
        tau_arr = np.linspace(0.05, 0.80, 16)
    else:
        tau_arr = np.asarray(list(tau_values), dtype=float)
        tau_arr = tau_arr[np.isfinite(tau_arr)]
        if tau_arr.size <= 0:
            raise ValueError('tau_values is empty after numeric filtering.')
    tau_arr = np.clip(tau_arr, 0.0, 1.0)
    tau_arr = np.unique(np.round(tau_arr.astype(float), 6))

    rng = np.random.default_rng(int(bootstrap_seed))
    rows: List[Dict[str, Any]] = []
    ci_metrics = (
        'accept_rate',
        'false_safe_rate',
        'safe_rejected_rate',
        'false_safe_cond',
        'safe_reject_cond',
        'feasible_set_rate',
        'fallback_rate',
    )

    for shift_suite, shift_df in work.groupby('shift_suite', sort=True):
        labels = _safe_label_array(shift_df[label_col])

        step_keys: Optional[np.ndarray] = None
        if ('scenario_id' in shift_df.columns) and ('step_idx' in shift_df.columns):
            step_keys = (
                shift_df['scenario_id'].astype(str).to_numpy(dtype=object)
                + '::'
                + shift_df['step_idx'].astype(str).to_numpy(dtype=object)
            )

        if 'scenario_id' in shift_df.columns:
            unit_ids = shift_df['scenario_id'].astype(str).to_numpy(dtype=object)
        elif step_keys is not None:
            unit_ids = step_keys
        else:
            unit_ids = np.asarray([str(i) for i in range(len(shift_df))], dtype=object)

        unique_units, inv = np.unique(unit_ids, return_inverse=True)
        unit_indices = [np.where(inv == i)[0] for i in range(len(unique_units))]

        for variant, prob_col in variant_map.items():
            probs = _safe_prob_array(shift_df[prob_col], default=0.5)
            variant_group = str(_variant_group(pd.Series([variant])).iloc[0])

            for tau in tau_arr:
                metrics = _decision_metrics_from_arrays(
                    probs,
                    labels,
                    tau=float(tau),
                    step_keys=step_keys,
                )
                row = {
                    'shift_suite': str(shift_suite),
                    'variant': str(variant),
                    'variant_group': variant_group,
                    'label': label_col,
                    'tau': float(tau),
                    'probability_column': str(prob_col),
                    **metrics,
                }

                n_boot = int(max(0, int(bootstrap_samples)))
                if n_boot > 0 and len(unit_indices) > 1:
                    boot_values: Dict[str, List[float]] = {k: [] for k in ci_metrics}
                    for _ in range(n_boot):
                        sampled = rng.integers(0, len(unit_indices), size=len(unit_indices))
                        idx = np.concatenate([unit_indices[int(i)] for i in sampled], axis=0)
                        boot_metrics = _decision_metrics_from_arrays(
                            probs[idx],
                            labels[idx],
                            tau=float(tau),
                            step_keys=(step_keys[idx] if step_keys is not None else None),
                        )
                        for key in ci_metrics:
                            boot_values[key].append(float(boot_metrics.get(key, np.nan)))
                    for key in ci_metrics:
                        lo, hi = _bootstrap_ci(boot_values[key])
                        row[f'{key}_ci_low'] = lo
                        row[f'{key}_ci_high'] = hi

                rows.append(row)

    return pd.DataFrame(rows)


def analyze_miscalibration_probe_bundle(
    bundle: MiscalibrationProbeBundle,
    *,
    run_prefix: str,
    focus_label: str = 'failure_proxy_h15',
    threshold: float = 0.20,
) -> MiscalibrationInterpretationBundle:
    per_shift = bundle.benchmark_bundle.per_shift_df.copy()
    reliability = bundle.benchmark_bundle.reliability_df.copy()
    threshold_df = bundle.threshold_df.copy()

    if not per_shift.empty and ('variant' in per_shift.columns):
        per_shift['variant_group'] = _variant_group(per_shift['variant'])
    else:
        per_shift['variant_group'] = 'other'

    if not threshold_df.empty and ('variant' in threshold_df.columns):
        threshold_df['variant_group'] = _variant_group(threshold_df['variant'])
    else:
        threshold_df['variant_group'] = 'other'

    if not reliability.empty and ('variant' in reliability.columns):
        reliability['variant_group'] = _variant_group(reliability['variant'])
    else:
        reliability['variant_group'] = 'other'

    per_shift_focus = per_shift.copy()
    if 'label' in per_shift_focus.columns:
        per_shift_focus = per_shift_focus[per_shift_focus['label'].astype(str).eq(str(focus_label))].copy()

    threshold_focus = threshold_df.copy()
    if 'label' in threshold_focus.columns:
        threshold_focus = threshold_focus[threshold_focus['label'].astype(str).eq(str(focus_label))].copy()

    reliability_focus = reliability.copy()
    if 'label' in reliability_focus.columns:
        reliability_focus = reliability_focus[reliability_focus['label'].astype(str).eq(str(focus_label))].copy()

    rows: List[Dict[str, Any]] = []
    scopes = {
        'nominal': {'shift_suite': 'nominal_clean'},
        'shift_only': {'shift_suite': [x for x in sorted(per_shift_focus.get('shift_suite', pd.Series(dtype=object)).astype(str).unique().tolist()) if x != 'nominal_clean']},
        'all': {},
    }
    for scope_name, filters in scopes.items():
        for group in ('raw', 'cal'):
            filters_now = dict(filters)
            filters_now['variant_group'] = group
            rows.append(
                {
                    'scope': scope_name,
                    'variant_group': group,
                    'ece': _mean_metric(per_shift_focus, 'ece', filters_now),
                    'nll': _mean_metric(per_shift_focus, 'nll', filters_now),
                    'brier': _mean_metric(per_shift_focus, 'brier', filters_now),
                    'auroc': _mean_metric(per_shift_focus, 'auroc', filters_now),
                    'auprc': _mean_metric(per_shift_focus, 'auprc', filters_now),
                    'empirical_failure_given_accepted': _mean_metric(threshold_focus, 'empirical_failure_given_accepted', filters_now),
                    'false_safe_rate': _mean_metric(threshold_focus, 'false_safe_rate', filters_now),
                    'safe_rejected_rate': _mean_metric(threshold_focus, 'safe_rejected_rate', filters_now),
                    'budget_violated_rate': _mean_metric(threshold_focus, 'budget_violated', filters_now),
                }
            )
    metric_summary_df = pd.DataFrame(rows)

    def get(scope: str, group: str, metric: str) -> float:
        sub = metric_summary_df[
            metric_summary_df['scope'].astype(str).eq(str(scope))
            & metric_summary_df['variant_group'].astype(str).eq(str(group))
        ]
        if sub.empty or (metric not in sub.columns):
            return float('nan')
        return _safe_float(sub.iloc[0][metric], default=float('nan'))

    raw_nom_ece = get('nominal', 'raw', 'ece')
    cal_nom_ece = get('nominal', 'cal', 'ece')
    raw_nom_nll = get('nominal', 'raw', 'nll')
    cal_nom_nll = get('nominal', 'cal', 'nll')
    raw_nom_emp_fail_acc = get('nominal', 'raw', 'empirical_failure_given_accepted')
    raw_nom_safe_rej = get('nominal', 'raw', 'safe_rejected_rate')
    cal_nom_safe_rej = get('nominal', 'cal', 'safe_rejected_rate')
    raw_shift_ece = get('shift_only', 'raw', 'ece')

    miscalibration_exists = None
    if np.isfinite(raw_nom_ece):
        miscalibration_exists = bool(raw_nom_ece >= 0.03)

    calibration_helps = None
    if np.isfinite(raw_nom_ece) and np.isfinite(cal_nom_ece):
        calibration_helps = bool(cal_nom_ece <= raw_nom_ece + 1e-8)
    if (calibration_helps is None) and np.isfinite(raw_nom_nll) and np.isfinite(cal_nom_nll):
        calibration_helps = bool(cal_nom_nll <= raw_nom_nll + 1e-8)

    over_confident_budget = None
    if np.isfinite(raw_nom_emp_fail_acc):
        over_confident_budget = bool(raw_nom_emp_fail_acc > float(threshold))

    under_confident_budget = None
    if np.isfinite(raw_nom_safe_rej):
        baseline = cal_nom_safe_rej if np.isfinite(cal_nom_safe_rej) else 0.05
        under_confident_budget = bool(raw_nom_safe_rej > max(0.05, baseline + 0.01))

    shift_worsens = None
    if np.isfinite(raw_shift_ece) and np.isfinite(raw_nom_ece):
        shift_worsens = bool(raw_shift_ece > raw_nom_ece + 0.005)

    problem_exists = None
    if (miscalibration_exists is not None) and ((over_confident_budget is not None) or (under_confident_budget is not None)):
        problem_exists = bool(miscalibration_exists and bool(over_confident_budget or under_confident_budget))

    verdict_rows = [
        _status_row(
            'Planner-side proxy miscalibration exists (nominal)',
            miscalibration_exists,
            f'raw_nominal_ece={raw_nom_ece:.4f}' if np.isfinite(raw_nom_ece) else 'raw_nominal_ece=NA',
        ),
        _status_row(
            'Calibration improves reliability (nominal)',
            calibration_helps,
            (
                f'ece raw={raw_nom_ece:.4f} vs cal={cal_nom_ece:.4f}; '
                f'nll raw={raw_nom_nll:.4f} vs cal={cal_nom_nll:.4f}'
            )
            if any(np.isfinite(x) for x in [raw_nom_ece, cal_nom_ece, raw_nom_nll, cal_nom_nll])
            else 'metrics unavailable',
        ),
        _status_row(
            f'Budgeted decisions show over-confidence at tau={threshold:.2f}',
            over_confident_budget,
            (
                f'empirical_failure_given_accepted(raw_nominal)={raw_nom_emp_fail_acc:.4f}'
                if np.isfinite(raw_nom_emp_fail_acc)
                else 'empirical_failure_given_accepted unavailable'
            ),
        ),
        _status_row(
            'Budgeted decisions show under-confidence (safe rejections)',
            under_confident_budget,
            (
                f'safe_rejected_rate raw={raw_nom_safe_rej:.4f}, cal={cal_nom_safe_rej:.4f}'
                if np.isfinite(raw_nom_safe_rej) or np.isfinite(cal_nom_safe_rej)
                else 'safe_rejected_rate unavailable'
            ),
        ),
        _status_row(
            'Miscalibration worsens under shift',
            shift_worsens,
            (
                f'raw_ece nominal={raw_nom_ece:.4f}, shift_mean={raw_shift_ece:.4f}'
                if np.isfinite(raw_nom_ece) or np.isfinite(raw_shift_ece)
                else 'shift comparison unavailable'
            ),
        ),
        _status_row(
            'Problem framing validated (miscalibration + decision impact)',
            problem_exists,
            (
                f'miscalibration={miscalibration_exists}, '
                f'over_confident_budget={over_confident_budget}, '
                f'under_confident_budget={under_confident_budget}'
            ),
        ),
    ]
    verdict_df = pd.DataFrame(verdict_rows)

    if not verdict_df.empty and (verdict_df['claim'].astype(str) == 'Problem framing validated (miscalibration + decision impact)').any():
        status = verdict_df.loc[
            verdict_df['claim'].astype(str).eq('Problem framing validated (miscalibration + decision impact)'),
            'status',
        ].iloc[0]
    else:
        status = 'inconclusive'

    narrative = (
        f'Overall verdict for focus label {focus_label}: {status}. '
        f'Use the claim-level verdict table to see exactly which links in the chain '
        f'(miscalibration -> budget decision error) are supported.'
    )

    return MiscalibrationInterpretationBundle(
        run_prefix=str(run_prefix),
        focus_label=str(focus_label),
        threshold=float(threshold),
        source_bundle=bundle,
        per_shift_focus_df=per_shift_focus,
        reliability_focus_df=reliability_focus,
        threshold_focus_df=threshold_focus,
        metric_summary_df=metric_summary_df,
        verdict_df=verdict_df,
        narrative=narrative,
        artifact_paths=dict(bundle.artifact_paths),
    )


def load_and_analyze_miscalibration_probe(
    *,
    run_prefix: str,
    focus_label: str = 'failure_proxy_h15',
    threshold: float = 0.20,
) -> MiscalibrationInterpretationBundle:
    if not has_existing_miscalibration_probe_artifacts(str(run_prefix)):
        raise FileNotFoundError(
            f'Missing miscalibration probe artifacts for run_prefix={run_prefix!r}. '
            'Run miscalibration_probe_colab.ipynb first.'
        )
    bundle = load_existing_miscalibration_probe_bundle(str(run_prefix))
    return analyze_miscalibration_probe_bundle(
        bundle,
        run_prefix=str(run_prefix),
        focus_label=str(focus_label),
        threshold=float(threshold),
    )
