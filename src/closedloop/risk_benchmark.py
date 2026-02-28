from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class BenchmarkBundle:
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_shift_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    reliability_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    selective_curve_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    shift_gap_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def _clip_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def binary_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    p = _clip_probs(probs)
    y = np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi >= 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return float(ece)


def adaptive_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    p = _clip_probs(probs)
    y = np.asarray(labels, dtype=float)
    q = int(max(2, min(int(n_bins), p.size)))
    order = np.argsort(p)
    chunks = np.array_split(order, q)
    ece = 0.0
    for chunk in chunks:
        if chunk.size == 0:
            continue
        ece += float(chunk.size / max(1, p.size)) * abs(float(np.mean(p[chunk])) - float(np.mean(y[chunk])))
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    p = _clip_probs(probs)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2))


def nll_score(probs: np.ndarray, labels: np.ndarray) -> float:
    p = _clip_probs(probs)
    y = np.asarray(labels, dtype=float)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def binary_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos = p[y == 1]
    neg = p[y == 0]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    wins = 0.0
    for val in pos:
        wins += float(np.sum(val > neg)) + 0.5 * float(np.sum(val == neg))
    return float(wins / max(1, pos.size * neg.size))


def binary_auprc(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    if np.sum(y == 1) == 0:
        return np.nan
    order = np.argsort(-p)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(1, np.sum(y == 1))
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    p = _clip_probs(probs)
    y = np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)
    rows = []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (p >= lo) & (p <= hi if i == len(bins) - 2 else p < hi)
        if not np.any(mask):
            rows.append({'bin_id': i, 'bin_lo': lo, 'bin_hi': hi, 'count': 0, 'mean_prob': np.nan, 'event_rate': np.nan})
            continue
        rows.append({
            'bin_id': i,
            'bin_lo': lo,
            'bin_hi': hi,
            'count': int(np.sum(mask)),
            'mean_prob': float(np.mean(p[mask])),
            'event_rate': float(np.mean(y[mask])),
        })
    return pd.DataFrame(rows)


def selective_risk_curve(probs: np.ndarray, labels: np.ndarray, n_points: int = 25) -> pd.DataFrame:
    p = _clip_probs(probs)
    y = np.asarray(labels, dtype=float)
    rows = []
    for coverage in np.linspace(0.05, 1.0, int(max(2, n_points))):
        k = max(1, int(np.ceil(float(coverage) * p.size)))
        idx = np.argsort(p)[:k]
        rows.append({'coverage': float(coverage), 'selective_risk': float(np.mean(y[idx])), 'threshold': float(np.max(p[idx]))})
    return pd.DataFrame(rows)


def _metric_row(name: str, probs: np.ndarray, labels: np.ndarray, shift_suite: str, variant: str) -> Dict[str, Any]:
    return {
        'label': name,
        'shift_suite': shift_suite,
        'variant': variant,
        'n_rows': int(len(labels)),
        'positive_rate': float(np.mean(labels)) if len(labels) > 0 else np.nan,
        'ece': binary_ece(probs, labels),
        'adaptive_ece': adaptive_ece(probs, labels),
        'brier': brier_score(probs, labels),
        'nll': nll_score(probs, labels),
        'auroc': binary_auroc(probs, labels),
        'auprc': binary_auprc(probs, labels),
    }


def run_uq_benchmark(
    df: pd.DataFrame,
    *,
    variants: Optional[Mapping[str, str]] = None,
    label_columns: Sequence[str] = ('failure_proxy_h15',),
    n_bins: int = 15,
) -> BenchmarkBundle:
    if df.empty:
        return BenchmarkBundle()
    if variants is None:
        variants = {'raw': 'risk_raw_failure_proxy_h15', 'cal': 'risk_cal_failure_proxy_h15'}

    work = df.copy()
    if 'shift_suite' not in work.columns:
        work['shift_suite'] = 'nominal_clean'

    summary_rows: List[Dict[str, Any]] = []
    rel_rows: List[pd.DataFrame] = []
    curve_rows: List[pd.DataFrame] = []
    for label in label_columns:
        if label not in work.columns:
            continue
        for variant, prob_col in variants.items():
            if prob_col not in work.columns:
                continue
            for shift_suite, grp in work.groupby('shift_suite', sort=True):
                probs = grp[prob_col].to_numpy(dtype=float)
                labs = grp[label].to_numpy(dtype=float)
                summary_rows.append(_metric_row(label, probs, labs, shift_suite=str(shift_suite), variant=str(variant)))
                rel = reliability_bins(probs, labs, n_bins=n_bins)
                rel['label'] = label
                rel['variant'] = variant
                rel['shift_suite'] = str(shift_suite)
                rel_rows.append(rel)
                curve = selective_risk_curve(probs, labs)
                curve['label'] = label
                curve['variant'] = variant
                curve['shift_suite'] = str(shift_suite)
                curve_rows.append(curve)

    per_shift_df = pd.DataFrame(summary_rows)
    if per_shift_df.empty:
        return BenchmarkBundle()
    summary_df = (
        per_shift_df.groupby(['label', 'variant'], as_index=False)
        .agg(
            n_rows=('n_rows', 'sum'),
            positive_rate=('positive_rate', 'mean'),
            ece=('ece', 'mean'),
            adaptive_ece=('adaptive_ece', 'mean'),
            brier=('brier', 'mean'),
            nll=('nll', 'mean'),
            auroc=('auroc', 'mean'),
            auprc=('auprc', 'mean'),
        )
    )
    reliability_df = pd.concat(rel_rows, ignore_index=True) if len(rel_rows) > 0 else pd.DataFrame()
    selective_df = pd.concat(curve_rows, ignore_index=True) if len(curve_rows) > 0 else pd.DataFrame()

    nominal = per_shift_df[per_shift_df['shift_suite'] == 'nominal_clean'][['label', 'variant', 'ece', 'nll']].rename(
        columns={'ece': 'nominal_ece', 'nll': 'nominal_nll'}
    )
    shift_gap_df = per_shift_df.merge(nominal, on=['label', 'variant'], how='left')
    shift_gap_df['ece_gap_vs_nominal'] = shift_gap_df['ece'] - shift_gap_df['nominal_ece']
    shift_gap_df['nll_gap_vs_nominal'] = shift_gap_df['nll'] - shift_gap_df['nominal_nll']

    return BenchmarkBundle(
        summary_df=summary_df,
        per_shift_df=per_shift_df,
        reliability_df=reliability_df,
        selective_curve_df=selective_df,
        shift_gap_df=shift_gap_df,
    )


def summarize_controller_tradeoff(
    base_df: pd.DataFrame,
    controller_df: pd.DataFrame,
    *,
    progress_col: str = 'progress_h6',
    failure_col: str = 'failure_proxy_h15',
) -> pd.DataFrame:
    left = base_df[['scenario_id', progress_col, failure_col]].rename(
        columns={progress_col: 'base_progress', failure_col: 'base_failure'}
    )
    right = controller_df[['scenario_id', progress_col, failure_col]].rename(
        columns={progress_col: 'ctrl_progress', failure_col: 'ctrl_failure'}
    )
    merged = left.merge(right, on='scenario_id', how='inner')
    if merged.empty:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            'n_scenarios': int(len(merged)),
            'base_progress_mean': float(merged['base_progress'].mean()),
            'ctrl_progress_mean': float(merged['ctrl_progress'].mean()),
            'relative_progress_change': float((merged['ctrl_progress'].mean() - merged['base_progress'].mean()) / max(1e-6, abs(merged['base_progress'].mean()))),
            'base_failure_rate': float(merged['base_failure'].mean()),
            'ctrl_failure_rate': float(merged['ctrl_failure'].mean()),
            'failure_rate_delta': float(merged['ctrl_failure'].mean() - merged['base_failure'].mean()),
        }
    ])
