from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


def compute_high_risk_event_flag(
    df: pd.DataFrame,
    risk_col: str = "risk_sks",
    failure_col: str = "failure_proxy",
    min_ttc_col: str = "min_ttc",
    risk_quantile: float = 0.9,
    absolute_risk_threshold: Optional[float] = None,
    near_miss_ttc_threshold: float = 2.0,
) -> tuple[pd.Series, Dict[str, Any]]:
    if df.empty:
        return pd.Series(False, index=df.index), {"risk_threshold": np.nan}

    out = pd.Series(False, index=df.index)

    risk_threshold = float("nan")
    if risk_col in df.columns:
        risk_vals = pd.to_numeric(df[risk_col], errors="coerce")
        if absolute_risk_threshold is not None:
            risk_threshold = float(absolute_risk_threshold)
        else:
            finite = risk_vals[np.isfinite(risk_vals)]
            if len(finite) > 0:
                risk_threshold = float(np.quantile(finite, float(np.clip(risk_quantile, 0.0, 1.0))))
        if np.isfinite(risk_threshold):
            out = out | (np.isfinite(risk_vals) & (risk_vals >= risk_threshold))

    if failure_col in df.columns:
        out = out | (pd.to_numeric(df[failure_col], errors="coerce").fillna(0.0) > 0.0)

    if min_ttc_col in df.columns:
        ttc = pd.to_numeric(df[min_ttc_col], errors="coerce")
        out = out | (np.isfinite(ttc) & (ttc <= float(near_miss_ttc_threshold)))

    meta = {
        "risk_col": risk_col,
        "failure_col": failure_col if failure_col in df.columns else None,
        "min_ttc_col": min_ttc_col if min_ttc_col in df.columns else None,
        "risk_threshold": float(risk_threshold) if np.isfinite(risk_threshold) else np.nan,
    }
    return out.astype(bool), meta


def scenario_factor_response(
    df: pd.DataFrame,
    factor_col: str,
    outcome_col: str = "is_high_risk_event",
    scenario_col: str = "scenario_id",
    method_col: str = "method",
    min_unique_values: int = 3,
) -> pd.DataFrame:
    required = [scenario_col, method_col, factor_col, outcome_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for scenario factor response: {missing}")
    if df.empty:
        return pd.DataFrame()

    rows: list[Dict[str, Any]] = []
    group_cols = [scenario_col, method_col]
    for keys, grp in df.groupby(group_cols, sort=True):
        gg = grp[[factor_col, outcome_col]].copy()
        gg[factor_col] = pd.to_numeric(gg[factor_col], errors="coerce")
        gg[outcome_col] = pd.to_numeric(gg[outcome_col], errors="coerce")
        gg = gg[np.isfinite(gg[factor_col]) & np.isfinite(gg[outcome_col])]
        if gg.empty:
            continue

        prof = gg.groupby(factor_col, as_index=False)[outcome_col].mean().sort_values(factor_col)
        if int(prof[factor_col].nunique()) < int(max(2, min_unique_values)):
            continue

        x = prof[factor_col].to_numpy(dtype=float)
        y = prof[outcome_col].to_numpy(dtype=float)

        slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else np.nan
        if len(x) >= 2:
            rho = float(pd.Series(x).rank().corr(pd.Series(y).rank(), method="pearson"))
        else:
            rho = np.nan

        baseline_idx = int(np.argmin(np.abs(x)))
        baseline_outcome = float(y[baseline_idx])
        peak_outcome = float(np.max(y))
        trough_outcome = float(np.min(y))
        response_range = float(peak_outcome - trough_outcome)

        row = {
            scenario_col: keys[0],
            method_col: keys[1],
            "factor_name": factor_col,
            "n_points": int(len(x)),
            "min_factor_value": float(np.min(x)),
            "max_factor_value": float(np.max(x)),
            "slope": slope,
            "abs_slope": float(abs(slope)) if np.isfinite(slope) else np.nan,
            "spearman_like_rho": rho,
            "baseline_outcome": baseline_outcome,
            "peak_outcome": peak_outcome,
            "trough_outcome": trough_outcome,
            "response_range": response_range,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_sensitivity_atlas(
    trace_df: pd.DataFrame,
    factor_columns: Sequence[str] = ("delta_x", "delta_y", "delta_l2", "step_scale"),
    outcome_col: Optional[str] = None,
    scenario_col: str = "scenario_id",
    method_col: str = "method",
    min_unique_values: int = 3,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if trace_df.empty:
        return pd.DataFrame(), {}

    working = trace_df.copy()
    outcome_metadata: Dict[str, Any] = {}
    if outcome_col is None:
        outcome_flag, meta = compute_high_risk_event_flag(working)
        working["is_high_risk_event"] = outcome_flag.astype(int)
        outcome_col = "is_high_risk_event"
        outcome_metadata = meta
    elif outcome_col not in working.columns:
        raise ValueError(f"Requested outcome_col not found: {outcome_col}")

    rows = []
    for factor in factor_columns:
        if factor not in working.columns:
            continue
        tab = scenario_factor_response(
            df=working,
            factor_col=factor,
            outcome_col=outcome_col,
            scenario_col=scenario_col,
            method_col=method_col,
            min_unique_values=min_unique_values,
        )
        if not tab.empty:
            rows.append(tab)

    atlas = pd.concat(rows, ignore_index=True) if len(rows) > 0 else pd.DataFrame()
    return atlas, outcome_metadata


def aggregate_factor_importance(atlas_df: pd.DataFrame) -> pd.DataFrame:
    if atlas_df.empty:
        return pd.DataFrame()
    required = ["factor_name", "abs_slope", "slope", "response_range", "spearman_like_rho"]
    missing = [c for c in required if c not in atlas_df.columns]
    if missing:
        raise ValueError(f"Missing columns for factor-importance aggregation: {missing}")

    out = (
        atlas_df.groupby("factor_name", as_index=False)
        .agg(
            n_groups=("abs_slope", "size"),
            mean_abs_slope=("abs_slope", "mean"),
            median_abs_slope=("abs_slope", "median"),
            mean_slope=("slope", "mean"),
            positive_slope_fraction=("slope", lambda x: float((x > 0).mean())),
            mean_response_range=("response_range", "mean"),
            median_response_range=("response_range", "median"),
            monotonic_fraction=("spearman_like_rho", lambda x: float((x.abs() >= 0.5).mean())),
        )
        .sort_values(["mean_abs_slope", "mean_response_range"], ascending=False)
        .reset_index(drop=True)
    )
    return out


def top_sensitive_scenarios(
    atlas_df: pd.DataFrame,
    top_n: int = 25,
    scenario_col: str = "scenario_id",
    method_col: str = "method",
) -> pd.DataFrame:
    if atlas_df.empty:
        return pd.DataFrame()
    required = [scenario_col, method_col, "factor_name", "abs_slope", "response_range"]
    missing = [c for c in required if c not in atlas_df.columns]
    if missing:
        raise ValueError(f"Missing columns for top-sensitive scenario extraction: {missing}")

    out = (
        atlas_df.sort_values(["abs_slope", "response_range"], ascending=False)
        .head(int(max(1, top_n)))
        .reset_index(drop=True)
    )
    return out


def method_factor_importance(
    atlas_df: pd.DataFrame,
    method_col: str = "method",
) -> pd.DataFrame:
    if atlas_df.empty:
        return pd.DataFrame()
    required = [method_col, "factor_name", "abs_slope", "slope", "response_range", "spearman_like_rho"]
    missing = [c for c in required if c not in atlas_df.columns]
    if missing:
        raise ValueError(f"Missing columns for method_factor_importance: {missing}")

    out = (
        atlas_df.groupby([method_col, "factor_name"], as_index=False)
        .agg(
            n_groups=("abs_slope", "size"),
            mean_abs_slope=("abs_slope", "mean"),
            median_abs_slope=("abs_slope", "median"),
            mean_slope=("slope", "mean"),
            mean_response_range=("response_range", "mean"),
            monotonic_fraction=("spearman_like_rho", lambda x: float((x.abs() >= 0.5).mean())),
        )
        .sort_values([method_col, "mean_abs_slope"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return out


def factor_response_profile(
    trace_df: pd.DataFrame,
    factor_col: str,
    outcome_col: str = "is_high_risk_event",
    method_col: str = "method",
    n_bins: int = 10,
) -> pd.DataFrame:
    required = [factor_col, outcome_col, method_col]
    missing = [c for c in required if c not in trace_df.columns]
    if missing:
        raise ValueError(f"Missing columns for factor_response_profile: {missing}")
    if trace_df.empty:
        return pd.DataFrame()

    sub = trace_df[[factor_col, outcome_col, method_col]].copy()
    sub[factor_col] = pd.to_numeric(sub[factor_col], errors="coerce")
    sub[outcome_col] = pd.to_numeric(sub[outcome_col], errors="coerce")
    sub = sub[np.isfinite(sub[factor_col]) & np.isfinite(sub[outcome_col])]
    if sub.empty:
        return pd.DataFrame()

    unique_vals = int(sub[factor_col].nunique())
    eff_bins = int(max(2, min(int(max(2, n_bins)), unique_vals)))
    sub["_bin"] = pd.qcut(sub[factor_col], q=eff_bins, duplicates="drop")

    prof = (
        sub.groupby([method_col, "_bin"], observed=True, as_index=False)
        .agg(
            bin_mean=(factor_col, "mean"),
            bin_min=(factor_col, "min"),
            bin_max=(factor_col, "max"),
            outcome_rate=(outcome_col, "mean"),
            n=(outcome_col, "size"),
        )
        .sort_values([method_col, "bin_mean"])
        .reset_index(drop=True)
    )
    prof["factor_name"] = factor_col
    return prof
