from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_METHODS: Tuple[str, ...] = ("random", "risk_only", "surprise_only", "joint")
SURPRISE_COL_CANDIDATES: Tuple[str, ...] = ("delta_surprise", "delta_surprise_pd", "surprise_pd")


def _resolve_surprise_col(df: pd.DataFrame) -> str:
    for col in SURPRISE_COL_CANDIDATES:
        if col in df.columns:
            return col
    return "delta_surprise"


def _ensure_surprise_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "delta_surprise" not in df.columns:
        if "delta_surprise_pd" in df.columns:
            df["delta_surprise"] = df["delta_surprise_pd"]
        elif "surprise_pd" in df.columns:
            df["delta_surprise"] = df["surprise_pd"]
    if ("delta_surprise_pd" not in df.columns) and ("delta_surprise" in df.columns):
        df["delta_surprise_pd"] = df["delta_surprise"]
    if ("surprise_pd" not in df.columns) and ("delta_surprise" in df.columns):
        df["surprise_pd"] = df["delta_surprise"]
    return df


def _corr_or_nan(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if (not np.isfinite(x).all()) or (not np.isfinite(y).all()):
        return float("nan")
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return float("nan")
    out = pd.Series(x).corr(pd.Series(y), method=method)
    return float(out) if pd.notna(out) else float("nan")


def analyze_surprise_signal_usefulness(
    closedloop_results_df: pd.DataFrame,
    methods: Optional[Sequence[str]] = None,
    n_bins: int = 10,
    top_fracs: Sequence[float] = (0.10, 0.20),
    scenario_min_points: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    methods = tuple(methods) if methods is not None else DEFAULT_METHODS
    closedloop_results_df = _ensure_surprise_alias_columns(closedloop_results_df.copy())
    surprise_col = _resolve_surprise_col(closedloop_results_df)
    required = {"scenario_id", "method", surprise_col, "delta_risk"}
    if not required.issubset(set(closedloop_results_df.columns)):
        missing = sorted(list(required.difference(set(closedloop_results_df.columns))))
        raise ValueError(
            "closedloop_results_df is missing required columns for signal diagnostics: "
            + ", ".join(missing)
        )

    base = closedloop_results_df[closedloop_results_df["method"].isin(methods)].copy()
    usable = base[
        np.isfinite(base[surprise_col].to_numpy(dtype=float))
        & np.isfinite(base["delta_risk"].to_numpy(dtype=float))
    ].copy()

    if usable.empty:
        empty = pd.DataFrame()
        summary = pd.DataFrame(
            [
                {
                    "n_rows_input": int(len(closedloop_results_df)),
                    "n_rows_methods": int(len(base)),
                    "n_usable_rows": 0,
                    "n_scenarios_usable": 0,
                    "surprise_std": float("nan"),
                    "delta_risk_std": float("nan"),
                    "delta_risk_mean": float("nan"),
                    "delta_risk_positive_rate": float("nan"),
                    "corr_pearson": float("nan"),
                    "corr_spearman": float("nan"),
                    "corr_kendall": float("nan"),
                    "monotonic_bin_count": 0,
                    "monotonic_non_decreasing": 0,
                    "monotonic_violation_rate": float("nan"),
                    "monotonic_bin_spearman": float("nan"),
                    "within_scenario_count": 0,
                    "within_spearman_mean": float("nan"),
                    "within_spearman_positive_rate": float("nan"),
                    "top10_lift_vs_all": float("nan"),
                    "top10_lift_vs_bottom": float("nan"),
                }
            ]
        )
        return summary, empty, empty, empty, empty

    signal = usable[surprise_col].to_numpy(dtype=float)
    outcome = usable["delta_risk"].to_numpy(dtype=float)

    corr_pearson = _corr_or_nan(signal, outcome, "pearson")
    corr_spearman = _corr_or_nan(signal, outcome, "spearman")
    corr_kendall = _corr_or_nan(signal, outcome, "kendall")

    method_rows = []
    for method_name, g in usable.groupby("method", as_index=False):
        s = g[surprise_col].to_numpy(dtype=float)
        y = g["delta_risk"].to_numpy(dtype=float)
        method_rows.append(
            {
                "method": str(method_name),
                "n": int(len(g)),
                "surprise_std": float(np.nanstd(s)) if s.size > 1 else float("nan"),
                "delta_risk_std": float(np.nanstd(y)) if y.size > 1 else float("nan"),
                "corr_pearson": _corr_or_nan(s, y, "pearson"),
                "corr_spearman": _corr_or_nan(s, y, "spearman"),
                "corr_kendall": _corr_or_nan(s, y, "kendall"),
                "delta_risk_positive_rate": float(np.mean(y > 0.0)),
            }
        )
    method_corr_df = pd.DataFrame(method_rows).sort_values("method").reset_index(drop=True)

    usable = usable.sort_values(surprise_col).reset_index(drop=True)
    uniq_signal = int(np.unique(signal).size)
    q = int(max(1, min(int(n_bins), uniq_signal)))
    if q <= 1:
        bin_codes = pd.Series(np.zeros((len(usable),), dtype=int))
    else:
        ranked = usable[surprise_col].rank(method="average")
        binned = pd.qcut(ranked, q=q, labels=False, duplicates="drop")
        if binned.isna().all():
            bin_codes = pd.Series(np.zeros((len(usable),), dtype=int))
        else:
            bin_codes = binned.fillna(0).astype(int)

    bdf = usable.copy()
    bdf["signal_bin"] = bin_codes.to_numpy(dtype=int)
    bin_df = (
        bdf.groupby("signal_bin", as_index=False)
        .agg(
            n=("delta_risk", "size"),
            signal_min=(surprise_col, "min"),
            signal_max=(surprise_col, "max"),
            signal_mean=(surprise_col, "mean"),
            delta_risk_mean=("delta_risk", "mean"),
            delta_risk_p50=("delta_risk", "median"),
            delta_risk_positive_rate=("delta_risk", lambda x: float(np.mean(np.asarray(x, dtype=float) > 0.0))),
        )
        .sort_values("signal_bin")
        .reset_index(drop=True)
    )
    bin_df["signal_bin"] = bin_df["signal_bin"].astype(int) + 1

    monotonic_bin_count = int(len(bin_df))
    if monotonic_bin_count >= 2:
        means = bin_df["delta_risk_mean"].to_numpy(dtype=float)
        diffs = np.diff(means)
        monotonic_non_decreasing = int(np.all(diffs >= -1e-12))
        monotonic_violation_rate = float(np.mean(diffs < -1e-12))
        monotonic_bin_spearman = _corr_or_nan(
            np.arange(monotonic_bin_count, dtype=float),
            means,
            "spearman",
        )
    else:
        monotonic_non_decreasing = 0
        monotonic_violation_rate = float("nan")
        monotonic_bin_spearman = float("nan")

    topk_rows = []
    n = int(len(usable))
    overall_mean = float(np.mean(outcome))
    overall_pos_rate = float(np.mean(outcome > 0.0))
    for frac in sorted(set(float(x) for x in top_fracs)):
        if (not np.isfinite(frac)) or frac <= 0.0 or frac >= 1.0:
            continue
        k = int(max(1, np.ceil(frac * n)))
        top = usable.nlargest(k, surprise_col)
        bottom = usable.nsmallest(k, surprise_col)
        top_mean = float(np.mean(top["delta_risk"].to_numpy(dtype=float)))
        bottom_mean = float(np.mean(bottom["delta_risk"].to_numpy(dtype=float)))
        top_pos_rate = float(np.mean(top["delta_risk"].to_numpy(dtype=float) > 0.0))
        bottom_pos_rate = float(np.mean(bottom["delta_risk"].to_numpy(dtype=float) > 0.0))
        topk_rows.append(
            {
                "top_fraction": float(frac),
                "k": int(k),
                "overall_mean_delta_risk": overall_mean,
                "top_mean_delta_risk": top_mean,
                "bottom_mean_delta_risk": bottom_mean,
                "lift_top_vs_all": float(top_mean - overall_mean),
                "lift_top_vs_bottom": float(top_mean - bottom_mean),
                "overall_delta_risk_positive_rate": overall_pos_rate,
                "top_delta_risk_positive_rate": top_pos_rate,
                "bottom_delta_risk_positive_rate": bottom_pos_rate,
                "positive_rate_lift_top_vs_all": float(top_pos_rate - overall_pos_rate),
                "positive_rate_lift_top_vs_bottom": float(top_pos_rate - bottom_pos_rate),
            }
        )
    topk_df = pd.DataFrame(topk_rows).sort_values("top_fraction").reset_index(drop=True)

    scenario_rows = []
    for scenario_id, g in usable.groupby("scenario_id", as_index=False):
        if len(g) < int(max(2, scenario_min_points)):
            continue
        s = g[surprise_col].to_numpy(dtype=float)
        y = g["delta_risk"].to_numpy(dtype=float)
        if np.unique(s).size < 2 or np.unique(y).size < 2:
            continue
        scenario_rows.append(
            {
                "scenario_id": int(scenario_id),
                "n": int(len(g)),
                "surprise_std": float(np.std(s)),
                "delta_risk_std": float(np.std(y)),
                "corr_spearman": _corr_or_nan(s, y, "spearman"),
                "corr_kendall": _corr_or_nan(s, y, "kendall"),
                "corr_pearson": _corr_or_nan(s, y, "pearson"),
            }
        )
    if len(scenario_rows) > 0:
        within_scenario_df = pd.DataFrame(scenario_rows).sort_values("scenario_id").reset_index(drop=True)
    else:
        within_scenario_df = pd.DataFrame(
            columns=[
                "scenario_id",
                "n",
                "surprise_std",
                "delta_risk_std",
                "corr_spearman",
                "corr_kendall",
                "corr_pearson",
            ]
        )

    within_count = int(len(within_scenario_df))
    within_spearman_mean = (
        float(np.nanmean(within_scenario_df["corr_spearman"].to_numpy(dtype=float)))
        if within_count > 0
        else float("nan")
    )
    within_spearman_positive_rate = (
        float(np.nanmean(within_scenario_df["corr_spearman"].to_numpy(dtype=float) > 0.0))
        if within_count > 0
        else float("nan")
    )

    top10_row = topk_df[np.isclose(topk_df["top_fraction"], 0.10)] if not topk_df.empty else pd.DataFrame()
    top10_lift_vs_all = (
        float(top10_row.iloc[0]["lift_top_vs_all"]) if len(top10_row) == 1 else float("nan")
    )
    top10_lift_vs_bottom = (
        float(top10_row.iloc[0]["lift_top_vs_bottom"]) if len(top10_row) == 1 else float("nan")
    )

    summary_df = pd.DataFrame(
        [
            {
                "n_rows_input": int(len(closedloop_results_df)),
                "n_rows_methods": int(len(base)),
                "n_usable_rows": int(len(usable)),
                "n_scenarios_usable": int(usable["scenario_id"].nunique()),
                "surprise_std": float(np.std(signal)) if signal.size > 1 else float("nan"),
                "delta_risk_std": float(np.std(outcome)) if outcome.size > 1 else float("nan"),
                "delta_risk_mean": overall_mean,
                "delta_risk_positive_rate": overall_pos_rate,
                "corr_pearson": corr_pearson,
                "corr_spearman": corr_spearman,
                "corr_kendall": corr_kendall,
                "monotonic_bin_count": int(monotonic_bin_count),
                "monotonic_non_decreasing": int(monotonic_non_decreasing),
                "monotonic_violation_rate": monotonic_violation_rate,
                "monotonic_bin_spearman": monotonic_bin_spearman,
                "within_scenario_count": within_count,
                "within_spearman_mean": within_spearman_mean,
                "within_spearman_positive_rate": within_spearman_positive_rate,
                "top10_lift_vs_all": top10_lift_vs_all,
                "top10_lift_vs_bottom": top10_lift_vs_bottom,
            }
        ]
    )

    return summary_df, method_corr_df, bin_df, topk_df, within_scenario_df


def save_surprise_signal_usefulness_artifacts(
    run_prefix: str,
    summary_df: pd.DataFrame,
    method_corr_df: pd.DataFrame,
    bin_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    within_scenario_df: pd.DataFrame,
) -> Dict[str, str]:
    base = Path(run_prefix)
    out = {
        "surprise_signal_summary": f"{run_prefix}_surprise_signal_summary.csv",
        "surprise_signal_method_corr": f"{run_prefix}_surprise_signal_method_corr.csv",
        "surprise_signal_bins": f"{run_prefix}_surprise_signal_bins.csv",
        "surprise_signal_topk_lift": f"{run_prefix}_surprise_signal_topk_lift.csv",
        "surprise_signal_within_scenario": f"{run_prefix}_surprise_signal_within_scenario.csv",
    }
    Path(base).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out["surprise_signal_summary"], index=False)
    method_corr_df.to_csv(out["surprise_signal_method_corr"], index=False)
    bin_df.to_csv(out["surprise_signal_bins"], index=False)
    topk_df.to_csv(out["surprise_signal_topk_lift"], index=False)
    within_scenario_df.to_csv(out["surprise_signal_within_scenario"], index=False)
    return out
