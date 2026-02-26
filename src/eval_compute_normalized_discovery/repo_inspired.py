from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


def _wasserstein_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    try:
        from scipy.stats import wasserstein_distance  # type: ignore

        return float(wasserstein_distance(a, b))
    except Exception:
        aq = np.quantile(a, np.linspace(0.0, 1.0, 101))
        bq = np.quantile(b, np.linspace(0.0, 1.0, 101))
        return float(np.mean(np.abs(aq - bq)))


def _js_divergence(a: np.ndarray, b: np.ndarray, n_bins: int = 20) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0
    bins = np.linspace(lo, hi, int(max(5, n_bins)) + 1)
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    ha = ha / max(ha.sum(), 1e-12)
    hb = hb / max(hb.sum(), 1e-12)
    m = 0.5 * (ha + hb)

    def _kl(p, q):
        mask = (p > 0) & (q > 0)
        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

    return float(0.5 * _kl(ha, m) + 0.5 * _kl(hb, m))


@dataclass(frozen=True)
class NaturalnessConfig:
    max_delta_l2_abs: Optional[float] = None
    max_delta_l2_quantile: float = 0.95
    max_fallback_ratio: float = 0.80
    max_actor_fallback_ratio: float = 0.80
    require_rollout_feasible: bool = True
    require_no_feasibility_violation: bool = True
    min_dist_non_null_ratio: float = 0.05


def derive_naturalness_mask(
    df: pd.DataFrame,
    config: Optional[NaturalnessConfig] = None,
) -> tuple[pd.Series, Dict[str, Any]]:
    cfg = config or NaturalnessConfig()
    if df.empty:
        return pd.Series(False, index=df.index), {}

    mask = pd.Series(True, index=df.index)
    meta: Dict[str, Any] = {}

    if cfg.require_rollout_feasible and "rollout_feasible" in df.columns:
        m = pd.to_numeric(df["rollout_feasible"], errors="coerce").fillna(0.0) > 0.5
        mask = mask & m
        meta["require_rollout_feasible"] = True

    if cfg.require_no_feasibility_violation and "feasibility_violation" in df.columns:
        m = pd.to_numeric(df["feasibility_violation"], errors="coerce").fillna(0.0) <= 0.0
        mask = mask & m
        meta["require_no_feasibility_violation"] = True

    if "dist_fallback_ratio" in df.columns:
        m = pd.to_numeric(df["dist_fallback_ratio"], errors="coerce").fillna(1.0) <= float(
            cfg.max_fallback_ratio
        )
        mask = mask & m
        meta["max_fallback_ratio"] = float(cfg.max_fallback_ratio)

    if "dist_actor_fallback_ratio" in df.columns:
        m = pd.to_numeric(df["dist_actor_fallback_ratio"], errors="coerce").fillna(1.0) <= float(
            cfg.max_actor_fallback_ratio
        )
        mask = mask & m
        meta["max_actor_fallback_ratio"] = float(cfg.max_actor_fallback_ratio)

    if "dist_non_null_ratio" in df.columns:
        m = pd.to_numeric(df["dist_non_null_ratio"], errors="coerce").fillna(0.0) >= float(
            cfg.min_dist_non_null_ratio
        )
        mask = mask & m
        meta["min_dist_non_null_ratio"] = float(cfg.min_dist_non_null_ratio)

    if "delta_l2" in df.columns:
        vals = pd.to_numeric(df["delta_l2"], errors="coerce")
        finite = vals[np.isfinite(vals)]
        if cfg.max_delta_l2_abs is not None:
            thr = float(cfg.max_delta_l2_abs)
        elif len(finite) > 0:
            thr = float(np.quantile(finite, float(np.clip(cfg.max_delta_l2_quantile, 0.0, 1.0))))
        else:
            thr = float("nan")
        if np.isfinite(thr):
            mask = mask & (vals.fillna(np.inf) <= thr)
        meta["delta_l2_threshold"] = float(thr) if np.isfinite(thr) else np.nan

    return mask.astype(bool), meta


def plausibility_filtered_summary(
    labeled_df: pd.DataFrame,
    naturalness_config: Optional[NaturalnessConfig] = None,
) -> tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    required = ["method", "is_blindspot", "blindspot_unit", "compute_units"]
    missing = [c for c in required if c not in labeled_df.columns]
    if missing:
        raise ValueError(f"Missing columns for plausibility summary: {missing}")

    natural_mask, meta = derive_naturalness_mask(labeled_df, config=naturalness_config)
    work = labeled_df.copy()
    work["is_natural"] = natural_mask.astype(int)
    work["is_plausible_blindspot"] = ((work["is_blindspot"] > 0) & (work["is_natural"] > 0)).astype(int)

    rows = []
    for method, grp in work.groupby("method", sort=True):
        n_rows = int(len(grp))
        natural = grp[grp["is_natural"] > 0]
        plausible = grp[grp["is_plausible_blindspot"] > 0]
        units = plausible["blindspot_unit"].dropna().astype(str)
        unique_units = int(units.nunique())
        total_compute = float(pd.to_numeric(grp["compute_units"], errors="coerce").fillna(0.0).sum())
        natural_compute = float(pd.to_numeric(natural["compute_units"], errors="coerce").fillna(0.0).sum())

        rows.append(
            {
                "method": str(method),
                "n_rows": n_rows,
                "natural_rows": int(len(natural)),
                "natural_pass_rate": float(len(natural) / max(1, n_rows)),
                "plausible_blindspot_events": int(len(plausible)),
                "plausible_blindspot_units": unique_units,
                "plausible_efficiency": float(unique_units / max(total_compute, 1e-9)),
                "plausible_efficiency_natural_compute": float(unique_units / max(natural_compute, 1e-9)),
            }
        )
    out = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    return out, natural_mask, meta


def cluster_coverage_diversity(
    labeled_df: pd.DataFrame,
    unit_col: str = "blindspot_unit",
) -> pd.DataFrame:
    required = ["method", "is_blindspot", unit_col]
    missing = [c for c in required if c not in labeled_df.columns]
    if missing:
        raise ValueError(f"Missing columns for cluster coverage diversity: {missing}")

    all_units = labeled_df[unit_col].dropna().astype(str).unique().tolist()
    total_units = max(1, len(all_units))

    rows = []
    for method, grp in labeled_df.groupby("method", sort=True):
        hits = grp[grp["is_blindspot"] > 0][unit_col].dropna().astype(str)
        n_hit = int(len(hits))
        if n_hit == 0:
            rows.append(
                {
                    "method": str(method),
                    "covered_units": 0,
                    "coverage_ratio": 0.0,
                    "normalized_entropy": 0.0,
                    "simpson_diversity": 0.0,
                }
            )
            continue
        freq = hits.value_counts(normalize=True)
        entropy = float(-(freq * np.log(freq + 1e-12)).sum())
        max_entropy = float(np.log(max(1, len(freq))))
        norm_entropy = float(entropy / max(max_entropy, 1e-12))
        simpson = float(1.0 - np.sum(np.square(freq.to_numpy(dtype=float))))
        rows.append(
            {
                "method": str(method),
                "covered_units": int(freq.index.nunique()),
                "coverage_ratio": float(freq.index.nunique() / total_units),
                "normalized_entropy": norm_entropy,
                "simpson_diversity": simpson,
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def realism_gap_summary(
    labeled_df: pd.DataFrame,
    baseline_method: str = "random",
    compare_on: str = "blindspots",
    feature_cols: Sequence[str] = ("risk_sks", "min_ttc", "delta_l2"),
) -> pd.DataFrame:
    required = ["method", "is_blindspot"]
    missing = [c for c in required if c not in labeled_df.columns]
    if missing:
        raise ValueError(f"Missing columns for realism gap summary: {missing}")
    work = labeled_df.copy()
    if compare_on == "blindspots":
        base = work[(work["method"] == baseline_method) & (work["is_blindspot"] > 0)]
        selector = work["is_blindspot"] > 0
    elif compare_on == "all":
        base = work[work["method"] == baseline_method]
        selector = pd.Series(True, index=work.index)
    else:
        raise ValueError(f"Unsupported compare_on: {compare_on}")

    rows = []
    for method, grp in work[selector].groupby("method", sort=True):
        row = {"method": str(method), "compare_on": compare_on}
        wd_vals = []
        js_vals = []
        for col in feature_cols:
            if col not in grp.columns or col not in base.columns:
                row[f"wd_{col}"] = np.nan
                row[f"js_{col}"] = np.nan
                continue
            a = pd.to_numeric(base[col], errors="coerce").dropna().to_numpy(dtype=float)
            b = pd.to_numeric(grp[col], errors="coerce").dropna().to_numpy(dtype=float)
            wd = _wasserstein_distance(a, b)
            js = _js_divergence(a, b, n_bins=20)
            row[f"wd_{col}"] = wd
            row[f"js_{col}"] = js
            if np.isfinite(wd):
                wd_vals.append(float(wd))
            if np.isfinite(js):
                js_vals.append(float(js))
        row["wd_mean"] = float(np.mean(wd_vals)) if len(wd_vals) > 0 else np.nan
        row["js_mean"] = float(np.mean(js_vals)) if len(js_vals) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def rulebook_lexicographic_ranking(
    method_df: pd.DataFrame,
    priorities: Sequence[str],
    maximize: Optional[Dict[str, bool]] = None,
) -> pd.DataFrame:
    if method_df.empty:
        return pd.DataFrame()
    maximize = dict(maximize or {})
    missing = [c for c in priorities if c not in method_df.columns]
    if missing:
        raise ValueError(f"Missing priority columns for rulebook ranking: {missing}")
    if "method" not in method_df.columns:
        raise ValueError("method_df must include a 'method' column.")

    rows = method_df.copy()
    sort_cols = []
    ascending = []
    for col in priorities:
        sort_cols.append(col)
        asc = not bool(maximize.get(col, True))
        ascending.append(asc)
    ranked = rows.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    ranked["rulebook_rank"] = np.arange(1, len(ranked) + 1)
    ranked["rulebook_priorities"] = " > ".join(list(priorities))
    return ranked


def combine_repo_inspired_method_table(
    base_method_scores_df: pd.DataFrame,
    plausible_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    realism_df: pd.DataFrame,
) -> pd.DataFrame:
    out = base_method_scores_df.copy()
    if "method" not in out.columns:
        raise ValueError("base_method_scores_df must include 'method'.")
    for extra in [plausible_df, coverage_df, realism_df]:
        if not extra.empty and "method" in extra.columns:
            out = out.merge(extra, on="method", how="left")
    return out
