from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

METHODS: List[str] = ["random", "risk_only", "surprise_only", "joint"]


def _subset_methods(df: pd.DataFrame, methods: Optional[Sequence[str]] = None) -> pd.DataFrame:
    methods = list(methods or METHODS)
    if "method" not in df.columns:
        raise ValueError("DataFrame must include a 'method' column.")
    return df[df["method"].isin(methods)].copy()


def method_summary(results_df: pd.DataFrame, methods: Optional[Sequence[str]] = None) -> pd.DataFrame:
    sub = _subset_methods(results_df, methods=methods)
    if sub.empty:
        return pd.DataFrame()

    out = (
        sub.groupby("method", as_index=False)
        .agg(
            n=("scenario_id", "nunique"),
            rows=("scenario_id", "size"),
            mean_risk=("risk_sks", "mean"),
            mean_surprise=("surprise_pd", "mean"),
            failure_rate=("failure_proxy", "mean"),
            q1_rate=("q1_hit", "mean"),
            q4_rate=("q4_hit", "mean"),
            blind_spot_rate=("blind_spot_proxy_hit", "mean"),
            mean_budget=("budget_units_used", "mean"),
            mean_objective_gain=("objective_gain", "mean"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
    return out


def budget_normalized_efficiency(
    results_df: pd.DataFrame,
    methods: Optional[Sequence[str]] = None,
    hit_col: str = "blind_spot_proxy_hit",
    budget_col: str = "budget_units_used",
) -> pd.DataFrame:
    sub = _subset_methods(results_df, methods=methods)
    required = ["method", "scenario_id", hit_col, budget_col]
    missing = [c for c in required if c not in sub.columns]
    if missing:
        raise ValueError(f"Missing columns for efficiency computation: {missing}")
    if sub.empty:
        return pd.DataFrame()

    grp = sub.groupby("method", as_index=False).agg(
        n_scenarios=("scenario_id", "nunique"),
        total_hits=(hit_col, "sum"),
        total_budget=(budget_col, "sum"),
        hit_rate=(hit_col, "mean"),
    )
    grp["hits_per_100_budget"] = 100.0 * grp["total_hits"] / np.maximum(grp["total_budget"], 1e-9)
    grp["hits_per_scenario"] = grp["total_hits"] / np.maximum(grp["n_scenarios"], 1.0)
    return grp.sort_values("method").reset_index(drop=True)


def _paired_delta(
    results_df: pd.DataFrame,
    treatment: str,
    control: str,
    value_col: str,
) -> pd.DataFrame:
    sub = results_df[results_df["method"].isin([treatment, control])].copy()
    if sub.empty:
        return pd.DataFrame(columns=["scenario_id", treatment, control, "delta"])

    pivot = (
        sub.pivot_table(index="scenario_id", columns="method", values=value_col, aggfunc="mean")
        .dropna(subset=[treatment, control])
        .reset_index()
    )
    if pivot.empty:
        return pd.DataFrame(columns=["scenario_id", treatment, control, "delta"])
    pivot["delta"] = pivot[treatment] - pivot[control]
    return pivot[["scenario_id", treatment, control, "delta"]]


def paired_bootstrap_ci(
    paired_df: pd.DataFrame,
    delta_col: str = "delta",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 17,
) -> Dict[str, float]:
    if paired_df.empty:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_pairs": 0}

    deltas = paired_df[delta_col].to_numpy(dtype=float)
    n = deltas.shape[0]
    rng = np.random.default_rng(seed)
    means = np.empty((n_boot,), dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(deltas[idx]))

    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {
        "mean": float(np.mean(deltas)),
        "ci_low": lo,
        "ci_high": hi,
        "n_pairs": int(n),
    }


def paired_permutation_pvalue(
    paired_df: pd.DataFrame,
    delta_col: str = "delta",
    n_perm: int = 5000,
    seed: int = 17,
    alternative: str = "greater",
) -> Dict[str, float]:
    if paired_df.empty:
        return {
            "mean": np.nan,
            "p_value": np.nan,
            "n_pairs": 0,
            "null_mean": np.nan,
            "null_std": np.nan,
        }

    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be one of {'greater', 'less', 'two-sided'}")

    deltas = paired_df[delta_col].to_numpy(dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return {
            "mean": np.nan,
            "p_value": np.nan,
            "n_pairs": 0,
            "null_mean": np.nan,
            "null_std": np.nan,
        }

    observed = float(np.mean(deltas))
    rng = np.random.default_rng(seed)
    n = int(deltas.size)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=(int(n_perm), n), replace=True)
    null_means = np.mean(signs * deltas.reshape(1, -1), axis=1)

    if alternative == "greater":
        p_value = float(np.mean(null_means >= observed))
    elif alternative == "less":
        p_value = float(np.mean(null_means <= observed))
    else:
        p_value = float(np.mean(np.abs(null_means) >= abs(observed)))

    return {
        "mean": observed,
        "p_value": p_value,
        "n_pairs": int(n),
        "null_mean": float(np.mean(null_means)),
        "null_std": float(np.std(null_means)),
    }


def paired_shuffle_control(
    paired_df: pd.DataFrame,
    treatment_col: str,
    control_col: str,
    n_shuffle: int = 5000,
    seed: int = 17,
) -> Dict[str, float]:
    if paired_df.empty:
        return {
            "observed_mean_delta": np.nan,
            "shuffle_mean_delta": np.nan,
            "shuffle_ci_low": np.nan,
            "shuffle_ci_high": np.nan,
            "shuffle_p_ge_observed": np.nan,
            "n_pairs": 0,
        }

    required = [treatment_col, control_col]
    missing = [c for c in required if c not in paired_df.columns]
    if missing:
        raise ValueError(f"Missing columns for paired shuffle control: {missing}")

    t = paired_df[treatment_col].to_numpy(dtype=float)
    c = paired_df[control_col].to_numpy(dtype=float)
    mask = np.isfinite(t) & np.isfinite(c)
    t = t[mask]
    c = c[mask]
    n = int(t.size)
    if n == 0:
        return {
            "observed_mean_delta": np.nan,
            "shuffle_mean_delta": np.nan,
            "shuffle_ci_low": np.nan,
            "shuffle_ci_high": np.nan,
            "shuffle_p_ge_observed": np.nan,
            "n_pairs": 0,
        }

    observed = float(np.mean(t - c))
    rng = np.random.default_rng(seed)
    null_means = np.empty((int(n_shuffle),), dtype=float)
    for i in range(int(n_shuffle)):
        perm = rng.permutation(n)
        null_means[i] = float(np.mean(t[perm] - c))

    return {
        "observed_mean_delta": observed,
        "shuffle_mean_delta": float(np.mean(null_means)),
        "shuffle_ci_low": float(np.quantile(null_means, 0.025)),
        "shuffle_ci_high": float(np.quantile(null_means, 0.975)),
        "shuffle_p_ge_observed": float(np.mean(null_means >= observed)),
        "n_pairs": int(n),
    }


def deterministic_scenario_split(
    results_df: pd.DataFrame,
    holdout_fraction: float = 0.20,
    seed: int = 17,
    scenario_col: str = "scenario_id",
) -> pd.DataFrame:
    if scenario_col not in results_df.columns:
        raise ValueError(f"Missing scenario column: {scenario_col}")
    holdout_fraction = float(holdout_fraction)
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError("holdout_fraction must be in (0, 1).")

    scenarios = np.asarray(sorted(results_df[scenario_col].dropna().unique().tolist()))
    n = int(scenarios.size)
    if n == 0:
        return pd.DataFrame(columns=[scenario_col, "eval_split"])

    n_holdout = int(max(1, round(holdout_fraction * n)))
    n_holdout = min(n_holdout, n - 1) if n > 1 else 1
    rng = np.random.default_rng(int(seed))
    shuffled = scenarios.copy()
    rng.shuffle(shuffled)
    holdout_set = set(shuffled[:n_holdout].tolist())

    rows = []
    for sid in scenarios.tolist():
        rows.append(
            {
                scenario_col: sid,
                "eval_split": "holdout" if sid in holdout_set else "explore",
            }
        )
    return pd.DataFrame(rows)


def paired_effect_significance_table(
    results_df: pd.DataFrame,
    treatment: str = "joint",
    control: str = "risk_only",
    outcome_col: str = "blind_spot_proxy_hit",
    split_col: Optional[str] = None,
    bootstrap_samples: int = 2000,
    permutation_samples: int = 5000,
    shuffle_samples: int = 5000,
    seed: int = 17,
) -> pd.DataFrame:
    required = ["scenario_id", "method", outcome_col]
    missing = [c for c in required if c not in results_df.columns]
    if missing:
        raise ValueError(f"Missing columns for paired significance table: {missing}")

    if split_col is not None and split_col not in results_df.columns:
        raise ValueError(f"split_col '{split_col}' not found in results_df.")

    if split_col is None:
        split_items = [("all", results_df.copy())]
    else:
        split_items = [("all", results_df.copy())]
        split_items.extend(
            [(str(name), g.copy()) for name, g in results_df.groupby(split_col, dropna=False)]
        )

    rows: List[Dict[str, Any]] = []
    for split_name, split_df in split_items:
        paired = _paired_delta(
            split_df,
            treatment=treatment,
            control=control,
            value_col=outcome_col,
        )
        ci = paired_bootstrap_ci(
            paired_df=paired,
            delta_col="delta",
            n_boot=int(bootstrap_samples),
            seed=int(seed),
        )
        perm = paired_permutation_pvalue(
            paired_df=paired,
            delta_col="delta",
            n_perm=int(permutation_samples),
            seed=int(seed),
            alternative="greater",
        )
        shuf = paired_shuffle_control(
            paired_df=paired,
            treatment_col=treatment,
            control_col=control,
            n_shuffle=int(shuffle_samples),
            seed=int(seed),
        )

        deltas = paired["delta"].to_numpy(dtype=float) if len(paired) > 0 else np.asarray([])
        delta_std = float(np.std(deltas)) if deltas.size > 1 else float("nan")
        effect_size = float(ci["mean"] / delta_std) if np.isfinite(delta_std) and delta_std > 1e-12 else float("nan")

        rows.append(
            {
                "split": split_name,
                "treatment": treatment,
                "control": control,
                "outcome_col": outcome_col,
                "n_pairs": int(ci["n_pairs"]),
                "mean_delta": float(ci["mean"]),
                "ci_low": float(ci["ci_low"]),
                "ci_high": float(ci["ci_high"]),
                "paired_effect_size_d": effect_size,
                "permutation_pvalue_one_sided": float(perm["p_value"]),
                "shuffle_null_p_ge_observed": float(shuf["shuffle_p_ge_observed"]),
                "shuffle_null_ci_low": float(shuf["shuffle_ci_low"]),
                "shuffle_null_ci_high": float(shuf["shuffle_ci_high"]),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    split_order = {"all": 0, "explore": 1, "holdout": 2}
    out["_order"] = out["split"].map(split_order).fillna(99).astype(int)
    out = out.sort_values(["_order", "split"]).drop(columns=["_order"]).reset_index(drop=True)
    return out


def conditional_lift_by_risk_bins(
    results_df: pd.DataFrame,
    treatment: str = "joint",
    control: str = "risk_only",
    risk_col: str = "risk_sks",
    outcome_col: str = "blind_spot_proxy_hit",
    n_bins: int = 10,
    min_bin_count: int = 5,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 17,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = ["scenario_id", "method", risk_col, outcome_col]
    missing = [c for c in required if c not in results_df.columns]
    if missing:
        raise ValueError(f"Missing columns for conditional-lift computation: {missing}")

    sub = results_df[results_df["method"].isin([treatment, control])].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    unique_risk = int(sub[risk_col].nunique())
    effective_bins = int(max(1, min(n_bins, unique_risk)))
    if effective_bins <= 1:
        sub["_risk_bin"] = "all"
    else:
        sub["_risk_bin"] = pd.qcut(sub[risk_col], q=effective_bins, duplicates="drop")

    grouped = (
        sub.groupby(["_risk_bin", "method"], observed=True)
        .agg(
            n=("scenario_id", "size"),
            n_scenarios=("scenario_id", "nunique"),
            outcome_rate=(outcome_col, "mean"),
            mean_risk=(risk_col, "mean"),
        )
        .reset_index()
    )

    pivot_rate = grouped.pivot(index="_risk_bin", columns="method", values="outcome_rate")
    pivot_n = grouped.pivot(index="_risk_bin", columns="method", values="n").fillna(0)
    pivot_risk = grouped.groupby("_risk_bin", observed=True)["mean_risk"].mean()

    rows: List[Dict[str, Any]] = []
    for risk_bin in pivot_rate.index:
        tr = float(pivot_rate.loc[risk_bin, treatment]) if treatment in pivot_rate.columns else np.nan
        ct = float(pivot_rate.loc[risk_bin, control]) if control in pivot_rate.columns else np.nan
        n_t = int(pivot_n.loc[risk_bin, treatment]) if treatment in pivot_n.columns else 0
        n_c = int(pivot_n.loc[risk_bin, control]) if control in pivot_n.columns else 0

        if (n_t < min_bin_count) or (n_c < min_bin_count):
            continue

        rows.append(
            {
                "risk_bin": str(risk_bin),
                "mean_risk": float(pivot_risk.loc[risk_bin]),
                "n_treatment": int(n_t),
                "n_control": int(n_c),
                "treatment_rate": tr,
                "control_rate": ct,
                "lift": float(tr - ct),
            }
        )

    bin_df = pd.DataFrame(rows)
    if not bin_df.empty:
        bin_df = bin_df.sort_values("mean_risk").reset_index(drop=True)

    paired = _paired_delta(sub, treatment=treatment, control=control, value_col=outcome_col)
    ci = paired_bootstrap_ci(
        paired_df=paired,
        delta_col="delta",
        n_boot=bootstrap_samples,
        seed=bootstrap_seed,
    )
    overall = pd.DataFrame(
        [
            {
                "treatment": treatment,
                "control": control,
                "outcome_col": outcome_col,
                "mean_delta": ci["mean"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "n_pairs": int(ci["n_pairs"]),
                "treatment_rate_overall": float(sub[sub["method"] == treatment][outcome_col].mean()),
                "control_rate_overall": float(sub[sub["method"] == control][outcome_col].mean()),
            }
        ]
    )
    return bin_df, overall


def compute_trace_event_flags(
    trace_df: pd.DataFrame,
    thresholds: Dict[str, Any],
    risk_col: str = "risk_sks",
    surprise_col: str = "surprise_pd",
    failure_col: str = "failure_proxy",
) -> pd.DataFrame:
    required = ["scenario_id", "method", "eval_index", risk_col, surprise_col, failure_col]
    missing = [c for c in required if c not in trace_df.columns]
    if missing:
        raise ValueError(f"Missing trace columns for event flags: {missing}")

    surprise_thr = float(thresholds.get("surprise_high_threshold", np.nan))
    risk_high_thr = float(thresholds.get("risk_high_threshold", np.nan))
    risk_low_thr = float(thresholds.get("risk_low_threshold", np.nan))
    if not np.isfinite(surprise_thr):
        raise ValueError("thresholds must contain surprise_high_threshold")

    out = trace_df.copy()
    out["blind_spot_proxy_hit_eval"] = (
        (out[failure_col] > 0.0) & (out[surprise_col] >= surprise_thr)
    ).astype(int)
    out["q1_hit_eval"] = (
        (out[risk_col] >= risk_high_thr) & (out[surprise_col] >= surprise_thr)
    ).astype(int)
    out["q4_hit_eval"] = (
        (out[risk_col] <= risk_low_thr) & (out[surprise_col] >= surprise_thr)
    ).astype(int)
    return out


def discovery_curve_from_trace(
    trace_df: pd.DataFrame,
    thresholds: Dict[str, Any],
    methods: Optional[Sequence[str]] = None,
    max_eval_index: Optional[int] = None,
) -> pd.DataFrame:
    methods = list(methods or METHODS)
    flagged = compute_trace_event_flags(trace_df, thresholds=thresholds)
    flagged = flagged[flagged["method"].isin(methods)].copy()
    if flagged.empty:
        return pd.DataFrame()

    out_rows: List[Dict[str, Any]] = []
    for method in methods:
        sub = flagged[flagged["method"] == method].copy()
        if sub.empty:
            continue

        scenario_ids = sorted(sub["scenario_id"].unique().tolist())
        if max_eval_index is None:
            max_k = int(sub["eval_index"].max())
        else:
            max_k = int(max_eval_index)
        eval_grid = np.arange(0, max_k + 1, dtype=int)

        curves = []
        for sid in scenario_ids:
            g = (
                sub[sub["scenario_id"] == sid][["eval_index", "blind_spot_proxy_hit_eval"]]
                .sort_values("eval_index")
                .drop_duplicates(subset=["eval_index"], keep="last")
                .set_index("eval_index")
            )
            g = g.reindex(eval_grid)
            g["blind_spot_proxy_hit_eval"] = g["blind_spot_proxy_hit_eval"].fillna(0).astype(float)
            discovered = g["blind_spot_proxy_hit_eval"].cummax().to_numpy(dtype=float)
            curves.append(discovered)

        mat = np.vstack(curves) if len(curves) > 0 else np.zeros((0, len(eval_grid)))
        mean_curve = mat.mean(axis=0) if mat.shape[0] > 0 else np.zeros((len(eval_grid),), dtype=float)

        for k, y in zip(eval_grid.tolist(), mean_curve.tolist()):
            out_rows.append(
                {
                    "method": method,
                    "eval_index": int(k),
                    "discovery_rate": float(y),
                    "n_scenarios": int(mat.shape[0]),
                    "expected_hits": float(y * mat.shape[0]),
                }
            )

    return pd.DataFrame(out_rows).sort_values(["method", "eval_index"]).reset_index(drop=True)


def discovery_auc(curve_df: pd.DataFrame) -> pd.DataFrame:
    required = ["method", "eval_index", "discovery_rate"]
    missing = [c for c in required if c not in curve_df.columns]
    if missing:
        raise ValueError(f"Missing curve columns for AUC: {missing}")
    if curve_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for method, g in curve_df.groupby("method"):
        gg = g.sort_values("eval_index")
        x = gg["eval_index"].to_numpy(dtype=float)
        y = gg["discovery_rate"].to_numpy(dtype=float)
        if x.size <= 1:
            auc = float(y[-1]) if y.size == 1 else np.nan
            norm_auc = auc
        else:
            auc = float(np.trapezoid(y, x))
            span = float(x[-1] - x[0])
            norm_auc = float(auc / span) if span > 0 else np.nan
        rows.append(
            {
                "method": str(method),
                "auc_discovery_vs_eval": auc,
                "normalized_auc": norm_auc,
                "max_eval_index": int(x[-1]) if x.size > 0 else -1,
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
