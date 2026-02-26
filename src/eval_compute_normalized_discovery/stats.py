from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _spearman_corr_no_scipy(a: pd.Series, b: pd.Series) -> float:
    ar = pd.to_numeric(a, errors="coerce")
    br = pd.to_numeric(b, errors="coerce")
    mask = np.isfinite(ar.to_numpy(dtype=float)) & np.isfinite(br.to_numpy(dtype=float))
    if int(mask.sum()) <= 1:
        return float("nan")
    ar_rank = ar[mask].rank(method="average")
    br_rank = br[mask].rank(method="average")
    if int(ar_rank.nunique()) <= 1 or int(br_rank.nunique()) <= 1:
        return float("nan")
    return float(ar_rank.corr(br_rank, method="pearson"))


def _paired_definition_deltas(
    metrics_df: pd.DataFrame,
    treatment: str,
    control: str,
    score_col: str,
) -> pd.DataFrame:
    required = ["definition_key", "method", score_col]
    missing = [c for c in required if c not in metrics_df.columns]
    if missing:
        raise ValueError(f"Missing columns for paired deltas: {missing}")

    sub = metrics_df[metrics_df["method"].isin([treatment, control])].copy()
    if sub.empty:
        return pd.DataFrame(columns=["definition_key", treatment, control, "delta"])

    piv = (
        sub.pivot_table(index="definition_key", columns="method", values=score_col, aggfunc="mean")
        .dropna(subset=[treatment, control])
        .reset_index()
    )
    if piv.empty:
        return pd.DataFrame(columns=["definition_key", treatment, control, "delta"])
    piv["delta"] = piv[treatment] - piv[control]
    return piv[["definition_key", treatment, control, "delta"]]


def rank_stability_table(
    metrics_df: pd.DataFrame,
    score_col: str = "discovery_efficiency",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = ["definition_key", "method", score_col]
    missing = [c for c in required if c not in metrics_df.columns]
    if missing:
        raise ValueError(f"Missing columns for rank stability: {missing}")
    if metrics_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    pivot = (
        metrics_df.pivot_table(index="method", columns="definition_key", values=score_col, aggfunc="mean")
        .sort_index()
    )
    rank_mat = pivot.rank(axis=0, ascending=False, method="average")
    rank_summary = (
        pd.DataFrame(
            {
                "method": rank_mat.index,
                "mean_rank": rank_mat.mean(axis=1),
                "std_rank": rank_mat.std(axis=1),
                "mean_score": pivot.mean(axis=1),
            }
        )
        .sort_values(["mean_rank", "mean_score"], ascending=[True, False])
        .reset_index(drop=True)
    )

    defs = list(rank_mat.columns)
    pair_rows = []
    for i, a in enumerate(defs):
        for b in defs[i + 1 :]:
            sa = rank_mat[a]
            sb = rank_mat[b]
            corr = _spearman_corr_no_scipy(sa, sb)
            pair_rows.append(
                {
                    "definition_a": str(a),
                    "definition_b": str(b),
                    "spearman_rank_corr": corr,
                }
            )
    pair_df = pd.DataFrame(pair_rows)
    return rank_summary, pair_df


def paired_bootstrap_delta(
    metrics_df: pd.DataFrame,
    treatment: str,
    control: str,
    score_col: str = "discovery_efficiency",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 17,
) -> Dict[str, Any]:
    paired = _paired_definition_deltas(
        metrics_df=metrics_df,
        treatment=treatment,
        control=control,
        score_col=score_col,
    )
    if paired.empty:
        return {
            "treatment": treatment,
            "control": control,
            "score_col": score_col,
            "n_pairs": 0,
            "mean_delta": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    deltas = paired["delta"].to_numpy(dtype=float)
    n = deltas.size
    rng = np.random.default_rng(seed)
    means = np.empty((int(max(1, n_boot)),), dtype=float)
    for i in range(means.size):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(deltas[idx]))

    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {
        "treatment": treatment,
        "control": control,
        "score_col": score_col,
        "n_pairs": int(n),
        "mean_delta": float(np.mean(deltas)),
        "ci_low": lo,
        "ci_high": hi,
    }


def paired_permutation_test(
    metrics_df: pd.DataFrame,
    treatment: str,
    control: str,
    score_col: str = "discovery_efficiency",
    n_perm: int = 5000,
    seed: int = 17,
) -> Dict[str, Any]:
    paired = _paired_definition_deltas(
        metrics_df=metrics_df,
        treatment=treatment,
        control=control,
        score_col=score_col,
    )
    if paired.empty:
        return {
            "treatment": treatment,
            "control": control,
            "score_col": score_col,
            "n_pairs": 0,
            "observed_mean_delta": np.nan,
            "p_value_two_sided": np.nan,
        }

    deltas = paired["delta"].to_numpy(dtype=float)
    obs = float(np.mean(deltas))
    rng = np.random.default_rng(seed)
    ge = 0
    n_perm = int(max(1, n_perm))
    for _ in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=deltas.size, replace=True)
        perm = float(np.mean(deltas * signs))
        if abs(perm) >= abs(obs):
            ge += 1
    p = float((ge + 1) / (n_perm + 1))
    return {
        "treatment": treatment,
        "control": control,
        "score_col": score_col,
        "n_pairs": int(deltas.size),
        "observed_mean_delta": obs,
        "p_value_two_sided": p,
    }
