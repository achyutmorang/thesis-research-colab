from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .stats import (
    paired_bootstrap_delta,
    paired_permutation_test,
    rank_stability_table,
)


@dataclass(frozen=True)
class DiscoveryHypothesisConfig:
    treatment: str = "joint"
    control: str = "random"
    score_col: str = "discovery_efficiency"
    alpha: float = 0.05
    min_win_fraction: float = 0.60
    min_top_win_fraction: float = 0.50
    max_rank_std: float = 1.0
    min_worst_case_delta: float = 0.0


def _definition_delta_table(
    metrics_df: pd.DataFrame,
    treatment: str,
    control: str,
    score_col: str,
) -> pd.DataFrame:
    sub = metrics_df[metrics_df["method"].isin([treatment, control])].copy()
    if sub.empty:
        return pd.DataFrame(columns=["definition_key", "delta"])
    piv = (
        sub.pivot_table(index="definition_key", columns="method", values=score_col, aggfunc="mean")
        .dropna(subset=[treatment, control])
        .reset_index()
    )
    if piv.empty:
        return pd.DataFrame(columns=["definition_key", "delta"])
    piv["delta"] = piv[treatment] - piv[control]
    return piv


def _winner_frequency(metrics_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(columns=["method", "wins", "win_fraction"])
    rows = []
    for key, grp in metrics_df.groupby("definition_key", sort=True):
        gg = grp.sort_values(score_col, ascending=False).reset_index(drop=True)
        if gg.empty:
            continue
        rows.append({"definition_key": key, "winner": str(gg.loc[0, "method"])})
    if len(rows) == 0:
        return pd.DataFrame(columns=["method", "wins", "win_fraction"])
    win_df = pd.DataFrame(rows)
    counts = win_df["winner"].value_counts().rename_axis("method").reset_index(name="wins")
    counts["win_fraction"] = counts["wins"] / max(1, len(win_df))
    return counts.sort_values("wins", ascending=False).reset_index(drop=True)


def evaluate_discovery_hypotheses(
    metrics_df: pd.DataFrame,
    config: Optional[DiscoveryHypothesisConfig] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = config or DiscoveryHypothesisConfig()
    required = ["definition_key", "method", cfg.score_col]
    missing = [c for c in required if c not in metrics_df.columns]
    if missing:
        raise ValueError(f"Missing columns for hypothesis evaluation: {missing}")

    deltas = _definition_delta_table(
        metrics_df=metrics_df,
        treatment=cfg.treatment,
        control=cfg.control,
        score_col=cfg.score_col,
    )
    boot = paired_bootstrap_delta(
        metrics_df=metrics_df,
        treatment=cfg.treatment,
        control=cfg.control,
        score_col=cfg.score_col,
        alpha=cfg.alpha,
    )
    perm = paired_permutation_test(
        metrics_df=metrics_df,
        treatment=cfg.treatment,
        control=cfg.control,
        score_col=cfg.score_col,
    )
    rank_summary, rank_pairs = rank_stability_table(metrics_df, score_col=cfg.score_col)
    winner_df = _winner_frequency(metrics_df, score_col=cfg.score_col)

    n_defs = int(metrics_df["definition_key"].nunique())
    win_fraction = (
        float((deltas["delta"] > 0).mean()) if (not deltas.empty and "delta" in deltas.columns) else float("nan")
    )
    worst_case_delta = float(deltas["delta"].min()) if not deltas.empty else float("nan")
    treatment_rank_std = float(
        rank_summary.loc[rank_summary["method"] == cfg.treatment, "std_rank"].iloc[0]
    ) if (not rank_summary.empty and (rank_summary["method"] == cfg.treatment).any()) else float("nan")
    treatment_top_win_fraction = float(
        winner_df.loc[winner_df["method"] == cfg.treatment, "win_fraction"].iloc[0]
    ) if (not winner_df.empty and (winner_df["method"] == cfg.treatment).any()) else 0.0

    h1_pass = (
        np.isfinite(boot.get("mean_delta", np.nan))
        and float(boot["mean_delta"]) > 0.0
        and float(boot.get("ci_low", np.nan)) > 0.0
        and float(perm.get("p_value_two_sided", np.nan)) < cfg.alpha
    )
    h2_pass = np.isfinite(win_fraction) and float(win_fraction) >= cfg.min_win_fraction
    h3_pass = np.isfinite(worst_case_delta) and float(worst_case_delta) >= cfg.min_worst_case_delta
    h4_pass = (
        np.isfinite(treatment_rank_std)
        and treatment_rank_std <= cfg.max_rank_std
        and treatment_top_win_fraction >= cfg.min_top_win_fraction
    )

    def _verdict(is_pass: bool, fallback: bool = False) -> str:
        if is_pass:
            return "PASS"
        return "INCONCLUSIVE" if fallback else "FAIL"

    rows = [
        {
            "hypothesis_id": "H1_efficiency_superiority",
            "statement": f"{cfg.treatment} has higher {cfg.score_col} than {cfg.control}.",
            "test": "paired bootstrap + paired sign permutation",
            "metric_value": float(boot.get("mean_delta", np.nan)),
            "threshold": "> 0 with CI low > 0 and p < alpha",
            "evidence": (
                f"mean_delta={boot.get('mean_delta', np.nan):.6g}; "
                f"ci_low={boot.get('ci_low', np.nan):.6g}; "
                f"ci_high={boot.get('ci_high', np.nan):.6g}; "
                f"p={perm.get('p_value_two_sided', np.nan):.6g}"
            ),
            "verdict": _verdict(h1_pass, fallback=True),
        },
        {
            "hypothesis_id": "H2_definition_robust_win_fraction",
            "statement": f"{cfg.treatment} wins in at least {cfg.min_win_fraction:.0%} of definition settings against {cfg.control}.",
            "test": "definition-wise delta sign",
            "metric_value": float(win_fraction) if np.isfinite(win_fraction) else np.nan,
            "threshold": f">= {cfg.min_win_fraction:.2f}",
            "evidence": f"positive_delta_fraction={win_fraction:.6g}; n_definitions={n_defs}",
            "verdict": _verdict(h2_pass, fallback=True),
        },
        {
            "hypothesis_id": "H3_worst_case_non_negative_lift",
            "statement": f"{cfg.treatment} has non-negative worst-case lift over {cfg.control} across definitions.",
            "test": "minimum definition-wise delta",
            "metric_value": float(worst_case_delta) if np.isfinite(worst_case_delta) else np.nan,
            "threshold": f">= {cfg.min_worst_case_delta:.6g}",
            "evidence": f"worst_case_delta={worst_case_delta:.6g}; n_definitions={n_defs}",
            "verdict": _verdict(h3_pass, fallback=True),
        },
        {
            "hypothesis_id": "H4_rank_stability",
            "statement": f"{cfg.treatment} rank remains stable across definition settings.",
            "test": "rank std + top-win frequency",
            "metric_value": float(treatment_rank_std) if np.isfinite(treatment_rank_std) else np.nan,
            "threshold": (
                f"std_rank <= {cfg.max_rank_std:.6g} and "
                f"top_win_fraction >= {cfg.min_top_win_fraction:.6g}"
            ),
            "evidence": (
                f"std_rank={treatment_rank_std:.6g}; "
                f"top_win_fraction={treatment_top_win_fraction:.6g}"
            ),
            "verdict": _verdict(h4_pass, fallback=True),
        },
    ]
    hypothesis_df = pd.DataFrame(rows)
    artifacts = {
        "delta_table": deltas,
        "bootstrap": boot,
        "permutation": perm,
        "rank_summary": rank_summary,
        "rank_pairwise_corr": rank_pairs,
        "winner_frequency": winner_df,
    }
    return hypothesis_df, artifacts
