from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .analysis import method_factor_importance


@dataclass(frozen=True)
class CounterfactualHypothesisConfig:
    alpha: float = 0.05
    min_mean_abs_slope: float = 0.02
    min_monotonic_fraction: float = 0.50
    min_top_factor_method_consistency: float = 0.60
    min_significant_factors: int = 1


def evaluate_counterfactual_hypotheses(
    atlas_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    config: Optional[CounterfactualHypothesisConfig] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = config or CounterfactualHypothesisConfig()
    if atlas_df.empty or importance_df.empty:
        rows = [
            {
                "hypothesis_id": "C0_data_adequacy",
                "statement": "Counterfactual atlas has enough valid rows for inference.",
                "test": "non-empty atlas + importance table",
                "metric_value": 0.0,
                "threshold": "> 0 rows",
                "evidence": "atlas or importance table is empty",
                "verdict": "FAIL",
            }
        ]
        return pd.DataFrame(rows), {"method_importance": pd.DataFrame()}

    perm_df = permutation_df.copy()
    if not perm_df.empty and "p_value_two_sided" in perm_df.columns:
        perm_df["is_significant"] = pd.to_numeric(perm_df["p_value_two_sided"], errors="coerce") < cfg.alpha
    else:
        perm_df["is_significant"] = False

    method_imp = method_factor_importance(atlas_df)

    n_sig = int(perm_df["is_significant"].sum()) if "is_significant" in perm_df.columns else 0
    top_factor = str(importance_df.iloc[0]["factor_name"]) if not importance_df.empty else ""
    top_mean_abs_slope = float(importance_df.iloc[0]["mean_abs_slope"]) if not importance_df.empty else np.nan
    top_monotonic_fraction = (
        float(importance_df.iloc[0]["monotonic_fraction"])
        if ("monotonic_fraction" in importance_df.columns and not importance_df.empty)
        else np.nan
    )

    top_factor_consistency = np.nan
    if not method_imp.empty:
        top_per_method = (
            method_imp.sort_values(["method", "mean_abs_slope"], ascending=[True, False])
            .groupby("method", as_index=False)
            .first()
        )
        if not top_per_method.empty:
            top_factor_consistency = float((top_per_method["factor_name"] == top_factor).mean())

    c1_pass = n_sig >= int(cfg.min_significant_factors)
    c2_pass = np.isfinite(top_mean_abs_slope) and top_mean_abs_slope >= cfg.min_mean_abs_slope
    c3_pass = np.isfinite(top_monotonic_fraction) and top_monotonic_fraction >= cfg.min_monotonic_fraction
    c4_pass = (
        np.isfinite(top_factor_consistency)
        and float(top_factor_consistency) >= float(cfg.min_top_factor_method_consistency)
    )

    rows = [
        {
            "hypothesis_id": "C1_nonzero_counterfactual_effects",
            "statement": "At least one factor has statistically non-zero mean slope.",
            "test": "per-factor sign permutation",
            "metric_value": float(n_sig),
            "threshold": f">= {int(cfg.min_significant_factors)} significant factors (p < {cfg.alpha})",
            "evidence": f"significant_factors={n_sig}",
            "verdict": "PASS" if c1_pass else "INCONCLUSIVE",
        },
        {
            "hypothesis_id": "C2_strong_primary_factor",
            "statement": "Top factor has practically meaningful effect size.",
            "test": "aggregate mean absolute slope",
            "metric_value": float(top_mean_abs_slope) if np.isfinite(top_mean_abs_slope) else np.nan,
            "threshold": f">= {cfg.min_mean_abs_slope}",
            "evidence": f"top_factor={top_factor}; mean_abs_slope={top_mean_abs_slope:.6g}",
            "verdict": "PASS" if c2_pass else "INCONCLUSIVE",
        },
        {
            "hypothesis_id": "C3_monotonic_response_consistency",
            "statement": "Top factor has mostly monotonic response across groups.",
            "test": "fraction(|rank corr| >= 0.5)",
            "metric_value": float(top_monotonic_fraction) if np.isfinite(top_monotonic_fraction) else np.nan,
            "threshold": f">= {cfg.min_monotonic_fraction}",
            "evidence": (
                f"top_factor={top_factor}; monotonic_fraction={top_monotonic_fraction:.6g}"
            ),
            "verdict": "PASS" if c3_pass else "INCONCLUSIVE",
        },
        {
            "hypothesis_id": "C4_method_robust_top_factor",
            "statement": "Top factor remains top across methods.",
            "test": "top-factor agreement rate across methods",
            "metric_value": float(top_factor_consistency) if np.isfinite(top_factor_consistency) else np.nan,
            "threshold": f">= {cfg.min_top_factor_method_consistency}",
            "evidence": (
                f"top_factor={top_factor}; method_consistency={top_factor_consistency:.6g}"
            ),
            "verdict": "PASS" if c4_pass else "INCONCLUSIVE",
        },
    ]
    hypothesis_df = pd.DataFrame(rows)
    artifacts = {
        "method_importance": method_imp,
        "permutation_with_significance": perm_df,
    }
    return hypothesis_df, artifacts
