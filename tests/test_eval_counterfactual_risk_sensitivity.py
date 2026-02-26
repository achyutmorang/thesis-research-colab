from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval_counterfactual_risk_sensitivity import (
    CounterfactualHypothesisConfig,
    aggregate_factor_importance,
    bootstrap_factor_importance_ci,
    build_sensitivity_atlas,
    evaluate_counterfactual_hypotheses,
    factor_response_profile,
    method_factor_importance,
    permutation_test_factor_nonzero,
    top_sensitive_scenarios,
)


def _mock_trace() -> pd.DataFrame:
    rows = []
    for sid in range(6):
        for k, dx in enumerate(np.linspace(-1.0, 1.0, 9)):
            dl2 = abs(dx)
            risk = 0.2 + 0.6 * dl2 + 0.02 * sid
            rows.append(
                {
                    "scenario_id": sid,
                    "method": "joint",
                    "eval_index": k,
                    "delta_x": dx,
                    "delta_y": 0.2 * dx,
                    "delta_l2": dl2,
                    "step_scale": 0.1 + 0.03 * k,
                    "risk_sks": risk,
                    "failure_proxy": int(dl2 >= 0.8),
                }
            )
    return pd.DataFrame(rows)



def test_build_sensitivity_atlas_and_importance():
    trace_df = _mock_trace()
    atlas_df, meta = build_sensitivity_atlas(
        trace_df,
        factor_columns=("delta_x", "delta_l2", "step_scale"),
        min_unique_values=4,
    )

    assert not atlas_df.empty
    assert "risk_threshold" in meta
    imp_df = aggregate_factor_importance(atlas_df)
    assert not imp_df.empty
    assert {"factor_name", "mean_abs_slope", "n_groups"}.issubset(imp_df.columns)

    # delta_l2 should dominate because mock risk is constructed from delta_l2.
    assert str(imp_df.iloc[0]["factor_name"]) == "delta_l2"



def test_factor_stats_helpers():
    trace_df = _mock_trace()
    atlas_df, _ = build_sensitivity_atlas(
        trace_df,
        factor_columns=("delta_x", "delta_l2", "step_scale"),
        min_unique_values=4,
    )

    top_df = top_sensitive_scenarios(atlas_df, top_n=10)
    assert len(top_df) <= 10
    assert not top_df.empty

    ci_df = bootstrap_factor_importance_ci(atlas_df, n_boot=200)
    assert not ci_df.empty
    assert np.isfinite(ci_df["mean_abs_slope"]).all()

    p = permutation_test_factor_nonzero(atlas_df, factor_name="delta_l2", n_perm=500)
    assert p["n_groups"] > 0
    assert 0.0 <= p["p_value_two_sided"] <= 1.0


def test_counterfactual_hypotheses_and_profiles():
    trace_df = _mock_trace()
    atlas_df, _ = build_sensitivity_atlas(
        trace_df,
        factor_columns=("delta_x", "delta_l2", "step_scale"),
        min_unique_values=4,
    )
    imp_df = aggregate_factor_importance(atlas_df)
    perm_rows = [
        permutation_test_factor_nonzero(atlas_df, factor_name=f, n_perm=300)
        for f in imp_df["factor_name"].tolist()
    ]
    perm_df = pd.DataFrame(perm_rows)

    h_df, artifacts = evaluate_counterfactual_hypotheses(
        atlas_df=atlas_df,
        importance_df=imp_df,
        permutation_df=perm_df,
        config=CounterfactualHypothesisConfig(min_significant_factors=0),
    )
    assert not h_df.empty
    assert {"hypothesis_id", "verdict"}.issubset(h_df.columns)
    assert "method_importance" in artifacts

    m_imp = method_factor_importance(atlas_df)
    assert not m_imp.empty
    assert {"method", "factor_name", "mean_abs_slope"}.issubset(m_imp.columns)

    trace_work = trace_df.copy()
    trace_work["is_high_risk_event"] = (trace_work["failure_proxy"] > 0).astype(int)
    prof_df = factor_response_profile(trace_work, factor_col="delta_l2", outcome_col="is_high_risk_event")
    assert not prof_df.empty
