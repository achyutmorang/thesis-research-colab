from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.analysis import (
    budget_normalized_efficiency,
    conditional_lift_by_risk_bins,
    discovery_auc,
    discovery_curve_from_trace,
    method_summary,
)


def _mock_results() -> pd.DataFrame:
    rows = []
    methods = ["random", "risk_only", "surprise_only", "joint"]
    for sid in range(6):
        base_risk = 1.0 + 0.1 * sid
        for m in methods:
            bs = int(m == "joint" and sid % 2 == 0)
            if m == "risk_only" and sid % 2 == 0:
                bs = 0
            rows.append(
                {
                    "scenario_id": sid,
                    "method": m,
                    "risk_sks": base_risk + (0.2 if m == "risk_only" else 0.0),
                    "surprise_pd": 0.1 + 0.05 * sid,
                    "failure_proxy": float(sid % 2 == 0),
                    "q1_hit": 0,
                    "q4_hit": 0,
                    "blind_spot_proxy_hit": bs,
                    "budget_units_used": 15,
                    "objective_gain": 0.1,
                }
            )
    return pd.DataFrame(rows)


def _mock_trace() -> pd.DataFrame:
    rows = []
    for sid in range(4):
        for method in ["risk_only", "joint"]:
            for k in range(5):
                hit = int(method == "joint" and k >= 2 and sid in [0, 2])
                rows.append(
                    {
                        "scenario_id": sid,
                        "method": method,
                        "eval_index": k,
                        "risk_sks": 1.0 + 0.1 * sid,
                        "surprise_pd": 0.9 if hit else 0.2,
                        "failure_proxy": float(hit),
                    }
                )
    return pd.DataFrame(rows)


def test_method_summary_and_efficiency_not_empty():
    df = _mock_results()
    summary = method_summary(df)
    eff = budget_normalized_efficiency(df)
    assert not summary.empty
    assert not eff.empty
    assert "blind_spot_rate" in summary.columns
    assert np.isfinite(eff["hits_per_100_budget"]).all()


def test_conditional_lift_returns_overall():
    df = _mock_results()
    bins_df, overall_df = conditional_lift_by_risk_bins(
        df,
        treatment="joint",
        control="risk_only",
        n_bins=3,
        min_bin_count=1,
        bootstrap_samples=200,
    )
    assert overall_df.shape[0] == 1
    assert overall_df.loc[0, "mean_delta"] >= 0.0
    assert "ci_low" in overall_df.columns
    assert bins_df.shape[0] >= 1


def test_discovery_curve_monotonic_and_auc():
    trace_df = _mock_trace()
    thresholds = {
        "surprise_high_threshold": 0.5,
        "risk_high_threshold": 100.0,
        "risk_low_threshold": -100.0,
    }
    curve_df = discovery_curve_from_trace(trace_df, thresholds=thresholds)
    assert not curve_df.empty

    for _, g in curve_df.groupby("method"):
        y = g.sort_values("eval_index")["discovery_rate"].to_numpy()
        assert np.all(np.diff(y) >= -1e-10)

    auc_df = discovery_auc(curve_df)
    assert not auc_df.empty
    assert np.isfinite(auc_df["normalized_auc"]).all()
