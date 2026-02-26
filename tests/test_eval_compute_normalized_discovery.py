from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval_compute_normalized_discovery import (
    DiscoveryHypothesisConfig,
    NaturalnessConfig,
    cluster_coverage_diversity,
    combine_repo_inspired_method_table,
    evaluate_discovery_grid,
    evaluate_discovery_hypotheses,
    paired_bootstrap_delta,
    paired_permutation_test,
    plausibility_filtered_summary,
    rank_stability_table,
    realism_gap_summary,
    rulebook_lexicographic_ranking,
    standard_blindspot_definitions,
    standard_risk_definitions,
)


def _mock_results() -> pd.DataFrame:
    rows = []
    methods = ["random", "risk_only", "surprise_only", "joint"]
    for sid in range(10):
        cluster = "common" if sid < 6 else f"rare_{sid}"
        for method in methods:
            risk = 0.2 + 0.12 * sid + (0.08 if method == "risk_only" else 0.0)
            failure = int(method == "joint" and sid >= 6)
            if method == "risk_only" and sid >= 8:
                failure = 1
            rows.append(
                {
                    "scenario_id": sid,
                    "method": method,
                    "scenario_cluster": cluster,
                    "risk_sks": risk,
                    "collision": int(sid == 9 and method in ["risk_only", "joint"]),
                    "min_ttc": 1.2 if sid >= 8 else 3.5,
                    "failure_proxy": failure,
                    "budget_units_used": 15,
                    "rollout_feasible": 1,
                    "feasibility_violation": 0,
                    "delta_l2": 0.1 + 0.04 * sid,
                    "dist_fallback_ratio": 0.05,
                }
            )
    return pd.DataFrame(rows)


def _mock_trace() -> pd.DataFrame:
    rows = []
    methods = ["random", "risk_only", "surprise_only", "joint"]
    for sid in range(10):
        for method in methods:
            for k in range(5):
                event = int(method == "joint" and sid >= 6 and k >= 2)
                if method == "risk_only" and sid >= 8 and k >= 4:
                    event = 1
                rows.append(
                    {
                        "scenario_id": sid,
                        "method": method,
                        "eval_index": k,
                        "risk_sks": 0.2 + 0.1 * sid,
                        "failure_proxy": event,
                    }
                )
    return pd.DataFrame(rows)


def test_evaluate_discovery_grid_outputs_non_empty_tables():
    results_df = _mock_results()
    trace_df = _mock_trace()

    metrics_df, labeled = evaluate_discovery_grid(
        results_df=results_df,
        trace_df=trace_df,
        risk_definitions=standard_risk_definitions(),
        blindspot_definitions=standard_blindspot_definitions(),
        k_values=(2, 4),
    )

    assert not metrics_df.empty
    assert len(labeled) == len(standard_risk_definitions()) * len(standard_blindspot_definitions())
    assert {"definition_key", "risk_definition", "blindspot_definition", "method"}.issubset(metrics_df.columns)
    assert (metrics_df["total_compute"] > 0).all()
    assert np.isfinite(metrics_df["discovery_efficiency"]).all()



def test_rank_and_delta_statistics_are_computable():
    results_df = _mock_results()
    trace_df = _mock_trace()
    metrics_df, _ = evaluate_discovery_grid(results_df=results_df, trace_df=trace_df)

    rank_summary, pair_df = rank_stability_table(metrics_df)
    assert not rank_summary.empty
    assert {"method", "mean_rank", "mean_score"}.issubset(rank_summary.columns)
    assert pair_df.shape[0] >= 1

    boot = paired_bootstrap_delta(metrics_df, treatment="joint", control="random")
    assert boot["n_pairs"] > 0
    assert np.isfinite(boot["mean_delta"])

    perm = paired_permutation_test(metrics_df, treatment="joint", control="random", n_perm=500)
    assert perm["n_pairs"] > 0
    assert 0.0 <= perm["p_value_two_sided"] <= 1.0


def test_hypothesis_table_is_well_formed():
    results_df = _mock_results()
    trace_df = _mock_trace()
    metrics_df, _ = evaluate_discovery_grid(results_df=results_df, trace_df=trace_df)

    h_df, artifacts = evaluate_discovery_hypotheses(
        metrics_df,
        config=DiscoveryHypothesisConfig(
            treatment="joint",
            control="random",
            min_win_fraction=0.3,
            min_top_win_fraction=0.25,
            max_rank_std=2.0,
        ),
    )
    assert not h_df.empty
    assert {"hypothesis_id", "statement", "verdict", "evidence"}.issubset(h_df.columns)
    assert "winner_frequency" in artifacts


def test_repo_inspired_summary_and_rulebook_rank():
    results_df = _mock_results()
    trace_df = _mock_trace()
    metrics_df, labeled_map = evaluate_discovery_grid(results_df=results_df, trace_df=trace_df, k_values=(2,))
    labeled_df = next(iter(labeled_map.values()))

    plausible_df, natural_mask, meta = plausibility_filtered_summary(
        labeled_df,
        naturalness_config=NaturalnessConfig(max_delta_l2_quantile=0.9),
    )
    assert not plausible_df.empty
    assert len(natural_mask) == len(labeled_df)
    assert "delta_l2_threshold" in meta

    cov_df = cluster_coverage_diversity(labeled_df)
    assert not cov_df.empty
    assert {"coverage_ratio", "normalized_entropy"}.issubset(cov_df.columns)

    real_df = realism_gap_summary(labeled_df, baseline_method="random", compare_on="all")
    assert not real_df.empty
    assert "wd_mean" in real_df.columns

    base_df = (
        metrics_df.groupby("method", as_index=False)
        .agg(discovery_efficiency=("discovery_efficiency", "mean"))
    )
    combined_df = combine_repo_inspired_method_table(base_df, plausible_df, cov_df, real_df)
    assert not combined_df.empty

    rank_df = rulebook_lexicographic_ranking(
        combined_df.fillna(0.0),
        priorities=["discovery_efficiency", "plausible_efficiency", "coverage_ratio", "wd_mean"],
        maximize={
            "discovery_efficiency": True,
            "plausible_efficiency": True,
            "coverage_ratio": True,
            "wd_mean": False,
        },
    )
    assert not rank_df.empty
    assert int(rank_df["rulebook_rank"].min()) == 1
