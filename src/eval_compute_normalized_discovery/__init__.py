from .definitions import (
    BlindspotDefinition,
    RiskDefinition,
    derive_blindspot_flag,
    derive_high_risk_flag,
    standard_blindspot_definitions,
    standard_risk_definitions,
)
from .io import (
    DiscoveryRunData,
    discover_and_load_run,
    load_results_and_trace_csv,
)
from .hypotheses import (
    DiscoveryHypothesisConfig,
    evaluate_discovery_hypotheses,
)
from .metrics import (
    best_method_per_definition,
    evaluate_discovery_grid,
    method_score_table,
    time_to_k_by_method,
)
from .repo_inspired import (
    NaturalnessConfig,
    cluster_coverage_diversity,
    combine_repo_inspired_method_table,
    derive_naturalness_mask,
    plausibility_filtered_summary,
    realism_gap_summary,
    rulebook_lexicographic_ranking,
)
from .stats import (
    paired_bootstrap_delta,
    paired_permutation_test,
    rank_stability_table,
)

try:
    from .plots import (
        plot_definition_heatmap,
        plot_method_score_distribution,
        plot_rank_heatmap,
        plot_time_to_k,
        save_figure,
    )
except Exception:  # pragma: no cover - optional plotting dependency
    plot_definition_heatmap = None
    plot_method_score_distribution = None
    plot_rank_heatmap = None
    plot_time_to_k = None
    save_figure = None

__all__ = [
    "BlindspotDefinition",
    "DiscoveryRunData",
    "DiscoveryHypothesisConfig",
    "NaturalnessConfig",
    "RiskDefinition",
    "derive_blindspot_flag",
    "derive_high_risk_flag",
    "cluster_coverage_diversity",
    "combine_repo_inspired_method_table",
    "discover_and_load_run",
    "best_method_per_definition",
    "derive_naturalness_mask",
    "evaluate_discovery_hypotheses",
    "evaluate_discovery_grid",
    "load_results_and_trace_csv",
    "method_score_table",
    "paired_bootstrap_delta",
    "paired_permutation_test",
    "plausibility_filtered_summary",
    "plot_definition_heatmap",
    "plot_method_score_distribution",
    "plot_rank_heatmap",
    "plot_time_to_k",
    "realism_gap_summary",
    "rank_stability_table",
    "rulebook_lexicographic_ranking",
    "save_figure",
    "standard_blindspot_definitions",
    "standard_risk_definitions",
    "time_to_k_by_method",
]
