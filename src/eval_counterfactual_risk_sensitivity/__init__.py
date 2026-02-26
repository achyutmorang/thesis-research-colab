from .analysis import (
    aggregate_factor_importance,
    build_sensitivity_atlas,
    compute_high_risk_event_flag,
    factor_response_profile,
    method_factor_importance,
    scenario_factor_response,
    top_sensitive_scenarios,
)
from .hypotheses import (
    CounterfactualHypothesisConfig,
    evaluate_counterfactual_hypotheses,
)
from .io import (
    CounterfactualRunData,
    discover_and_load_trace,
    load_intervention_tables,
    trace_to_intervention_table,
)
from .stats import (
    bootstrap_factor_importance_ci,
    permutation_test_factor_nonzero,
)

try:
    from .plots import (
        plot_factor_importance_ci,
        plot_factor_slope_distribution,
        plot_method_factor_heatmap,
        plot_response_profile,
        save_figure,
    )
except Exception:  # pragma: no cover - optional plotting dependency
    plot_factor_importance_ci = None
    plot_factor_slope_distribution = None
    plot_method_factor_heatmap = None
    plot_response_profile = None
    save_figure = None

__all__ = [
    "CounterfactualRunData",
    "CounterfactualHypothesisConfig",
    "aggregate_factor_importance",
    "bootstrap_factor_importance_ci",
    "build_sensitivity_atlas",
    "compute_high_risk_event_flag",
    "discover_and_load_trace",
    "evaluate_counterfactual_hypotheses",
    "factor_response_profile",
    "load_intervention_tables",
    "method_factor_importance",
    "permutation_test_factor_nonzero",
    "plot_factor_importance_ci",
    "plot_factor_slope_distribution",
    "plot_method_factor_heatmap",
    "plot_response_profile",
    "save_figure",
    "scenario_factor_response",
    "top_sensitive_scenarios",
    "trace_to_intervention_table",
]
