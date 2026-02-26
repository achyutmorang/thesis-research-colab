"""Backward-compatibility shim for relocated notebook workflow helpers.

New canonical location: `src.workflows.closedloop_flow`.
"""

from importlib import import_module

__all__ = [
    "SimulationContextBundle",
    "analyze_signal_if_available",
    "build_full_simulation_context",
    "initialize_run_context",
    "report_export_bundle",
    "report_main_loop_bundle",
    "report_preflight_bundle",
    "report_quick_probe_bundle",
    "report_run_context",
    "report_signal_bundle",
    "report_surprise_gate_bundle",
    "resolve_main_loop_policy",
    "run_main_loop_with_policy",
    "run_preflight_bundle",
    "run_quick_probe_with_auto_escalation",
    "run_surprise_gate_with_policy",
    "summarize_and_export_if_available",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("src.workflows.closedloop_flow")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
