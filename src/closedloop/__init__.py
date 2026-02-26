from .config import (
    SearchConfig,
    ClosedLoopConfig,
    align_dataset_scale,
    auto_select_shard_id,
    build_run_artifact_paths,
    configure_persistent_run_prefix,
    initialize_configs,
    inspect_shard_progress,
    resolve_latentdriver_checkpoint,
    restore_artifacts_via_upload,
    shard_run_prefix,
)
from .calibration import diagnose_surprise_root_cause, run_surprise_quality_gate
from .core import (
    build_closedloop_runner_and_splits,
    make_waymax_data_iter,
    run_quick_surprise_probe,
    run_preflight_and_calibration,
    run_closed_loop,
)
from .resume_io import export_closedloop_artifacts, summarize_method_outputs
from .signal_analysis import (
    analyze_surprise_signal_usefulness,
    save_surprise_signal_usefulness_artifacts,
)
from .colab_runtime import (
    ensure_drive_ready,
    ensure_repo_checkout,
    prepare_repo_imports,
    run_cached_deterministic_setup,
)

_WORKFLOW_EXPORTS = [
    'SimulationContextBundle',
    'analyze_signal_if_available',
    'build_full_simulation_context',
    'initialize_run_context',
    'report_export_bundle',
    'report_main_loop_bundle',
    'report_preflight_bundle',
    'report_quick_probe_bundle',
    'report_run_context',
    'report_signal_bundle',
    'report_surprise_gate_bundle',
    'resolve_main_loop_policy',
    'run_main_loop_with_policy',
    'run_preflight_bundle',
    'run_quick_probe_with_auto_escalation',
    'run_surprise_gate_with_policy',
    'summarize_and_export_if_available',
]

__all__ = [
    'SearchConfig',
    'ClosedLoopConfig',
    'align_dataset_scale',
    'auto_select_shard_id',
    'build_run_artifact_paths',
    'build_closedloop_runner_and_splits',
    'configure_persistent_run_prefix',
    'diagnose_surprise_root_cause',
    'export_closedloop_artifacts',
    'initialize_configs',
    'inspect_shard_progress',
    'make_waymax_data_iter',
    'run_quick_surprise_probe',
    'resolve_latentdriver_checkpoint',
    'restore_artifacts_via_upload',
    'run_preflight_and_calibration',
    'run_surprise_quality_gate',
    'run_closed_loop',
    'save_surprise_signal_usefulness_artifacts',
    'shard_run_prefix',
    'summarize_method_outputs',
    'analyze_surprise_signal_usefulness',
    'ensure_drive_ready',
    'ensure_repo_checkout',
    'prepare_repo_imports',
    'run_cached_deterministic_setup',
] + _WORKFLOW_EXPORTS


def __getattr__(name: str):
    if name in _WORKFLOW_EXPORTS:
        # Avoid eager import of workflow symbols; eager import creates a circular
        # dependency when src.workflows imports src.closedloop internals.
        from src.workflows import closedloop_flow as _closedloop_flow

        value = getattr(_closedloop_flow, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
