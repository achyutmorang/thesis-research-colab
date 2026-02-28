from __future__ import annotations

from importlib import import_module
from typing import Dict

__all__ = [
    'TemperatureScaler',
    'NumpyEnsembleMLP',
    'NumpyEnsembleMLPConfig',
    'RiskTrainingBundle',
    'add_eval_splits',
    'apply_temperature_scalers',
    'build_risk_dataset',
    'deterministic_scenario_split',
    'extract_candidate_risk_features',
    'extract_dist_trace_features',
    'extract_rollout_summary_features',
    'fit_binary_conformal_threshold',
    'fit_temperature_scaler',
    'fit_temperature_scalers',
    'label_candidate_rollout_events',
    'load_risk_artifacts',
    'predict_calibrated_risk',
    'predict_raw_risk',
    'save_risk_artifacts',
    'save_risk_dataset_artifacts',
    'save_risk_evaluation_artifacts',
    'train_risk_ensemble',
]

_SYMBOL_MODULE: Dict[str, str] = {
    'load_risk_artifacts': 'src.risk_model.artifacts',
    'save_risk_artifacts': 'src.risk_model.artifacts',
    'save_risk_dataset_artifacts': 'src.risk_model.artifacts',
    'save_risk_evaluation_artifacts': 'src.risk_model.artifacts',
    'TemperatureScaler': 'src.risk_model.calibration',
    'apply_temperature_scalers': 'src.risk_model.calibration',
    'fit_binary_conformal_threshold': 'src.risk_model.calibration',
    'fit_temperature_scaler': 'src.risk_model.calibration',
    'fit_temperature_scalers': 'src.risk_model.calibration',
    'add_eval_splits': 'src.risk_model.dataset',
    'build_risk_dataset': 'src.risk_model.dataset',
    'deterministic_scenario_split': 'src.risk_model.dataset',
    'extract_candidate_risk_features': 'src.risk_model.features',
    'extract_dist_trace_features': 'src.risk_model.features',
    'extract_rollout_summary_features': 'src.risk_model.features',
    'predict_calibrated_risk': 'src.risk_model.inference',
    'predict_raw_risk': 'src.risk_model.inference',
    'label_candidate_rollout_events': 'src.risk_model.labels',
    'NumpyEnsembleMLP': 'src.risk_model.model',
    'NumpyEnsembleMLPConfig': 'src.risk_model.model',
    'RiskTrainingBundle': 'src.risk_model.train',
    'train_risk_ensemble': 'src.risk_model.train',
}


def __getattr__(name: str):
    module_name = _SYMBOL_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = import_module(module_name)
    value = getattr(mod, name)
    globals()[name] = value
    return value
