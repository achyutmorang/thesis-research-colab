from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from .calibration import TemperatureScaler
from .model import NumpyEnsembleMLP
from .train import RiskTrainingBundle


def _mkdir(path: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_frame(path: str, df: pd.DataFrame) -> str:
    p = _mkdir(path)
    suffix = p.suffix.lower()
    if suffix == '.parquet':
        try:
            df.to_parquet(p, index=False)
            return str(p)
        except Exception:
            csv_path = p.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            return str(csv_path)
    df.to_csv(p, index=False)
    return str(p)


def _write_uq_artifact_schema_manifest(run_prefix: str, artifact_names: list[str]) -> str:
    p = _mkdir(f'{run_prefix}_uq_artifact_schema.json')
    p.write_text(
        json.dumps(
            {
                'schema_version': '1.0.0',
                'artifacts': sorted(list(artifact_names)),
            },
            indent=2,
        )
    )
    return str(p)


def save_risk_dataset_artifacts(run_prefix: str, dataset_df: pd.DataFrame) -> Dict[str, str]:
    paths = {
        'risk_dataset': _write_frame(f'{run_prefix}_risk_dataset.parquet', dataset_df),
        'risk_dataset_summary': _write_frame(
            f'{run_prefix}_risk_dataset_summary.csv',
            dataset_df.groupby('eval_split', as_index=False).agg(n_rows=('scenario_id', 'size')) if not dataset_df.empty else pd.DataFrame(),
        ),
    }
    return paths


def save_risk_artifacts(
    run_prefix: str,
    bundle: RiskTrainingBundle,
    scalers: Mapping[str, Any],
    conformal_thresholds: Mapping[str, float],
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    state = bundle.model.state_dict()
    for idx, params in enumerate(state['params']):
        p = _mkdir(f'{run_prefix}_risk_ensemble_member_{idx}.npz')
        np.savez_compressed(p, **params)
        out[f'member_{idx}'] = str(p)

    meta = {
        'feature_columns': list(bundle.feature_columns),
        'label_columns': list(bundle.label_columns),
        'feature_mean': bundle.feature_mean.tolist(),
        'feature_std': bundle.feature_std.tolist(),
        'model_config': state['config'],
    }
    meta_path = _mkdir(f'{run_prefix}_risk_model_metadata.json')
    meta_path.write_text(json.dumps(meta, indent=2))
    out['metadata'] = str(meta_path)

    scaler_payload = {k: float(v.temperature) for k, v in scalers.items()}
    scalers_path = _mkdir(f'{run_prefix}_risk_temperature_scalers.json')
    scalers_path.write_text(json.dumps(scaler_payload, indent=2))
    out['temperature_scalers'] = str(scalers_path)

    conf_path = _mkdir(f'{run_prefix}_risk_conformal_thresholds.json')
    conf_path.write_text(json.dumps({k: float(v) for k, v in conformal_thresholds.items()}, indent=2))
    out['conformal_thresholds'] = str(conf_path)

    summary_path = _write_frame(f'{run_prefix}_risk_train_summary.csv', bundle.train_summary)
    out['train_summary'] = summary_path
    summary_json_path = _mkdir(f'{run_prefix}_risk_train_summary.json')
    summary_json_path.write_text(bundle.train_summary.to_json(orient='records', indent=2))
    out['train_summary_json'] = str(summary_json_path)
    val_pred_path = _write_frame(f'{run_prefix}_risk_validation_predictions.parquet', bundle.validation_predictions)
    out['validation_predictions'] = val_pred_path
    return out


def load_risk_artifacts(run_prefix: str) -> Dict[str, Any]:
    meta_path = Path(f'{run_prefix}_risk_model_metadata.json')
    if not meta_path.exists():
        raise FileNotFoundError(str(meta_path))
    meta = json.loads(meta_path.read_text())
    member_files = sorted(Path(meta_path.parent).glob(Path(run_prefix).name + '_risk_ensemble_member_*.npz'))
    state = {'config': meta['model_config'], 'params': []}
    for fp in member_files:
        payload = np.load(fp)
        state['params'].append({k: np.asarray(payload[k]) for k in payload.files})
    model = NumpyEnsembleMLP.from_state_dict(state)
    scalers_path = Path(f'{run_prefix}_risk_temperature_scalers.json')
    conformal_path = Path(f'{run_prefix}_risk_conformal_thresholds.json')
    raw_scalers = json.loads(scalers_path.read_text()) if scalers_path.exists() else {}
    scalers = {k: TemperatureScaler(float(v)) for k, v in raw_scalers.items()}
    conformal = json.loads(conformal_path.read_text()) if conformal_path.exists() else {}
    return {
        'model': model,
        'feature_columns': list(meta['feature_columns']),
        'label_columns': list(meta['label_columns']),
        'feature_mean': np.asarray(meta['feature_mean'], dtype=float),
        'feature_std': np.asarray(meta['feature_std'], dtype=float),
        'temperature_scalers': scalers,
        'conformal_thresholds': conformal,
    }


def save_risk_evaluation_artifacts(run_prefix: str, frames: Mapping[str, pd.DataFrame]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name, df in frames.items():
        ext = '.csv'
        if 'predictions' in name or 'dataset' in name:
            ext = '.parquet'
        out[name] = _write_frame(f'{run_prefix}_{name}{ext}', df)
    out['artifact_schema'] = _write_uq_artifact_schema_manifest(run_prefix, sorted(out.keys()))
    return out
