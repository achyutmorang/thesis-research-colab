from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .calibration import TemperatureScaler, apply_temperature_scalers
from .model import NumpyEnsembleMLP


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def _standardize_features(df: pd.DataFrame, feature_columns: Sequence[str], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = df.loc[:, list(feature_columns)].astype(float).to_numpy(copy=True)
    return (x - mean[None, :]) / std[None, :]


def predict_raw_risk(
    model: NumpyEnsembleMLP,
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    label_columns: Sequence[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> pd.DataFrame:
    x = _standardize_features(df, feature_columns, feature_mean, feature_std)
    pred = model.predict_with_uncertainty(x)
    out = df.copy()
    for idx, label in enumerate(label_columns):
        out[f'risk_raw_{label}'] = pred['mean_probs'][:, idx]
        out[f'risk_logit_{label}'] = pred['mean_logits'][:, idx]
        out[f'risk_epistemic_{label}'] = pred['epistemic_var'][:, idx]
    return out


def predict_calibrated_risk(
    model: NumpyEnsembleMLP,
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    label_columns: Sequence[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    scalers: Mapping[str, TemperatureScaler],
    conformal_thresholds: Mapping[str, float],
) -> pd.DataFrame:
    raw = predict_raw_risk(model, df, feature_columns, label_columns, feature_mean, feature_std)
    logits = np.stack([raw[f'risk_logit_{label}'].to_numpy(dtype=float) for label in label_columns], axis=1)
    cal_logits = apply_temperature_scalers(logits, scalers, label_columns)
    cal_probs = _sigmoid(cal_logits)
    for idx, label in enumerate(label_columns):
        raw[f'risk_cal_{label}'] = cal_probs[:, idx]
        thr = float(conformal_thresholds.get(label, 1.1))
        raw[f'risk_conformal_flag_{label}'] = (cal_probs[:, idx] >= thr).astype(int)
    return raw
