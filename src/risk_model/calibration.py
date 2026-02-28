from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def apply(self, logits: np.ndarray) -> np.ndarray:
        t = float(max(1e-3, self.temperature))
        return np.asarray(logits, dtype=float) / t


def _bce_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=float)
    loss = np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    return float(np.mean(loss))


def fit_temperature_scaler(logits: np.ndarray, targets: np.ndarray) -> TemperatureScaler:
    logits = np.asarray(logits, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    if logits.size == 0:
        return TemperatureScaler(1.0)

    base_loss = _bce_from_logits(logits, targets)

    def objective(temp: float) -> float:
        return _bce_from_logits(logits / max(1e-3, float(temp)), targets)

    res = minimize_scalar(objective, bounds=(0.05, 20.0), method='bounded')
    if not bool(res.success):
        return TemperatureScaler(1.0)
    best_temp = float(res.x)
    best_loss = float(res.fun)
    if best_loss > base_loss + 1e-10:
        return TemperatureScaler(1.0)
    return TemperatureScaler(best_temp)


def fit_temperature_scalers(
    logits: np.ndarray,
    targets: np.ndarray,
    label_names: Sequence[str],
) -> Dict[str, TemperatureScaler]:
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=float)
    out: Dict[str, TemperatureScaler] = {}
    for idx, label in enumerate(label_names):
        out[str(label)] = fit_temperature_scaler(logits[:, idx], targets[:, idx])
    return out


def apply_temperature_scalers(
    logits: np.ndarray,
    scalers: Mapping[str, TemperatureScaler],
    label_names: Sequence[str],
) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    out = np.zeros_like(logits, dtype=float)
    for idx, label in enumerate(label_names):
        scaler = scalers.get(str(label), TemperatureScaler(1.0))
        out[:, idx] = scaler.apply(logits[:, idx])
    return out


def fit_binary_conformal_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.10,
) -> float:
    probs = np.asarray(probs, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    pos = probs[labels > 0.5]
    if pos.size == 0:
        return 0.5
    nonconformity = 1.0 - np.clip(pos, 0.0, 1.0)
    q = float(np.quantile(nonconformity, min(1.0, max(0.0, 1.0 - float(alpha))), method='higher'))
    return float(np.clip(1.0 - q, 0.0, 1.0))
