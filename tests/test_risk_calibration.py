from __future__ import annotations

import numpy as np

from src.risk_model.calibration import (
    apply_temperature_scalers,
    fit_binary_conformal_threshold,
    fit_temperature_scaler,
    fit_temperature_scalers,
)


def _bce_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=float)
    loss = np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    return float(np.mean(loss))


def test_temperature_scaler_not_worse_than_base_nll() -> None:
    rng = np.random.default_rng(11)
    logits = rng.normal(loc=0.0, scale=2.0, size=500)
    probs = 1.0 / (1.0 + np.exp(-0.6 * logits))
    labels = (rng.random(500) < probs).astype(float)

    scaler = fit_temperature_scaler(logits, labels)
    base_nll = _bce_from_logits(logits, labels)
    scaled_nll = _bce_from_logits(logits / max(1e-6, scaler.temperature), labels)

    assert scaled_nll <= base_nll + 1e-8


def test_apply_temperature_scalers_preserves_per_label_ordering() -> None:
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(100, 2))
    labels = (rng.random((100, 2)) < 0.5).astype(float)
    scalers = fit_temperature_scalers(logits, labels, ['a', 'b'])
    scaled = apply_temperature_scalers(logits, scalers, ['a', 'b'])

    for idx in range(2):
        order_before = np.argsort(logits[:, idx])
        order_after = np.argsort(scaled[:, idx])
        assert np.array_equal(order_before, order_after)


def test_binary_conformal_threshold_hits_target_coverage_on_positive_set() -> None:
    rng = np.random.default_rng(5)
    labels = (rng.random(400) < 0.3).astype(float)
    probs = np.where(labels > 0.5, rng.uniform(0.4, 1.0, size=400), rng.uniform(0.0, 0.6, size=400))
    alpha = 0.10
    thr = fit_binary_conformal_threshold(probs, labels, alpha=alpha)

    pos = probs[labels > 0.5]
    coverage = float(np.mean(pos >= thr)) if pos.size > 0 else 1.0
    assert coverage >= 1.0 - alpha - 0.05
