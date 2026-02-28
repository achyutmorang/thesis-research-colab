from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np

from .metrics import compute_risk_metrics, risk_kwargs_from_cfg


def _offroad_proxy(valid: np.ndarray) -> float:
    vv = np.asarray(valid, dtype=bool)
    if vv.ndim != 2 or vv.shape[0] == 0:
        return 0.0
    ego_valid = vv[0]
    return float(np.any(~ego_valid))


def label_candidate_rollout_events(
    xy: np.ndarray,
    valid: np.ndarray,
    cfg: Any,
    horizons: Iterable[int] = (5, 10, 15),
) -> Dict[str, int]:
    xy = np.asarray(xy, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if xy.ndim != 3 or valid.ndim != 2:
        raise ValueError('Expected xy shape (N,T,2) and valid shape (N,T).')

    labels: Dict[str, int] = {}
    for h in tuple(int(x) for x in horizons):
        hh = max(1, min(int(h), xy.shape[1], valid.shape[1]))
        risk = compute_risk_metrics(xy[:, :hh, :], valid[:, :hh], **risk_kwargs_from_cfg(cfg))
        labels[f'collision_h{h}'] = int(float(risk['collision']) > 0.0)
        labels[f'offroad_h{h}'] = int(_offroad_proxy(valid[:, :hh]) > 0.0)
        labels[f'failure_proxy_h{h}'] = int(float(risk['failure_extended_proxy']) > 0.0)
    return labels
