from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass
class RiskControlSelection:
    selected_action: np.ndarray
    selected_row: Dict[str, Any]
    candidate_predictions: pd.DataFrame


DEFAULT_FAILURE_LABEL = 'failure_proxy_h15'


def _score_candidates(df: pd.DataFrame, cfg: Any, failure_label: str = DEFAULT_FAILURE_LABEL) -> pd.DataFrame:
    out = df.copy()
    fail_col = f'risk_cal_{failure_label}'
    epistemic_col = f'risk_epistemic_{failure_label}'
    horizon = int(max(1, getattr(cfg, 'risk_dataset_control_horizon_steps', 6)))
    progress_col = f'progress_h{horizon}'
    comfort_acc_col = f'max_abs_acc_h{horizon}'
    comfort_jerk_col = f'max_abs_jerk_h{horizon}'

    out[progress_col] = pd.to_numeric(out.get(progress_col, 0.0), errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out[fail_col] = pd.to_numeric(out.get(fail_col, 1.0), errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out[epistemic_col] = pd.to_numeric(out.get(epistemic_col, 0.0), errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    comfort_cost = (
        pd.to_numeric(out.get(comfort_acc_col, 0.0), errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
        + pd.to_numeric(out.get(comfort_jerk_col, 0.0), errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    out['_comfort_cost'] = comfort_cost
    out['_primary_score'] = (
        float(getattr(cfg, 'risk_control_progress_weight', 1.0)) * out[progress_col]
        - float(getattr(cfg, 'risk_control_fail_weight', 2.0)) * out[fail_col]
        - float(getattr(cfg, 'risk_control_uncertainty_weight', 0.5)) * out[epistemic_col]
        - float(getattr(cfg, 'risk_control_comfort_weight', 0.25)) * out['_comfort_cost']
    )
    out['_budget_ok'] = (out[fail_col] <= float(getattr(cfg, 'risk_control_fail_budget', 0.20))).astype(int)
    return out


def select_action_with_calibrated_risk(
    candidate_predictions: pd.DataFrame,
    cfg: Any,
    failure_label: str = DEFAULT_FAILURE_LABEL,
) -> RiskControlSelection:
    if candidate_predictions.empty:
        raise ValueError('candidate_predictions is empty')
    scored = _score_candidates(candidate_predictions, cfg=cfg, failure_label=failure_label)
    action_cols = [c for c in ['action_0', 'action_1', 'action_2'] if c in scored.columns]
    fail_col = f'risk_cal_{failure_label}'
    if fail_col not in scored.columns:
        raise ValueError(f'Missing required column: {fail_col}')
    budget_ok = scored[scored['_budget_ok'] > 0]
    if not budget_ok.empty:
        chosen = budget_ok.sort_values(['_primary_score', fail_col], ascending=[False, True]).iloc[0]
    else:
        chosen = scored.sort_values([fail_col, '_primary_score'], ascending=[True, False]).iloc[0]
    action = chosen[action_cols].to_numpy(dtype=np.float32)
    return RiskControlSelection(
        selected_action=action,
        selected_row=chosen.to_dict(),
        candidate_predictions=scored,
    )

