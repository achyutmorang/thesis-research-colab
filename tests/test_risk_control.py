from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.closedloop.risk_control import select_action_with_calibrated_risk


class _IdentityScaler:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def apply(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=float)


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        risk_dataset_control_horizon_steps=6,
        risk_control_progress_weight=1.0,
        risk_control_fail_weight=2.0,
        risk_control_uncertainty_weight=0.5,
        risk_control_comfort_weight=0.25,
        risk_control_fail_budget=0.20,
    )


def test_select_action_prefers_lower_risk_when_progress_equal() -> None:
    df = pd.DataFrame(
        [
            {
                'action_0': 1.0,
                'action_1': 0.0,
                'action_2': 0.0,
                'progress_h6': 5.0,
                'max_abs_acc_h6': 0.1,
                'max_abs_jerk_h6': 0.1,
                'risk_cal_failure_proxy_h15': 0.35,
                'risk_epistemic_failure_proxy_h15': 0.01,
            },
            {
                'action_0': 2.0,
                'action_1': 0.0,
                'action_2': 0.0,
                'progress_h6': 5.0,
                'max_abs_acc_h6': 0.1,
                'max_abs_jerk_h6': 0.1,
                'risk_cal_failure_proxy_h15': 0.05,
                'risk_epistemic_failure_proxy_h15': 0.01,
            },
        ]
    )

    selection = select_action_with_calibrated_risk(df, _cfg())
    assert np.isclose(selection.selected_action[0], 2.0)


def test_select_action_handles_non_finite_risk_values() -> None:
    df = pd.DataFrame(
        [
            {
                'action_0': 1.0,
                'action_1': 0.0,
                'action_2': 0.0,
                'progress_h6': 3.0,
                'max_abs_acc_h6': 0.1,
                'max_abs_jerk_h6': 0.1,
                'risk_cal_failure_proxy_h15': np.nan,
                'risk_epistemic_failure_proxy_h15': np.nan,
            },
            {
                'action_0': 2.0,
                'action_1': 0.0,
                'action_2': 0.0,
                'progress_h6': 2.0,
                'max_abs_acc_h6': 0.1,
                'max_abs_jerk_h6': 0.1,
                'risk_cal_failure_proxy_h15': 0.2,
                'risk_epistemic_failure_proxy_h15': 0.2,
            },
        ]
    )

    selection = select_action_with_calibrated_risk(df, _cfg())
    assert np.isfinite(selection.selected_action).all()
