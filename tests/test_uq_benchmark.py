from __future__ import annotations

import numpy as np
import pandas as pd

from src.closedloop.risk_benchmark import run_uq_benchmark, summarize_controller_tradeoff


def test_run_uq_benchmark_returns_expected_tables() -> None:
    rng = np.random.default_rng(13)
    n = 300
    labels = (rng.random(n) < 0.25).astype(float)
    raw = np.clip(rng.normal(loc=0.25, scale=0.2, size=n), 0.0, 1.0)
    cal = np.clip(0.75 * raw + 0.25 * labels, 0.0, 1.0)
    shifts = np.where(np.arange(n) % 2 == 0, 'nominal_clean', 'hist_prim_shift')

    df = pd.DataFrame(
        {
            'shift_suite': shifts,
            'failure_proxy_h15': labels,
            'risk_raw_failure_proxy_h15': raw,
            'risk_cal_failure_proxy_h15': cal,
        }
    )

    bundle = run_uq_benchmark(
        df,
        variants={'raw': 'risk_raw_failure_proxy_h15', 'cal': 'risk_cal_failure_proxy_h15'},
        label_columns=('failure_proxy_h15',),
        n_bins=10,
    )

    assert not bundle.summary_df.empty
    assert not bundle.per_shift_df.empty
    assert not bundle.reliability_df.empty
    assert not bundle.selective_curve_df.empty
    assert not bundle.shift_gap_df.empty


def test_summarize_controller_tradeoff_outputs_single_row() -> None:
    base_df = pd.DataFrame(
        {
            'scenario_id': [1, 2, 3],
            'progress_h6': [10.0, 8.0, 9.0],
            'failure_proxy_h15': [0.0, 1.0, 0.0],
        }
    )
    ctrl_df = pd.DataFrame(
        {
            'scenario_id': [1, 2, 3],
            'progress_h6': [9.5, 8.2, 8.8],
            'failure_proxy_h15': [0.0, 0.0, 0.0],
        }
    )

    summary = summarize_controller_tradeoff(base_df, ctrl_df)
    assert len(summary) == 1
    assert summary.iloc[0]['n_scenarios'] == 3
