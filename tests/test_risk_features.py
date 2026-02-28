from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.risk_model.features import extract_candidate_risk_features


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        collision_distance=1.5,
        risk_w_dist=1.0,
        risk_w_ttc=0.5,
        risk_w_sks=0.01,
        ttc_fail_seconds=2.0,
        no_hazard_ttc_seconds=3.0,
        no_hazard_dist_m=8.0,
        hard_brake_mps2=6.0,
        hard_jerk_mps3=8.0,
        enable_intervention_proxy=True,
    )


def test_extract_candidate_risk_features_returns_finite_core_fields() -> None:
    xy = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
            [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [4.0, 2.0], [5.0, 2.0]],
        ],
        dtype=float,
    )
    valid = np.ones((2, 6), dtype=bool)

    dist_step = {
        'weights': np.array([0.7, 0.3], dtype=float),
        'means': np.array([[1.0, 0.0, 0.0], [0.5, 0.2, 0.0]], dtype=float),
        'stds': np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]], dtype=float),
        'belief_kl_step': np.array(0.2, dtype=float),
        'belief_kl_available': np.array(1, dtype=int),
    }
    dist_trace = [dist_step, dist_step]

    feats = extract_candidate_risk_features(
        dist_step=dist_step,
        dist_trace=dist_trace,
        xy=xy,
        valid=valid,
        cfg=_cfg(),
        control_horizon_steps=6,
        target_interaction_score=0.9,
    )

    required = [
        'dist_entropy',
        'dist_top1_weight',
        'dist_num_components',
        'belief_kl_current',
        'belief_kl_rolling_mean',
        'progress_h6',
        'min_distance_h6',
        'risk_sks_short_h6',
        'target_interaction_score',
    ]
    for key in required:
        assert key in feats
        assert np.isfinite(float(feats[key]))
    assert 'min_ttc_h6' in feats
    assert float(feats['min_ttc_h6']) >= 0.0
