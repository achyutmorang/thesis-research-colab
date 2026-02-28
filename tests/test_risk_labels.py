from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.risk_model.labels import label_candidate_rollout_events


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


def test_label_candidate_rollout_events_flags_collision_offroad_and_failure() -> None:
    # Two agents overlap by step 3 to trigger collision proxy.
    xy = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0], [7.0, 0.0], [8.0, 0.0], [9.0, 0.0], [10.0, 0.0], [11.0, 0.0], [12.0, 0.0], [13.0, 0.0], [14.0, 0.0]],
            [[3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0]],
        ],
        dtype=float,
    )
    valid = np.ones((2, 15), dtype=bool)
    valid[0, 9:] = False  # ego invalid later -> offroad proxy for >=10 step horizon

    labels = label_candidate_rollout_events(xy, valid, cfg=_cfg(), horizons=(5, 10, 15))

    assert labels['collision_h5'] == 1
    assert labels['collision_h10'] == 1
    assert labels['offroad_h5'] == 0
    assert labels['offroad_h10'] == 1
    assert labels['offroad_h15'] == 1
    assert labels['failure_proxy_h15'] == 1
