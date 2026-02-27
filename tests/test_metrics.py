from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module(module_name: str, rel_path: str):
    root = Path(__file__).resolve().parents[1]
    path = root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


metrics = _load_module("closedloop_metrics", "src/closedloop/metrics.py")


def test_compute_finite_differences_shape_and_values():
    xy = np.asarray([[[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]], dtype=float)
    vel, acc, jerk = metrics.compute_finite_differences(xy)

    assert vel.shape == xy.shape
    assert acc.shape == xy.shape
    assert jerk.shape == xy.shape

    np.testing.assert_allclose(vel[0, 1], [1.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(vel[0, 2], [2.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(acc[0, 2], [1.0, 0.0], atol=1e-8)


def test_compute_risk_metrics_collision_flag():
    xy = np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.5, 0.0], [0.5, 0.0]],
        ],
        dtype=float,
    )
    valid = np.ones((2, 2), dtype=bool)
    out = metrics.compute_risk_metrics(
        xy=xy,
        valid=valid,
        collision_distance=1.0,
        w_dist=1.0,
        w_ttc=0.5,
        w_sks=0.01,
    )
    assert out["collision"] == 1.0
    assert out["failure_strict_proxy"] == 1.0
    assert out["failure_extended_proxy"] == 1.0
    assert out["risk_sks"] > 0.0


def test_robust_scale_has_min_floor():
    vals = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=float)
    s = metrics.robust_scale(vals, min_scale=1e-3)
    assert s == 1e-3


def test_planner_action_surprise_kl_zero_when_identical():
    actions = np.asarray([[0.1, 0.2], [0.1, 0.2]], dtype=float)
    valid = np.asarray([True, True], dtype=bool)
    kl = metrics.planner_action_surprise_kl(
        actions=actions,
        action_valid=valid,
        base_actions=actions.copy(),
        base_action_valid=valid.copy(),
        sigma=0.25,
    )
    assert abs(kl) < 1e-12


def test_counterfactual_surprise_score_has_nonzero_floor_when_perturbation_realized():
    score, diag = metrics.compute_counterfactual_surprise_score(
        trace_change_diag={},
        effect_l2_mean=0.0,
        effect_l2_budget=1.5,
        base_surprise_abs=0.0,
        proposal_surprise_abs=0.0,
        action_divergence=0.0,
        metric_hint="predictive_seq_w2",
        proposal_delta_l2=1.0,
        perturb_floor_weight=0.02,
        response_floor_weight=0.35,
        use_additive_score=True,
    )
    assert score > 0.0
    assert float(diag["surprise_signal_floor"]) > 0.0
    assert float(diag["surprise_belief_shift"]) >= float(diag["surprise_signal_floor"])
    assert float(diag["surprise_policy_shift"]) >= float(diag["surprise_signal_floor"])


def test_counterfactual_surprise_score_increases_with_stronger_policy_shift():
    low_score, _ = metrics.compute_counterfactual_surprise_score(
        trace_change_diag={"step_w2_all_mean": 1e-4, "step_logit_l1_all_mean": 1e-4},
        effect_l2_mean=0.2,
        effect_l2_budget=1.0,
        base_surprise_abs=0.0,
        proposal_surprise_abs=0.0,
        action_divergence=1e-4,
        metric_hint="predictive_seq_w2",
        proposal_delta_l2=0.5,
    )
    high_score, _ = metrics.compute_counterfactual_surprise_score(
        trace_change_diag={"step_w2_all_mean": 1.0, "step_logit_l1_all_mean": 1.0},
        effect_l2_mean=0.2,
        effect_l2_budget=1.0,
        base_surprise_abs=0.0,
        proposal_surprise_abs=0.0,
        action_divergence=1.0,
        metric_hint="predictive_seq_w2",
        proposal_delta_l2=0.5,
    )
    assert high_score > low_score
