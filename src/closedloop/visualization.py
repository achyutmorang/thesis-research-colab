from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import ClosedLoopConfig, SearchConfig
from .planner_backends import (
    _choose_target_non_ego,
    _gaussian_kl,
    _gaussian_w2,
    _moment_match_diag_gmm,
    _trace_step_is_fallback,
    closed_loop_rollout_selected,
    dist_trace_change_stats,
    dist_trace_diagnostics,
    make_behavioral_delta_proposal,
    make_closed_loop_components,
    predictive_divergence_from_dist_traces,
    project_delta_vec,
)


def _safe_array(x: Any, ndim: int = 1) -> np.ndarray:
    arr = np.asarray(x)
    while arr.ndim < ndim:
        arr = np.expand_dims(arr, axis=0)
    return arr


def _safe_step_dist(trace: List[Optional[Dict[str, np.ndarray]]], step: int) -> Optional[Dict[str, np.ndarray]]:
    if len(trace) <= 0:
        return None
    s = int(np.clip(int(step), 0, len(trace) - 1))
    out = trace[s]
    if out is None:
        return None
    if not isinstance(out, dict):
        return None
    if not all(k in out for k in ("weights", "means", "stds")):
        return None
    return out


def _step_metrics_df(
    base_trace: List[Optional[Dict[str, np.ndarray]]],
    proposal_trace: List[Optional[Dict[str, np.ndarray]]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n = int(min(len(base_trace), len(proposal_trace)))
    for t in range(n):
        b = base_trace[t]
        p = proposal_trace[t]
        if b is None or p is None:
            rows.append(
                {
                    "step": int(t),
                    "paired": 0,
                    "fallback_pair": np.nan,
                    "w2": np.nan,
                    "sym_kl": np.nan,
                    "mean_shift_l2": np.nan,
                    "base_source": str((b or {}).get("source", "")),
                    "proposal_source": str((p or {}).get("source", "")),
                }
            )
            continue
        try:
            mu_b, cov_b = _moment_match_diag_gmm(b)
            mu_p, cov_p = _moment_match_diag_gmm(p)
            d = int(min(mu_b.shape[0], mu_p.shape[0]))
            if d <= 0:
                raise ValueError("invalid moment dims")
            w2 = float(_gaussian_w2(mu_p[:d], cov_p[:d, :d], mu_b[:d], cov_b[:d, :d]))
            kl_pb = float(_gaussian_kl(mu_p[:d], cov_p[:d, :d], mu_b[:d], cov_b[:d, :d]))
            kl_bp = float(_gaussian_kl(mu_b[:d], cov_b[:d, :d], mu_p[:d], cov_p[:d, :d]))
            sym_kl = 0.5 * (kl_pb + kl_bp)
            mean_shift_l2 = float(np.linalg.norm(mu_p[:d] - mu_b[:d]))
            fallback_pair = bool(_trace_step_is_fallback(p) or _trace_step_is_fallback(b))
            rows.append(
                {
                    "step": int(t),
                    "paired": 1,
                    "fallback_pair": int(fallback_pair),
                    "w2": w2,
                    "sym_kl": sym_kl,
                    "mean_shift_l2": mean_shift_l2,
                    "base_source": str(b.get("source", "")),
                    "proposal_source": str(p.get("source", "")),
                }
            )
        except Exception:
            rows.append(
                {
                    "step": int(t),
                    "paired": 0,
                    "fallback_pair": np.nan,
                    "w2": np.nan,
                    "sym_kl": np.nan,
                    "mean_shift_l2": np.nan,
                    "base_source": str(b.get("source", "")),
                    "proposal_source": str(p.get("source", "")),
                }
            )
    return pd.DataFrame(rows)


def preview_predictive_distribution_pair(
    runner: Any,
    cfg: ClosedLoopConfig,
    search_cfg: SearchConfig,
    eval_idx: np.ndarray,
    scenario_rank: int = 0,
    primitive_hint: str = "toward_ego",
    delta_scale: float = 1.0,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    if len(eval_idx) <= 0:
        raise ValueError("eval_idx is empty; build simulation context first.")

    sid = int(eval_idx[int(np.clip(int(scenario_rank), 0, len(eval_idx) - 1))])
    rec = runner.data["scenarios"][sid]
    if "state" not in rec:
        raise ValueError(f"scenario_id={sid} has no retained state.")

    selected_idx = np.asarray(rec["selected_indices"], dtype=np.int32)
    target_idx = _choose_target_non_ego(rec["state"], selected_idx, cfg=cfg)
    planner_bundle = make_closed_loop_components(rec["state"], cfg.planner_kind, cfg.planner_name, cfg)

    base_seed = int(cfg.global_seed + sid + seed_offset + 17)
    prop_seed = int(cfg.global_seed + sid + seed_offset + 1017)

    base_xy, base_valid, base_actions, base_action_valid, base_trace, base_feasible, base_note = closed_loop_rollout_selected(
        base_state=rec["state"],
        selected_idx=selected_idx,
        target_obj_idx=target_idx,
        delta_xy=np.zeros((2,), dtype=float),
        cfg=cfg,
        planner_bundle=planner_bundle,
        seed=base_seed,
    )

    delta_xy, delta_meta = make_behavioral_delta_proposal(
        base_state=rec["state"],
        target_obj_idx=target_idx,
        cfg=cfg,
        primitive_hint=str(primitive_hint),
        scale_mult=float(delta_scale),
        return_meta=True,
    )
    delta_xy = project_delta_vec(
        delta_xy=np.asarray(delta_xy, dtype=float),
        delta_clip=float(search_cfg.delta_clip),
        delta_l2_budget=float(search_cfg.delta_l2_budget),
    )

    prop_xy, prop_valid, prop_actions, prop_action_valid, prop_trace, prop_feasible, prop_note = closed_loop_rollout_selected(
        base_state=rec["state"],
        selected_idx=selected_idx,
        target_obj_idx=target_idx,
        delta_xy=delta_xy,
        cfg=cfg,
        planner_bundle=planner_bundle,
        seed=prop_seed,
    )

    w2, w2_source = predictive_divergence_from_dist_traces(
        trace_p=prop_trace,
        trace_q=base_trace,
        metric="predictive_w2",
        skip_fallback_steps=bool(cfg.predictive_kl_skip_fallback_steps),
    )
    kl, kl_source = predictive_divergence_from_dist_traces(
        trace_p=prop_trace,
        trace_q=base_trace,
        metric="predictive_kl",
        estimator=str(cfg.predictive_kl_estimator),
        n_mc_samples=int(cfg.predictive_kl_mc_samples),
        seed=int(cfg.predictive_kl_mc_seed + sid),
        eps=float(cfg.predictive_kl_eps),
        symmetric=bool(cfg.predictive_kl_symmetric),
        skip_fallback_steps=bool(cfg.predictive_kl_skip_fallback_steps),
    )

    base_diag = dist_trace_diagnostics(base_trace)
    prop_diag = dist_trace_diagnostics(prop_trace)
    change_diag = dist_trace_change_stats(prop_trace, base_trace)
    step_df = _step_metrics_df(base_trace, prop_trace)

    summary_df = pd.DataFrame(
        [
            {
                "scenario_id": int(sid),
                "planner_kind": str(cfg.planner_kind),
                "planner_used": str(planner_bundle.get("planner_used", "")),
                "target_obj_idx": int(target_idx),
                "delta_x": float(delta_xy[0]),
                "delta_y": float(delta_xy[1]),
                "delta_l2": float(np.linalg.norm(delta_xy)),
                "primitive_hint": str(delta_meta.get("primitive", primitive_hint)),
                "base_rollout_feasible": int(bool(base_feasible)),
                "proposal_rollout_feasible": int(bool(prop_feasible)),
                "base_rollout_note": str(base_note),
                "proposal_rollout_note": str(prop_note),
                "predictive_w2": float(w2),
                "predictive_kl": float(kl),
                "predictive_w2_source": str(w2_source),
                "predictive_kl_source": str(kl_source),
                "trace_pair_ratio": float(change_diag.get("trace_pair_ratio", np.nan)),
                "trace_pair_ratio_all": float(change_diag.get("trace_pair_ratio_all", np.nan)),
                "trace_fallback_pair_ratio": float(change_diag.get("trace_fallback_pair_ratio", np.nan)),
                "step_mean_l2_all_mean": float(change_diag.get("step_mean_l2_all_mean", np.nan)),
                "step_logit_l1_all_mean": float(change_diag.get("step_logit_l1_all_mean", np.nan)),
                "base_dist_source_model_ratio": float(base_diag.get("dist_source_model_ratio", np.nan)),
                "proposal_dist_source_model_ratio": float(prop_diag.get("dist_source_model_ratio", np.nan)),
                "proposal_dist_source_proxy_ratio": float(prop_diag.get("dist_source_proxy_ratio", np.nan)),
            }
        ]
    )

    return {
        "summary_df": summary_df,
        "step_metrics_df": step_df,
        "base_dist_trace": base_trace,
        "proposal_dist_trace": prop_trace,
        "base_actions": np.asarray(base_actions),
        "proposal_actions": np.asarray(prop_actions),
        "base_action_valid": np.asarray(base_action_valid),
        "proposal_action_valid": np.asarray(prop_action_valid),
    }


def plot_predictive_distribution_step(
    base_trace: List[Optional[Dict[str, np.ndarray]]],
    proposal_trace: List[Optional[Dict[str, np.ndarray]]],
    step: int = 0,
    title: str = "Predictive action distribution comparison",
    max_components: int = 6,
) -> Any:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except Exception as e:
        raise RuntimeError("matplotlib is required to render predictive distribution plots.") from e

    base = _safe_step_dist(base_trace, step=step)
    prop = _safe_step_dist(proposal_trace, step=step)
    if base is None or prop is None:
        raise ValueError(f"No valid paired predictive distribution at step={step}.")

    w_b = _safe_array(base["weights"], ndim=1).reshape(-1)
    m_b = _safe_array(base["means"], ndim=2)
    s_b = _safe_array(base["stds"], ndim=2)
    w_p = _safe_array(prop["weights"], ndim=1).reshape(-1)
    m_p = _safe_array(prop["means"], ndim=2)
    s_p = _safe_array(prop["stds"], ndim=2)

    k_b = int(min(len(w_b), len(m_b), len(s_b), max_components))
    k_p = int(min(len(w_p), len(m_p), len(s_p), max_components))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_xy, ax_yaw = axes

    for k in range(k_b):
        mu = np.asarray(m_b[k], dtype=float).reshape(-1)
        sd = np.maximum(np.asarray(s_b[k], dtype=float).reshape(-1), 1e-6)
        if mu.shape[0] < 2 or sd.shape[0] < 2:
            continue
        ax_xy.scatter(mu[0], mu[1], color="tab:blue", s=45, alpha=0.7)
        e = Ellipse(
            xy=(mu[0], mu[1]),
            width=float(2.0 * sd[0]),
            height=float(2.0 * sd[1]),
            edgecolor="tab:blue",
            facecolor="none",
            alpha=float(0.25 + 0.45 * min(1.0, float(w_b[k]))),
            lw=1.2,
        )
        ax_xy.add_patch(e)

    for k in range(k_p):
        mu = np.asarray(m_p[k], dtype=float).reshape(-1)
        sd = np.maximum(np.asarray(s_p[k], dtype=float).reshape(-1), 1e-6)
        if mu.shape[0] < 2 or sd.shape[0] < 2:
            continue
        ax_xy.scatter(mu[0], mu[1], color="tab:orange", s=45, alpha=0.7, marker="x")
        e = Ellipse(
            xy=(mu[0], mu[1]),
            width=float(2.0 * sd[0]),
            height=float(2.0 * sd[1]),
            edgecolor="tab:orange",
            facecolor="none",
            alpha=float(0.25 + 0.45 * min(1.0, float(w_p[k]))),
            lw=1.2,
            ls="--",
        )
        ax_xy.add_patch(e)

    ax_xy.set_title(f"Step {int(step)}: $(\\Delta x,\\Delta y)$ components")
    ax_xy.set_xlabel("$\\Delta x$")
    ax_xy.set_ylabel("$\\Delta y$")
    ax_xy.grid(alpha=0.25)
    ax_xy.axis("equal")

    def _gmm_pdf_1d(x: np.ndarray, w: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x, dtype=float)
        for i in range(min(len(w), len(m), len(s))):
            si = max(float(s[i]), 1e-6)
            y += float(w[i]) * (1.0 / (np.sqrt(2.0 * np.pi) * si)) * np.exp(-0.5 * ((x - float(m[i])) / si) ** 2)
        return y

    mu_b_yaw = m_b[:k_b, 2] if (k_b > 0 and m_b.shape[1] > 2) else np.zeros((k_b,), dtype=float)
    sd_b_yaw = s_b[:k_b, 2] if (k_b > 0 and s_b.shape[1] > 2) else np.ones((k_b,), dtype=float) * 0.2
    mu_p_yaw = m_p[:k_p, 2] if (k_p > 0 and m_p.shape[1] > 2) else np.zeros((k_p,), dtype=float)
    sd_p_yaw = s_p[:k_p, 2] if (k_p > 0 and s_p.shape[1] > 2) else np.ones((k_p,), dtype=float) * 0.2

    yaw_min = float(min(np.min(mu_b_yaw - 4.0 * sd_b_yaw), np.min(mu_p_yaw - 4.0 * sd_p_yaw))) if (k_b > 0 and k_p > 0) else -1.0
    yaw_max = float(max(np.max(mu_b_yaw + 4.0 * sd_b_yaw), np.max(mu_p_yaw + 4.0 * sd_p_yaw))) if (k_b > 0 and k_p > 0) else 1.0
    if not np.isfinite(yaw_min) or not np.isfinite(yaw_max) or yaw_max <= yaw_min:
        yaw_min, yaw_max = -1.0, 1.0
    x_grid = np.linspace(yaw_min, yaw_max, 300)

    y_b = _gmm_pdf_1d(x_grid, w_b[:k_b], mu_b_yaw, sd_b_yaw) if k_b > 0 else np.zeros_like(x_grid)
    y_p = _gmm_pdf_1d(x_grid, w_p[:k_p], mu_p_yaw, sd_p_yaw) if k_p > 0 else np.zeros_like(x_grid)

    ax_yaw.plot(x_grid, y_b, color="tab:blue", lw=2.0, label="base")
    ax_yaw.plot(x_grid, y_p, color="tab:orange", lw=2.0, ls="--", label="proposal")
    ax_yaw.set_title(f"Step {int(step)}: $\\Delta\\psi$ mixture density")
    ax_yaw.set_xlabel("$\\Delta\\psi$")
    ax_yaw.set_ylabel("density")
    ax_yaw.grid(alpha=0.25)
    ax_yaw.legend(loc="best")

    fig.suptitle(str(title), fontsize=12, y=1.02)
    fig.tight_layout()
    return fig
