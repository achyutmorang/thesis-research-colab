from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.risk_model.benchmark import (
    adaptive_ece,
    binary_auprc,
    binary_auroc,
    binary_ece,
    brier_score,
    nll_score,
)
from .miscalibration_interpretation_flow import discover_probe_run_prefixes
from .miscalibration_probe_flow import (
    _apply_binary_logistic_1d,
    _fit_binary_logistic_1d,
    has_existing_miscalibration_probe_artifacts,
    load_existing_miscalibration_probe_bundle,
)


DEFAULT_CONTROLLER_VARIANTS: Tuple[str, ...] = (
    "combo:platt",
    "combo:raw",
    "racp_style:platt",
    "ccmpc_style:platt",
    "radius_style:platt",
)

DEFAULT_RACP_LAMBDAS: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)

DEFAULT_REPO_REFERENCES: Tuple[Tuple[str, str], ...] = (
    ("rap", "https://github.com/TRI-ML/RAP"),
    ("racp", "https://github.com/KhMustafa/Risk-aware-contingency-planning-with-multi-modal-predictions"),
    ("radius", "https://github.com/roahmlab/RADIUS"),
    ("cuqds", "https://github.com/HuiqunHuang/CUQDS"),
)


@dataclass
class PlannerMethodVariantAuditBundle:
    output_prefix: str
    focus_label: str
    risk_budget_tau: float
    loaded_run_prefixes: List[str] = field(default_factory=list)
    loaded_sources_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibrator_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    signal_metrics_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    signal_tau_sweep_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    signal_tau_at_budget_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    controller_metrics_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    controller_step_trace_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    controller_at_budget_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    conformal_threshold_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnosis_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    repo_reference_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    artifact_paths: Dict[str, str] = field(default_factory=dict)


def _safe_float(values: Any, default: float = 0.0) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = np.where(np.isfinite(arr), arr, float(default))
    return arr


def _clip_prob(values: Any, default: float = 0.5) -> np.ndarray:
    return np.clip(_safe_float(values, default=default), 1e-6, 1.0 - 1e-6)


def _parse_prefixes(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        out: List[str] = []
        for item in values:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(values).strip()
    if not text:
        return []
    text = text.replace(chr(10), ",").replace(";", ",")
    return [part.strip() for part in text.split(",") if part.strip()]


def _with_planner_variant(df: pd.DataFrame, run_prefix: str) -> pd.DataFrame:
    out = df.copy()
    run_tag = Path(str(run_prefix)).name
    planner_cols = ["planner_variant", "planner_backend", "planner_kind", "planner_used"]
    planner_value: Optional[pd.Series] = None
    for col in planner_cols:
        if col in out.columns:
            series = out[col].astype(str).str.strip()
            if (series != "").any():
                planner_value = series.where(series.str.len() > 0, f"run:{run_tag}")
                break
    if planner_value is None:
        planner_value = pd.Series([f"run:{run_tag}"] * len(out), index=out.index, dtype="object")
    out["planner_variant"] = planner_value.astype(str)
    out["analysis_run_prefix"] = str(run_prefix)
    out["analysis_run_tag"] = str(run_tag)
    return out


def _load_run_predictions(run_prefix: str) -> Tuple[pd.DataFrame, str]:
    cross_path = Path(f"{run_prefix}_cross_signal_decision_predictions.parquet")
    if cross_path.exists():
        try:
            return pd.read_parquet(cross_path), "cross_signal_decision_predictions"
        except Exception:
            pass
    bundle = load_existing_miscalibration_probe_bundle(run_prefix)
    return bundle.predictions_df.copy(), "miscalibration_probe_predictions"


def load_multi_run_prediction_frame(
    *,
    current_run_prefix: str,
    persist_root: str,
    analysis_run_prefix: str = "",
    analysis_run_prefixes: Sequence[str] = (),
    allow_previous_run_fallback: bool = True,
    max_discovered_runs: int = 50,
    max_runs_to_load: int = 8,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    requested: List[str] = []
    if str(analysis_run_prefix).strip():
        requested.append(str(analysis_run_prefix).strip())
    requested.extend(_parse_prefixes(list(analysis_run_prefixes)))
    if len(requested) <= 0:
        requested.append(str(current_run_prefix))

    seen = set()
    requested = [p for p in requested if not (p in seen or seen.add(p))]

    discovered_df = discover_probe_run_prefixes(persist_root, limit=int(max(1, max_discovered_runs)))
    prefixes_to_try = list(requested)
    if bool(allow_previous_run_fallback) and (not discovered_df.empty):
        for pfx in discovered_df["run_prefix"].astype(str).tolist():
            if pfx not in prefixes_to_try:
                prefixes_to_try.append(pfx)
    prefixes_to_try = prefixes_to_try[: int(max(1, max_runs_to_load))]

    frames: List[pd.DataFrame] = []
    rows: List[Dict[str, Any]] = []
    loaded_prefixes: List[str] = []
    for pfx in prefixes_to_try:
        exists = has_existing_miscalibration_probe_artifacts(pfx)
        if not exists:
            rows.append({"run_prefix": str(pfx), "loaded": 0, "reason": "missing_probe_artifacts", "source": ""})
            continue
        try:
            pred, source = _load_run_predictions(pfx)
        except Exception as exc:
            rows.append({"run_prefix": str(pfx), "loaded": 0, "reason": f"load_failed:{exc}", "source": ""})
            continue
        if pred.empty:
            rows.append({"run_prefix": str(pfx), "loaded": 0, "reason": "empty_predictions", "source": source})
            continue
        pred = _with_planner_variant(pred, pfx)
        frames.append(pred)
        loaded_prefixes.append(str(pfx))
        rows.append({"run_prefix": str(pfx), "loaded": 1, "reason": "ok", "source": source, "rows": int(len(pred))})

    loaded_df = pd.DataFrame(rows)
    if len(frames) <= 0:
        return pd.DataFrame(), [], loaded_df
    out = pd.concat(frames, axis=0, ignore_index=True)
    return out, loaded_prefixes, loaded_df


def _fit_quantile_scaler(values: Any, mask: np.ndarray, q_lo: float = 0.01, q_hi: float = 0.99) -> Tuple[float, float]:
    arr = _safe_float(values, default=np.nan)
    ref = arr[np.asarray(mask, dtype=bool)]
    ref = ref[np.isfinite(ref)]
    if ref.size <= 0:
        ref = arr[np.isfinite(arr)]
    if ref.size <= 0:
        return 0.0, 1.0
    lo = float(np.quantile(ref, float(q_lo)))
    hi = float(np.quantile(ref, float(q_hi)))
    if not np.isfinite(lo):
        lo = float(np.nanmin(ref))
    if not np.isfinite(hi):
        hi = float(np.nanmax(ref))
    if hi <= lo + 1e-9:
        hi = lo + 1.0
    return float(lo), float(hi)


def _scale_quantile(values: Any, lo: float, hi: float) -> np.ndarray:
    arr = _safe_float(values, default=float(lo))
    out = (arr - float(lo)) / max(1e-9, float(hi - lo))
    return np.clip(out, 0.0, 1.0)


def _inverse_ttc(values: Any, cap: float = 8.0) -> np.ndarray:
    ttc = _safe_float(values, default=cap)
    ttc = np.clip(ttc, 0.0, cap)
    return np.clip(1.0 - (ttc / max(1e-6, cap)), 0.0, 1.0)


def _inverse_distance(values: Any, cap: float = 20.0) -> np.ndarray:
    dist = _safe_float(values, default=cap)
    dist = np.clip(dist, 0.0, cap)
    return np.clip(1.0 - (dist / max(1e-6, cap)), 0.0, 1.0)


def _prepare_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
    out = df.copy()
    if "shift_suite" not in out.columns:
        out["shift_suite"] = "nominal_clean"
    else:
        out["shift_suite"] = out["shift_suite"].fillna("nominal_clean").astype(str)
    if "scenario_id" not in out.columns:
        out["scenario_id"] = np.arange(len(out)).astype(str)
    if "step_idx" not in out.columns:
        out["step_idx"] = 0
    out["scenario_id"] = out["scenario_id"].astype(str)
    out["step_idx"] = pd.to_numeric(out["step_idx"], errors="coerce").fillna(0).astype(int)
    out["planner_variant"] = out.get("planner_variant", "planner_unknown").astype(str)

    if "eval_split" in out.columns:
        split = out["eval_split"].astype(str)
        cal_mask = split.eq("val").to_numpy(dtype=bool)
        eval_mask = split.isin(["test", "high_interaction_holdout"]).to_numpy(dtype=bool)
        split_source = "eval_split column from dataset artifacts"
    else:
        unique_ids = pd.Series(out["scenario_id"].astype(str).unique())
        shuffled = unique_ids.sample(frac=1.0, random_state=17).to_numpy(dtype=object)
        n_total = len(shuffled)
        n_val = max(1, int(round(0.15 * n_total)))
        n_test = max(1, int(round(0.15 * n_total)))
        val_ids = set(shuffled[:n_val].tolist())
        test_ids = set(shuffled[n_val : n_val + n_test].tolist())
        sid = out["scenario_id"].astype(str)
        cal_mask = sid.isin(val_ids).to_numpy(dtype=bool)
        eval_mask = sid.isin(test_ids).to_numpy(dtype=bool)
        split_source = "fallback scenario-level random split (15% val, 15% eval)"

    if int(np.sum(cal_mask)) < 50:
        cal_mask = np.ones((len(out),), dtype=bool)
    if int(np.sum(eval_mask)) < 50:
        eval_mask = np.ones((len(out),), dtype=bool)
    return out, cal_mask, eval_mask, split_source


def _build_raw_signals(df: pd.DataFrame, cal_mask: np.ndarray) -> Dict[str, np.ndarray]:
    out = df.copy()
    signals: Dict[str, np.ndarray] = {}

    if "planner_risk_top1_proxy" in out.columns:
        signals["top1"] = _clip_prob(out["planner_risk_top1_proxy"])
    elif "dist_top1_weight" in out.columns:
        signals["top1"] = _clip_prob(1.0 - _clip_prob(out["dist_top1_weight"]))

    if "planner_risk_entropy_proxy" in out.columns:
        signals["entropy"] = _clip_prob(out["planner_risk_entropy_proxy"])
    elif ("dist_entropy" in out.columns) and ("dist_num_components" in out.columns):
        entropy = _safe_float(out["dist_entropy"], default=0.0)
        n_comp = np.maximum(2.0, _safe_float(out["dist_num_components"], default=2.0))
        entropy_max = np.log(n_comp)
        entropy_max = np.where(np.isfinite(entropy_max) & (entropy_max > 1e-8), entropy_max, 1.0)
        conf_entropy = 1.0 - np.clip(entropy / entropy_max, 0.0, 1.0)
        signals["entropy"] = _clip_prob(1.0 - conf_entropy)

    if "planner_risk_combo_proxy" in out.columns:
        signals["combo"] = _clip_prob(out["planner_risk_combo_proxy"])
    elif ("top1" in signals) and ("entropy" in signals):
        conf_top1 = 1.0 - _clip_prob(signals["top1"])
        conf_entropy = 1.0 - _clip_prob(signals["entropy"])
        combo_conf = np.clip(0.70 * conf_top1 + 0.30 * conf_entropy, 0.0, 1.0)
        signals["combo"] = _clip_prob(1.0 - combo_conf)

    if "dist_std_max" in out.columns:
        lo, hi = _fit_quantile_scaler(out["dist_std_max"], cal_mask)
        signals["stdmax"] = _clip_prob(_scale_quantile(out["dist_std_max"], lo, hi))
    if "belief_kl_current" in out.columns:
        lo, hi = _fit_quantile_scaler(out["belief_kl_current"], cal_mask)
        signals["belief_kl"] = _clip_prob(_scale_quantile(out["belief_kl_current"], lo, hi))
    if "min_ttc_h6" in out.columns:
        signals["inv_ttc"] = _clip_prob(_inverse_ttc(out["min_ttc_h6"], cap=8.0))
    if "min_distance_h6" in out.columns:
        signals["inv_distance"] = _clip_prob(_inverse_distance(out["min_distance_h6"], cap=20.0))

    interaction = np.zeros((len(out),), dtype=float)
    if "target_interaction_score" in out.columns:
        lo, hi = _fit_quantile_scaler(out["target_interaction_score"], cal_mask, q_lo=0.05, q_hi=0.95)
        interaction = _clip_prob(_scale_quantile(out["target_interaction_score"], lo, hi))
    inv_ttc = signals.get("inv_ttc", np.zeros((len(out),), dtype=float))
    inv_dist = signals.get("inv_distance", np.zeros((len(out),), dtype=float))
    combo = signals.get("combo", np.zeros((len(out),), dtype=float))
    stdmax = signals.get("stdmax", np.zeros((len(out),), dtype=float))
    belief = signals.get("belief_kl", np.zeros((len(out),), dtype=float))

    if "combo" in signals:
        # RACP-inspired: blend multimodal planner confidence, interaction pressure, and short-horizon hazard.
        signals["racp_style"] = _clip_prob(0.45 * combo + 0.25 * inv_ttc + 0.20 * interaction + 0.10 * belief)
    # Chance-constrained MPC-inspired collision proxy.
    signals["ccmpc_style"] = _clip_prob(np.clip(0.50 * inv_ttc + 0.30 * inv_dist + 0.20 * stdmax, 0.0, 1.0))
    # Reachability-inspired conservative upper-bound proxy.
    signals["radius_style"] = _clip_prob(np.clip(0.80 * np.maximum.reduce([inv_ttc, inv_dist, interaction]) + 0.20 * belief, 0.0, 1.0))
    return signals


def _calibrate_signals(
    df: pd.DataFrame,
    *,
    focus_label: str,
    signals: Mapping[str, np.ndarray],
    cal_mask: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    out = df.copy()
    calibrator_rows: List[Dict[str, Any]] = []
    variant_to_column: Dict[str, str] = {}

    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore

        has_isotonic = True
    except Exception:
        IsotonicRegression = None  # type: ignore
        has_isotonic = False

    for signal_name, values in signals.items():
        raw_col = f"risk_{signal_name}_raw"
        out[raw_col] = _clip_prob(values)
        variant_to_column[f"{signal_name}:raw"] = raw_col

    planners = sorted(out["planner_variant"].astype(str).unique().tolist())
    for planner in planners:
        planner_mask = out["planner_variant"].astype(str).eq(str(planner)).to_numpy(dtype=bool)
        planner_cal_mask = planner_mask & np.asarray(cal_mask, dtype=bool)
        if int(np.sum(planner_cal_mask)) < 30:
            planner_cal_mask = planner_mask

        for signal_name in sorted(signals.keys()):
            raw_col = f"risk_{signal_name}_raw"
            platt_col = f"risk_{signal_name}_platt"
            iso_col = f"risk_{signal_name}_iso"

            out.loc[planner_mask, platt_col] = out.loc[planner_mask, raw_col].to_numpy(dtype=float)
            out.loc[planner_mask, iso_col] = out.loc[planner_mask, raw_col].to_numpy(dtype=float)
            variant_to_column[f"{signal_name}:platt"] = platt_col
            variant_to_column[f"{signal_name}:iso"] = iso_col

            x_cal = _safe_float(out.loc[planner_cal_mask, raw_col], default=0.5)
            y_cal = (_safe_float(out.loc[planner_cal_mask, focus_label], default=0.0) > 0.5).astype(float)
            has_pos = bool(np.any(y_cal > 0.5))
            has_neg = bool(np.any(y_cal < 0.5))
            fit_ok = bool((len(y_cal) >= 30) and has_pos and has_neg)

            alpha = np.nan
            beta = np.nan
            platt_fitted = 0
            if fit_ok:
                alpha, beta = _fit_binary_logistic_1d(x_cal, y_cal)
                p_platt = _apply_binary_logistic_1d(_safe_float(out.loc[planner_mask, raw_col], default=0.5), alpha, beta)
                out.loc[planner_mask, platt_col] = _clip_prob(p_platt)
                platt_fitted = 1

            iso_fitted = 0
            if has_isotonic and fit_ok and (IsotonicRegression is not None):
                unique_x = np.unique(np.round(x_cal, 6))
                if len(unique_x) >= 10:
                    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                    iso.fit(x_cal, y_cal)
                    p_iso = iso.predict(_safe_float(out.loc[planner_mask, raw_col], default=0.5))
                    out.loc[planner_mask, iso_col] = _clip_prob(p_iso)
                    iso_fitted = 1

            calibrator_rows.append(
                {
                    "planner_variant": str(planner),
                    "signal": str(signal_name),
                    "calibration": "platt",
                    "fitted": int(platt_fitted),
                    "alpha": float(alpha) if np.isfinite(alpha) else np.nan,
                    "beta": float(beta) if np.isfinite(beta) else np.nan,
                    "n_cal_rows": int(len(y_cal)),
                    "pos_rate_cal": float(np.mean(y_cal)) if len(y_cal) > 0 else np.nan,
                    "isotonic_available": int(has_isotonic),
                }
            )
            calibrator_rows.append(
                {
                    "planner_variant": str(planner),
                    "signal": str(signal_name),
                    "calibration": "iso",
                    "fitted": int(iso_fitted),
                    "alpha": np.nan,
                    "beta": np.nan,
                    "n_cal_rows": int(len(y_cal)),
                    "pos_rate_cal": float(np.mean(y_cal)) if len(y_cal) > 0 else np.nan,
                    "isotonic_available": int(has_isotonic),
                }
            )

    return out, pd.DataFrame(calibrator_rows), variant_to_column


def _within_step_auc(df: pd.DataFrame, prob_col: str, label_col: str) -> Tuple[float, int]:
    aucs: List[float] = []
    for _, grp in df.groupby("_step_key", sort=False):
        labels = (_safe_float(grp[label_col], default=0.0) > 0.5).astype(float)
        if (len(labels) < 2) or (len(np.unique(labels)) < 2):
            continue
        probs = _clip_prob(grp[prob_col])
        auc = binary_auroc(probs, labels)
        if np.isfinite(auc):
            aucs.append(float(auc))
    if len(aucs) <= 0:
        return np.nan, 0
    return float(np.mean(aucs)), int(len(aucs))


def _local_calibration(df: pd.DataFrame, prob_col: str, label_col: str, tau: float, window: float) -> Dict[str, float]:
    probs = _clip_prob(df[prob_col])
    labels = (_safe_float(df[label_col], default=0.0) > 0.5).astype(float)
    mask = np.abs(probs - float(tau)) <= float(window)
    n_local = int(np.sum(mask))
    if n_local <= 0:
        return {"local_n": 0.0, "local_mean_pred": np.nan, "local_event_rate": np.nan, "local_gap": np.nan}
    mean_pred = float(np.mean(probs[mask]))
    event_rate = float(np.mean(labels[mask]))
    return {
        "local_n": float(n_local),
        "local_mean_pred": mean_pred,
        "local_event_rate": event_rate,
        "local_gap": float(event_rate - mean_pred),
    }


def _decision_metrics(df: pd.DataFrame, prob_col: str, label_col: str, tau: float) -> Dict[str, float]:
    probs = _clip_prob(df[prob_col])
    labels = (_safe_float(df[label_col], default=0.0) > 0.5).astype(float)
    accepted = probs <= float(tau)
    rejected = ~accepted
    accept_count = int(np.sum(accepted))
    reject_count = int(np.sum(rejected))

    feasible_set_rate = np.nan
    fallback_rate = np.nan
    if "_step_key" in df.columns:
        tmp = pd.DataFrame({"step_key": df["_step_key"].astype(str), "accepted": accepted.astype(int)})
        if len(tmp) > 0:
            feasible_set_rate = float(tmp.groupby("step_key", sort=False)["accepted"].max().mean())
            fallback_rate = float(1.0 - feasible_set_rate) if np.isfinite(feasible_set_rate) else np.nan

    return {
        "n_rows": float(len(df)),
        "n_pos": float(np.sum(labels > 0.5)),
        "accept_count": float(accept_count),
        "reject_count": float(reject_count),
        "accept_rate": float(np.mean(accepted)) if len(df) > 0 else np.nan,
        "false_safe": float(np.mean(labels[accepted])) if accept_count > 0 else np.nan,
        "safe_reject": float(np.mean((labels <= 0.5)[rejected])) if reject_count > 0 else np.nan,
        "feasible_set_rate": feasible_set_rate,
        "fallback_rate": fallback_rate,
    }


def _bootstrap_ci(metric_fn: Any, df: pd.DataFrame, n_boot: int, seed: int) -> Dict[str, float]:
    n_boot = int(max(0, int(n_boot)))
    if (n_boot <= 0) or df.empty:
        return {}
    work = df.reset_index(drop=True)
    groups = {k: np.asarray(v, dtype=int) for k, v in work.groupby(work["scenario_id"].astype(str), sort=False).indices.items()}
    unit_ids = list(groups.keys())
    if len(unit_ids) <= 1:
        return {}

    rng = np.random.default_rng(int(seed))
    samples: List[Dict[str, float]] = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(unit_ids), size=len(unit_ids))
        idx = np.concatenate([groups[unit_ids[int(i)]] for i in picked], axis=0)
        sub = work.iloc[idx]
        samples.append(metric_fn(sub))

    keys = sorted(samples[0].keys())
    out: Dict[str, float] = {}
    for key in keys:
        arr = np.asarray([s.get(key, np.nan) for s in samples], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size <= 0:
            continue
        out[f"{key}_ci_low"] = float(np.quantile(arr, 0.025))
        out[f"{key}_ci_high"] = float(np.quantile(arr, 0.975))
    return out


def evaluate_signal_variants(
    df: pd.DataFrame,
    *,
    focus_label: str,
    variant_to_column: Mapping[str, str],
    risk_budget_tau: float,
    tau_values: Sequence[float],
    local_tau_window: float,
    n_bins: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eval_df = df.copy()
    eval_df["_step_key"] = (
        eval_df["planner_variant"].astype(str)
        + "::"
        + eval_df["scenario_id"].astype(str)
        + "::"
        + eval_df["step_idx"].astype(str)
    )

    metric_rows: List[Dict[str, Any]] = []
    tau_rows: List[Dict[str, Any]] = []
    for planner in sorted(eval_df["planner_variant"].astype(str).unique().tolist()):
        planner_df = eval_df[eval_df["planner_variant"].astype(str).eq(str(planner))].copy()
        for shift_suite, shift_df in planner_df.groupby("shift_suite", sort=True):
            for variant_key, prob_col in sorted(variant_to_column.items()):
                if prob_col not in shift_df.columns:
                    continue
                signal_name, cal_name = str(variant_key).split(":", 1)

                def _metric_fn(sub_df: pd.DataFrame) -> Dict[str, float]:
                    probs = _clip_prob(sub_df[prob_col])
                    labels = (_safe_float(sub_df[focus_label], default=0.0) > 0.5).astype(float)
                    ws_auc, _ = _within_step_auc(sub_df, prob_col, focus_label)
                    local = _local_calibration(sub_df, prob_col, focus_label, risk_budget_tau, local_tau_window)
                    return {
                        "auroc": float(binary_auroc(probs, labels)),
                        "auprc": float(binary_auprc(probs, labels)),
                        "within_step_auc": float(ws_auc),
                        "ece": float(binary_ece(probs, labels, n_bins=int(max(10, n_bins)))),
                        "adaptive_ece": float(adaptive_ece(probs, labels, n_bins=int(max(10, n_bins)))),
                        "nll": float(nll_score(probs, labels)),
                        "brier": float(brier_score(probs, labels)),
                        "local_n": float(local["local_n"]),
                        "local_gap": float(local["local_gap"]),
                    }

                metrics = _metric_fn(shift_df)
                ci = _bootstrap_ci(_metric_fn, shift_df, n_boot=bootstrap_samples, seed=bootstrap_seed)
                ws_auc, ws_n = _within_step_auc(shift_df, prob_col, focus_label)
                local = _local_calibration(shift_df, prob_col, focus_label, risk_budget_tau, local_tau_window)
                labels = (_safe_float(shift_df[focus_label], default=0.0) > 0.5).astype(float)
                metric_rows.append(
                    {
                        "planner_variant": str(planner),
                        "shift_suite": str(shift_suite),
                        "domain": "nominal" if str(shift_suite) == "nominal_clean" else "shifted",
                        "signal": str(signal_name),
                        "calibration": str(cal_name),
                        "variant": str(variant_key),
                        "prob_col": str(prob_col),
                        "n_rows": int(len(shift_df)),
                        "positive_rate": float(np.mean(labels)) if len(labels) > 0 else np.nan,
                        "within_step_groups_used": int(ws_n),
                        "local_tau": float(risk_budget_tau),
                        "local_tau_window": float(local_tau_window),
                        **metrics,
                        **ci,
                    }
                )

                for tau in tau_values:

                    def _tau_metric_fn(sub_df: pd.DataFrame) -> Dict[str, float]:
                        return _decision_metrics(sub_df, prob_col, focus_label, tau=float(tau))

                    dec = _tau_metric_fn(shift_df)
                    tau_ci = _bootstrap_ci(_tau_metric_fn, shift_df, n_boot=bootstrap_samples, seed=bootstrap_seed)

                    gate_reasons: List[str] = []
                    if int(dec["n_rows"]) < 200:
                        gate_reasons.append("low_rows")
                    if int(dec["n_pos"]) < 20:
                        gate_reasons.append("low_positives")
                    if int(dec["accept_count"]) < 30:
                        gate_reasons.append("low_accept_count")
                    if int(dec["reject_count"]) < 30:
                        gate_reasons.append("low_reject_count")
                    if shift_df["_step_key"].nunique() < 30:
                        gate_reasons.append("low_step_count")

                    tau_rows.append(
                        {
                            "planner_variant": str(planner),
                            "shift_suite": str(shift_suite),
                            "domain": "nominal" if str(shift_suite) == "nominal_clean" else "shifted",
                            "signal": str(signal_name),
                            "calibration": str(cal_name),
                            "variant": str(variant_key),
                            "tau": float(tau),
                            **dec,
                            **tau_ci,
                            "status": "ok" if len(gate_reasons) <= 0 else "inconclusive",
                            "inconclusive_reason": ";".join(gate_reasons),
                        }
                    )

    metrics_df = pd.DataFrame(metric_rows)
    tau_df = pd.DataFrame(tau_rows)
    tau_at_budget_df = pd.DataFrame()
    if not tau_df.empty:
        tau_at_budget_df = (
            tau_df.assign(_dist=(tau_df["tau"] - float(risk_budget_tau)).abs())
            .sort_values("_dist")
            .groupby(["planner_variant", "shift_suite", "signal", "calibration"], as_index=False)
            .first()
            .drop(columns=["_dist"])
            .sort_values(["planner_variant", "shift_suite", "signal", "calibration"])
            .reset_index(drop=True)
        )
    return metrics_df, tau_df, tau_at_budget_df


def _resolve_variant_columns(df: pd.DataFrame, preferred_variants: Sequence[str]) -> Dict[str, str]:
    available: Dict[str, str] = {}
    for col in df.columns:
        text = str(col)
        if not text.startswith("risk_"):
            continue
        parts = text.split("_")
        if len(parts) < 3:
            continue
        key = f"{'_'.join(parts[1:-1])}:{parts[-1]}"
        available[key] = text

    def _resolve_one(key: str) -> Optional[str]:
        if key in available:
            return key
        if ":" in key:
            signal = key.split(":", 1)[0]
            for cal in ("platt", "iso", "raw"):
                k = f"{signal}:{cal}"
                if k in available:
                    return k
        return None

    resolved: Dict[str, str] = {}
    for req in preferred_variants:
        key = _resolve_one(str(req))
        if key is None:
            continue
        if key not in resolved:
            resolved[key] = available[key]
    if len(resolved) <= 0:
        for key in sorted(available.keys()):
            resolved[key] = available[key]
            break
    return resolved


def _fit_conformal_tau(cal_df: pd.DataFrame, risk_col: str, tau_target: float, tau_values: Sequence[float], min_rows: int, min_accept: int) -> Tuple[float, str]:
    if cal_df.empty or (risk_col not in cal_df.columns):
        return float(tau_target), "fallback:no_cal_data"
    probs = _clip_prob(cal_df[risk_col])
    labels = (_safe_float(cal_df["y_true"], default=0.0) > 0.5).astype(float)
    if len(probs) < int(min_rows):
        return float(tau_target), "fallback:low_cal_rows"

    best_tau: Optional[float] = None
    for tau in sorted([float(x) for x in tau_values]):
        accepted = probs <= tau
        n_accept = int(np.sum(accepted))
        if n_accept < int(min_accept):
            continue
        false_safe = float(np.mean(labels[accepted]))
        if np.isfinite(false_safe) and (false_safe <= float(tau_target)):
            best_tau = float(tau)
    if best_tau is None:
        return float(np.min(np.asarray(list(tau_values), dtype=float))), "fallback:no_tau_meets_target"
    return float(best_tau), "ok"


def _select_chance(step_df: pd.DataFrame, risk_col: str, tau: float) -> Dict[str, Any]:
    risk = _clip_prob(step_df[risk_col])
    accepted = risk <= float(tau)
    if np.any(accepted):
        sub = step_df.loc[accepted].copy()
        idx = int(sub["progress_h6"].astype(float).idxmax())
        fallback = 0
    else:
        idx = int(step_df[risk_col].astype(float).idxmin())
        fallback = 1
    row = step_df.loc[idx]
    selected_risk = float(np.clip(float(row[risk_col]), 1e-6, 1.0 - 1e-6))
    return {
        "selected_idx": int(idx),
        "selected_risk": selected_risk,
        "accepted": int(selected_risk <= float(tau)),
        "fallback": int(fallback),
        "failure": int(row["y_true"]),
        "progress": float(row["progress_h6"]),
        "comfort": float(row["comfort_cost"]),
    }


def _select_racp(step_df: pd.DataFrame, risk_col: str, lam: float, report_tau: float) -> Dict[str, Any]:
    risk = _clip_prob(step_df[risk_col])
    score = step_df["progress_h6"].astype(float).to_numpy(dtype=float) - float(lam) * risk
    idx = int(step_df.index.to_numpy()[int(np.argmax(score))])
    row = step_df.loc[idx]
    selected_risk = float(np.clip(float(row[risk_col]), 1e-6, 1.0 - 1e-6))
    return {
        "selected_idx": int(idx),
        "selected_risk": selected_risk,
        "accepted": int(selected_risk <= float(report_tau)),
        "fallback": 0,
        "failure": int(row["y_true"]),
        "progress": float(row["progress_h6"]),
        "comfort": float(row["comfort_cost"]),
    }


def _select_oracle_risk(step_df: pd.DataFrame, tau: float) -> Dict[str, Any]:
    safe = step_df["y_true"].to_numpy(dtype=int) == 0
    if np.any(safe):
        sub = step_df.loc[safe].copy()
        idx = int(sub["comfort_cost"].astype(float).idxmin())
        fallback = 0
    else:
        idx = int(step_df["comfort_cost"].astype(float).idxmin())
        fallback = 1
    row = step_df.loc[idx]
    risk = float(row["y_true"])
    return {
        "selected_idx": int(idx),
        "selected_risk": risk,
        "accepted": int(risk <= float(tau)),
        "fallback": int(fallback),
        "failure": int(row["y_true"]),
        "progress": float(row["progress_h6"]),
        "comfort": float(row["comfort_cost"]),
    }


def _select_oracle_best(step_df: pd.DataFrame, tau: float) -> Dict[str, Any]:
    safe = step_df["y_true"].to_numpy(dtype=int) == 0
    if np.any(safe):
        sub = step_df.loc[safe].copy()
        idx = int(sub["progress_h6"].astype(float).idxmax())
        fallback = 0
    else:
        idx = int(step_df["progress_h6"].astype(float).idxmax())
        fallback = 1
    row = step_df.loc[idx]
    risk = float(row["y_true"])
    return {
        "selected_idx": int(idx),
        "selected_risk": risk,
        "accepted": int(risk <= float(tau)),
        "fallback": int(fallback),
        "failure": int(row["y_true"]),
        "progress": float(row["progress_h6"]),
        "comfort": float(row["comfort_cost"]),
    }


def _evaluate_controller(df: pd.DataFrame, spec: Mapping[str, Any], tau: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    mode = str(spec.get("mode", "chance"))
    risk_col = str(spec.get("risk_col", ""))
    lam = float(spec.get("lambda", 0.0))
    tau_effective = float(spec.get("tau_effective", tau))
    rows: List[Dict[str, Any]] = []
    for (planner, scenario_id, step_idx, shift_suite), step_df in df.groupby(
        ["planner_variant", "scenario_id", "step_idx", "shift_suite"], sort=False
    ):
        if mode == "chance":
            out = _select_chance(step_df, risk_col=risk_col, tau=float(tau))
            decision_tau = float(tau)
        elif mode == "conformal":
            out = _select_chance(step_df, risk_col=risk_col, tau=float(tau_effective))
            decision_tau = float(tau_effective)
        elif mode == "racp":
            out = _select_racp(step_df, risk_col=risk_col, lam=float(lam), report_tau=float(tau))
            decision_tau = float(tau)
        elif mode == "oracle-risk":
            out = _select_oracle_risk(step_df, tau=float(tau))
            decision_tau = float(tau)
        elif mode == "oracle-best":
            out = _select_oracle_best(step_df, tau=float(tau))
            decision_tau = float(tau)
        else:
            raise ValueError(f"Unsupported controller mode: {mode}")

        feasible = np.nan
        if risk_col and (risk_col in step_df.columns):
            feasible = int(np.any(_clip_prob(step_df[risk_col]) <= float(decision_tau)))

        rows.append(
            {
                "planner_variant": str(planner),
                "scenario_id": str(scenario_id),
                "step_idx": int(step_idx),
                "shift_suite": str(shift_suite),
                "controller": str(spec.get("name", mode)),
                "mode": mode,
                "signal_variant": str(spec.get("variant_key", "oracle")),
                "risk_col": risk_col,
                "lambda": float(spec.get("lambda", np.nan)) if mode == "racp" else np.nan,
                "tau": float(tau),
                "tau_effective": float(tau_effective),
                "decision_tau": float(decision_tau),
                "safe_candidate_exists": int(np.any(step_df["y_true"].to_numpy(dtype=int) == 0)),
                "feasible_set": feasible,
                **out,
            }
        )

    step_df = pd.DataFrame(rows)
    if step_df.empty:
        return step_df, {}

    accepted = step_df["accepted"].to_numpy(dtype=float)
    accepted_mask = np.isfinite(accepted) & (accepted > 0.5)
    rejected_mask = np.isfinite(accepted) & (accepted <= 0.5)
    failures = step_df["failure"].to_numpy(dtype=float)
    metrics = {
        "n_steps": float(len(step_df)),
        "n_scenarios": float(step_df["scenario_id"].nunique()),
        "failure_rate": float(np.mean(failures)),
        "progress_mean": float(np.mean(step_df["progress"].to_numpy(dtype=float))),
        "accept_rate": float(np.mean(accepted_mask)) if len(step_df) > 0 else np.nan,
        "false_safe": float(np.mean(failures[accepted_mask])) if int(np.sum(accepted_mask)) > 0 else np.nan,
        "safe_reject": float(np.mean((failures <= 0.5)[rejected_mask])) if int(np.sum(rejected_mask)) > 0 else np.nan,
        "feasible_set_rate": float(np.nanmean(step_df["feasible_set"].to_numpy(dtype=float))) if "feasible_set" in step_df.columns else np.nan,
        "fallback_rate": float(np.mean(step_df["fallback"].to_numpy(dtype=float))),
        "safe_candidate_rate": float(np.mean(step_df["safe_candidate_exists"].to_numpy(dtype=float))),
    }
    return step_df, metrics


def evaluate_controller_variants(
    eval_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    *,
    variant_columns: Mapping[str, str],
    risk_budget_tau: float,
    tau_values: Sequence[float],
    racp_lambda_values: Sequence[float],
    bootstrap_samples: int,
    bootstrap_seed: int,
    conformal_min_cal_rows: int,
    conformal_min_accept_count: int,
    conformal_shift_local: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if eval_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    specs: List[Dict[str, Any]] = []
    for variant_key, risk_col in variant_columns.items():
        specs.append({"name": f"chance:{variant_key}", "mode": "chance", "variant_key": variant_key, "risk_col": risk_col})
        for lam in racp_lambda_values:
            specs.append(
                {
                    "name": f"racp:{variant_key}:lam={float(lam):g}",
                    "mode": "racp",
                    "variant_key": variant_key,
                    "risk_col": risk_col,
                    "lambda": float(lam),
                }
            )
    for spec in specs:
        if spec["name"] == "chance:combo:platt":
            spec["name"] = "current"
            break
    specs.append({"name": "oracle-risk", "mode": "oracle-risk", "variant_key": "oracle", "risk_col": ""})
    specs.append({"name": "oracle-best", "mode": "oracle-best", "variant_key": "oracle", "risk_col": ""})

    conformal_rows: List[Dict[str, Any]] = []
    conformal_map: Dict[Tuple[str, str, str], float] = {}
    for planner in sorted(eval_df["planner_variant"].astype(str).unique().tolist()):
        planner_cal = cal_df[cal_df["planner_variant"].astype(str).eq(str(planner))].copy()
        for variant_key, risk_col in variant_columns.items():
            shifts = ["_global_"]
            if bool(conformal_shift_local) and ("shift_suite" in planner_cal.columns):
                shifts = sorted(planner_cal["shift_suite"].astype(str).unique().tolist())
            for shift in shifts:
                sub = planner_cal if shift == "_global_" else planner_cal[planner_cal["shift_suite"].astype(str).eq(str(shift))]
                tau_eff, status = _fit_conformal_tau(
                    sub,
                    risk_col=risk_col,
                    tau_target=float(risk_budget_tau),
                    tau_values=tau_values,
                    min_rows=int(max(1, conformal_min_cal_rows)),
                    min_accept=int(max(1, conformal_min_accept_count)),
                )
                conformal_map[(str(planner), str(variant_key), str(shift))] = float(tau_eff)
                conformal_rows.append(
                    {
                        "planner_variant": str(planner),
                        "variant_key": str(variant_key),
                        "shift_suite": str(shift),
                        "risk_col": str(risk_col),
                        "tau_target": float(risk_budget_tau),
                        "tau_effective": float(tau_eff),
                        "status": str(status),
                        "n_cal_rows": int(len(sub)),
                        "pos_rate_cal": float(np.mean(sub["y_true"].to_numpy(dtype=float))) if len(sub) > 0 else np.nan,
                    }
                )
    conformal_df = pd.DataFrame(conformal_rows)

    for variant_key, risk_col in variant_columns.items():
        specs.append(
            {
                "name": f"conformal:{variant_key}",
                "mode": "conformal",
                "variant_key": variant_key,
                "risk_col": risk_col,
            }
        )

    metric_rows: List[Dict[str, Any]] = []
    step_rows: List[pd.DataFrame] = []
    planners = sorted(eval_df["planner_variant"].astype(str).unique().tolist())
    for planner in planners:
        planner_eval = eval_df[eval_df["planner_variant"].astype(str).eq(str(planner))].copy()
        for shift_suite, shift_df in planner_eval.groupby("shift_suite", sort=True):
            for spec in specs:
                mode = str(spec.get("mode", "chance"))
                if mode in {"chance"}:
                    tau_grid = [float(t) for t in tau_values]
                else:
                    tau_grid = [float(risk_budget_tau)]
                for tau in tau_grid:
                    local_spec = dict(spec)
                    if mode == "conformal":
                        variant_key = str(spec.get("variant_key", ""))
                        key_shift = str(shift_suite) if bool(conformal_shift_local) else "_global_"
                        tau_eff = conformal_map.get((str(planner), variant_key, key_shift), float(risk_budget_tau))
                        local_spec["tau_effective"] = float(tau_eff)
                    else:
                        local_spec["tau_effective"] = float(tau)

                    step_df, metrics = _evaluate_controller(shift_df, local_spec, tau=float(tau))
                    if step_df.empty:
                        continue

                    def _metric_fn(sub_df: pd.DataFrame) -> Dict[str, float]:
                        accepted = sub_df["accepted"].to_numpy(dtype=float)
                        accepted_mask = np.isfinite(accepted) & (accepted > 0.5)
                        rejected_mask = np.isfinite(accepted) & (accepted <= 0.5)
                        y = sub_df["failure"].to_numpy(dtype=float)
                        return {
                            "failure_rate": float(np.mean(y)),
                            "progress_mean": float(np.mean(sub_df["progress"].to_numpy(dtype=float))),
                            "accept_rate": float(np.mean(accepted_mask)) if len(sub_df) > 0 else np.nan,
                            "false_safe": float(np.mean(y[accepted_mask])) if int(np.sum(accepted_mask)) > 0 else np.nan,
                            "safe_reject": float(np.mean((y <= 0.5)[rejected_mask])) if int(np.sum(rejected_mask)) > 0 else np.nan,
                            "feasible_set_rate": float(np.nanmean(sub_df["feasible_set"].to_numpy(dtype=float))),
                            "fallback_rate": float(np.mean(sub_df["fallback"].to_numpy(dtype=float))),
                        }

                    ci = _bootstrap_ci(_metric_fn, step_df, n_boot=bootstrap_samples, seed=bootstrap_seed)
                    gate_reasons: List[str] = []
                    if int(metrics.get("n_steps", 0)) < 40:
                        gate_reasons.append("low_steps")
                    if int(np.sum(step_df["failure"].to_numpy(dtype=float) > 0.5)) < 10:
                        gate_reasons.append("low_failures")
                    if int(np.sum(step_df["accepted"].to_numpy(dtype=float) > 0.5)) < 20:
                        gate_reasons.append("low_accept_count")
                    metric_rows.append(
                        {
                            "planner_variant": str(planner),
                            "shift_suite": str(shift_suite),
                            "domain": "nominal" if str(shift_suite) == "nominal_clean" else "shifted",
                            "controller": str(local_spec.get("name", mode)),
                            "mode": str(mode),
                            "signal_variant": str(local_spec.get("variant_key", "oracle")),
                            "risk_col": str(local_spec.get("risk_col", "")),
                            "lambda": float(local_spec.get("lambda", np.nan)) if str(mode) == "racp" else np.nan,
                            "tau": float(tau),
                            "tau_effective": float(local_spec.get("tau_effective", tau)),
                            **metrics,
                            **ci,
                            "status": "ok" if len(gate_reasons) <= 0 else "inconclusive",
                            "inconclusive_reason": ";".join(gate_reasons),
                        }
                    )
                    step_df = step_df.copy()
                    step_df["domain"] = "nominal" if str(shift_suite) == "nominal_clean" else "shifted"
                    step_df["status"] = "ok" if len(gate_reasons) <= 0 else "inconclusive"
                    step_df["inconclusive_reason"] = ";".join(gate_reasons)
                    step_rows.append(step_df)

    controller_metrics_df = pd.DataFrame(metric_rows)
    controller_step_df = pd.concat(step_rows, ignore_index=True) if len(step_rows) > 0 else pd.DataFrame()
    controller_at_budget_df = pd.DataFrame()
    if not controller_metrics_df.empty:
        controller_at_budget_df = (
            controller_metrics_df.assign(_dist=(controller_metrics_df["tau"] - float(risk_budget_tau)).abs())
            .sort_values("_dist")
            .groupby(["planner_variant", "shift_suite", "controller"], as_index=False)
            .first()
            .drop(columns=["_dist"])
            .sort_values(["planner_variant", "shift_suite", "controller"])
            .reset_index(drop=True)
        )
    diagnosis_df = diagnose_controller_bottlenecks(controller_at_budget_df)
    return controller_metrics_df, controller_step_df, controller_at_budget_df, conformal_df, diagnosis_df


def diagnose_controller_bottlenecks(controller_at_budget_df: pd.DataFrame) -> pd.DataFrame:
    if controller_at_budget_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for planner, planner_df in controller_at_budget_df.groupby("planner_variant", sort=True):
        focus = planner_df[planner_df["shift_suite"].astype(str).eq("nominal_clean")].copy()
        if focus.empty:
            focus = planner_df.copy()

        def _row(name: str) -> Optional[pd.Series]:
            sub = focus[focus["controller"].astype(str).eq(str(name))]
            if sub.empty:
                return None
            return sub.iloc[0]

        cur = _row("current")
        if cur is None:
            chance = focus[focus["controller"].astype(str).str.startswith("chance:")]
            if not chance.empty:
                cur = chance.iloc[0]
        oracle_r = _row("oracle-risk")
        oracle_b = _row("oracle-best")
        out = {
            "planner_variant": str(planner),
            "signal_or_calibration_bottleneck": np.nan,
            "controller_rule_bottleneck": np.nan,
            "candidate_quality_bottleneck": np.nan,
        }
        if (cur is not None) and (oracle_r is not None):
            gap = float(cur["failure_rate"]) - float(oracle_r["failure_rate"])
            out["signal_or_calibration_bottleneck"] = int(gap > 0.02)
            out["signal_fail_gap_current_minus_oracle_risk"] = gap
        if (oracle_r is not None) and (oracle_b is not None):
            progress_gap = float(oracle_b["progress_mean"]) - float(oracle_r["progress_mean"])
            fail_gap = float(oracle_b["failure_rate"]) - float(oracle_r["failure_rate"])
            out["controller_rule_bottleneck"] = int((progress_gap > 0.02) and (fail_gap <= 0.01))
            out["rule_progress_gap_oracle_best_minus_oracle_risk"] = progress_gap
            out["rule_failure_gap_oracle_best_minus_oracle_risk"] = fail_gap
        if oracle_b is not None:
            safe_rate = float(oracle_b["safe_candidate_rate"]) if np.isfinite(float(oracle_b["safe_candidate_rate"])) else np.nan
            fail_rate = float(oracle_b["failure_rate"])
            out["candidate_quality_bottleneck"] = int((np.isfinite(safe_rate) and safe_rate < 0.75) or (fail_rate > 0.15))
            out["oracle_best_safe_candidate_rate"] = safe_rate
            out["oracle_best_failure_rate"] = fail_rate
        out["note"] = "1 indicates likely bottleneck under fixed candidate-set offline audit."
        rows.append(out)
    return pd.DataFrame(rows)


def _repo_reference_table(reference_repo_root: str, repo_refs: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    root = Path(str(reference_repo_root)).expanduser()
    rows: List[Dict[str, Any]] = []
    for name, url in repo_refs:
        repo_dir = root / str(name)
        exists = repo_dir.exists() and (repo_dir / ".git").exists()
        rows.append(
            {
                "repo_name": str(name),
                "repo_url": str(url),
                "expected_path": str(repo_dir),
                "local_snapshot_present": int(bool(exists)),
            }
        )
    return pd.DataFrame(rows)


def _export_artifacts(bundle: PlannerMethodVariantAuditBundle) -> Dict[str, str]:
    prefix = str(bundle.output_prefix)
    paths = {
        "planner_variant_loaded_sources": f"{prefix}_planner_variant_loaded_sources.csv",
        "planner_variant_predictions": f"{prefix}_planner_variant_predictions.parquet",
        "planner_variant_calibrator_summary": f"{prefix}_planner_variant_calibrator_summary.csv",
        "planner_variant_signal_metrics": f"{prefix}_planner_variant_signal_metrics.csv",
        "planner_variant_signal_tau_sweep": f"{prefix}_planner_variant_signal_tau_sweep.csv",
        "planner_variant_signal_tau_at_budget": f"{prefix}_planner_variant_signal_tau_at_budget.csv",
        "planner_variant_controller_metrics": f"{prefix}_planner_variant_controller_metrics.csv",
        "planner_variant_controller_step_trace": f"{prefix}_planner_variant_controller_step_trace.csv",
        "planner_variant_controller_at_budget": f"{prefix}_planner_variant_controller_at_budget.csv",
        "planner_variant_conformal_thresholds": f"{prefix}_planner_variant_conformal_thresholds.csv",
        "planner_variant_diagnosis": f"{prefix}_planner_variant_diagnosis.csv",
        "planner_variant_repo_reference": f"{prefix}_planner_variant_repo_reference.csv",
    }
    if not bundle.loaded_sources_df.empty:
        bundle.loaded_sources_df.to_csv(paths["planner_variant_loaded_sources"], index=False)
    if not bundle.predictions_df.empty:
        bundle.predictions_df.to_parquet(paths["planner_variant_predictions"], index=False)
    if not bundle.calibrator_df.empty:
        bundle.calibrator_df.to_csv(paths["planner_variant_calibrator_summary"], index=False)
    if not bundle.signal_metrics_df.empty:
        bundle.signal_metrics_df.to_csv(paths["planner_variant_signal_metrics"], index=False)
    if not bundle.signal_tau_sweep_df.empty:
        bundle.signal_tau_sweep_df.to_csv(paths["planner_variant_signal_tau_sweep"], index=False)
    if not bundle.signal_tau_at_budget_df.empty:
        bundle.signal_tau_at_budget_df.to_csv(paths["planner_variant_signal_tau_at_budget"], index=False)
    if not bundle.controller_metrics_df.empty:
        bundle.controller_metrics_df.to_csv(paths["planner_variant_controller_metrics"], index=False)
    if not bundle.controller_step_trace_df.empty:
        bundle.controller_step_trace_df.to_csv(paths["planner_variant_controller_step_trace"], index=False)
    if not bundle.controller_at_budget_df.empty:
        bundle.controller_at_budget_df.to_csv(paths["planner_variant_controller_at_budget"], index=False)
    if not bundle.conformal_threshold_df.empty:
        bundle.conformal_threshold_df.to_csv(paths["planner_variant_conformal_thresholds"], index=False)
    if not bundle.diagnosis_df.empty:
        bundle.diagnosis_df.to_csv(paths["planner_variant_diagnosis"], index=False)
    if not bundle.repo_reference_df.empty:
        bundle.repo_reference_df.to_csv(paths["planner_variant_repo_reference"], index=False)
    return paths


def run_planner_method_variant_audit(
    *,
    output_prefix: str,
    current_run_prefix: str,
    persist_root: str,
    focus_label: str = "failure_proxy_h15",
    risk_budget_tau: float = 0.20,
    analysis_run_prefix: str = "",
    analysis_run_prefixes: Sequence[str] = (),
    allow_previous_run_fallback: bool = True,
    max_discovered_runs: int = 50,
    max_runs_to_load: int = 8,
    tau_values: Sequence[float] = tuple(np.linspace(0.05, 0.80, 16).tolist()),
    local_tau_window: float = 0.05,
    uq_probability_bins: int = 15,
    bootstrap_samples: int = 300,
    bootstrap_seed: int = 17,
    controller_variant_keys: Sequence[str] = DEFAULT_CONTROLLER_VARIANTS,
    racp_lambda_values: Sequence[float] = DEFAULT_RACP_LAMBDAS,
    conformal_min_cal_rows: int = 80,
    conformal_min_accept_count: int = 20,
    conformal_shift_local: bool = False,
    reference_repo_root: str = "~/waymax_risk_uq_external_refs",
    write_artifacts: bool = True,
) -> PlannerMethodVariantAuditBundle:
    pred_df, loaded_prefixes, loaded_sources_df = load_multi_run_prediction_frame(
        current_run_prefix=str(current_run_prefix),
        persist_root=str(persist_root),
        analysis_run_prefix=str(analysis_run_prefix),
        analysis_run_prefixes=analysis_run_prefixes,
        allow_previous_run_fallback=bool(allow_previous_run_fallback),
        max_discovered_runs=int(max_discovered_runs),
        max_runs_to_load=int(max_runs_to_load),
    )
    if pred_df.empty:
        raise FileNotFoundError(
            "No prediction artifacts available. Run decision_audit_artifact_builder_colab.ipynb (or 00_probe/miscalibration_probe_colab.ipynb) first, or set analysis_run_prefix(es) to existing runs."
        )
    if str(focus_label) not in pred_df.columns:
        raise ValueError(f"Focus label missing from predictions: {focus_label!r}")

    work_df, cal_mask, eval_mask, _split_source = _prepare_splits(pred_df)
    raw_signals = _build_raw_signals(work_df, cal_mask=cal_mask)
    for required in ("top1", "entropy", "combo"):
        if required not in raw_signals:
            raise RuntimeError(f"Missing required signal: {required}")
    calibrated_df, calibrator_df, variant_to_column = _calibrate_signals(
        work_df,
        focus_label=str(focus_label),
        signals=raw_signals,
        cal_mask=cal_mask,
    )

    eval_df = calibrated_df.loc[np.asarray(eval_mask, dtype=bool)].copy()
    if eval_df.empty:
        eval_df = calibrated_df.copy()
    cal_df = calibrated_df.loc[np.asarray(cal_mask, dtype=bool)].copy()
    if cal_df.empty:
        cal_df = calibrated_df.copy()

    eval_df["y_true"] = (_safe_float(eval_df[focus_label], default=0.0) > 0.5).astype(int)
    cal_df["y_true"] = (_safe_float(cal_df[focus_label], default=0.0) > 0.5).astype(int)

    comfort_cost = np.zeros((len(eval_df),), dtype=float)
    if "max_abs_acc_h6" in eval_df.columns:
        comfort_cost += np.abs(_safe_float(eval_df["max_abs_acc_h6"], default=0.0))
    if "max_abs_jerk_h6" in eval_df.columns:
        comfort_cost += np.abs(_safe_float(eval_df["max_abs_jerk_h6"], default=0.0))
    eval_df["comfort_cost"] = comfort_cost

    signal_metrics_df, signal_tau_sweep_df, signal_tau_at_budget_df = evaluate_signal_variants(
        eval_df,
        focus_label=str(focus_label),
        variant_to_column=variant_to_column,
        risk_budget_tau=float(risk_budget_tau),
        tau_values=[float(x) for x in tau_values],
        local_tau_window=float(local_tau_window),
        n_bins=int(max(10, uq_probability_bins)),
        bootstrap_samples=int(max(0, bootstrap_samples)),
        bootstrap_seed=int(bootstrap_seed),
    )

    resolved_variants = _resolve_variant_columns(eval_df, [str(x) for x in controller_variant_keys])
    if len(resolved_variants) <= 0:
        raise RuntimeError("No controller risk variants resolved from available columns.")
    controller_metrics_df, controller_step_trace_df, controller_at_budget_df, conformal_threshold_df, diagnosis_df = (
        evaluate_controller_variants(
            eval_df,
            cal_df,
            variant_columns=resolved_variants,
            risk_budget_tau=float(risk_budget_tau),
            tau_values=[float(x) for x in tau_values],
            racp_lambda_values=[float(x) for x in racp_lambda_values],
            bootstrap_samples=int(max(0, bootstrap_samples)),
            bootstrap_seed=int(bootstrap_seed),
            conformal_min_cal_rows=int(max(1, conformal_min_cal_rows)),
            conformal_min_accept_count=int(max(1, conformal_min_accept_count)),
            conformal_shift_local=bool(conformal_shift_local),
        )
    )

    repo_reference_df = _repo_reference_table(reference_repo_root=str(reference_repo_root), repo_refs=DEFAULT_REPO_REFERENCES)
    bundle = PlannerMethodVariantAuditBundle(
        output_prefix=str(output_prefix),
        focus_label=str(focus_label),
        risk_budget_tau=float(risk_budget_tau),
        loaded_run_prefixes=loaded_prefixes,
        loaded_sources_df=loaded_sources_df,
        predictions_df=calibrated_df,
        calibrator_df=calibrator_df,
        signal_metrics_df=signal_metrics_df,
        signal_tau_sweep_df=signal_tau_sweep_df,
        signal_tau_at_budget_df=signal_tau_at_budget_df,
        controller_metrics_df=controller_metrics_df,
        controller_step_trace_df=controller_step_trace_df,
        controller_at_budget_df=controller_at_budget_df,
        conformal_threshold_df=conformal_threshold_df,
        diagnosis_df=diagnosis_df,
        repo_reference_df=repo_reference_df,
    )
    if bool(write_artifacts):
        bundle.artifact_paths = _export_artifacts(bundle)
    return bundle
