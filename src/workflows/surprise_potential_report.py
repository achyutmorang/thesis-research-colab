from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .surprise_potential_flow import SurprisePotentialSweepBundle

METHOD_ORDER = ("random", "risk_only", "surprise_only", "joint")
SURPRISE_COLS = ("delta_surprise", "delta_surprise_pd", "surprise_pd")
FAILURE_COLS = ("failure_proxy", "failure_extended_proxy", "failure_strict_proxy")
BLINDSPOT_COLS = ("blind_spot_proxy_hit", "bsdr_proxy_hit")


@dataclass
class SurprisePotentialReportFrames:
    metric_method_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    metric_method_rollup_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    metric_rank_df: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class SurprisePotentialReportExport:
    frames: SurprisePotentialReportFrames = field(default_factory=SurprisePotentialReportFrames)
    table_paths: Dict[str, str] = field(default_factory=dict)
    plot_paths: Dict[str, str] = field(default_factory=dict)
    manifest_path: str = ""


@dataclass
class SurprisePotentialRigorousEvalFrames:
    per_scenario_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    paired_delta_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    metric_health_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    rigorous_rank_df: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class SurprisePotentialReportLoad:
    exists: bool = False
    is_usable: bool = False
    output_root: str = ""
    manifest: Dict[str, Any] = field(default_factory=dict)
    frames: SurprisePotentialReportFrames = field(default_factory=SurprisePotentialReportFrames)
    rigorous_frames: SurprisePotentialRigorousEvalFrames = field(default_factory=SurprisePotentialRigorousEvalFrames)
    table_paths: Dict[str, str] = field(default_factory=dict)
    plot_paths: Dict[str, str] = field(default_factory=dict)
    integrity_issues: Tuple[str, ...] = field(default_factory=tuple)


def _pick_existing_column(columns: Iterable[str], candidates: Iterable[str], fallback: str = "") -> str:
    present = set(str(c) for c in columns)
    for name in candidates:
        key = str(name)
        if key in present:
            return key
    return str(fallback)


def _safe_mean(series: pd.Series) -> float:
    if not isinstance(series, pd.Series) or len(series) <= 0:
        return np.nan
    out = float(np.nanmean(pd.to_numeric(series, errors="coerce")))
    return out


def _metric_label(metric: str, family: str) -> str:
    return f"{family}:{metric}"


def _aggregate_metric_method_rows(bundle: SurprisePotentialSweepBundle) -> pd.DataFrame:
    rows = []
    for _, run_bundle in dict(bundle.metric_runs).items():
        metric = str(run_bundle.metric)
        family = str(run_bundle.counterfactual_family)
        run_tag = str(run_bundle.run_tag)
        run_prefix = str(run_bundle.run_prefix)
        run_df = run_bundle.main_loop_bundle.closedloop_results_df
        if not isinstance(run_df, pd.DataFrame) or len(run_df) <= 0:
            continue
        if "method" not in run_df.columns:
            continue

        surprise_col = _pick_existing_column(run_df.columns, SURPRISE_COLS, fallback="delta_surprise")
        failure_col = _pick_existing_column(run_df.columns, FAILURE_COLS, fallback="")
        blindspot_col = _pick_existing_column(run_df.columns, BLINDSPOT_COLS, fallback="")

        for method in METHOD_ORDER:
            sub = run_df[run_df["method"] == method]
            if len(sub) <= 0:
                continue
            row = {
                "metric": metric,
                "counterfactual_family": family,
                "metric_label": _metric_label(metric=metric, family=family),
                "run_tag": run_tag,
                "run_prefix": run_prefix,
                "method": method,
                "n_rows": int(len(sub)),
                "n_scenarios": int(sub["scenario_id"].nunique()) if "scenario_id" in sub.columns else 0,
                "mean_risk_sks": _safe_mean(sub["risk_sks"]) if "risk_sks" in sub.columns else np.nan,
                "mean_surprise": _safe_mean(sub[surprise_col]) if surprise_col in sub.columns else np.nan,
                "failure_rate": _safe_mean(sub[failure_col]) if failure_col in sub.columns else np.nan,
                "blind_spot_rate": _safe_mean(sub[blindspot_col]) if blindspot_col in sub.columns else np.nan,
                "q1_rate": _safe_mean(sub["q1_hit"]) if "q1_hit" in sub.columns else np.nan,
                "q4_rate": _safe_mean(sub["q4_hit"]) if "q4_hit" in sub.columns else np.nan,
                "mean_objective_gain": _safe_mean(sub["objective_gain"]) if "objective_gain" in sub.columns else np.nan,
                "mean_budget_used": _safe_mean(sub["budget_units_used"]) if "budget_units_used" in sub.columns else np.nan,
                "feasibility_violation_rate": _safe_mean(sub["feasibility_violation"]) if "feasibility_violation" in sub.columns else np.nan,
            }
            rows.append(row)
    if len(rows) <= 0:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _build_metric_method_rollup(metric_method_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(metric_method_df, pd.DataFrame) or len(metric_method_df) <= 0:
        return pd.DataFrame()
    value_cols = [
        "n_rows",
        "n_scenarios",
        "mean_risk_sks",
        "mean_surprise",
        "failure_rate",
        "blind_spot_rate",
        "q1_rate",
        "q4_rate",
        "mean_objective_gain",
        "mean_budget_used",
        "feasibility_violation_rate",
    ]
    use_cols = [c for c in value_cols if c in metric_method_df.columns]
    rollup = (
        metric_method_df
        .groupby(["metric", "counterfactual_family", "metric_label", "method"], as_index=False)[use_cols]
        .mean(numeric_only=True)
    )
    return rollup


def _extract_method_value(group: pd.DataFrame, method: str, col: str) -> float:
    if col not in group.columns:
        return np.nan
    sub = group[group["method"] == method]
    if len(sub) <= 0:
        return np.nan
    return float(sub.iloc[0][col])


def _build_metric_rank(metric_method_rollup_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(metric_method_rollup_df, pd.DataFrame) or len(metric_method_rollup_df) <= 0:
        return pd.DataFrame()
    rows = []
    group_cols = ["metric", "counterfactual_family", "metric_label"]
    for keys, sub in metric_method_rollup_df.groupby(group_cols, sort=False):
        metric, family, metric_label = keys
        random_failure = _extract_method_value(sub, "random", "failure_rate")
        risk_failure = _extract_method_value(sub, "risk_only", "failure_rate")
        joint_failure = _extract_method_value(sub, "joint", "failure_rate")

        random_bs = _extract_method_value(sub, "random", "blind_spot_rate")
        risk_bs = _extract_method_value(sub, "risk_only", "blind_spot_rate")
        joint_bs = _extract_method_value(sub, "joint", "blind_spot_rate")

        random_obj = _extract_method_value(sub, "random", "mean_objective_gain")
        risk_obj = _extract_method_value(sub, "risk_only", "mean_objective_gain")
        joint_obj = _extract_method_value(sub, "joint", "mean_objective_gain")

        joint_vs_random_failure_gain = float(joint_failure - random_failure) if np.isfinite(joint_failure) and np.isfinite(random_failure) else np.nan
        joint_vs_risk_failure_gain = float(joint_failure - risk_failure) if np.isfinite(joint_failure) and np.isfinite(risk_failure) else np.nan
        joint_vs_random_bs_gain = float(joint_bs - random_bs) if np.isfinite(joint_bs) and np.isfinite(random_bs) else np.nan
        joint_vs_risk_bs_gain = float(joint_bs - risk_bs) if np.isfinite(joint_bs) and np.isfinite(risk_bs) else np.nan
        joint_vs_random_obj_gain = float(joint_obj - random_obj) if np.isfinite(joint_obj) and np.isfinite(random_obj) else np.nan
        joint_vs_risk_obj_gain = float(joint_obj - risk_obj) if np.isfinite(joint_obj) and np.isfinite(risk_obj) else np.nan

        score_terms = [
            joint_vs_random_failure_gain,
            joint_vs_risk_failure_gain,
            joint_vs_random_bs_gain,
            joint_vs_risk_bs_gain,
            joint_vs_random_obj_gain,
            joint_vs_risk_obj_gain,
        ]
        finite_terms = [float(v) for v in score_terms if np.isfinite(v)]
        composite_score = float(np.mean(finite_terms)) if len(finite_terms) > 0 else np.nan
        rows.append(
            {
                "metric": str(metric),
                "counterfactual_family": str(family),
                "metric_label": str(metric_label),
                "joint_vs_random_failure_gain": joint_vs_random_failure_gain,
                "joint_vs_risk_failure_gain": joint_vs_risk_failure_gain,
                "joint_vs_random_blind_spot_gain": joint_vs_random_bs_gain,
                "joint_vs_risk_blind_spot_gain": joint_vs_risk_bs_gain,
                "joint_vs_random_objective_gain": joint_vs_random_obj_gain,
                "joint_vs_risk_objective_gain": joint_vs_risk_obj_gain,
                "composite_score": composite_score,
            }
        )
    if len(rows) <= 0:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("composite_score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def build_surprise_potential_report_frames(bundle: SurprisePotentialSweepBundle) -> SurprisePotentialReportFrames:
    metric_method_df = _aggregate_metric_method_rows(bundle=bundle)
    metric_method_rollup_df = _build_metric_method_rollup(metric_method_df=metric_method_df)
    metric_rank_df = _build_metric_rank(metric_method_rollup_df=metric_method_rollup_df)
    return SurprisePotentialReportFrames(
        metric_method_df=metric_method_df,
        metric_method_rollup_df=metric_method_rollup_df,
        metric_rank_df=metric_rank_df,
    )


def _plot_metric_method_bars(
    metric_method_rollup_df: pd.DataFrame,
    *,
    value_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> bool:
    if (not isinstance(metric_method_rollup_df, pd.DataFrame)) or (len(metric_method_rollup_df) <= 0):
        return False
    if value_col not in metric_method_rollup_df.columns:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plot_df = (
        metric_method_rollup_df
        .pivot_table(index="metric_label", columns="method", values=value_col, aggfunc="mean")
        .reindex(columns=list(METHOD_ORDER))
        .sort_index()
    )
    if len(plot_df) <= 0:
        return False

    fig_w = max(9.0, 1.6 * len(plot_df.index))
    fig, ax = plt.subplots(figsize=(fig_w, 5.4))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_xlabel("metric (family:metric)")
    ax.set_ylabel(str(ylabel))
    ax.set_title(str(title))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="method", loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _plot_metric_rank(metric_rank_df: pd.DataFrame, out_path: Path) -> bool:
    if (not isinstance(metric_rank_df, pd.DataFrame)) or (len(metric_rank_df) <= 0):
        return False
    if "composite_score" not in metric_rank_df.columns:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plot_df = metric_rank_df.copy()
    fig_w = max(8.0, 1.3 * len(plot_df.index))
    fig, ax = plt.subplots(figsize=(fig_w, 5.2))
    ax.bar(plot_df["metric_label"], plot_df["composite_score"])
    ax.set_xlabel("metric (family:metric)")
    ax.set_ylabel("composite_gain_score")
    ax.set_title("Metric Ranking By Joint-Method Gains")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True, default=str)


def _safe_read_csv(path: str) -> pd.DataFrame:
    p = Path(str(path))
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _resolve_manifest_paths(root: Path, raw_paths: Mapping[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not isinstance(raw_paths, Mapping):
        return out
    for key, value in dict(raw_paths).items():
        p_raw = str(value or "").strip()
        if not p_raw:
            continue
        p = Path(p_raw).expanduser()
        if not p.is_absolute():
            p = (root / p).resolve()
        out[str(key)] = str(p)
    return out


def _relativize_manifest_paths(root: Path, abs_paths: Mapping[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not isinstance(abs_paths, Mapping):
        return out
    for key, value in dict(abs_paths).items():
        p_raw = str(value or "").strip()
        if not p_raw:
            continue
        p = Path(p_raw).expanduser()
        try:
            rel = p.resolve().relative_to(root.resolve())
            out[str(key)] = str(rel)
        except Exception:
            out[str(key)] = str(p)
    return out


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int = 1200,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, mean, mean
    n = int(arr.size)
    reps = int(max(200, n_bootstrap))
    sample_idx = rng.integers(0, n, size=(reps, n))
    boot = np.mean(arr[sample_idx], axis=1)
    low = float(np.quantile(boot, alpha / 2.0))
    high = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return mean, low, high


def _load_per_scenario_from_artifact_index(artifact_index_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(artifact_index_df, pd.DataFrame) or len(artifact_index_df) <= 0:
        return pd.DataFrame()
    need_cols = {"metric", "counterfactual_family", "run_tag", "run_prefix", "artifact_key", "artifact_path"}
    if not need_cols.issubset(set(artifact_index_df.columns)):
        return pd.DataFrame()

    rows = []
    filtered = artifact_index_df[artifact_index_df["artifact_key"] == "per_scenario_results"].copy()
    for _, meta in filtered.iterrows():
        csv_path = str(meta["artifact_path"])
        df = _safe_read_csv(csv_path)
        if len(df) <= 0:
            continue
        if ("scenario_id" not in df.columns) or ("method" not in df.columns):
            continue
        df = df.copy()
        df["metric"] = str(meta["metric"])
        df["counterfactual_family"] = str(meta["counterfactual_family"])
        df["metric_label"] = _metric_label(metric=str(meta["metric"]), family=str(meta["counterfactual_family"]))
        df["run_tag"] = str(meta["run_tag"])
        df["run_prefix"] = str(meta["run_prefix"])
        rows.append(df)
    if len(rows) <= 0:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _paired_delta_records(
    per_scenario_df: pd.DataFrame,
    *,
    n_bootstrap: int = 1200,
    random_seed: int = 17,
) -> pd.DataFrame:
    if not isinstance(per_scenario_df, pd.DataFrame) or len(per_scenario_df) <= 0:
        return pd.DataFrame()

    endpoint_specs = [
        ("failure_rate", _pick_existing_column(per_scenario_df.columns, FAILURE_COLS, fallback=""), 1.0),
        ("blind_spot_rate", _pick_existing_column(per_scenario_df.columns, BLINDSPOT_COLS, fallback=""), 1.0),
        ("objective_gain", "objective_gain" if "objective_gain" in per_scenario_df.columns else "", 1.0),
        ("risk_increase", "risk_sks" if "risk_sks" in per_scenario_df.columns else "", 1.0),
    ]
    endpoint_specs = [(name, col, sign) for name, col, sign in endpoint_specs if str(col).strip()]
    if len(endpoint_specs) <= 0:
        return pd.DataFrame()

    pair_specs = [
        ("joint", "random"),
        ("joint", "risk_only"),
        ("surprise_only", "random"),
        ("risk_only", "random"),
    ]
    out_rows = []
    rng = np.random.default_rng(int(random_seed))

    group_cols = ["metric", "counterfactual_family", "metric_label"]
    for keys, g in per_scenario_df.groupby(group_cols, sort=False):
        metric, family, metric_label = keys
        for endpoint_name, col, sign in endpoint_specs:
            wide = (
                g.pivot_table(index="scenario_id", columns="method", values=col, aggfunc="mean")
                .reindex(columns=list(METHOD_ORDER))
            )
            if len(wide) <= 0:
                continue
            for method_a, method_b in pair_specs:
                if (method_a not in wide.columns) or (method_b not in wide.columns):
                    continue
                arr_a = pd.to_numeric(wide[method_a], errors="coerce")
                arr_b = pd.to_numeric(wide[method_b], errors="coerce")
                mask = np.isfinite(arr_a.values) & np.isfinite(arr_b.values)
                if not bool(np.any(mask)):
                    continue
                delta = float(sign) * (arr_a.values[mask] - arr_b.values[mask])
                mean, ci_low, ci_high = _bootstrap_mean_ci(
                    delta,
                    rng=rng,
                    n_bootstrap=int(n_bootstrap),
                    alpha=0.05,
                )
                win_rate = float(np.mean(delta > 0.0))
                out_rows.append(
                    {
                        "metric": str(metric),
                        "counterfactual_family": str(family),
                        "metric_label": str(metric_label),
                        "endpoint": str(endpoint_name),
                        "value_col": str(col),
                        "method_a": str(method_a),
                        "method_b": str(method_b),
                        "n_pairs": int(delta.size),
                        "delta_mean": float(mean),
                        "delta_ci_low": float(ci_low),
                        "delta_ci_high": float(ci_high),
                        "win_rate": float(win_rate),
                    }
                )
    if len(out_rows) <= 0:
        return pd.DataFrame()
    return pd.DataFrame(out_rows)


def _metric_health_from_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(summary_df, pd.DataFrame) or len(summary_df) <= 0:
        return pd.DataFrame()
    required = {"metric", "quick_probe_collapsed", "ready_for_main_loop", "ready_for_optimization", "main_loop_executed"}
    if not required.issubset(set(summary_df.columns)):
        return pd.DataFrame()
    out = (
        summary_df
        .groupby(["metric", "counterfactual_family"], as_index=False)
        .agg(
            n_runs=("metric", "size"),
            quick_probe_collapsed_rate=("quick_probe_collapsed", "mean"),
            ready_for_main_loop_rate=("ready_for_main_loop", "mean"),
            ready_for_optimization_rate=("ready_for_optimization", "mean"),
            main_loop_executed_rate=("main_loop_executed", "mean"),
        )
    )
    return out


def _rigorous_rank_from_pairwise(paired_delta_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(paired_delta_df, pd.DataFrame) or len(paired_delta_df) <= 0:
        return pd.DataFrame()
    sub = paired_delta_df[
        (paired_delta_df["method_a"] == "joint")
        & (paired_delta_df["method_b"] == "risk_only")
        & (paired_delta_df["endpoint"].isin(["failure_rate", "blind_spot_rate", "objective_gain", "risk_increase", "risk_reduction"]))
    ].copy()
    if len(sub) <= 0:
        return pd.DataFrame()

    # Backward-compatibility: older exports used risk_reduction with inverted sign.
    is_risk_reduction = sub["endpoint"] == "risk_reduction"
    if bool(is_risk_reduction.any()):
        sub.loc[is_risk_reduction, "delta_mean"] = -pd.to_numeric(sub.loc[is_risk_reduction, "delta_mean"], errors="coerce")
        sub.loc[is_risk_reduction, "delta_ci_low"] = -pd.to_numeric(sub.loc[is_risk_reduction, "delta_ci_low"], errors="coerce")
        sub.loc[is_risk_reduction, "delta_ci_high"] = -pd.to_numeric(sub.loc[is_risk_reduction, "delta_ci_high"], errors="coerce")
        sub.loc[is_risk_reduction, "endpoint"] = "risk_increase"

    out = (
        sub.groupby(["metric", "counterfactual_family", "metric_label"], as_index=False)
        .agg(
            rigorous_score=("delta_mean", "mean"),
            rigorous_score_lower_mean=("delta_ci_low", "mean"),
            rigorous_score_lower=("delta_ci_low", "min"),
            avg_joint_win_rate=("win_rate", "mean"),
            n_endpoints=("endpoint", "nunique"),
        )
        .sort_values(["rigorous_score_lower", "rigorous_score"], ascending=False)
        .reset_index(drop=True)
    )
    out.insert(0, "rigorous_rank", np.arange(1, len(out) + 1))
    return out


def build_surprise_potential_rigorous_eval(
    *,
    artifact_index_df: pd.DataFrame,
    summary_df: Optional[pd.DataFrame] = None,
    n_bootstrap: int = 1200,
    random_seed: int = 17,
) -> SurprisePotentialRigorousEvalFrames:
    per_scenario_df = _load_per_scenario_from_artifact_index(artifact_index_df)
    paired_delta_df = _paired_delta_records(
        per_scenario_df=per_scenario_df,
        n_bootstrap=int(n_bootstrap),
        random_seed=int(random_seed),
    )
    metric_health_df = _metric_health_from_summary(summary_df if isinstance(summary_df, pd.DataFrame) else pd.DataFrame())
    rigorous_rank_df = _rigorous_rank_from_pairwise(paired_delta_df)
    return SurprisePotentialRigorousEvalFrames(
        per_scenario_df=per_scenario_df,
        paired_delta_df=paired_delta_df,
        metric_health_df=metric_health_df,
        rigorous_rank_df=rigorous_rank_df,
    )


def _plot_joint_vs_risk_delta(paired_delta_df: pd.DataFrame, out_path: Path) -> bool:
    if not isinstance(paired_delta_df, pd.DataFrame) or len(paired_delta_df) <= 0:
        return False
    sub = paired_delta_df[
        (paired_delta_df["method_a"] == "joint")
        & (paired_delta_df["method_b"] == "risk_only")
    ].copy()
    if len(sub) <= 0:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    use = sub[sub["endpoint"].isin(["failure_rate", "blind_spot_rate", "objective_gain", "risk_increase", "risk_reduction"])].copy()
    if len(use) <= 0:
        return False

    # Backward-compatibility: older exports used risk_reduction with inverted sign.
    mask = use["endpoint"] == "risk_reduction"
    if bool(mask.any()):
        use.loc[mask, "delta_mean"] = -pd.to_numeric(use.loc[mask, "delta_mean"], errors="coerce")
        use.loc[mask, "endpoint"] = "risk_increase"
    pivot = (
        use.pivot_table(index="metric_label", columns="endpoint", values="delta_mean", aggfunc="mean")
        .sort_index()
    )
    if len(pivot) <= 0:
        return False
    fig_w = max(9.0, 1.7 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(fig_w, 5.6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("metric (family:metric)")
    ax.set_ylabel("joint - risk_only (higher is better)")
    ax.set_title("Joint vs Risk-Only: Paired Delta Means")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="endpoint", loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def load_surprise_potential_report(output_root: str) -> SurprisePotentialReportLoad:
    root = Path(str(output_root)).expanduser()
    manifest_path = root / "report_manifest.json"
    if not manifest_path.exists():
        return SurprisePotentialReportLoad(exists=False, output_root=str(root))

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        manifest = {}

    table_paths = _resolve_manifest_paths(root, manifest.get("table_paths", {})) if isinstance(manifest, dict) else {}
    plot_paths = _resolve_manifest_paths(root, manifest.get("plot_paths", {})) if isinstance(manifest, dict) else {}

    metric_method_df = _safe_read_csv(table_paths.get("metric_method_rows", ""))
    metric_method_rollup_df = _safe_read_csv(table_paths.get("metric_method_rollup", ""))
    metric_rank_df = _safe_read_csv(table_paths.get("metric_rank", ""))
    summary_df = _safe_read_csv(table_paths.get("sweep_summary", ""))
    artifact_index_df = _safe_read_csv(table_paths.get("artifact_index", ""))
    paired_delta_df = _safe_read_csv(table_paths.get("rigorous_pairwise_delta", ""))
    metric_health_df = _safe_read_csv(table_paths.get("rigorous_metric_health", ""))
    rigorous_rank_df = _safe_read_csv(table_paths.get("rigorous_metric_rank", ""))

    rigorous_per_scenario_df = pd.DataFrame()
    if paired_delta_df.empty or metric_health_df.empty or rigorous_rank_df.empty:
        recomputed = build_surprise_potential_rigorous_eval(
            artifact_index_df=artifact_index_df,
            summary_df=summary_df,
            n_bootstrap=int(manifest.get("rigorous_bootstrap", 1200)) if isinstance(manifest, dict) else 1200,
            random_seed=int(manifest.get("rigorous_seed", 17)) if isinstance(manifest, dict) else 17,
        )
        rigorous_per_scenario_df = recomputed.per_scenario_df
        if paired_delta_df.empty:
            paired_delta_df = recomputed.paired_delta_df
        if metric_health_df.empty:
            metric_health_df = recomputed.metric_health_df
        if rigorous_rank_df.empty:
            rigorous_rank_df = recomputed.rigorous_rank_df

    integrity_issues = []
    if not isinstance(manifest, dict):
        integrity_issues.append("manifest_parse_failed")
    if metric_method_rollup_df.empty:
        integrity_issues.append("metric_method_rollup_missing_or_empty")
    if metric_rank_df.empty:
        integrity_issues.append("metric_rank_missing_or_empty")
    is_usable = (len(integrity_issues) == 0)

    return SurprisePotentialReportLoad(
        exists=True,
        is_usable=bool(is_usable),
        output_root=str(root),
        manifest=manifest if isinstance(manifest, dict) else {},
        frames=SurprisePotentialReportFrames(
            metric_method_df=metric_method_df,
            metric_method_rollup_df=metric_method_rollup_df,
            metric_rank_df=metric_rank_df,
        ),
        rigorous_frames=SurprisePotentialRigorousEvalFrames(
            per_scenario_df=rigorous_per_scenario_df,
            paired_delta_df=paired_delta_df,
            metric_health_df=metric_health_df,
            rigorous_rank_df=rigorous_rank_df,
        ),
        table_paths=table_paths,
        plot_paths=plot_paths,
        integrity_issues=tuple(str(x) for x in integrity_issues),
    )


def export_surprise_potential_report(
    bundle: SurprisePotentialSweepBundle,
    output_root: str,
    *,
    write_plots: bool = True,
    write_rigorous_eval: bool = True,
    rigorous_bootstrap: int = 1200,
    rigorous_seed: int = 17,
) -> SurprisePotentialReportExport:
    frames = build_surprise_potential_report_frames(bundle=bundle)

    root = Path(str(output_root)).expanduser()
    tables_dir = root / "tables"
    plots_dir = root / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if bool(write_plots):
        plots_dir.mkdir(parents=True, exist_ok=True)

    table_paths: Dict[str, str] = {}
    plot_paths: Dict[str, str] = {}

    if len(frames.metric_method_df) > 0:
        p = tables_dir / "metric_method_rows.csv"
        frames.metric_method_df.to_csv(p, index=False)
        table_paths["metric_method_rows"] = str(p)
    if len(frames.metric_method_rollup_df) > 0:
        p = tables_dir / "metric_method_rollup.csv"
        frames.metric_method_rollup_df.to_csv(p, index=False)
        table_paths["metric_method_rollup"] = str(p)
    if len(frames.metric_rank_df) > 0:
        p = tables_dir / "metric_rank.csv"
        frames.metric_rank_df.to_csv(p, index=False)
        table_paths["metric_rank"] = str(p)
    if isinstance(bundle.summary_df, pd.DataFrame) and len(bundle.summary_df) > 0:
        p = tables_dir / "sweep_summary.csv"
        bundle.summary_df.to_csv(p, index=False)
        table_paths["sweep_summary"] = str(p)
    if isinstance(bundle.errors_df, pd.DataFrame) and len(bundle.errors_df) > 0:
        p = tables_dir / "sweep_errors.csv"
        bundle.errors_df.to_csv(p, index=False)
        table_paths["sweep_errors"] = str(p)
    if isinstance(bundle.artifacts_df, pd.DataFrame) and len(bundle.artifacts_df) > 0:
        p = tables_dir / "artifact_index.csv"
        bundle.artifacts_df.to_csv(p, index=False)
        table_paths["artifact_index"] = str(p)

    if bool(write_rigorous_eval):
        rigorous = build_surprise_potential_rigorous_eval(
            artifact_index_df=bundle.artifacts_df if isinstance(bundle.artifacts_df, pd.DataFrame) else pd.DataFrame(),
            summary_df=bundle.summary_df if isinstance(bundle.summary_df, pd.DataFrame) else pd.DataFrame(),
            n_bootstrap=int(rigorous_bootstrap),
            random_seed=int(rigorous_seed),
        )
        if len(rigorous.paired_delta_df) > 0:
            p = tables_dir / "rigorous_pairwise_delta.csv"
            rigorous.paired_delta_df.to_csv(p, index=False)
            table_paths["rigorous_pairwise_delta"] = str(p)
        if len(rigorous.metric_health_df) > 0:
            p = tables_dir / "rigorous_metric_health.csv"
            rigorous.metric_health_df.to_csv(p, index=False)
            table_paths["rigorous_metric_health"] = str(p)
        if len(rigorous.rigorous_rank_df) > 0:
            p = tables_dir / "rigorous_metric_rank.csv"
            rigorous.rigorous_rank_df.to_csv(p, index=False)
            table_paths["rigorous_metric_rank"] = str(p)

    if bool(write_plots):
        bar_specs = [
            ("mean_risk_sks", "mean_risk_sks", "Risk (mean) by Metric and Method", "risk_by_metric_method"),
            ("failure_rate", "failure_rate", "Failure Rate by Metric and Method", "failure_by_metric_method"),
            ("blind_spot_rate", "blind_spot_rate", "Blind-Spot Proxy Rate by Metric and Method", "blindspot_by_metric_method"),
            ("mean_objective_gain", "objective_gain", "Objective Gain by Metric and Method", "objective_gain_by_metric_method"),
        ]
        for col, ylabel, title, stem in bar_specs:
            p = plots_dir / f"{stem}.png"
            ok = _plot_metric_method_bars(
                frames.metric_method_rollup_df,
                value_col=col,
                ylabel=ylabel,
                title=title,
                out_path=p,
            )
            if ok:
                plot_paths[stem] = str(p)

        rank_path = plots_dir / "metric_rank_composite.png"
        if _plot_metric_rank(frames.metric_rank_df, rank_path):
            plot_paths["metric_rank_composite"] = str(rank_path)
        if bool(write_rigorous_eval):
            rigorous_df = _safe_read_csv(table_paths.get("rigorous_pairwise_delta", ""))
            p = plots_dir / "rigorous_joint_vs_risk_delta.png"
            if _plot_joint_vs_risk_delta(rigorous_df, p):
                plot_paths["rigorous_joint_vs_risk_delta"] = str(p)

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_version": 2,
        "output_root": str(root),
        "n_metric_runs": int(len(bundle.metric_runs)),
        "n_summary_rows": int(len(bundle.summary_df)) if isinstance(bundle.summary_df, pd.DataFrame) else 0,
        "n_error_rows": int(len(bundle.errors_df)) if isinstance(bundle.errors_df, pd.DataFrame) else 0,
        "n_metric_method_rows": int(len(frames.metric_method_df)),
        "n_metric_rank_rows": int(len(frames.metric_rank_df)),
        "methods": sorted(set(str(x) for x in frames.metric_method_df.get("method", pd.Series(dtype=object)).dropna().tolist())) if len(frames.metric_method_df) > 0 else [],
        "metrics": sorted(set(str(x) for x in frames.metric_method_df.get("metric", pd.Series(dtype=object)).dropna().tolist())) if len(frames.metric_method_df) > 0 else [],
        "counterfactual_families": sorted(set(str(x) for x in frames.metric_method_df.get("counterfactual_family", pd.Series(dtype=object)).dropna().tolist())) if len(frames.metric_method_df) > 0 else [],
        "write_rigorous_eval": bool(write_rigorous_eval),
        "rigorous_bootstrap": int(rigorous_bootstrap),
        "rigorous_seed": int(rigorous_seed),
        "table_paths": _relativize_manifest_paths(root, table_paths),
        "plot_paths": _relativize_manifest_paths(root, plot_paths),
    }
    manifest_path = root / "report_manifest.json"
    _write_json(manifest_path, manifest)

    return SurprisePotentialReportExport(
        frames=frames,
        table_paths=table_paths,
        plot_paths=plot_paths,
        manifest_path=str(manifest_path),
    )
