from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


RISK_EVENT_COLUMNS: Tuple[str, ...] = (
    "collision",
    "collision_proxy",
    "has_collision",
    "offroad",
    "offroad_proxy",
    "is_offroad",
    "route_violation",
    "route_violation_proxy",
    "is_route_violation",
)

FAILURE_COLUMNS: Tuple[str, ...] = (
    "failure_proxy",
    "failure_extended_proxy",
    "failure_strict_proxy",
)

NEAR_MISS_COLUMNS: Tuple[str, ...] = (
    "min_ttc",
    "ttc_min",
    "ttc",
)


@dataclass(frozen=True)
class RiskDefinition:
    name: str
    kind: str
    severity_col: str = "risk_sks"
    severity_quantile: float = 0.9
    near_miss_threshold: float = 2.0
    include_failure_proxy: bool = True


@dataclass(frozen=True)
class BlindspotDefinition:
    name: str
    kind: str
    cluster_col: str = "scenario_cluster"
    rare_quantile: float = 0.2
    rare_max_cluster_size: Optional[int] = None
    stability_event_col: str = "failure_proxy"
    stability_min_rate: float = 0.6
    stability_min_evals: int = 5


def standard_risk_definitions() -> list[RiskDefinition]:
    return [
        RiskDefinition(name="D1_hard_failure", kind="hard_events"),
        RiskDefinition(name="D2_hard_plus_near_miss", kind="hard_plus_near_miss"),
        RiskDefinition(
            name="D3_severity_quantile",
            kind="severity_quantile",
            severity_col="risk_sks",
            severity_quantile=0.9,
        ),
    ]


def standard_blindspot_definitions() -> list[BlindspotDefinition]:
    return [
        BlindspotDefinition(
            name="B1_rare_high_risk_template",
            kind="rare_high_risk_template",
            cluster_col="scenario_cluster",
            rare_quantile=0.2,
        ),
        BlindspotDefinition(
            name="B2_counterfactual_stable_high_risk",
            kind="counterfactual_stable_high_risk",
            stability_event_col="failure_proxy",
            stability_min_rate=0.6,
            stability_min_evals=5,
        ),
    ]


def _first_existing(df: pd.DataFrame, columns: Sequence[str]) -> Optional[str]:
    for col in columns:
        if col in df.columns:
            return col
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _binary_or(df: pd.DataFrame, columns: Sequence[str]) -> Tuple[pd.Series, list[str]]:
    hits: list[str] = []
    out = pd.Series(False, index=df.index)
    for col in columns:
        if col in df.columns:
            hits.append(col)
            out = out | (_to_numeric(df[col]).fillna(0.0) > 0.0)
    return out, hits


def _severity_threshold(
    df: pd.DataFrame,
    severity_col: str,
    severity_quantile: float,
) -> float:
    if severity_col not in df.columns:
        return float("nan")
    vals = _to_numeric(df[severity_col]).to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    q = float(np.clip(severity_quantile, 0.0, 1.0))
    return float(np.quantile(vals, q))


def derive_high_risk_flag(
    df: pd.DataFrame,
    risk_definition: RiskDefinition,
    severity_threshold: Optional[float] = None,
) -> tuple[pd.Series, Dict[str, Any]]:
    if df.empty:
        return pd.Series(False, index=df.index), {"severity_threshold": float("nan")}

    hard_flag, hard_cols = _binary_or(df, RISK_EVENT_COLUMNS)
    fail_flag = pd.Series(False, index=df.index)
    fail_cols: list[str] = []
    if risk_definition.include_failure_proxy:
        fail_flag, fail_cols = _binary_or(df, FAILURE_COLUMNS)
    hard = hard_flag | fail_flag

    near_col = _first_existing(df, NEAR_MISS_COLUMNS)
    if near_col is not None:
        near_vals = _to_numeric(df[near_col])
        near = np.isfinite(near_vals) & (near_vals <= float(risk_definition.near_miss_threshold))
    else:
        near = pd.Series(False, index=df.index)

    sev_col = str(risk_definition.severity_col)
    if severity_threshold is None:
        sev_thr = _severity_threshold(df, sev_col, float(risk_definition.severity_quantile))
    else:
        sev_thr = float(severity_threshold)
    if sev_col in df.columns and np.isfinite(sev_thr):
        sev_vals = _to_numeric(df[sev_col])
        sev = np.isfinite(sev_vals) & (sev_vals >= sev_thr)
    else:
        sev = pd.Series(False, index=df.index)

    kind = str(risk_definition.kind)
    if kind == "hard_events":
        risk_flag = hard if len(hard_cols) + len(fail_cols) > 0 else sev
    elif kind == "hard_plus_near_miss":
        base = hard if len(hard_cols) + len(fail_cols) > 0 else sev
        risk_flag = base | near
    elif kind == "severity_quantile":
        risk_flag = sev if sev_col in df.columns else hard
    else:
        raise ValueError(f"Unknown risk definition kind: {kind}")

    meta = {
        "risk_definition": risk_definition.name,
        "kind": kind,
        "severity_col": sev_col,
        "severity_threshold": float(sev_thr) if np.isfinite(sev_thr) else float("nan"),
        "hard_columns_used": hard_cols + fail_cols,
        "near_miss_column_used": near_col,
    }
    return risk_flag.astype(bool), meta


def derive_cluster_label(df: pd.DataFrame, cluster_col: str = "scenario_cluster") -> pd.Series:
    if cluster_col in df.columns:
        cluster = df[cluster_col].astype(str).fillna("missing")
    elif "scenario_id" in df.columns:
        cluster = df["scenario_id"].astype(str)
    else:
        cluster = pd.Series("unknown", index=df.index)
    return cluster


def _rare_cluster_mask(
    df: pd.DataFrame,
    cluster_label: pd.Series,
    rare_quantile: float,
    rare_max_cluster_size: Optional[int] = None,
) -> tuple[pd.Series, Dict[str, Any]]:
    if "scenario_id" in df.columns:
        dedup = pd.DataFrame(
            {
                "scenario_id": df["scenario_id"],
                "cluster": cluster_label,
            }
        ).drop_duplicates(subset=["scenario_id", "cluster"])
    else:
        dedup = pd.DataFrame({"cluster": cluster_label})

    counts = dedup.groupby("cluster", as_index=False).size().rename(columns={"size": "count"})
    if counts.empty:
        return pd.Series(False, index=df.index), {"rare_cluster_cutoff": 0}

    if rare_max_cluster_size is not None:
        cutoff = int(max(1, rare_max_cluster_size))
    else:
        q = float(np.clip(rare_quantile, 0.0, 1.0))
        cutoff = int(max(1, np.floor(np.quantile(counts["count"].to_numpy(dtype=float), q))))

    rare_clusters = set(counts.loc[counts["count"] <= cutoff, "cluster"].astype(str).tolist())
    mask = cluster_label.astype(str).isin(rare_clusters)
    meta = {
        "rare_cluster_cutoff": int(cutoff),
        "n_clusters": int(len(counts)),
        "n_rare_clusters": int(len(rare_clusters)),
    }
    return mask.astype(bool), meta


def _counterfactual_stability_map(
    trace_df: pd.DataFrame,
    event_flag: pd.Series,
    min_rate: float,
    min_evals: int,
) -> pd.DataFrame:
    required = ["scenario_id", "method"]
    if trace_df.empty or any(c not in trace_df.columns for c in required):
        return pd.DataFrame(columns=["scenario_id", "method", "is_counterfactual_stable"])

    tmp = trace_df.copy()
    tmp["_event"] = event_flag.astype(float)
    grp = (
        tmp.groupby(["scenario_id", "method"], as_index=False)
        .agg(
            cf_eval_count=("_event", "size"),
            cf_event_rate=("_event", "mean"),
        )
    )
    grp["is_counterfactual_stable"] = (
        (grp["cf_eval_count"] >= int(max(1, min_evals)))
        & (grp["cf_event_rate"] >= float(min_rate))
    ).astype(int)
    return grp


def derive_blindspot_flag(
    results_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    high_risk_flag: pd.Series,
    blindspot_definition: BlindspotDefinition,
    risk_definition: RiskDefinition,
    risk_severity_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    out = {
        "is_blindspot": pd.Series(False, index=results_df.index),
        "blindspot_unit": pd.Series(pd.NA, index=results_df.index, dtype="object"),
        "metadata": {
            "blindspot_definition": blindspot_definition.name,
            "kind": blindspot_definition.kind,
        },
        "counterfactual_stability_df": pd.DataFrame(),
    }

    if results_df.empty:
        return out

    kind = str(blindspot_definition.kind)
    high = high_risk_flag.astype(bool)

    if kind == "rare_high_risk_template":
        cluster = derive_cluster_label(results_df, cluster_col=blindspot_definition.cluster_col)
        rare_mask, rare_meta = _rare_cluster_mask(
            results_df,
            cluster_label=cluster,
            rare_quantile=float(blindspot_definition.rare_quantile),
            rare_max_cluster_size=blindspot_definition.rare_max_cluster_size,
        )
        out["is_blindspot"] = (high & rare_mask).astype(bool)
        out["blindspot_unit"] = cluster.astype(str)
        out["metadata"].update(rare_meta)
        return out

    if kind == "counterfactual_stable_high_risk":
        if trace_df.empty:
            out["is_blindspot"] = pd.Series(False, index=results_df.index)
            out["blindspot_unit"] = (
                results_df["scenario_id"].astype(str)
                if "scenario_id" in results_df.columns
                else pd.Series("unknown", index=results_df.index)
            )
            out["metadata"].update(
                {"stability_event_col": blindspot_definition.stability_event_col, "stable_count": 0}
            )
            return out

        trace_event = pd.Series(False, index=trace_df.index)
        event_col = str(blindspot_definition.stability_event_col)
        if event_col in trace_df.columns:
            trace_event = trace_event | (_to_numeric(trace_df[event_col]).fillna(0.0) > 0.0)
        trace_risk, _ = derive_high_risk_flag(
            trace_df,
            risk_definition=risk_definition,
            severity_threshold=risk_severity_threshold,
        )
        trace_event = (trace_event | trace_risk).astype(bool)

        stable = _counterfactual_stability_map(
            trace_df=trace_df,
            event_flag=trace_event,
            min_rate=float(blindspot_definition.stability_min_rate),
            min_evals=int(blindspot_definition.stability_min_evals),
        )
        out["counterfactual_stability_df"] = stable

        if {"scenario_id", "method"}.issubset(results_df.columns):
            join = results_df[["scenario_id", "method"]].merge(
                stable[["scenario_id", "method", "is_counterfactual_stable"]],
                on=["scenario_id", "method"],
                how="left",
            )
            stable_flag = join["is_counterfactual_stable"].fillna(0).astype(int) > 0
        else:
            stable_flag = pd.Series(False, index=results_df.index)

        out["is_blindspot"] = (high & stable_flag).astype(bool)
        if "scenario_id" in results_df.columns:
            out["blindspot_unit"] = results_df["scenario_id"].astype(str)
        else:
            out["blindspot_unit"] = pd.Series("unknown", index=results_df.index)

        out["metadata"].update(
            {
                "stability_event_col": event_col,
                "stability_min_rate": float(blindspot_definition.stability_min_rate),
                "stability_min_evals": int(blindspot_definition.stability_min_evals),
                "stable_count": int(stable["is_counterfactual_stable"].sum())
                if "is_counterfactual_stable" in stable.columns
                else 0,
            }
        )
        return out

    raise ValueError(f"Unknown blindspot definition kind: {kind}")
