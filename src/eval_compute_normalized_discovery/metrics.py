from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .definitions import (
    BlindspotDefinition,
    RiskDefinition,
    derive_blindspot_flag,
    derive_high_risk_flag,
    standard_blindspot_definitions,
    standard_risk_definitions,
)

DEFAULT_METHODS: list[str] = ["random", "risk_only", "surprise_only", "joint"]
DEFAULT_ORDER_COLS: list[str] = ["eval_index", "best_eval_index", "scenario_id"]
DEFAULT_COMPUTE_COLUMNS: list[str] = [
    "gpu_hours_used",
    "gpu_seconds_used",
    "gpu_seconds",
    "wall_time_seconds",
    "wall_seconds",
    "budget_units_used",
]


def _subset_methods(df: pd.DataFrame, methods: Optional[Sequence[str]]) -> pd.DataFrame:
    methods = list(methods or DEFAULT_METHODS)
    if "method" not in df.columns:
        raise ValueError("DataFrame must include a 'method' column.")
    return df[df["method"].isin(methods)].copy()


def _compute_units(
    df: pd.DataFrame,
    compute_col: Optional[str] = None,
) -> tuple[pd.Series, str]:
    if compute_col and compute_col in df.columns:
        chosen = compute_col
    else:
        chosen = next((c for c in DEFAULT_COMPUTE_COLUMNS if c in df.columns), "")

    if chosen == "":
        return pd.Series(1.0, index=df.index, dtype=float), "row_count_units"

    vals = pd.to_numeric(df[chosen], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "second" in chosen:
        vals = vals / 3600.0
    if "budget_units" in chosen:
        vals = vals.astype(float)
    return vals.astype(float), chosen


def _safe_unique_blindspot_units(grp: pd.DataFrame) -> int:
    sub = grp[grp["is_blindspot"] > 0]
    if sub.empty:
        return 0
    unit = sub["blindspot_unit"].dropna().astype(str)
    if unit.empty:
        if "scenario_id" in sub.columns:
            return int(sub["scenario_id"].nunique())
        return 0
    return int(unit.nunique())


def time_to_k_by_method(
    labeled_df: pd.DataFrame,
    k_values: Sequence[int] = (10, 25),
    event_col: str = "is_blindspot",
    unit_col: str = "blindspot_unit",
    compute_col: str = "compute_units",
    order_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    required = ["method", event_col, compute_col]
    missing = [c for c in required if c not in labeled_df.columns]
    if missing:
        raise ValueError(f"Missing columns for time_to_k computation: {missing}")

    order_cols = list(order_cols or DEFAULT_ORDER_COLS)
    rows: list[Dict[str, Any]] = []
    ks = sorted(set(int(max(1, k)) for k in k_values))
    if len(ks) == 0:
        return pd.DataFrame(columns=["method", "k", "compute_to_k", "rows_to_k", "reached_k"])

    for method, grp in labeled_df.groupby("method", sort=True):
        sort_cols = [c for c in order_cols if c in grp.columns]
        if len(sort_cols) == 0:
            sort_cols = ["scenario_id"] if "scenario_id" in grp.columns else []
        gg = grp.sort_values(sort_cols) if len(sort_cols) > 0 else grp.copy()

        seen: set[str] = set()
        cum_compute = 0.0
        progress: Dict[int, Dict[str, Any]] = {
            k: {"compute_to_k": np.nan, "rows_to_k": np.nan, "reached_k": 0} for k in ks
        }

        for row_idx, (_, row) in enumerate(gg.iterrows(), start=1):
            cu = float(pd.to_numeric(pd.Series([row.get(compute_col)]), errors="coerce").fillna(0.0).iloc[0])
            cum_compute += max(0.0, cu)

            is_event = bool(float(row.get(event_col, 0.0)) > 0.0)
            if is_event:
                unit = row.get(unit_col, None)
                if pd.isna(unit) or unit is None or str(unit).strip() == "":
                    if "scenario_id" in gg.columns:
                        unit = row.get("scenario_id", None)
                if unit is not None and not pd.isna(unit):
                    seen.add(str(unit))

            for k in ks:
                if progress[k]["reached_k"] == 0 and len(seen) >= k:
                    progress[k] = {
                        "compute_to_k": float(cum_compute),
                        "rows_to_k": int(row_idx),
                        "reached_k": 1,
                    }

        for k in ks:
            rows.append({"method": str(method), "k": int(k), **progress[k]})

    return pd.DataFrame(rows).sort_values(["method", "k"]).reset_index(drop=True)


def evaluate_discovery_grid(
    results_df: pd.DataFrame,
    trace_df: Optional[pd.DataFrame] = None,
    methods: Optional[Sequence[str]] = None,
    risk_definitions: Optional[Sequence[RiskDefinition]] = None,
    blindspot_definitions: Optional[Sequence[BlindspotDefinition]] = None,
    compute_col: Optional[str] = None,
    k_values: Sequence[int] = (10, 25),
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    trace_df = trace_df if isinstance(trace_df, pd.DataFrame) else pd.DataFrame()
    sub = _subset_methods(results_df, methods=methods)
    if sub.empty:
        return pd.DataFrame(), {}

    risk_definitions = list(risk_definitions or standard_risk_definitions())
    blindspot_definitions = list(blindspot_definitions or standard_blindspot_definitions())

    compute_units, compute_source = _compute_units(sub, compute_col=compute_col)
    sub = sub.copy()
    sub["compute_units"] = compute_units.astype(float)

    metric_rows: list[Dict[str, Any]] = []
    labeled_outputs: Dict[str, pd.DataFrame] = {}

    for risk_def in risk_definitions:
        high_risk, risk_meta = derive_high_risk_flag(sub, risk_definition=risk_def)
        sev_thr = risk_meta.get("severity_threshold", np.nan)

        for blind_def in blindspot_definitions:
            key = f"{risk_def.name}__{blind_def.name}"
            blind_payload = derive_blindspot_flag(
                results_df=sub,
                trace_df=trace_df,
                high_risk_flag=high_risk,
                blindspot_definition=blind_def,
                risk_definition=risk_def,
                risk_severity_threshold=float(sev_thr) if np.isfinite(sev_thr) else None,
            )
            labeled = sub.copy()
            labeled["is_high_risk"] = high_risk.astype(int)
            labeled["is_blindspot"] = blind_payload["is_blindspot"].astype(int)
            labeled["blindspot_unit"] = blind_payload["blindspot_unit"]
            labeled["risk_definition"] = risk_def.name
            labeled["blindspot_definition"] = blind_def.name
            labeled["definition_key"] = key
            labeled_outputs[key] = labeled

            grouped = (
                labeled.groupby("method", as_index=False)
                .agg(
                    n_rows=("scenario_id", "size"),
                    n_scenarios=("scenario_id", "nunique"),
                    high_risk_count=("is_high_risk", "sum"),
                    blindspot_count=("is_blindspot", "sum"),
                    total_compute=("compute_units", "sum"),
                )
            )
            unique_counts = (
                labeled.groupby("method")
                .apply(_safe_unique_blindspot_units)
                .rename("unique_blindspot_units")
                .reset_index()
            )
            grouped = grouped.merge(unique_counts, on="method", how="left")
            grouped["high_risk_rate"] = grouped["high_risk_count"] / np.maximum(grouped["n_rows"], 1.0)
            grouped["blindspot_rate"] = grouped["blindspot_count"] / np.maximum(grouped["n_rows"], 1.0)
            grouped["discovery_efficiency"] = grouped["unique_blindspot_units"] / np.maximum(
                grouped["total_compute"], 1e-9
            )
            grouped["blindspot_event_efficiency"] = grouped["blindspot_count"] / np.maximum(
                grouped["total_compute"], 1e-9
            )

            tt = time_to_k_by_method(
                labeled_df=labeled,
                k_values=k_values,
                event_col="is_blindspot",
                unit_col="blindspot_unit",
                compute_col="compute_units",
            )
            if not tt.empty:
                tt_wide = tt.pivot(index="method", columns="k", values="compute_to_k").add_prefix("compute_to_k_")
                tt_wide = tt_wide.reset_index()
                grouped = grouped.merge(tt_wide, on="method", how="left")

                tt_rows = tt.pivot(index="method", columns="k", values="rows_to_k").add_prefix("rows_to_k_")
                tt_rows = tt_rows.reset_index()
                grouped = grouped.merge(tt_rows, on="method", how="left")

            for _, row in grouped.iterrows():
                metric_rows.append(
                    {
                        "definition_key": key,
                        "risk_definition": risk_def.name,
                        "blindspot_definition": blind_def.name,
                        "method": row["method"],
                        "compute_source": compute_source,
                        "severity_threshold": float(risk_meta.get("severity_threshold", np.nan)),
                        **{k: row[k] for k in grouped.columns if k != "method"},
                    }
                )

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["risk_definition", "blindspot_definition", "method"]
    ).reset_index(drop=True)
    return metrics_df, labeled_outputs


def best_method_per_definition(
    metrics_df: pd.DataFrame,
    score_col: str = "discovery_efficiency",
) -> pd.DataFrame:
    required = ["definition_key", "method", score_col]
    missing = [c for c in required if c not in metrics_df.columns]
    if missing:
        raise ValueError(f"Missing columns for best-method computation: {missing}")
    if metrics_df.empty:
        return pd.DataFrame(columns=["definition_key", "method", score_col])

    rows = []
    for key, grp in metrics_df.groupby("definition_key", sort=True):
        gg = grp.sort_values(score_col, ascending=False)
        top = gg.iloc[0]
        rows.append({"definition_key": key, "method": top["method"], score_col: top[score_col]})
    return pd.DataFrame(rows).sort_values("definition_key").reset_index(drop=True)


def method_score_table(
    metrics_df: pd.DataFrame,
    score_col: str = "discovery_efficiency",
) -> pd.DataFrame:
    required = ["method", score_col]
    missing = [c for c in required if c not in metrics_df.columns]
    if missing:
        raise ValueError(f"Missing columns for method-score table: {missing}")
    if metrics_df.empty:
        return pd.DataFrame()

    out = (
        metrics_df.groupby("method", as_index=False)
        .agg(
            n_definitions=("definition_key", "nunique"),
            mean_score=(score_col, "mean"),
            median_score=(score_col, "median"),
            std_score=(score_col, "std"),
        )
        .sort_values("mean_score", ascending=False)
        .reset_index(drop=True)
    )
    return out
