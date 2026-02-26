from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import pandas as pd

from src.eval.io import discover_run_prefixes, load_run_artifacts, select_default_run_prefix


@dataclass
class CounterfactualRunData:
    candidates_df: pd.DataFrame
    selected_run_prefix: str
    artifact_paths: Dict[str, str]
    results_df: pd.DataFrame
    trace_df: pd.DataFrame
    thresholds: Dict[str, Any]


def discover_and_load_trace(
    run_tag: str,
    persist_root: str,
    n_shards: Optional[int] = None,
    out_run_tag: Optional[str] = None,
    prefer_kind: str = "merged",
) -> CounterfactualRunData:
    candidates = discover_run_prefixes(
        run_tag=run_tag,
        persist_root=persist_root,
        n_shards=n_shards,
        out_run_tag=out_run_tag,
    )
    selected = select_default_run_prefix(candidates, prefer_kind=prefer_kind)
    loaded = load_run_artifacts(selected, require_trace=True)
    return CounterfactualRunData(
        candidates_df=candidates,
        selected_run_prefix=selected,
        artifact_paths=loaded.artifact_paths,
        results_df=loaded.results_df.copy(),
        trace_df=loaded.trace_df.copy(),
        thresholds=dict(loaded.thresholds),
    )


def load_intervention_tables(csv_paths: Sequence[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in csv_paths:
        fp = Path(str(p)).expanduser()
        if not fp.exists():
            raise FileNotFoundError(f"Intervention CSV not found: {fp}")
        df = pd.read_csv(fp)
        df["source_path"] = str(fp)
        frames.append(df)
    if len(frames) == 0:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def trace_to_intervention_table(
    trace_df: pd.DataFrame,
    factor_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if trace_df.empty:
        return pd.DataFrame()

    factor_columns = list(factor_columns or ["delta_x", "delta_y", "delta_l2", "step_scale"])
    present_factors = [c for c in factor_columns if c in trace_df.columns]
    if len(present_factors) == 0:
        return pd.DataFrame()

    id_cols = [c for c in ["scenario_id", "method", "eval_index", "seed_used"] if c in trace_df.columns]
    value_cols = [
        c
        for c in [
            "risk_sks",
            "failure_proxy",
            "surprise_pd",
            "accepted",
            "is_best_so_far",
            "feasible",
            "rollout_feasible",
        ]
        if c in trace_df.columns
    ]
    long_df = trace_df[id_cols + value_cols + present_factors].melt(
        id_vars=id_cols + value_cols,
        value_vars=present_factors,
        var_name="factor_name",
        value_name="factor_value",
    )
    long_df["factor_value"] = pd.to_numeric(long_df["factor_value"], errors="coerce")
    return long_df
