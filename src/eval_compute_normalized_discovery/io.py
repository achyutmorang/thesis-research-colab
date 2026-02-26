from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.eval.io import (
    discover_run_prefixes,
    load_run_artifacts,
    select_default_run_prefix,
)


@dataclass
class DiscoveryRunData:
    candidates_df: pd.DataFrame
    selected_run_prefix: str
    artifact_paths: Dict[str, str]
    results_df: pd.DataFrame
    trace_df: pd.DataFrame
    thresholds: Dict[str, Any]


def discover_and_load_run(
    run_tag: str,
    persist_root: str,
    n_shards: Optional[int] = None,
    out_run_tag: Optional[str] = None,
    prefer_kind: str = "merged",
    require_trace: bool = True,
) -> DiscoveryRunData:
    candidates = discover_run_prefixes(
        run_tag=run_tag,
        persist_root=persist_root,
        n_shards=n_shards,
        out_run_tag=out_run_tag,
    )
    selected = select_default_run_prefix(candidates, prefer_kind=prefer_kind)
    loaded = load_run_artifacts(selected, require_trace=require_trace)
    return DiscoveryRunData(
        candidates_df=candidates,
        selected_run_prefix=selected,
        artifact_paths=loaded.artifact_paths,
        results_df=loaded.results_df.copy(),
        trace_df=loaded.trace_df.copy(),
        thresholds=dict(loaded.thresholds),
    )


def load_results_and_trace_csv(
    results_csv: str,
    trace_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results_path = Path(str(results_csv)).expanduser()
    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_path}")
    results_df = pd.read_csv(results_path)

    trace_df = pd.DataFrame()
    if trace_csv:
        trace_path = Path(str(trace_csv)).expanduser()
        if not trace_path.exists():
            raise FileNotFoundError(f"Trace CSV not found: {trace_path}")
        trace_df = pd.read_csv(trace_path)
    return results_df, trace_df
