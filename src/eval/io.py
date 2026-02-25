from __future__ import annotations

import json
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

METHODS: List[str] = ["random", "risk_only", "surprise_only", "prism_joint"]
RESULTS_REQUIRED_COLUMNS: List[str] = ["scenario_id", "method"]
TRACE_REQUIRED_COLUMNS: List[str] = ["scenario_id", "method", "eval_index"]


def _load_trackb_config_module():
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "src" / "trackb" / "config.py"
    spec = importlib.util.spec_from_file_location("trackb_config_for_eval", str(config_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["trackb_config_for_eval"] = module
    spec.loader.exec_module(module)
    return module


_trackb_config = _load_trackb_config_module()
build_run_artifact_paths = _trackb_config.build_run_artifact_paths
shard_run_prefix = _trackb_config.shard_run_prefix


@dataclass
class LoadedRunArtifacts:
    run_prefix: str
    artifact_paths: Dict[str, str]
    results_df: pd.DataFrame
    trace_df: pd.DataFrame
    thresholds: Dict[str, Any]
    calibration_df: pd.DataFrame
    calibration_diag_df: pd.DataFrame
    calibration_quant_df: pd.DataFrame


def _read_csv(path: str, required: bool = False) -> pd.DataFrame:
    fp = Path(path)
    if not fp.exists():
        if required:
            raise FileNotFoundError(f"Required artifact not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(fp)


def _read_json(path: str, required: bool = False) -> Dict[str, Any]:
    fp = Path(path)
    if not fp.exists():
        if required:
            raise FileNotFoundError(f"Required artifact not found: {path}")
        return {}
    with open(fp, "r") as f:
        return json.load(f)


def _completed_scenarios_from_results(path: str, methods: Sequence[str]) -> int:
    fp = Path(path)
    if not fp.exists():
        return 0
    try:
        df = pd.read_csv(fp)
    except Exception:
        return 0
    if df.empty or ("scenario_id" not in df.columns) or ("method" not in df.columns):
        return 0
    sub = df[df["method"].isin(methods)]
    if sub.empty:
        return 0
    counts = sub.groupby("scenario_id")["method"].nunique()
    return int((counts >= len(methods)).sum())


def discover_run_prefixes(
    run_tag: str,
    persist_root: str,
    n_shards: Optional[int] = None,
    out_run_tag: Optional[str] = None,
    include_unsharded: bool = True,
    include_shards: bool = True,
) -> pd.DataFrame:
    root = Path(str(persist_root)).expanduser()
    candidates: List[Dict[str, Any]] = []

    merged_tag = out_run_tag or f"{run_tag}_merged"
    merged_prefix = str(root / merged_tag)
    candidates.append({"kind": "merged", "run_tag": merged_tag, "run_prefix": merged_prefix})

    if include_unsharded:
        base_prefix = str(root / run_tag)
        candidates.append({"kind": "base", "run_tag": run_tag, "run_prefix": base_prefix})

    if include_shards and n_shards is not None:
        n_shards = int(max(1, n_shards))
        for shard_id in range(n_shards):
            run_prefix = shard_run_prefix(run_tag, persist_root, shard_id, n_shards)
            candidates.append(
                {
                    "kind": "shard",
                    "run_tag": run_tag,
                    "shard_id": int(shard_id),
                    "n_shards": int(n_shards),
                    "run_prefix": run_prefix,
                }
            )

    rows: List[Dict[str, Any]] = []
    for item in candidates:
        run_prefix = item["run_prefix"]
        paths = build_run_artifact_paths(run_prefix)
        results_path = paths["per_scenario_results"]
        trace_path = paths["per_eval_trace"]
        thresholds_path = paths["thresholds"]
        calib_path = paths["closedloop_calibration"]

        n_rows = 0
        n_trace_rows = 0
        if Path(results_path).exists():
            try:
                n_rows = int(len(pd.read_csv(results_path)))
            except Exception:
                n_rows = 0
        if Path(trace_path).exists():
            try:
                n_trace_rows = int(len(pd.read_csv(trace_path)))
            except Exception:
                n_trace_rows = 0

        row = {
            **item,
            "results_exists": int(Path(results_path).exists()),
            "trace_exists": int(Path(trace_path).exists()),
            "thresholds_exists": int(Path(thresholds_path).exists()),
            "calibration_exists": int(Path(calib_path).exists()),
            "n_rows": int(n_rows),
            "n_trace_rows": int(n_trace_rows),
            "n_completed_scenarios": int(
                _completed_scenarios_from_results(results_path, methods=METHODS)
            ),
        }
        rows.append(row)

    cols = [
        "kind",
        "run_tag",
        "shard_id",
        "n_shards",
        "run_prefix",
        "results_exists",
        "trace_exists",
        "thresholds_exists",
        "calibration_exists",
        "n_rows",
        "n_trace_rows",
        "n_completed_scenarios",
    ]
    out = pd.DataFrame(rows)
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[cols].sort_values(
        ["results_exists", "n_completed_scenarios", "n_rows", "kind", "shard_id"],
        ascending=[False, False, False, True, True],
    )
    return out.reset_index(drop=True)


def select_default_run_prefix(
    candidates_df: pd.DataFrame,
    prefer_kind: str = "merged",
) -> str:
    if candidates_df.empty:
        raise ValueError("No candidate run prefixes were discovered.")

    preferred = candidates_df[
        (candidates_df["kind"] == prefer_kind) & (candidates_df["results_exists"] == 1)
    ]
    if not preferred.empty:
        return str(preferred.iloc[0]["run_prefix"])

    available = candidates_df[candidates_df["results_exists"] == 1]
    if not available.empty:
        ranked = available.sort_values(
            ["n_completed_scenarios", "n_rows", "trace_exists"],
            ascending=[False, False, False],
        )
        return str(ranked.iloc[0]["run_prefix"])

    return str(candidates_df.iloc[0]["run_prefix"])


def load_run_artifacts(run_prefix: str, require_trace: bool = False) -> LoadedRunArtifacts:
    paths = build_run_artifact_paths(run_prefix)

    results_df = _read_csv(paths["per_scenario_results"], required=True)
    trace_df = _read_csv(paths["per_eval_trace"], required=require_trace)
    thresholds = _read_json(paths["thresholds"], required=False)
    calibration_df = _read_csv(paths["closedloop_calibration"], required=False)
    calibration_diag_df = _read_csv(paths["calibration_diagnostics"], required=False)
    calibration_quant_df = _read_csv(paths["calibration_quantiles"], required=False)

    missing_res_cols = [c for c in RESULTS_REQUIRED_COLUMNS if c not in results_df.columns]
    if missing_res_cols:
        raise ValueError(
            f"Missing required columns in per_scenario_results ({paths['per_scenario_results']}): "
            f"{missing_res_cols}"
        )

    if not trace_df.empty:
        missing_trace_cols = [c for c in TRACE_REQUIRED_COLUMNS if c not in trace_df.columns]
        if missing_trace_cols:
            raise ValueError(
                f"Missing required columns in per_eval_trace ({paths['per_eval_trace']}): "
                f"{missing_trace_cols}"
            )

    return LoadedRunArtifacts(
        run_prefix=run_prefix,
        artifact_paths=paths,
        results_df=results_df,
        trace_df=trace_df,
        thresholds=thresholds,
        calibration_df=calibration_df,
        calibration_diag_df=calibration_diag_df,
        calibration_quant_df=calibration_quant_df,
    )


def discover_and_load(
    run_tag: str,
    persist_root: str,
    n_shards: Optional[int] = None,
    out_run_tag: Optional[str] = None,
    prefer_kind: str = "merged",
    require_trace: bool = False,
) -> tuple[pd.DataFrame, LoadedRunArtifacts]:
    candidates = discover_run_prefixes(
        run_tag=run_tag,
        persist_root=persist_root,
        n_shards=n_shards,
        out_run_tag=out_run_tag,
    )
    selected = select_default_run_prefix(candidates, prefer_kind=prefer_kind)
    loaded = load_run_artifacts(selected, require_trace=require_trace)
    return candidates, loaded
