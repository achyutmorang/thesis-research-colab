from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _load_closedloop_config_module():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "src" / "closedloop" / "config.py"
    spec = importlib.util.spec_from_file_location("closedloop_config", str(config_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["closedloop_config"] = module
    spec.loader.exec_module(module)
    return module


closedloop_config = _load_closedloop_config_module()
build_run_artifact_paths = closedloop_config.build_run_artifact_paths
shard_run_prefix = closedloop_config.shard_run_prefix

RESULTS_REQUIRED_COLUMNS = ["scenario_id", "method"]
TRACE_REQUIRED_COLUMNS = ["scenario_id", "method", "eval_index"]


def _load_required_csv(path: Path, required_cols: List[str], artifact_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {artifact_name}: {path}")

    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{artifact_name} missing required columns {missing}: {path}")
    return df


def _merge_per_scenario_results(parts: List[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(parts, ignore_index=True)
    # Keep last row if duplicates occur.
    merged = merged.drop_duplicates(subset=["scenario_id", "method"], keep="last")
    return merged.sort_values(["scenario_id", "method"]).reset_index(drop=True)


def _merge_per_eval_trace(parts: List[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.drop_duplicates(
        subset=["scenario_id", "method", "eval_index"], keep="last"
    )
    return merged.sort_values(["scenario_id", "method", "eval_index"]).reset_index(drop=True)


def merge_shards(
    run_tag: str,
    persist_root: str,
    n_shards: int,
    out_run_tag: str | None = None,
) -> Dict[str, Any]:
    n_shards = int(max(1, n_shards))
    methods = ["random", "risk_only", "surprise_only", "joint"]

    scenario_parts: List[pd.DataFrame] = []
    trace_parts: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, Any]] = []

    for shard_id in range(n_shards):
        shard_prefix = shard_run_prefix(run_tag, persist_root, shard_id, n_shards)
        shard_paths = build_run_artifact_paths(shard_prefix)

        results_path = Path(shard_paths["per_scenario_results"])
        shard_df = _load_required_csv(
            results_path,
            required_cols=RESULTS_REQUIRED_COLUMNS,
            artifact_name="per_scenario_results",
        )
        shard_df["source_shard_id"] = int(shard_id)
        scenario_parts.append(shard_df)

        trace_path = Path(shard_paths["per_eval_trace"])
        trace_exists = trace_path.exists()
        if trace_exists:
            trace_df = _load_required_csv(
                trace_path,
                required_cols=TRACE_REQUIRED_COLUMNS,
                artifact_name="per_eval_trace",
            )
            trace_df["source_shard_id"] = int(shard_id)
            trace_parts.append(trace_df)

        sub = shard_df[shard_df["method"].isin(methods)]
        completed = 0
        if not sub.empty:
            counts = sub.groupby("scenario_id")["method"].nunique()
            completed = int((counts >= len(methods)).sum())

        manifest_rows.append(
            {
                "shard_id": int(shard_id),
                "run_prefix": shard_prefix,
                "results_path": str(results_path),
                "results_rows": int(len(shard_df)),
                "trace_path": str(trace_path),
                "trace_exists": int(trace_exists),
                "trace_rows": int(len(trace_parts[-1])) if trace_exists else 0,
                "completed_scenarios": int(completed),
            }
        )

    merged_results = _merge_per_scenario_results(scenario_parts)
    merged_trace = _merge_per_eval_trace(trace_parts) if len(trace_parts) > 0 else pd.DataFrame()

    out_tag = out_run_tag or f"{run_tag}_merged"
    out_prefix = str(Path(persist_root).expanduser() / out_tag)
    out_results_path = f"{out_prefix}_per_scenario_results.csv"
    out_trace_path = f"{out_prefix}_per_eval_trace.csv"
    out_manifest_path = f"{out_prefix}_merge_manifest.json"

    Path(out_results_path).parent.mkdir(parents=True, exist_ok=True)
    merged_results.to_csv(out_results_path, index=False)
    if len(merged_trace) > 0:
        merged_trace.to_csv(out_trace_path, index=False)

    merge_manifest: Dict[str, Any] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_run_tag": run_tag,
        "output_run_tag": out_tag,
        "persist_root": str(Path(persist_root).expanduser()),
        "n_shards": int(n_shards),
        "shards": manifest_rows,
        "outputs": {
            "merged_per_scenario_results": out_results_path,
            "merged_per_eval_trace": out_trace_path if len(merged_trace) > 0 else None,
        },
        "merged_counts": {
            "per_scenario_rows": int(len(merged_results)),
            "per_eval_trace_rows": int(len(merged_trace)),
            "n_unique_scenarios": int(merged_results["scenario_id"].nunique()),
        },
    }

    with open(out_manifest_path, "w") as f:
        json.dump(merge_manifest, f, indent=2)

    return {
        "merged_per_scenario_results": out_results_path,
        "merged_per_eval_trace": out_trace_path if len(merged_trace) > 0 else "",
        "merge_manifest": out_manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded Closed-Loop artifacts.")
    parser.add_argument("--run-tag", required=True, help="Base run tag used for shard runs.")
    parser.add_argument("--persist-root", required=True, help="Persistent root directory.")
    parser.add_argument("--n-shards", required=True, type=int, help="Number of shards to merge.")
    parser.add_argument(
        "--out-run-tag",
        default=None,
        help="Optional output run tag for merged artifacts (default: <run-tag>_merged).",
    )
    args = parser.parse_args()

    outputs = merge_shards(
        run_tag=args.run_tag,
        persist_root=args.persist_root,
        n_shards=args.n_shards,
        out_run_tag=args.out_run_tag,
    )
    print("[merge] wrote outputs:")
    for key, value in outputs.items():
        if value:
            print(f" - {key}: {value}")


if __name__ == "__main__":
    main()
