from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


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


config = _load_module("trackb_config", "src/trackb/config.py")


def test_auto_select_shard_prefers_missing_shard(tmp_path: Path):
    run_tag = "unit_shard_missing"
    n_shards = 3
    persist_root = str(tmp_path)

    shard0_prefix = config.shard_run_prefix(run_tag, persist_root, 0, n_shards)
    path0 = Path(f"{shard0_prefix}_per_scenario_results.csv")
    pd.DataFrame(
        [
            {"scenario_id": 11, "method": "random"},
            {"scenario_id": 11, "method": "risk_only"},
            {"scenario_id": 11, "method": "surprise_only"},
            {"scenario_id": 11, "method": "prism_joint"},
        ]
    ).to_csv(path0, index=False)

    # shard 1 and 2 are missing; selector should pick the first missing shard.
    selected = config.auto_select_shard_id(run_tag, persist_root, n_shards)
    assert selected == 1


def test_auto_select_shard_prefers_least_complete_when_all_exist(tmp_path: Path):
    run_tag = "unit_shard_complete_scan"
    n_shards = 3
    persist_root = str(tmp_path)

    methods = ["random", "risk_only", "surprise_only", "prism_joint"]
    for sid in range(n_shards):
        prefix = config.shard_run_prefix(run_tag, persist_root, sid, n_shards)
        path = Path(f"{prefix}_per_scenario_results.csv")
        rows = []
        # shard0 complete for 2 scenarios, shard1 complete for 1 scenario, shard2 none
        n_complete = {0: 2, 1: 1, 2: 0}[sid]
        for scenario_id in range(100 + sid * 10, 100 + sid * 10 + n_complete):
            rows.extend({"scenario_id": scenario_id, "method": m} for m in methods)
        pd.DataFrame(rows if rows else [{"scenario_id": -1, "method": "other"}]).to_csv(path, index=False)

    selected = config.auto_select_shard_id(run_tag, persist_root, n_shards)
    assert selected == 2


def test_inspect_shard_progress_columns(tmp_path: Path):
    df = config.inspect_shard_progress(
        run_tag="unit_progress",
        persist_root=str(tmp_path),
        n_shards=2,
    )
    expected_cols = {
        "shard_id",
        "run_prefix",
        "results_exists",
        "n_rows",
        "n_touched_scenarios",
        "n_completed_scenarios",
        "status",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 2
