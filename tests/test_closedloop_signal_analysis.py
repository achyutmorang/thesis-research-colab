from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
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


signal_analysis = _load_module("closedloop_signal_analysis", "src/closedloop/signal_analysis.py")


def _mock_results_monotonic() -> pd.DataFrame:
    rows = []
    methods = ["random", "risk_only", "surprise_only", "joint"]
    for sid in range(12):
        for midx, method in enumerate(methods):
            surprise = 0.2 * sid + 0.05 * midx
            delta_risk = 0.8 * surprise + 0.01 * (midx - 1.5)
            rows.append(
                {
                    "scenario_id": sid,
                    "method": method,
                    "surprise_pd": float(surprise),
                    "delta_risk": float(delta_risk),
                }
            )
    return pd.DataFrame(rows)


def test_signal_usefulness_detects_positive_monotonic_relation():
    df = _mock_results_monotonic()
    summary_df, method_corr_df, bin_df, topk_df, within_scenario_df = (
        signal_analysis.analyze_surprise_signal_usefulness(df, n_bins=8)
    )

    assert summary_df.shape[0] == 1
    assert float(summary_df.loc[0, "corr_spearman"]) > 0.95
    assert int(summary_df.loc[0, "monotonic_non_decreasing"]) == 1
    assert float(summary_df.loc[0, "top10_lift_vs_all"]) > 0.0
    assert not method_corr_df.empty
    assert set(method_corr_df["method"].tolist()) == {"random", "risk_only", "surprise_only", "joint"}
    assert not bin_df.empty
    assert not topk_df.empty
    assert within_scenario_df.shape[0] == 12


def test_signal_usefulness_handles_constant_surprise():
    df = _mock_results_monotonic()
    df["surprise_pd"] = 0.0
    summary_df, method_corr_df, bin_df, topk_df, within_scenario_df = (
        signal_analysis.analyze_surprise_signal_usefulness(df, n_bins=8)
    )

    assert int(summary_df.loc[0, "n_usable_rows"]) == len(df)
    assert np.isnan(float(summary_df.loc[0, "corr_spearman"]))
    assert bin_df.shape[0] == 1
    assert method_corr_df.shape[0] == 4
    assert topk_df.shape[0] >= 1
    assert within_scenario_df.empty


def test_signal_usefulness_artifact_writer(tmp_path: Path):
    df = _mock_results_monotonic()
    out = signal_analysis.analyze_surprise_signal_usefulness(df, n_bins=8)
    run_prefix = str(tmp_path / "runs" / "closedloop_v1")
    paths = signal_analysis.save_surprise_signal_usefulness_artifacts(
        run_prefix=run_prefix,
        summary_df=out[0],
        method_corr_df=out[1],
        bin_df=out[2],
        topk_df=out[3],
        within_scenario_df=out[4],
    )
    for p in paths.values():
        assert Path(p).exists()
