from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / 'src/workflows/miscalibration_probe_flow.py'
# Build a lightweight package stub so relative imports inside the module work
# without importing src.workflows.__init__ (which has optional heavy deps).
pkg_name = 'src.workflows'
if pkg_name not in sys.modules:
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(MODULE_PATH.parent)]  # type: ignore[attr-defined]
    sys.modules[pkg_name] = pkg

SPEC = importlib.util.spec_from_file_location('src.workflows.miscalibration_probe_flow', MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
flow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = flow
SPEC.loader.exec_module(flow)


def _synthetic_probe_df(n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(41)
    x = rng.normal(size=n)
    probs = 1.0 / (1.0 + np.exp(-x))
    labels = (rng.random(n) < probs).astype(float)
    shift = np.where(np.arange(n) % 3 == 0, 'nominal_clean', 'hist_prim_shift')

    eval_split = np.full((n,), 'test', dtype=object)
    eval_split[: int(0.15 * n)] = 'train'
    eval_split[int(0.15 * n): int(0.30 * n)] = 'val'
    eval_split[int(0.90 * n):] = 'high_interaction_holdout'

    return pd.DataFrame(
        {
            'scenario_id': np.arange(n, dtype=int),
            'candidate_id': 0,
            'shift_suite': shift,
            'eval_split': eval_split,
            'dist_top1_weight': np.clip(0.55 + 0.25 * x, 0.01, 0.99),
            'dist_entropy': np.clip(0.90 - 0.30 * x + 0.1 * rng.normal(size=n), 0.0, 2.2),
            'dist_num_components': np.where(np.arange(n) % 2 == 0, 8.0, 6.0),
            'failure_proxy_h15': labels,
            'collision_h15': (labels * (rng.random(n) < 0.35)).astype(float),
            'offroad_h15': (labels * (rng.random(n) < 0.20)).astype(float),
        }
    )


def _cfg(run_prefix: str) -> SimpleNamespace:
    return SimpleNamespace(
        run_prefix=str(run_prefix),
        uq_eval_probability_bins=12,
        risk_control_fail_budget=0.20,
    )


def test_miscalibration_probe_flow_writes_artifacts_and_supports_resume(tmp_path: Path) -> None:
    run_prefix = str(tmp_path / 'miscal_probe')
    cfg = _cfg(run_prefix)
    df = _synthetic_probe_df()

    first = flow.run_miscalibration_probe_flow(
        cfg=cfg,
        dataset_df=df,
        run_prefix=run_prefix,
        resume_mode='fresh',
    )
    assert not first.loaded_from_existing
    assert not first.benchmark_bundle.summary_df.empty
    assert not first.benchmark_bundle.per_shift_df.empty
    assert not first.benchmark_bundle.reliability_df.empty
    assert not first.threshold_df.empty
    assert flow.has_existing_miscalibration_probe_artifacts(run_prefix)

    required_keys = {
        'miscalibration_probe_summary',
        'miscalibration_probe_per_shift',
        'miscalibration_probe_reliability_bins',
        'miscalibration_probe_selective_risk_curve',
        'miscalibration_probe_shift_gap_summary',
        'miscalibration_probe_threshold_diagnostics',
        'miscalibration_probe_predictions',
        'artifact_schema',
    }
    assert required_keys.issubset(set(first.artifact_paths.keys()))
    for key in required_keys:
        path = Path(first.artifact_paths[key])
        assert path.exists(), f'missing artifact for key={key}: {path}'

    second = flow.run_miscalibration_probe_flow(
        cfg=cfg,
        run_prefix=run_prefix,
        resume_mode='resume',
    )
    assert second.loaded_from_existing
    assert not second.benchmark_bundle.summary_df.empty
    assert not second.threshold_df.empty

    loaded = flow.load_existing_miscalibration_probe_bundle(run_prefix)
    assert loaded.loaded_from_existing
    assert not loaded.predictions_df.empty
