from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd

WORKFLOWS_DIR = Path(__file__).resolve().parents[1] / 'src/workflows'
PROBE_MODULE_PATH = WORKFLOWS_DIR / 'miscalibration_probe_flow.py'
INTERP_MODULE_PATH = WORKFLOWS_DIR / 'miscalibration_interpretation_flow.py'

pkg_name = 'src.workflows'
if pkg_name not in sys.modules:
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(WORKFLOWS_DIR)]  # type: ignore[attr-defined]
    sys.modules[pkg_name] = pkg

PROBE_SPEC = importlib.util.spec_from_file_location('src.workflows.miscalibration_probe_flow', PROBE_MODULE_PATH)
assert PROBE_SPEC is not None and PROBE_SPEC.loader is not None
probe_flow = importlib.util.module_from_spec(PROBE_SPEC)
sys.modules[PROBE_SPEC.name] = probe_flow
PROBE_SPEC.loader.exec_module(probe_flow)

INTERP_SPEC = importlib.util.spec_from_file_location('src.workflows.miscalibration_interpretation_flow', INTERP_MODULE_PATH)
assert INTERP_SPEC is not None and INTERP_SPEC.loader is not None
interp_flow = importlib.util.module_from_spec(INTERP_SPEC)
sys.modules[INTERP_SPEC.name] = interp_flow
INTERP_SPEC.loader.exec_module(interp_flow)


def _synthetic_probe_df(n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(119)
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
        }
    )


def _cfg(run_prefix: str) -> SimpleNamespace:
    return SimpleNamespace(
        run_prefix=str(run_prefix),
        uq_eval_probability_bins=12,
        risk_control_fail_budget=0.20,
    )


def test_miscalibration_interpretation_flow_discovers_and_analyzes(tmp_path: Path) -> None:
    run_prefix = str(tmp_path / 'miscal_probe_interp')
    cfg = _cfg(run_prefix)
    df = _synthetic_probe_df()

    probe_bundle = probe_flow.run_miscalibration_probe_flow(
        cfg=cfg,
        dataset_df=df,
        run_prefix=run_prefix,
        resume_mode='fresh',
    )
    assert not probe_bundle.benchmark_bundle.summary_df.empty

    discovered = interp_flow.discover_probe_run_prefixes(tmp_path, limit=10)
    assert not discovered.empty
    assert run_prefix in set(discovered['run_prefix'].astype(str).tolist())

    interpreted = interp_flow.load_and_analyze_miscalibration_probe(
        run_prefix=run_prefix,
        focus_label='failure_proxy_h15',
        threshold=0.20,
    )
    assert not interpreted.metric_summary_df.empty
    assert not interpreted.verdict_df.empty
    assert 'Problem framing validated (miscalibration + decision impact)' in set(
        interpreted.verdict_df['claim'].astype(str).tolist()
    )
    assert str(interpreted.narrative).strip()
