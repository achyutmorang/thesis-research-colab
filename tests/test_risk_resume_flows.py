from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.risk_model.train import train_risk_ensemble


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


risk_training_flow = _load_module('risk_training_flow_direct', 'src/workflows/risk_training_flow.py')
uq_benchmark_flow = _load_module('uq_benchmark_flow_direct', 'src/workflows/uq_benchmark_flow.py')


LABEL_COLS = [
    'collision_h5', 'collision_h10', 'collision_h15',
    'offroad_h5', 'offroad_h10', 'offroad_h15',
    'failure_proxy_h5', 'failure_proxy_h10', 'failure_proxy_h15',
]


def _cfg(run_prefix: str) -> SimpleNamespace:
    return SimpleNamespace(
        run_prefix=run_prefix,
        global_seed=17,
        rollout_seed_stride=10000,
        planner_kind='latentdriver',
        planner_name='latentdriver_waypoint_sdc',
        risk_model_ensemble_size=2,
        risk_model_hidden_dims=(32, 16),
        risk_model_dropout=0.10,
        risk_model_learning_rate=1e-3,
        risk_model_batch_size=64,
        risk_model_max_epochs=4,
        risk_model_patience=2,
        risk_model_checkpoint_every_epochs=1,
        risk_model_resume_from_checkpoints=True,
        uq_eval_probability_bins=10,
    )


def _dataset(n: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.8 * x1 - 0.3 * x2 + 0.4 * x3)))
    y = (rng.random(n) < p).astype(float)

    scenario_id = np.arange(n, dtype=int)
    split = np.where(scenario_id < int(0.7 * n), 'train', np.where(scenario_id < int(0.85 * n), 'val', 'test'))
    df = pd.DataFrame(
        {
            'scenario_id': scenario_id,
            'eval_split': split,
            'shift_suite': np.where(scenario_id % 2 == 0, 'nominal_clean', 'hist_prim_shift'),
            'step_idx': 0,
            'candidate_id': 0,
            'candidate_source': 'mode_0',
            'planner_backend': 'latentdriver',
            'seed': 17,
            'feature_a': x1,
            'feature_b': x2,
            'feature_c': x3,
            'action_0': rng.normal(size=n),
            'action_1': rng.normal(size=n),
            'action_2': rng.normal(size=n),
            'progress_h6': np.abs(rng.normal(loc=5.0, scale=1.0, size=n)),
            'max_abs_acc_h6': np.abs(rng.normal(loc=1.0, scale=0.2, size=n)),
            'max_abs_jerk_h6': np.abs(rng.normal(loc=1.0, scale=0.2, size=n)),
            'target_interaction_score': np.abs(rng.normal(size=n)),
        }
    )
    for idx, label in enumerate(LABEL_COLS):
        noise = (rng.random(n) < (0.05 + 0.01 * idx)).astype(float)
        df[label] = np.clip(y + noise, 0.0, 1.0)
    return df


def test_risk_training_flow_resume_loads_existing_artifacts(tmp_path: Path) -> None:
    run_prefix = str(tmp_path / 'risk_resume_smoke')
    cfg = _cfg(run_prefix)
    df = _dataset()

    first = risk_training_flow.run_risk_training_flow(
        cfg=cfg,
        dataset_df=df,
        run_prefix=run_prefix,
        resume_mode='fresh',
    )
    assert first.loaded_from_existing is False
    assert (tmp_path / 'risk_resume_smoke_risk_model_metadata.json').exists()

    second = risk_training_flow.run_risk_training_flow(
        cfg=cfg,
        run_prefix=run_prefix,
        resume_mode='resume',
    )
    assert second.loaded_from_existing is True
    assert not second.dataset_bundle.dataset_df.empty


def test_uq_benchmark_flow_resume_loads_existing_artifacts(tmp_path: Path) -> None:
    run_prefix = str(tmp_path / 'uq_resume_smoke')
    cfg = _cfg(run_prefix)
    df = _dataset()

    risk_training_flow.run_risk_training_flow(
        cfg=cfg,
        dataset_df=df,
        run_prefix=run_prefix,
        resume_mode='fresh',
    )

    first = uq_benchmark_flow.run_uq_benchmark_flow(
        cfg=cfg,
        dataset_df=df,
        run_prefix=run_prefix,
        resume_mode='fresh',
    )
    assert first.loaded_from_existing is False
    assert (tmp_path / 'uq_resume_smoke_uq_benchmark_summary.csv').exists()
    assert (tmp_path / 'uq_resume_smoke_risk_control_per_step_trace.csv').exists()
    assert (tmp_path / 'uq_resume_smoke_risk_control_per_scenario_results.csv').exists()
    assert not first.controller_per_step_df.empty
    assert not first.controller_per_scenario_df.empty
    assert not first.controller_summary_df.empty

    second = uq_benchmark_flow.run_uq_benchmark_flow(
        cfg=cfg,
        dataset_df=pd.DataFrame(),
        run_prefix=run_prefix,
        resume_mode='resume',
    )
    assert second.loaded_from_existing is True
    assert not second.benchmark_bundle.summary_df.empty
    assert not second.controller_per_step_df.empty


def test_risk_training_flow_resume_uses_checkpoint_only_state(tmp_path: Path) -> None:
    run_prefix = str(tmp_path / 'risk_resume_ckpt_only')
    cfg = _cfg(run_prefix)
    df = _dataset()

    train_risk_ensemble(
        df,
        label_columns=LABEL_COLS,
        ensemble_size=int(cfg.risk_model_ensemble_size),
        hidden_dims=tuple(cfg.risk_model_hidden_dims),
        dropout=float(cfg.risk_model_dropout),
        learning_rate=float(cfg.risk_model_learning_rate),
        batch_size=int(cfg.risk_model_batch_size),
        max_epochs=int(cfg.risk_model_max_epochs),
        patience=int(cfg.risk_model_patience),
        seed=int(cfg.global_seed),
        checkpoint_prefix=run_prefix,
        checkpoint_every_epochs=1,
        resume_from_checkpoints=True,
    )
    assert not (tmp_path / 'risk_resume_ckpt_only_risk_model_metadata.json').exists()
    assert risk_training_flow.has_existing_risk_training_checkpoints(run_prefix) is True

    resumed = risk_training_flow.run_risk_training_flow(
        cfg=cfg,
        dataset_df=df,
        run_prefix=run_prefix,
        resume_mode='resume',
    )
    assert resumed.loaded_from_existing is False
    assert (tmp_path / 'risk_resume_ckpt_only_risk_model_metadata.json').exists()
