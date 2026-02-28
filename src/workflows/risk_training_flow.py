from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.closedloop.planner_backends import make_closed_loop_components
from src.closedloop.risk_benchmark import run_uq_benchmark
from src.closedloop.risk_candidates import build_candidate_risk_dataset_rows
from src.risk_model.artifacts import (
    save_risk_artifacts,
    save_risk_dataset_artifacts,
    save_risk_evaluation_artifacts,
)
from src.risk_model.calibration import (
    TemperatureScaler,
    fit_binary_conformal_threshold,
    fit_temperature_scalers,
)
from src.risk_model.dataset import add_eval_splits, build_risk_dataset
from src.risk_model.train import RiskTrainingBundle, train_risk_ensemble


@dataclass
class RiskDatasetBundle:
    dataset_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    processed_scenarios: List[int] = field(default_factory=list)
    skipped_scenarios: List[int] = field(default_factory=list)


@dataclass
class RiskTrainingFlowBundle:
    dataset_bundle: RiskDatasetBundle
    training_bundle: RiskTrainingBundle
    scalers: Dict[str, TemperatureScaler]
    conformal_thresholds: Dict[str, float]
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    calibration_bundle: Any = None


DEFAULT_FAILURE_LABEL = 'failure_proxy_h15'


def _meta_map(runner: Any) -> Dict[int, Dict[str, Any]]:
    meta = getattr(runner, 'data', {}).get('meta', pd.DataFrame())
    if not isinstance(meta, pd.DataFrame) or meta.empty or 'scenario_id' not in meta.columns:
        return {}
    rows = meta.to_dict(orient='records')
    return {int(row['scenario_id']): row for row in rows}


def build_risk_dataset_from_runner(
    runner: Any,
    cfg: Any,
    *,
    scenario_ids: Optional[Iterable[int]] = None,
    shift_suite: str = 'nominal_clean',
    persist: bool = True,
    run_prefix: Optional[str] = None,
) -> RiskDatasetBundle:
    scenarios = getattr(runner, 'data', {}).get('scenarios', [])
    if scenario_ids is None:
        scenario_ids = [int(rec['scenario_id']) for rec in scenarios if isinstance(rec, dict)]
    meta_by_id = _meta_map(runner)

    rows: List[Dict[str, Any]] = []
    processed: List[int] = []
    skipped: List[int] = []
    for sid in scenario_ids:
        sid_int = int(sid)
        rec = scenarios[sid_int]
        if 'state' not in rec:
            skipped.append(sid_int)
            continue
        planner_bundle = make_closed_loop_components(
            rec['state'],
            planner_kind=cfg.planner_kind,
            planner_name=cfg.planner_name,
            cfg=cfg,
        )
        meta = meta_by_id.get(sid_int, {})
        target_interaction_score = float(meta.get('risk_sks', np.nan)) if meta else np.nan
        seed = int(cfg.global_seed + sid_int * max(1, int(cfg.rollout_seed_stride)))
        rows.extend(
            build_candidate_risk_dataset_rows(
                scenario_id=sid_int,
                state=rec['state'],
                selected_idx=np.asarray(rec['selected_indices'], dtype=np.int32),
                planner_bundle=planner_bundle,
                cfg=cfg,
                seed=seed,
                shift_suite=shift_suite,
                target_interaction_score=target_interaction_score,
            )
        )
        processed.append(sid_int)

    dataset_df = build_risk_dataset(rows)
    dataset_df = add_eval_splits(
        dataset_df,
        holdout_fraction=0.20,
        interaction_score_col='target_interaction_score',
        shift_suites=getattr(cfg, 'uq_shift_suites', ('nominal_clean',)),
    )

    artifact_paths: Dict[str, str] = {}
    if persist:
        artifact_paths.update(save_risk_dataset_artifacts(run_prefix or cfg.run_prefix, dataset_df))
    return RiskDatasetBundle(
        dataset_df=dataset_df,
        artifact_paths=artifact_paths,
        processed_scenarios=processed,
        skipped_scenarios=skipped,
    )


def _build_calibration_outputs(bundle: RiskTrainingBundle) -> tuple[Dict[str, TemperatureScaler], Dict[str, float], pd.DataFrame]:
    val_pred = bundle.validation_predictions.copy()
    logits = np.column_stack([val_pred[f'logit_{label}'].to_numpy(dtype=float) for label in bundle.label_columns])
    labels = np.column_stack([val_pred[label].to_numpy(dtype=float) for label in bundle.label_columns])
    scalers = fit_temperature_scalers(logits, labels, bundle.label_columns)

    cal_logits = np.column_stack([
        scalers[label].apply(val_pred[f'logit_{label}'].to_numpy(dtype=float))
        for label in bundle.label_columns
    ])
    cal_probs = 1.0 / (1.0 + np.exp(-cal_logits))
    for idx, label in enumerate(bundle.label_columns):
        val_pred[f'risk_raw_{label}'] = val_pred[f'prob_{label}'].to_numpy(dtype=float)
        val_pred[f'risk_cal_{label}'] = cal_probs[:, idx]
        val_pred[f'risk_epistemic_{label}'] = val_pred[f'epistemic_{label}'].to_numpy(dtype=float)

    alpha = 0.10
    failure_idx = bundle.label_columns.index(DEFAULT_FAILURE_LABEL) if DEFAULT_FAILURE_LABEL in bundle.label_columns else len(bundle.label_columns) - 1
    conformal_thresholds = {
        DEFAULT_FAILURE_LABEL: fit_binary_conformal_threshold(cal_probs[:, failure_idx], labels[:, failure_idx], alpha=alpha),
    }
    return scalers, conformal_thresholds, val_pred


def train_and_calibrate_risk_model(
    dataset_df: pd.DataFrame,
    cfg: Any,
    *,
    run_prefix: Optional[str] = None,
) -> RiskTrainingFlowBundle:
    run_prefix = run_prefix or cfg.run_prefix
    train_bundle = train_risk_ensemble(
        dataset_df,
        label_columns=[
            'collision_h5', 'collision_h10', 'collision_h15',
            'offroad_h5', 'offroad_h10', 'offroad_h15',
            'failure_proxy_h5', 'failure_proxy_h10', 'failure_proxy_h15',
        ],
        ensemble_size=int(getattr(cfg, 'risk_model_ensemble_size', 5)),
        hidden_dims=tuple(getattr(cfg, 'risk_model_hidden_dims', (128, 128))),
        dropout=float(getattr(cfg, 'risk_model_dropout', 0.10)),
        learning_rate=float(getattr(cfg, 'risk_model_learning_rate', 1e-3)),
        batch_size=int(getattr(cfg, 'risk_model_batch_size', 1024)),
        max_epochs=int(getattr(cfg, 'risk_model_max_epochs', 50)),
        patience=int(getattr(cfg, 'risk_model_patience', 8)),
        seed=int(getattr(cfg, 'global_seed', 17)),
    )
    scalers, conformal_thresholds, calibration_predictions = _build_calibration_outputs(train_bundle)
    artifact_paths = save_risk_artifacts(run_prefix, train_bundle, scalers, conformal_thresholds)

    calibration_bundle = run_uq_benchmark(
        calibration_predictions,
        variants={
            'raw': 'risk_raw_failure_proxy_h15',
            'cal': 'risk_cal_failure_proxy_h15',
        },
        label_columns=(DEFAULT_FAILURE_LABEL,),
        n_bins=int(getattr(cfg, 'uq_eval_probability_bins', 15)),
    )
    artifact_paths.update(
        save_risk_evaluation_artifacts(
            run_prefix,
            {
                'risk_calibration_summary': calibration_bundle.summary_df,
                'risk_reliability_bins': calibration_bundle.reliability_df,
            },
        )
    )
    dataset_bundle = RiskDatasetBundle(dataset_df=dataset_df)
    return RiskTrainingFlowBundle(
        dataset_bundle=dataset_bundle,
        training_bundle=train_bundle,
        scalers=scalers,
        conformal_thresholds=conformal_thresholds,
        artifact_paths=artifact_paths,
        calibration_bundle=calibration_bundle,
    )


def run_risk_training_flow(
    *,
    cfg: Any,
    runner: Optional[Any] = None,
    dataset_df: Optional[pd.DataFrame] = None,
    scenario_ids: Optional[Iterable[int]] = None,
    shift_suite: str = 'nominal_clean',
    run_prefix: Optional[str] = None,
) -> RiskTrainingFlowBundle:
    run_prefix = run_prefix or cfg.run_prefix
    if dataset_df is None:
        if runner is None:
            raise ValueError('runner or dataset_df is required.')
        dataset_bundle = build_risk_dataset_from_runner(
            runner,
            cfg,
            scenario_ids=scenario_ids,
            shift_suite=shift_suite,
            persist=True,
            run_prefix=run_prefix,
        )
        dataset_df = dataset_bundle.dataset_df
    else:
        dataset_bundle = RiskDatasetBundle(
            dataset_df=dataset_df,
            artifact_paths=save_risk_dataset_artifacts(run_prefix, dataset_df),
        )
    bundle = train_and_calibrate_risk_model(dataset_df, cfg, run_prefix=run_prefix)
    bundle.dataset_bundle = dataset_bundle
    return bundle
