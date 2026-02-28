from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.closedloop.risk_benchmark import run_uq_benchmark
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
    loaded_from_existing: bool = False


DEFAULT_FAILURE_LABEL = 'failure_proxy_h15'
DEFAULT_LABEL_COLUMNS = [
    'collision_h5', 'collision_h10', 'collision_h15',
    'offroad_h5', 'offroad_h10', 'offroad_h15',
    'failure_proxy_h5', 'failure_proxy_h10', 'failure_proxy_h15',
]


def _normalize_resume_mode(value: Any) -> str:
    mode = str(value if value is not None else 'auto').strip().lower()
    if mode not in {'auto', 'fresh', 'resume'}:
        raise ValueError(f"Invalid resume_mode={value!r}. Expected one of: auto, fresh, resume.")
    return mode


def _read_frame_with_parquet_fallback(preferred_path: str) -> pd.DataFrame:
    p = Path(preferred_path)
    if p.exists():
        try:
            if p.suffix.lower() == '.parquet':
                return pd.read_parquet(p)
            return pd.read_csv(p)
        except (pd.errors.EmptyDataError, ValueError):
            return pd.DataFrame()

    if p.suffix.lower() == '.parquet':
        csv_path = p.with_suffix('.csv')
        if csv_path.exists():
            try:
                return pd.read_csv(csv_path)
            except (pd.errors.EmptyDataError, ValueError):
                return pd.DataFrame()
    return pd.DataFrame()


def _risk_artifact_paths(run_prefix: str) -> Dict[str, str]:
    return {
        'dataset': f'{run_prefix}_risk_dataset.parquet',
        'dataset_summary': f'{run_prefix}_risk_dataset_summary.csv',
        'train_summary': f'{run_prefix}_risk_train_summary.csv',
        'validation_predictions': f'{run_prefix}_risk_validation_predictions.parquet',
        'metadata': f'{run_prefix}_risk_model_metadata.json',
        'temperature_scalers': f'{run_prefix}_risk_temperature_scalers.json',
        'conformal_thresholds': f'{run_prefix}_risk_conformal_thresholds.json',
        'calibration_summary': f'{run_prefix}_risk_calibration_summary.csv',
        'reliability_bins': f'{run_prefix}_risk_reliability_bins.csv',
    }


def load_existing_risk_dataset_artifact(run_prefix: str) -> pd.DataFrame:
    return _read_frame_with_parquet_fallback(_risk_artifact_paths(run_prefix)['dataset'])


def has_existing_risk_model_artifacts(run_prefix: str) -> bool:
    paths = _risk_artifact_paths(run_prefix)
    members = list(Path(Path(paths['metadata']).parent).glob(Path(run_prefix).name + '_risk_ensemble_member_*.npz'))
    return bool(
        Path(paths['metadata']).exists()
        and Path(paths['temperature_scalers']).exists()
        and Path(paths['conformal_thresholds']).exists()
        and len(members) > 0
    )


def has_existing_risk_training_checkpoints(run_prefix: str) -> bool:
    root = Path(run_prefix).parent
    pattern = Path(run_prefix).name + '_risk_train_ckpt_member_*.npz'
    return any(root.glob(pattern))


def load_existing_risk_training_bundle(
    run_prefix: str,
    dataset_df: Optional[pd.DataFrame] = None,
) -> RiskTrainingFlowBundle:
    from src.risk_model.artifacts import load_risk_artifacts

    artifacts = load_risk_artifacts(run_prefix)
    paths = _risk_artifact_paths(run_prefix)

    train_summary = _read_frame_with_parquet_fallback(paths['train_summary'])
    val_pred = _read_frame_with_parquet_fallback(paths['validation_predictions'])
    if dataset_df is None:
        dataset_df = load_existing_risk_dataset_artifact(run_prefix)

    label_columns = list(artifacts.get('label_columns', DEFAULT_LABEL_COLUMNS))
    train_bundle = RiskTrainingBundle(
        model=artifacts['model'],
        feature_columns=list(artifacts['feature_columns']),
        label_columns=label_columns,
        feature_mean=np.asarray(artifacts['feature_mean'], dtype=float),
        feature_std=np.asarray(artifacts['feature_std'], dtype=float),
        train_summary=train_summary,
        validation_predictions=val_pred,
    )

    calibration_bundle = None
    if (not val_pred.empty) and all(f'risk_raw_{label}' in val_pred.columns for label in label_columns):
        calibration_bundle = run_uq_benchmark(
            val_pred,
            variants={
                'raw': 'risk_raw_failure_proxy_h15',
                'cal': 'risk_cal_failure_proxy_h15',
            },
            label_columns=(DEFAULT_FAILURE_LABEL,),
            n_bins=15,
        )

    artifact_paths = {
        'risk_dataset': paths['dataset'] if Path(paths['dataset']).exists() or Path(paths['dataset']).with_suffix('.csv').exists() else '',
        'risk_dataset_summary': paths['dataset_summary'] if Path(paths['dataset_summary']).exists() else '',
        'risk_train_summary': paths['train_summary'] if Path(paths['train_summary']).exists() else '',
        'risk_validation_predictions': paths['validation_predictions'] if Path(paths['validation_predictions']).exists() or Path(paths['validation_predictions']).with_suffix('.csv').exists() else '',
        'risk_model_metadata': paths['metadata'] if Path(paths['metadata']).exists() else '',
        'risk_temperature_scalers': paths['temperature_scalers'] if Path(paths['temperature_scalers']).exists() else '',
        'risk_conformal_thresholds': paths['conformal_thresholds'] if Path(paths['conformal_thresholds']).exists() else '',
        'risk_calibration_summary': paths['calibration_summary'] if Path(paths['calibration_summary']).exists() else '',
        'risk_reliability_bins': paths['reliability_bins'] if Path(paths['reliability_bins']).exists() else '',
    }
    return RiskTrainingFlowBundle(
        dataset_bundle=RiskDatasetBundle(dataset_df=dataset_df),
        training_bundle=train_bundle,
        scalers=dict(artifacts.get('temperature_scalers', {})),
        conformal_thresholds={k: float(v) for k, v in dict(artifacts.get('conformal_thresholds', {})).items()},
        artifact_paths=artifact_paths,
        calibration_bundle=calibration_bundle,
        loaded_from_existing=True,
    )


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
    from src.closedloop.planner_backends import make_closed_loop_components
    from src.closedloop.risk_candidates import build_candidate_risk_dataset_rows

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
        seed=int(getattr(cfg, 'global_seed', 17)),
        train_fraction=0.70,
        val_fraction=0.15,
        high_interaction_fraction=0.20,
        interaction_col='target_interaction_score',
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
    checkpoint_prefix: Optional[str] = None,
    checkpoint_every_epochs: Optional[int] = None,
    resume_from_checkpoints: Optional[bool] = None,
) -> RiskTrainingFlowBundle:
    run_prefix = run_prefix or cfg.run_prefix
    checkpoint_prefix = checkpoint_prefix or run_prefix
    checkpoint_every_epochs = int(max(1, checkpoint_every_epochs if checkpoint_every_epochs is not None else int(getattr(cfg, 'risk_model_checkpoint_every_epochs', 1))))
    if resume_from_checkpoints is None:
        resume_from_checkpoints = bool(getattr(cfg, 'risk_model_resume_from_checkpoints', True))

    train_bundle = train_risk_ensemble(
        dataset_df,
        label_columns=list(DEFAULT_LABEL_COLUMNS),
        ensemble_size=int(getattr(cfg, 'risk_model_ensemble_size', 5)),
        hidden_dims=tuple(getattr(cfg, 'risk_model_hidden_dims', (128, 128))),
        dropout=float(getattr(cfg, 'risk_model_dropout', 0.10)),
        learning_rate=float(getattr(cfg, 'risk_model_learning_rate', 1e-3)),
        batch_size=int(getattr(cfg, 'risk_model_batch_size', 1024)),
        max_epochs=int(getattr(cfg, 'risk_model_max_epochs', 50)),
        patience=int(getattr(cfg, 'risk_model_patience', 8)),
        seed=int(getattr(cfg, 'global_seed', 17)),
        checkpoint_prefix=checkpoint_prefix,
        checkpoint_every_epochs=checkpoint_every_epochs,
        resume_from_checkpoints=bool(resume_from_checkpoints),
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
        loaded_from_existing=False,
    )


def run_risk_training_flow(
    *,
    cfg: Any,
    runner: Optional[Any] = None,
    dataset_df: Optional[pd.DataFrame] = None,
    scenario_ids: Optional[Iterable[int]] = None,
    shift_suite: str = 'nominal_clean',
    run_prefix: Optional[str] = None,
    resume_mode: str = 'auto',
    force_rebuild_dataset: bool = False,
    force_retrain_model: bool = False,
) -> RiskTrainingFlowBundle:
    run_prefix = run_prefix or cfg.run_prefix
    mode = _normalize_resume_mode(resume_mode)

    existing_dataset_df = load_existing_risk_dataset_artifact(run_prefix)
    existing_model_ready = has_existing_risk_model_artifacts(run_prefix)
    existing_checkpoint_ready = has_existing_risk_training_checkpoints(run_prefix)
    if mode == 'resume':
        if dataset_df is None and existing_dataset_df.empty:
            raise FileNotFoundError(
                f"resume_mode='resume' but dataset artifact not found for run_prefix={run_prefix!r}."
            )
        if (not existing_model_ready) and (not existing_checkpoint_ready) and bool(not force_retrain_model):
            raise FileNotFoundError(
                f"resume_mode='resume' but no completed artifacts or checkpoints were found for run_prefix={run_prefix!r}."
            )

    if dataset_df is None:
        can_reuse_dataset = bool((mode in {'auto', 'resume'}) and (not force_rebuild_dataset) and (not existing_dataset_df.empty))
        if can_reuse_dataset:
            dataset_df = existing_dataset_df
            dataset_bundle = RiskDatasetBundle(
                dataset_df=dataset_df,
                artifact_paths={'risk_dataset': _risk_artifact_paths(run_prefix)['dataset']},
            )
        else:
            if runner is None:
                raise ValueError('runner or dataset_df is required when dataset artifact is unavailable.')
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

    can_reuse_model = bool((mode in {'auto', 'resume'}) and (not force_retrain_model) and has_existing_risk_model_artifacts(run_prefix))
    if can_reuse_model:
        existing_bundle = load_existing_risk_training_bundle(run_prefix=run_prefix, dataset_df=dataset_df)
        existing_bundle.dataset_bundle = dataset_bundle
        return existing_bundle

    should_resume_checkpoints = bool(getattr(cfg, 'risk_model_resume_from_checkpoints', True)) and bool(mode != 'fresh')
    bundle = train_and_calibrate_risk_model(
        dataset_df,
        cfg,
        run_prefix=run_prefix,
        checkpoint_prefix=run_prefix,
        checkpoint_every_epochs=int(getattr(cfg, 'risk_model_checkpoint_every_epochs', 1)),
        resume_from_checkpoints=should_resume_checkpoints,
    )
    bundle.dataset_bundle = dataset_bundle
    return bundle
