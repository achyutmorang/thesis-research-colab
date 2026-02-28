from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .model import NumpyEnsembleMLP, NumpyEnsembleMLPConfig


@dataclass
class RiskTrainingBundle:
    model: NumpyEnsembleMLP
    feature_columns: List[str]
    label_columns: List[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    train_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


DEFAULT_LABEL_COLUMNS = [
    'collision_h5', 'collision_h10', 'collision_h15',
    'offroad_h5', 'offroad_h10', 'offroad_h15',
    'failure_proxy_h5', 'failure_proxy_h10', 'failure_proxy_h15',
]

DEFAULT_EXCLUDE_COLUMNS = {
    'scenario_id', 'eval_split', 'shift_suite', 'step_idx', 'candidate_id', 'candidate_source', 'planner_backend', 'seed'
}


def infer_feature_columns(df: pd.DataFrame, label_columns: Sequence[str]) -> List[str]:
    cols: List[str] = []
    label_set = set(label_columns)
    for col in df.columns:
        if col in label_set or col in DEFAULT_EXCLUDE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _standardize_frame(df: pd.DataFrame, feature_columns: Sequence[str], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = df.loc[:, list(feature_columns)].astype(float).to_numpy(copy=True)
    return (x - mean[None, :]) / std[None, :]


def _validation_metrics(logits: np.ndarray, y: np.ndarray, label_columns: Sequence[str]) -> pd.DataFrame:
    probs = 1.0 / (1.0 + np.exp(-logits))
    rows = []
    for idx, label in enumerate(label_columns):
        yy = y[:, idx]
        pp = probs[:, idx]
        pp = np.clip(pp, 1e-6, 1.0 - 1e-6)
        nll = float(np.mean(-(yy * np.log(pp) + (1.0 - yy) * np.log(1.0 - pp))))
        brier = float(np.mean((pp - yy) ** 2))
        rows.append({'label': label, 'nll': nll, 'brier': brier, 'positive_rate': float(np.mean(yy))})
    return pd.DataFrame(rows)


def train_risk_ensemble(
    dataset_df: pd.DataFrame,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    label_columns: Sequence[str] = DEFAULT_LABEL_COLUMNS,
    ensemble_size: int = 5,
    hidden_dims: Tuple[int, int] = (128, 128),
    dropout: float = 0.10,
    learning_rate: float = 1e-3,
    batch_size: int = 1024,
    max_epochs: int = 50,
    patience: int = 8,
    seed: int = 17,
    checkpoint_prefix: Optional[str] = None,
    checkpoint_every_epochs: int = 1,
    resume_from_checkpoints: bool = True,
) -> RiskTrainingBundle:
    if dataset_df.empty:
        raise ValueError('dataset_df is empty.')
    if 'eval_split' not in dataset_df.columns:
        raise ValueError('dataset_df must contain eval_split.')

    label_columns = list(label_columns)
    feature_columns = list(feature_columns) if feature_columns is not None else infer_feature_columns(dataset_df, label_columns)
    if len(feature_columns) == 0:
        raise ValueError('No numeric feature columns found for risk training.')

    train_df = dataset_df[dataset_df['eval_split'] == 'train'].copy()
    val_df = dataset_df[dataset_df['eval_split'] == 'val'].copy()
    if train_df.empty or val_df.empty:
        raise ValueError('Need non-empty train and val splits for risk training.')

    y_train = train_df.loc[:, label_columns].astype(float).to_numpy(copy=True)
    y_val = val_df.loc[:, label_columns].astype(float).to_numpy(copy=True)
    x_train_raw = train_df.loc[:, feature_columns].astype(float).to_numpy(copy=True)

    feat_mean = np.nanmean(x_train_raw, axis=0)
    feat_std = np.nanstd(x_train_raw, axis=0)
    feat_std = np.where(np.isfinite(feat_std) & (feat_std > 1e-6), feat_std, 1.0)

    x_train = _standardize_frame(train_df, feature_columns, feat_mean, feat_std)
    x_val = _standardize_frame(val_df, feature_columns, feat_mean, feat_std)

    cfg = NumpyEnsembleMLPConfig(
        input_dim=len(feature_columns),
        output_dim=len(label_columns),
        ensemble_size=ensemble_size,
        hidden_dims=hidden_dims,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
    )
    model = NumpyEnsembleMLP(cfg)
    histories = model.fit(
        x_train,
        y_train,
        x_val,
        y_val,
        checkpoint_prefix=checkpoint_prefix,
        checkpoint_every_epochs=int(max(1, checkpoint_every_epochs)),
        resume_from_checkpoints=bool(resume_from_checkpoints),
    )

    pred = model.predict_with_uncertainty(x_val)
    val_metrics = _validation_metrics(pred['mean_logits'], y_val, label_columns)
    hist_df = pd.DataFrame([
        {
            'member_index': int(h.member_index),
            'best_epoch': int(h.best_epoch),
            'best_val_loss': float(h.best_val_loss),
            'epochs_ran': int(h.epochs_ran),
            'resumed_from_checkpoint': int(bool(h.resumed_from_checkpoint)),
            'checkpoint_path': str(h.checkpoint_path),
        }
        for h in histories
    ])
    train_summary = hist_df.merge(val_metrics.assign(key=1), how='cross') if not hist_df.empty else val_metrics

    val_pred_df = val_df[['scenario_id', 'eval_split', 'shift_suite', 'candidate_id']].copy()
    for idx, label in enumerate(label_columns):
        val_pred_df[f'logit_{label}'] = pred['mean_logits'][:, idx]
        val_pred_df[f'prob_{label}'] = pred['mean_probs'][:, idx]
        val_pred_df[f'epistemic_{label}'] = pred['epistemic_var'][:, idx]
        val_pred_df[label] = y_val[:, idx]

    return RiskTrainingBundle(
        model=model,
        feature_columns=feature_columns,
        label_columns=label_columns,
        feature_mean=feat_mean,
        feature_std=feat_std,
        train_summary=train_summary,
        validation_predictions=val_pred_df,
    )
