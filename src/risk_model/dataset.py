from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_ID_COLUMNS = [
    'scenario_id',
    'eval_split',
    'shift_suite',
    'step_idx',
    'candidate_id',
    'candidate_source',
    'planner_backend',
    'seed',
]


def deterministic_scenario_split(
    scenario_ids: Sequence[Any],
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    seed: int = 17,
) -> pd.DataFrame:
    train_fraction = float(np.clip(train_fraction, 0.0, 1.0))
    val_fraction = float(np.clip(val_fraction, 0.0, 1.0 - train_fraction))
    rows: List[Dict[str, Any]] = []
    for sid in pd.Series(list(scenario_ids)).drop_duplicates().tolist():
        token = f'{seed}:{sid}'.encode('utf-8')
        digest = hashlib.sha256(token).hexdigest()
        unit = int(digest[:8], 16) / float(16 ** 8)
        if unit < train_fraction:
            split = 'train'
        elif unit < (train_fraction + val_fraction):
            split = 'val'
        else:
            split = 'test'
        rows.append({'scenario_id': sid, 'eval_split': split})
    return pd.DataFrame(rows)


def add_eval_splits(
    df: pd.DataFrame,
    seed: int = 17,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    high_interaction_fraction: float = 0.20,
    interaction_col: str = 'target_interaction_score',
) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        if 'eval_split' not in out.columns:
            out['eval_split'] = pd.Series(dtype=object)
        return out

    out = df.copy()
    split_df = deterministic_scenario_split(
        scenario_ids=out['scenario_id'].tolist(),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=seed,
    )
    out = out.drop(columns=['eval_split'], errors='ignore').merge(split_df, on='scenario_id', how='left')

    if interaction_col in out.columns:
        score_tab = (
            out.groupby('scenario_id', as_index=False)[interaction_col]
            .max()
            .rename(columns={interaction_col: '_interaction_max'})
        )
        if not score_tab.empty:
            threshold = float(score_tab['_interaction_max'].quantile(max(0.0, 1.0 - float(high_interaction_fraction))))
            holdout_ids = set(score_tab.loc[score_tab['_interaction_max'] >= threshold, 'scenario_id'].tolist())
            mask = out['scenario_id'].isin(holdout_ids)
            out.loc[mask, 'eval_split'] = 'high_interaction_holdout'
    return out


def build_risk_dataset(
    rows: Iterable[Dict[str, Any]],
    *,
    seed: int = 17,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    high_interaction_fraction: float = 0.20,
    interaction_col: str = 'target_interaction_score',
) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df

    for col in DEFAULT_ID_COLUMNS:
        if col not in df.columns:
            if col == 'eval_split':
                df[col] = 'train'
            elif col == 'shift_suite':
                df[col] = 'nominal_clean'
            elif col == 'step_idx':
                df[col] = 0
            elif col == 'candidate_id':
                df[col] = np.arange(len(df), dtype=int)
            elif col == 'candidate_source':
                df[col] = 'unspecified'
            elif col == 'planner_backend':
                df[col] = 'latentdriver'
            elif col == 'seed':
                df[col] = int(seed)
            else:
                df[col] = ''

    df = add_eval_splits(
        df,
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        high_interaction_fraction=high_interaction_fraction,
        interaction_col=interaction_col,
    )
    return df
