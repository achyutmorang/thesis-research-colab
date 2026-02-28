from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.risk_model.model import NumpyEnsembleMLP, NumpyEnsembleMLPConfig
from src.risk_model.train import train_risk_ensemble


LABEL_COLS = [
    'collision_h5', 'collision_h10', 'collision_h15',
    'offroad_h5', 'offroad_h10', 'offroad_h15',
    'failure_proxy_h5', 'failure_proxy_h10', 'failure_proxy_h15',
]


def _synthetic_dataset(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.9 * x1 - 0.6 * x2 + 0.4 * x3)))
    y = (rng.random(n) < p).astype(float)

    df = pd.DataFrame(
        {
            'scenario_id': np.arange(n, dtype=int),
            'eval_split': np.where(np.arange(n) < int(0.7 * n), 'train', 'val'),
            'shift_suite': 'nominal_clean',
            'candidate_id': 0,
            'feature_a': x1,
            'feature_b': x2,
            'feature_c': x3,
        }
    )
    for col in LABEL_COLS:
        if col.endswith('h5'):
            df[col] = y
        elif col.endswith('h10'):
            df[col] = np.clip(y + (rng.random(n) < 0.1).astype(float), 0.0, 1.0)
        else:
            df[col] = np.clip(y + (rng.random(n) < 0.15).astype(float), 0.0, 1.0)
    return df


def test_numpy_ensemble_shape_and_round_trip_state_dict() -> None:
    cfg = NumpyEnsembleMLPConfig(input_dim=4, output_dim=3, ensemble_size=3, hidden_dims=(16, 8), max_epochs=3, patience=2)
    model = NumpyEnsembleMLP(cfg)
    x = np.random.default_rng(3).normal(size=(10, 4))
    pred = model.predict_with_uncertainty(x)
    assert pred['mean_logits'].shape == (10, 3)
    assert pred['mean_probs'].shape == (10, 3)
    assert pred['epistemic_var'].shape == (10, 3)

    restored = NumpyEnsembleMLP.from_state_dict(model.state_dict())
    pred_restored = restored.predict_with_uncertainty(x)
    assert np.allclose(pred['mean_probs'], pred_restored['mean_probs'])


def test_train_risk_ensemble_produces_expected_validation_columns() -> None:
    df = _synthetic_dataset()
    bundle = train_risk_ensemble(
        df,
        ensemble_size=2,
        hidden_dims=(32, 16),
        max_epochs=4,
        patience=2,
        batch_size=64,
        seed=19,
    )

    assert len(bundle.feature_columns) >= 3
    assert list(bundle.label_columns) == LABEL_COLS
    assert not bundle.validation_predictions.empty
    for col in LABEL_COLS:
        assert f'logit_{col}' in bundle.validation_predictions.columns
        assert f'prob_{col}' in bundle.validation_predictions.columns
        assert f'epistemic_{col}' in bundle.validation_predictions.columns


def test_train_risk_ensemble_reuses_completed_member_checkpoints(tmp_path: Path) -> None:
    df = _synthetic_dataset()
    run_prefix = str(tmp_path / 'risk_model_ckpt_reuse')

    first = train_risk_ensemble(
        df,
        ensemble_size=2,
        hidden_dims=(32, 16),
        max_epochs=4,
        patience=2,
        batch_size=64,
        seed=19,
        checkpoint_prefix=run_prefix,
        checkpoint_every_epochs=1,
        resume_from_checkpoints=True,
    )
    assert not first.train_summary.empty

    second = train_risk_ensemble(
        df,
        ensemble_size=2,
        hidden_dims=(32, 16),
        max_epochs=4,
        patience=2,
        batch_size=64,
        seed=19,
        checkpoint_prefix=run_prefix,
        checkpoint_every_epochs=1,
        resume_from_checkpoints=True,
    )
    resumed_flags = second.train_summary['resumed_from_checkpoint'].astype(int).to_numpy()
    assert np.all(resumed_flags == 1)
