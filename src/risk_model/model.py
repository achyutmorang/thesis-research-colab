from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


ArrayDict = Dict[str, np.ndarray]


@dataclass
class NumpyEnsembleMLPConfig:
    input_dim: int
    output_dim: int
    ensemble_size: int = 5
    hidden_dims: Tuple[int, int] = (128, 128)
    dropout: float = 0.10
    learning_rate: float = 1e-3
    batch_size: int = 1024
    max_epochs: int = 50
    patience: int = 8
    seed: int = 17


@dataclass
class FitHistory:
    member_index: int
    best_epoch: int
    best_val_loss: float
    epochs_ran: int
    resumed_from_checkpoint: bool = False
    checkpoint_path: str = ''


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def _bce_with_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=float)
    loss = np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    return float(np.mean(loss))


class NumpyEnsembleMLP:
    def __init__(self, config: NumpyEnsembleMLPConfig):
        self.config = config
        self.params: List[ArrayDict] = [self._init_member(member_idx=i) for i in range(int(config.ensemble_size))]
        self.histories: List[FitHistory] = []

    def _init_member(self, member_idx: int) -> ArrayDict:
        rng = np.random.default_rng(int(self.config.seed + member_idx * 9973))
        dims = [int(self.config.input_dim), *[int(x) for x in self.config.hidden_dims], int(self.config.output_dim)]
        params: ArrayDict = {}
        for layer_idx, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
            scale = np.sqrt(2.0 / max(1, din + dout))
            params[f'W{layer_idx}'] = rng.normal(loc=0.0, scale=scale, size=(din, dout)).astype(np.float64)
            params[f'b{layer_idx}'] = np.zeros((dout,), dtype=np.float64)
        return params

    def _forward(self, params: ArrayDict, x: np.ndarray, training: bool, rng: np.random.Generator) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
        a = np.asarray(x, dtype=np.float64)
        caches: List[Dict[str, np.ndarray]] = []
        n_hidden = len(self.config.hidden_dims)
        for layer_idx in range(n_hidden):
            z = a @ params[f'W{layer_idx}'] + params[f'b{layer_idx}']
            h = _relu(z)
            mask = None
            if training and float(self.config.dropout) > 0.0:
                keep = max(1e-6, 1.0 - float(self.config.dropout))
                mask = (rng.random(h.shape) < keep).astype(np.float64) / keep
                h = h * mask
            caches.append({'a_prev': a, 'z': z, 'h': h, 'mask': mask})
            a = h
        logits = a @ params[f'W{n_hidden}'] + params[f'b{n_hidden}']
        caches.append({'a_prev': a})
        return logits, caches

    def _loss_and_grads(self, params: ArrayDict, x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> Tuple[float, ArrayDict]:
        logits, caches = self._forward(params, x, training=True, rng=rng)
        probs = _sigmoid(logits)
        batch = max(1, x.shape[0])
        dlogits = (probs - y) / float(batch)

        n_hidden = len(self.config.hidden_dims)
        grads: ArrayDict = {}
        a_last = caches[-1]['a_prev']
        grads[f'W{n_hidden}'] = a_last.T @ dlogits
        grads[f'b{n_hidden}'] = np.sum(dlogits, axis=0)

        da = dlogits @ params[f'W{n_hidden}'].T
        for layer_idx in reversed(range(n_hidden)):
            cache = caches[layer_idx]
            z = cache['z']
            hmask = cache['mask']
            dz = da * (z > 0.0).astype(np.float64)
            if hmask is not None:
                dz = dz * hmask
            a_prev = cache['a_prev']
            grads[f'W{layer_idx}'] = a_prev.T @ dz
            grads[f'b{layer_idx}'] = np.sum(dz, axis=0)
            da = dz @ params[f'W{layer_idx}'].T

        return _bce_with_logits(logits, y), grads

    def _iterate_minibatches(self, n_samples: int, rng: np.random.Generator) -> List[np.ndarray]:
        order = np.arange(n_samples, dtype=int)
        rng.shuffle(order)
        step = int(max(1, self.config.batch_size))
        return [order[i:i + step] for i in range(0, n_samples, step)]

    @staticmethod
    def _pack_params(prefix: str, params: ArrayDict) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name, value in params.items():
            out[f'{prefix}{name}'] = np.asarray(value, dtype=np.float64)
        return out

    @staticmethod
    def _unpack_params(payload: Any, prefix: str) -> ArrayDict:
        out: ArrayDict = {}
        for name in payload.files:
            if name.startswith(prefix):
                out[name[len(prefix):]] = np.asarray(payload[name], dtype=np.float64)
        return out

    def _member_checkpoint_path(self, checkpoint_prefix: str, member_idx: int) -> Path:
        return Path(f'{checkpoint_prefix}_risk_train_ckpt_member_{int(member_idx)}.npz')

    def _save_member_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        member_idx: int,
        current_params: ArrayDict,
        best_params: ArrayDict,
        last_epoch: int,
        best_epoch: int,
        best_val_loss: float,
        wait: int,
        epochs_ran: int,
        status_done: bool,
    ) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, np.ndarray] = {
            'member_index': np.asarray([int(member_idx)], dtype=np.int32),
            'last_epoch': np.asarray([int(last_epoch)], dtype=np.int32),
            'best_epoch': np.asarray([int(best_epoch)], dtype=np.int32),
            'best_val_loss': np.asarray([float(best_val_loss)], dtype=np.float64),
            'wait': np.asarray([int(wait)], dtype=np.int32),
            'epochs_ran': np.asarray([int(max(0, epochs_ran))], dtype=np.int32),
            'status_done': np.asarray([1 if bool(status_done) else 0], dtype=np.int32),
            'config_input_dim': np.asarray([int(self.config.input_dim)], dtype=np.int32),
            'config_output_dim': np.asarray([int(self.config.output_dim)], dtype=np.int32),
            'config_hidden_dims': np.asarray([int(x) for x in self.config.hidden_dims], dtype=np.int32),
        }
        payload.update(self._pack_params('cur__', current_params))
        payload.update(self._pack_params('best__', best_params))
        np.savez_compressed(str(checkpoint_path), **payload)

    def _load_member_checkpoint(
        self,
        checkpoint_path: Path,
        member_idx: int,
        *,
        expected_params: Optional[ArrayDict] = None,
    ) -> Optional[Dict[str, Any]]:
        if not checkpoint_path.exists():
            return None
        try:
            payload = np.load(str(checkpoint_path), allow_pickle=False)
            ckpt_member = int(np.asarray(payload['member_index']).reshape(-1)[0])
            if ckpt_member != int(member_idx):
                return None
            if 'config_input_dim' in payload.files:
                ckpt_input_dim = int(np.asarray(payload['config_input_dim']).reshape(-1)[0])
                if ckpt_input_dim != int(self.config.input_dim):
                    return None
            if 'config_output_dim' in payload.files:
                ckpt_output_dim = int(np.asarray(payload['config_output_dim']).reshape(-1)[0])
                if ckpt_output_dim != int(self.config.output_dim):
                    return None
            if 'config_hidden_dims' in payload.files:
                ckpt_hidden = tuple(int(x) for x in np.asarray(payload['config_hidden_dims']).reshape(-1))
                if ckpt_hidden != tuple(int(x) for x in self.config.hidden_dims):
                    return None
            current_params = self._unpack_params(payload, 'cur__')
            best_params = self._unpack_params(payload, 'best__')
            if len(current_params) == 0 or len(best_params) == 0:
                return None
            if expected_params is not None:
                expected_keys = set(expected_params.keys())
                if set(current_params.keys()) != expected_keys or set(best_params.keys()) != expected_keys:
                    return None
                for name in expected_keys:
                    exp_shape = tuple(expected_params[name].shape)
                    if tuple(current_params[name].shape) != exp_shape:
                        return None
                    if tuple(best_params[name].shape) != exp_shape:
                        return None
            return {
                'member_index': int(ckpt_member),
                'last_epoch': int(np.asarray(payload['last_epoch']).reshape(-1)[0]),
                'best_epoch': int(np.asarray(payload['best_epoch']).reshape(-1)[0]),
                'best_val_loss': float(np.asarray(payload['best_val_loss']).reshape(-1)[0]),
                'wait': int(np.asarray(payload['wait']).reshape(-1)[0]),
                'epochs_ran': int(np.asarray(payload['epochs_ran']).reshape(-1)[0]),
                'status_done': bool(int(np.asarray(payload['status_done']).reshape(-1)[0]) > 0),
                'current_params': current_params,
                'best_params': best_params,
            }
        except Exception:
            return None

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        *,
        checkpoint_prefix: Optional[str] = None,
        checkpoint_every_epochs: int = 1,
        resume_from_checkpoints: bool = True,
    ) -> List[FitHistory]:
        x_train = np.asarray(x_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        x_val = np.asarray(x_val, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.float64)
        checkpoint_every_epochs = int(max(1, checkpoint_every_epochs))

        histories: List[FitHistory] = []
        for member_idx, params in enumerate(self.params):
            rng = np.random.default_rng(int(self.config.seed + member_idx * 7919))
            best_params = {k: v.copy() for k, v in params.items()}
            best_val = float('inf')
            best_epoch = -1
            wait = 0
            start_epoch = 0
            resumed_from_checkpoint = False
            checkpoint_path: Optional[Path] = None
            if checkpoint_prefix is not None and str(checkpoint_prefix).strip():
                checkpoint_path = self._member_checkpoint_path(str(checkpoint_prefix), int(member_idx))
                if bool(resume_from_checkpoints):
                    ckpt = self._load_member_checkpoint(
                        checkpoint_path,
                        member_idx=int(member_idx),
                        expected_params=params,
                    )
                    if ckpt is not None:
                        resumed_from_checkpoint = True
                        params = {k: v.copy() for k, v in ckpt['current_params'].items()}
                        best_params = {k: v.copy() for k, v in ckpt['best_params'].items()}
                        best_val = float(ckpt['best_val_loss'])
                        best_epoch = int(ckpt['best_epoch'])
                        wait = int(max(0, ckpt['wait']))
                        start_epoch = int(max(0, ckpt['last_epoch'] + 1))

                        if bool(ckpt['status_done']) or start_epoch >= int(self.config.max_epochs):
                            self.params[member_idx] = best_params
                            histories.append(
                                FitHistory(
                                    member_index=int(member_idx),
                                    best_epoch=int(best_epoch),
                                    best_val_loss=float(best_val),
                                    epochs_ran=int(max(0, ckpt['epochs_ran'])),
                                    resumed_from_checkpoint=True,
                                    checkpoint_path=str(checkpoint_path),
                                )
                            )
                            continue

            last_epoch_completed = int(start_epoch - 1)
            for epoch in range(int(start_epoch), int(self.config.max_epochs)):
                for batch_idx in self._iterate_minibatches(x_train.shape[0], rng):
                    loss, grads = self._loss_and_grads(params, x_train[batch_idx], y_train[batch_idx], rng)
                    del loss
                    for name in params:
                        params[name] = params[name] - float(self.config.learning_rate) * grads[name]

                val_logits, _ = self._forward(params, x_val, training=False, rng=rng)
                val_loss = _bce_with_logits(val_logits, y_val)
                last_epoch_completed = int(epoch)
                if val_loss + 1e-10 < best_val:
                    best_val = float(val_loss)
                    best_epoch = int(epoch)
                    best_params = {k: v.copy() for k, v in params.items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= int(self.config.patience):
                        if checkpoint_path is not None:
                            self._save_member_checkpoint(
                                checkpoint_path=checkpoint_path,
                                member_idx=int(member_idx),
                                current_params=params,
                                best_params=best_params,
                                last_epoch=int(epoch),
                                best_epoch=int(best_epoch),
                                best_val_loss=float(best_val),
                                wait=int(wait),
                                epochs_ran=int(epoch + 1),
                                status_done=True,
                            )
                        break

                if checkpoint_path is not None and (((int(epoch) + 1) % checkpoint_every_epochs) == 0):
                    self._save_member_checkpoint(
                        checkpoint_path=checkpoint_path,
                        member_idx=int(member_idx),
                        current_params=params,
                        best_params=best_params,
                        last_epoch=int(epoch),
                        best_epoch=int(best_epoch),
                        best_val_loss=float(best_val),
                        wait=int(wait),
                        epochs_ran=int(epoch + 1),
                        status_done=False,
                    )

            self.params[member_idx] = best_params
            epochs_ran = int(max(0, last_epoch_completed + 1))
            if checkpoint_path is not None:
                self._save_member_checkpoint(
                    checkpoint_path=checkpoint_path,
                    member_idx=int(member_idx),
                    current_params=params,
                    best_params=best_params,
                    last_epoch=int(max(last_epoch_completed, start_epoch - 1)),
                    best_epoch=int(best_epoch),
                    best_val_loss=float(best_val),
                    wait=int(wait),
                    epochs_ran=int(epochs_ran),
                    status_done=True,
                )
            histories.append(
                FitHistory(
                    member_index=int(member_idx),
                    best_epoch=int(best_epoch),
                    best_val_loss=float(best_val),
                    epochs_ran=int(epochs_ran),
                    resumed_from_checkpoint=bool(resumed_from_checkpoint),
                    checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else '',
                )
            )
        self.histories = histories
        return histories

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        logits = []
        rng = np.random.default_rng(int(self.config.seed))
        for params in self.params:
            out, _ = self._forward(params, x, training=False, rng=rng)
            logits.append(out.astype(np.float64))
        return np.stack(logits, axis=0)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(self.predict_logits(x))

    def predict_with_uncertainty(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        logits = self.predict_logits(x)
        probs = _sigmoid(logits)
        return {
            'member_logits': logits,
            'member_probs': probs,
            'mean_logits': np.mean(logits, axis=0),
            'mean_probs': np.mean(probs, axis=0),
            'epistemic_var': np.var(probs, axis=0),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            'config': {
                'input_dim': int(self.config.input_dim),
                'output_dim': int(self.config.output_dim),
                'ensemble_size': int(self.config.ensemble_size),
                'hidden_dims': tuple(int(x) for x in self.config.hidden_dims),
                'dropout': float(self.config.dropout),
                'learning_rate': float(self.config.learning_rate),
                'batch_size': int(self.config.batch_size),
                'max_epochs': int(self.config.max_epochs),
                'patience': int(self.config.patience),
                'seed': int(self.config.seed),
            },
            'params': [{k: v.copy() for k, v in member.items()} for member in self.params],
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> 'NumpyEnsembleMLP':
        cfg = NumpyEnsembleMLPConfig(**state['config'])
        obj = cls(cfg)
        obj.params = [{k: np.asarray(v, dtype=np.float64) for k, v in member.items()} for member in state['params']]
        return obj
