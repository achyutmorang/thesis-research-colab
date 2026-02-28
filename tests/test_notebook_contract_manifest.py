from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def _load_module(module_name: str, rel_path: str):
    root = Path(__file__).resolve().parents[1]
    path = root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load module {module_name} from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


notebook_contract = _load_module('notebook_contract_direct', 'src/workflows/notebook_contract.py')


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        global_seed=17,
        run_prefix='unused',
        planner_name='latentdriver_waypoint_sdc',
        risk_model_ensemble_size=5,
    )


def _search_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        budget_evals=15,
        random_scale=0.35,
    )


def test_manifest_write_load_validate_roundtrip(tmp_path: Path) -> None:
    run_prefix = str(tmp_path / 'risk_uq_20260228_010203')
    path = notebook_contract.write_notebook_contract_manifest(
        run_prefix=run_prefix,
        run_tag='risk_uq_20260228_010203',
        cfg=_cfg(),
        search_cfg=_search_cfg(),
        n_shards=1,
        shard_id=0,
        notebook_name='risk_model_training_colab',
        stage='gates_passed',
        git_commit='abc123',
        quick_probe_pass=True,
        preflight_pass=True,
        extra_fields={'foo': 1},
    )
    assert Path(path).exists()
    manifest = notebook_contract.load_notebook_contract_manifest(run_prefix)
    ok, reasons = notebook_contract.validate_notebook_contract_manifest(
        manifest,
        require_quick_probe=True,
        require_preflight=True,
        required_stages=('gates_passed',),
    )
    assert ok, reasons
    assert notebook_contract.manifest_has_stage(manifest, 'gates_passed') is True
    assert manifest.get('git_commit') == 'abc123'


def test_manifest_validation_detects_missing_stages(tmp_path: Path) -> None:
    run_prefix = str(tmp_path / 'risk_uq_20260228_999999')
    notebook_contract.write_notebook_contract_manifest(
        run_prefix=run_prefix,
        run_tag='risk_uq_20260228_999999',
        cfg=_cfg(),
        search_cfg=_search_cfg(),
        n_shards=1,
        shard_id=0,
        notebook_name='uq_benchmark_colab',
        stage='uq_benchmark_completed',
        git_commit='def456',
        quick_probe_pass=True,
        preflight_pass=True,
    )
    manifest = notebook_contract.load_notebook_contract_manifest(run_prefix)
    ok, reasons = notebook_contract.validate_notebook_contract_manifest(
        manifest,
        require_quick_probe=True,
        require_preflight=True,
        required_stages=('risk_training_completed',),
    )
    assert ok is False
    assert any(str(r).startswith('missing_stage:risk_training_completed') for r in reasons)


def test_write_contract_storage_mirror_creates_expected_layout(tmp_path: Path) -> None:
    persist_root = tmp_path / 'persist'
    flat_run_prefix = str(tmp_path / 'risk_uq_20260228_010203')
    metrics_path = tmp_path / 'metrics.csv'
    metrics_path.write_text('metric,value\nece,0.1\n')

    out = notebook_contract.write_contract_storage_mirror(
        persist_root=str(persist_root),
        run_prefix='risk_uq',
        run_name='20260228_010203',
        run_prefix_path=flat_run_prefix,
        cfg=_cfg(),
        search_cfg=_search_cfg(),
        n_shards=1,
        shard_id=0,
        stage='risk_training_completed',
        git_commit='abc123',
        resume_from_existing=True,
        run_enabled=True,
        artifact_paths={'risk_dataset': f'{flat_run_prefix}_risk_dataset.parquet'},
        metrics_csv_path=str(metrics_path),
        extra_fields={'foo': 1},
    )

    run_dir = Path(out['contract_run_dir'])
    assert run_dir.exists()
    assert (run_dir / 'config.json').exists()
    assert (run_dir / 'env_manifest.json').exists()
    assert (run_dir / 'run_manifest.json').exists()
    assert (run_dir / 'progress' / 'shard_0.json').exists()
    assert (run_dir / 'checkpoints' / 'latest.json').exists()
    assert (run_dir / 'outputs' / 'metrics.csv').exists()
    assert (run_dir / 'outputs' / 'artifacts' / 'artifact_index.json').exists()

    run_manifest = json.loads((run_dir / 'run_manifest.json').read_text())
    assert run_manifest.get('run_name') == '20260228_010203'
    assert run_manifest.get('run_prefix') == 'risk_uq'
    assert run_manifest.get('git_commit') == 'abc123'
