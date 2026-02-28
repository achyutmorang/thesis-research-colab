from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .spec import ExperimentPack, normalize_slug


_PACKS: Sequence[ExperimentPack] = (
    ExperimentPack(
        slug='closedloop-core',
        title='Closed-Loop Core (Simulation + Evaluation)',
        objective='Run closed-loop simulation and post-hoc evaluation from one unified experiment pack.',
        notebooks=(
            'experiments/closedloop-core/notebooks/closedloop_simulation_colab.ipynb',
            'experiments/closedloop-core/notebooks/closedloop_evaluation_colab.ipynb',
            'experiments/closedloop-core/notebooks/compute_normalized_blindspot_discovery_colab.ipynb',
            'experiments/closedloop-core/notebooks/counterfactual_risk_sensitivity_atlas_colab.ipynb',
        ),
        workflows=(
            'src/workflows/closedloop_flow.py',
        ),
        modules=(
            'src/closedloop',
            'src/eval',
            'src/eval_compute_normalized_discovery',
            'src/eval_counterfactual_risk_sensitivity',
            'src/workflows/closedloop_flow.py',
        ),
        config_paths=(
            'configs/experiments/closedloop-core.json',
        ),
        tags=('closedloop', 'simulation', 'evaluation', 'search', 'waymax', 'womd'),
    ),
    ExperimentPack(
        slug='surprise-potential',
        title='Surprise Potential Sweep',
        objective='Compare surprise signal instantiations and counterfactual families before expensive runs.',
        notebooks=(
            'experiments/surprise-potential/notebooks/surprise_potential_closedloop_colab.ipynb',
        ),
        workflows=(
            'src/workflows/surprise_potential_flow.py',
        ),
        modules=(
            'src/workflows/surprise_potential_flow.py',
            'src/closedloop/signal_analysis.py',
            'src/closedloop/calibration.py',
        ),
        config_paths=(
            'configs/experiments/surprise-potential.json',
        ),
        tags=('closedloop', 'surprise', 'counterfactual', 'diagnostics'),
    ),
    ExperimentPack(
        slug='risk-uq-suite',
        title='Risk And Uncertainty Suite',
        objective='Train calibrated risk models and benchmark robustness/calibration with publication exports.',
        notebooks=(
            'experiments/risk-uq-suite/notebooks/risk_model_training_colab.ipynb',
            'experiments/risk-uq-suite/notebooks/uq_benchmark_colab.ipynb',
            'experiments/risk-uq-suite/notebooks/paper_tables_figures_colab.ipynb',
        ),
        workflows=(
            'src/workflows/risk_training_flow.py',
            'src/workflows/uq_benchmark_flow.py',
            'src/workflows/paper_export_flow.py',
        ),
        modules=(
            'src/risk_model',
            'src/workflows/risk_training_flow.py',
            'src/workflows/uq_benchmark_flow.py',
            'src/workflows/paper_export_flow.py',
        ),
        config_paths=(
            'configs/experiments/risk-uq-suite.json',
        ),
        tags=('risk', 'uq', 'training', 'benchmark', 'paper'),
    ),
)


def list_experiment_packs() -> List[ExperimentPack]:
    return list(_PACKS)


def get_experiment_pack(slug: str) -> ExperimentPack:
    key = normalize_slug(slug)
    for pack in _PACKS:
        if pack.slug == key:
            return pack
    raise KeyError(f'Unknown experiment pack: {slug!r}')


def find_experiment_packs(query: str, tags: Optional[Iterable[str]] = None) -> List[ExperimentPack]:
    q = str(query).strip().lower()
    wanted_tags = {normalize_slug(t) for t in list(tags or ()) if str(t).strip()}
    out: List[ExperimentPack] = []
    for pack in _PACKS:
        hay = ' '.join([pack.slug, pack.title, pack.objective, ' '.join(pack.tags)]).lower()
        if q and (q not in hay):
            continue
        if wanted_tags and not wanted_tags.issubset(set(pack.tags)):
            continue
        out.append(pack)
    return out


def validate_pack_paths(repo_root: str | Path, pack: ExperimentPack) -> Dict[str, List[str]]:
    root = Path(repo_root).expanduser().resolve()
    required: List[str] = []
    required.extend(pack.notebooks)
    required.extend(pack.workflows)
    required.extend(pack.modules)
    required.extend(pack.config_paths)

    existing: List[str] = []
    missing: List[str] = []
    for rel in required:
        p = root / rel
        if p.exists():
            existing.append(rel)
        else:
            missing.append(rel)
    return {'existing': existing, 'missing': missing}


def validate_registry(repo_root: str | Path) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    for pack in _PACKS:
        out[pack.slug] = validate_pack_paths(repo_root, pack)
    return out
