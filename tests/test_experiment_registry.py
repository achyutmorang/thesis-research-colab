from __future__ import annotations

from pathlib import Path

from src.experiments import (
    find_experiment_packs,
    get_experiment_pack,
    list_experiment_packs,
    validate_registry,
)


def test_registry_contains_core_packs() -> None:
    slugs = {pack.slug for pack in list_experiment_packs()}
    assert 'closedloop-simulation' in slugs
    assert 'surprise-potential' in slugs
    assert 'closedloop-evaluation' in slugs
    assert 'risk-uq-suite' in slugs


def test_get_pack_and_find_by_tag() -> None:
    pack = get_experiment_pack('closedloop-simulation')
    assert pack.slug == 'closedloop-simulation'
    assert 'waymax' in pack.tags

    risk = find_experiment_packs(query='', tags=['risk'])
    risk_slugs = {p.slug for p in risk}
    assert 'risk-uq-suite' in risk_slugs


def test_registry_paths_are_valid() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    report = validate_registry(repo_root)
    assert report
    for _, status in report.items():
        assert status['missing'] == []

