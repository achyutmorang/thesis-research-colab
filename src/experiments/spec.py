from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence, Tuple


def normalize_slug(value: str) -> str:
    slug = ''.join(ch if (ch.isalnum() or ch in {'-', '_'}) else '-' for ch in str(value).strip().lower())
    slug = slug.strip('-_')
    if not slug:
        raise ValueError('Experiment slug cannot be empty.')
    return slug


def normalize_tags(values: Sequence[str]) -> Tuple[str, ...]:
    out = []
    for value in values:
        token = normalize_slug(str(value))
        if token not in out:
            out.append(token)
    return tuple(out)


@dataclass(frozen=True)
class ExperimentPack:
    slug: str
    title: str
    objective: str
    notebooks: Tuple[str, ...] = field(default_factory=tuple)
    workflows: Tuple[str, ...] = field(default_factory=tuple)
    modules: Tuple[str, ...] = field(default_factory=tuple)
    config_paths: Tuple[str, ...] = field(default_factory=tuple)
    tags: Tuple[str, ...] = field(default_factory=tuple)
    status: str = 'active'
    notes: str = ''

    def __post_init__(self) -> None:
        object.__setattr__(self, 'slug', normalize_slug(self.slug))
        object.__setattr__(self, 'tags', normalize_tags(self.tags))
        object.__setattr__(self, 'status', str(self.status).strip().lower() or 'active')

    def to_dict(self) -> Mapping[str, object]:
        return {
            'slug': self.slug,
            'title': self.title,
            'objective': self.objective,
            'notebooks': list(self.notebooks),
            'workflows': list(self.workflows),
            'modules': list(self.modules),
            'config_paths': list(self.config_paths),
            'tags': list(self.tags),
            'status': self.status,
            'notes': self.notes,
        }

