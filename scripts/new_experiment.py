#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments import scaffold_experiment_pack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Scaffold a new paper/idea experiment pack.')
    parser.add_argument('--repo-root', type=str, default='.', help='Repository root path.')
    parser.add_argument('--slug', type=str, required=True, help='Experiment slug (e.g., social-lstm-replication).')
    parser.add_argument('--title', type=str, required=True, help='Human-readable experiment title.')
    parser.add_argument('--objective', type=str, required=True, help='One-line experiment objective.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing scaffold files.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = scaffold_experiment_pack(
        repo_root=Path(args.repo_root),
        slug=args.slug,
        title=args.title,
        objective=args.objective,
        overwrite=bool(args.overwrite),
    )
    print('created:')
    for path in summary['created']:
        print(f'  - {path}')
    print('skipped:')
    for path in summary['skipped']:
        print(f'  - {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

