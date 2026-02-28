from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.closedloop.risk_benchmark import BenchmarkBundle
try:
    from .living_report import update_living_report_from_paper_export
except ImportError:  # pragma: no cover - supports direct module loading in tests
    from src.workflows.living_report import update_living_report_from_paper_export


@dataclass
class PaperExportBundle:
    output_dir: str
    exported_paths: Dict[str, str] = field(default_factory=dict)


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _save_frame(path: Path, df: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def _maybe_render_figures(output_dir: Path, benchmark: BenchmarkBundle) -> Dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    out: Dict[str, str] = {}
    if not benchmark.reliability_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        rel = benchmark.reliability_df.copy()
        rel = rel[rel['variant'] == rel['variant'].iloc[0]]
        ax.plot(rel['mean_prob'], rel['event_rate'], marker='o')
        ax.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1)
        ax.set_xlabel('Mean predicted risk')
        ax.set_ylabel('Observed event rate')
        ax.set_title('Reliability diagram')
        p = output_dir / 'reliability_diagram.png'
        fig.tight_layout()
        fig.savefig(p, dpi=160)
        plt.close(fig)
        out['reliability_diagram'] = str(p)

    if not benchmark.selective_curve_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        curve = benchmark.selective_curve_df.copy()
        for variant, grp in curve.groupby('variant', sort=True):
            ax.plot(grp['coverage'], grp['selective_risk'], label=str(variant))
        ax.set_xlabel('Coverage')
        ax.set_ylabel('Selective risk')
        ax.set_title('Coverage-risk tradeoff')
        ax.legend()
        p = output_dir / 'coverage_risk_curve.png'
        fig.tight_layout()
        fig.savefig(p, dpi=160)
        plt.close(fig)
        out['coverage_risk_curve'] = str(p)
    return out


def export_paper_tables_and_figures(
    *,
    run_prefix: str,
    benchmark_bundle: Optional[BenchmarkBundle] = None,
    output_dir: Optional[str] = None,
    cfg: Optional[Any] = None,
) -> PaperExportBundle:
    output = Path(output_dir or f'{run_prefix}_paper_exports')
    output.mkdir(parents=True, exist_ok=True)

    if benchmark_bundle is None:
        benchmark_bundle = BenchmarkBundle(
            summary_df=_load_frame(Path(f'{run_prefix}_uq_benchmark_summary.csv')),
            per_shift_df=_load_frame(Path(f'{run_prefix}_uq_benchmark_per_shift.csv')),
            reliability_df=_load_frame(Path(f'{run_prefix}_uq_reliability_bins.csv')),
            selective_curve_df=_load_frame(Path(f'{run_prefix}_uq_selective_risk_curve.csv')),
            shift_gap_df=_load_frame(Path(f'{run_prefix}_uq_shift_gap_summary.csv')),
        )

    exported = {
        'calibration_table': _save_frame(output / 'calibration_table.csv', benchmark_bundle.summary_df),
        'shift_robustness_table': _save_frame(output / 'shift_robustness_table.csv', benchmark_bundle.per_shift_df),
        'shift_gap_table': _save_frame(output / 'shift_gap_table.csv', benchmark_bundle.shift_gap_df),
        'reliability_bins': _save_frame(output / 'reliability_bins.csv', benchmark_bundle.reliability_df),
        'coverage_risk_table': _save_frame(output / 'coverage_risk_table.csv', benchmark_bundle.selective_curve_df),
    }
    exported.update(_maybe_render_figures(output, benchmark_bundle))
    exported.update(
        update_living_report_from_paper_export(
            cfg=cfg,
            run_prefix=run_prefix,
            exported_paths=exported,
        )
    )
    return PaperExportBundle(output_dir=str(output), exported_paths=exported)
