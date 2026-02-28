from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


REPORT_SCHEMA_VERSION = '1.0.0'
REPORT_REL_PATH = Path('experiments/risk-uq-suite/LIVING_REPORT.md')
REPORT_STATE_REL_PATH = Path('experiments/risk-uq-suite/LIVING_REPORT_STATE.json')


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _safe_float(value: Any, *, default: float = float('nan')) -> float:
    try:
        val = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(val):
        return float(default)
    return float(val)


def _mean_metric(df: pd.DataFrame, *, metric: str, filters: Mapping[str, Any]) -> float:
    if df.empty or (metric not in df.columns):
        return float('nan')
    mask = np.ones((len(df),), dtype=bool)
    for col, value in filters.items():
        if col not in df.columns:
            return float('nan')
        series = df[col]
        if isinstance(value, (list, tuple, set)):
            mask &= series.isin(list(value)).to_numpy(dtype=bool)
        else:
            mask &= series.astype(str).eq(str(value)).to_numpy(dtype=bool)
    values = pd.to_numeric(df.loc[mask, metric], errors='coerce').to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan')
    return float(np.mean(values))


def _format_float(value: float, *, places: int = 4) -> str:
    if not np.isfinite(value):
        return 'NA'
    return f'{float(value):.{int(max(0, places))}f}'


def _status_label(raw: str) -> str:
    key = str(raw).strip().lower()
    mapping = {
        'open': 'Open',
        'inconclusive': 'Inconclusive',
        'mixed': 'Mixed',
        'supported': 'Supported',
        'not_supported': 'Not Supported',
    }
    return mapping.get(key, key.title() if key else 'Open')


def _repo_root(repo_root: Optional[str | Path] = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _resolve_paths(
    *,
    run_prefix: Optional[str],
    repo_root: Optional[str | Path] = None,
) -> Dict[str, Path]:
    root = _repo_root(repo_root)
    report_path = root / REPORT_REL_PATH
    state_path = root / REPORT_STATE_REL_PATH
    out = {
        'report_path': report_path,
        'state_path': state_path,
    }
    if run_prefix:
        run_prefix_path = Path(str(run_prefix))
        out['run_report_path'] = Path(f'{run_prefix_path}_living_report.md')
        out['run_state_path'] = Path(f'{run_prefix_path}_living_report_state.json')
    return out


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=False))


def _base_state() -> Dict[str, Any]:
    now = _utc_now_iso()
    return {
        'schema_version': REPORT_SCHEMA_VERSION,
        'created_utc': now,
        'updated_utc': now,
        'problem_framing': {
            'objective': (
                'Build and validate a calibrated risk-to-control pipeline in Waymax so action selection '
                'uses reliable risk probabilities under nominal and shifted closed-loop conditions.'
            ),
            'causal_chain': (
                'Miscalibration -> wrong risk threshold decisions -> false-safe action acceptance -> '
                'closed-loop failures.'
            ),
        },
        'research_questions': [
            'RQ1: Are planner-side risk proxies miscalibrated and does shift make this worse?',
            'RQ2: Does post-hoc calibration improve reliability of learned risk probabilities?',
            'RQ3: Does calibrated risk-aware reranking reduce failures with bounded progress loss?',
            'RQ4: Do these gains persist across benchmark shift suites?',
        ],
        'hypotheses': {
            'H1': {
                'statement': 'Planner-side risk proxies are miscalibrated, especially under shift.',
                'status': 'open',
                'last_updated_utc': '',
                'evidence': '',
            },
            'H2': {
                'statement': 'Post-hoc calibration improves reliability (NLL/Brier/ECE) vs raw risk.',
                'status': 'open',
                'last_updated_utc': '',
                'evidence': '',
            },
            'H3': {
                'statement': (
                    'Calibrated risk-aware reranking reduces failure rate with at most 5% relative '
                    'progress loss on nominal conditions.'
                ),
                'status': 'open',
                'last_updated_utc': '',
                'evidence': '',
            },
            'H4': {
                'statement': 'Calibration and controller gains transfer to shifted suites.',
                'status': 'open',
                'last_updated_utc': '',
                'evidence': '',
            },
        },
        'stages': {},
        'history': [],
    }


def _load_state(paths: Mapping[str, Path]) -> Dict[str, Any]:
    candidates = []
    if 'run_state_path' in paths:
        candidates.append(paths['run_state_path'])
    candidates.append(paths['state_path'])
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return _base_state()


def _record_hypothesis(
    state: Dict[str, Any],
    *,
    hypothesis_id: str,
    status: str,
    evidence: str,
    now_utc: str,
) -> None:
    hypotheses = state.setdefault('hypotheses', {})
    h = hypotheses.setdefault(
        str(hypothesis_id),
        {
            'statement': '',
            'status': 'open',
            'last_updated_utc': '',
            'evidence': '',
        },
    )
    h['status'] = str(status)
    h['evidence'] = str(evidence)
    h['last_updated_utc'] = str(now_utc)


def _to_serializable_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in metrics.items():
        key = str(k)
        if isinstance(v, (str, int, bool)):
            out[key] = v
            continue
        if isinstance(v, (float, np.floating)):
            val = float(v)
            out[key] = val if np.isfinite(val) else None
            continue
        if isinstance(v, (np.integer,)):
            out[key] = int(v)
            continue
        out[key] = str(v)
    return out


def _update_stage(
    state: Dict[str, Any],
    *,
    stage: str,
    run_prefix: str,
    headline: str,
    metrics: Mapping[str, Any],
    artifacts: Optional[Mapping[str, str]],
    now_utc: str,
) -> None:
    stages = state.setdefault('stages', {})
    stages[str(stage)] = {
        'updated_utc': str(now_utc),
        'run_prefix': str(run_prefix),
        'headline': str(headline),
        'metrics': _to_serializable_metrics(metrics),
        'artifacts': {str(k): str(v) for k, v in dict(artifacts or {}).items()},
    }
    history = state.setdefault('history', [])
    history.append(
        {
            'updated_utc': str(now_utc),
            'stage': str(stage),
            'run_prefix': str(run_prefix),
            'headline': str(headline),
        }
    )
    state['history'] = list(reversed(list(reversed(history))[-40:]))
    state['updated_utc'] = str(now_utc)


def _render_markdown(state: Mapping[str, Any]) -> str:
    problem = dict(state.get('problem_framing', {}))
    hypotheses = dict(state.get('hypotheses', {}))
    stages = dict(state.get('stages', {}))
    history = list(state.get('history', []))
    rqs = list(state.get('research_questions', []))

    lines = []
    lines.append('# Risk-UQ Living Report')
    lines.append('')
    lines.append('_Auto-updated by workflow runs in this repository._')
    lines.append('')
    lines.append(f"- Schema version: `{state.get('schema_version', REPORT_SCHEMA_VERSION)}`")
    lines.append(f"- Last updated (UTC): `{state.get('updated_utc', '')}`")
    lines.append('')
    lines.append('## Current Problem Framing')
    lines.append(problem.get('objective', ''))
    lines.append('')
    lines.append(f"Causal chain: `{problem.get('causal_chain', '')}`")
    lines.append('')
    lines.append('## Research Questions')
    for rq in rqs:
        lines.append(f'1. {rq}')
    lines.append('')
    lines.append('## Hypotheses And Evidence')
    lines.append('| Hypothesis | Statement | Status | Last Updated (UTC) | Latest Evidence |')
    lines.append('|---|---|---|---|---|')
    for hid in ['H1', 'H2', 'H3', 'H4']:
        record = dict(hypotheses.get(hid, {}))
        statement = str(record.get('statement', '')).replace('|', '\\|')
        status = _status_label(str(record.get('status', 'open')))
        updated = str(record.get('last_updated_utc', ''))
        evidence = str(record.get('evidence', '')).replace('|', '\\|')
        lines.append(f'| {hid} | {statement} | {status} | {updated} | {evidence} |')
    lines.append('')
    lines.append('## Stage Snapshots')
    if len(stages) == 0:
        lines.append('- No stage snapshots yet.')
    for stage in ['miscalibration_probe', 'risk_training', 'uq_benchmark', 'paper_export']:
        snapshot = stages.get(stage)
        if not isinstance(snapshot, dict):
            continue
        lines.append(f'### {stage}')
        lines.append(f"- Updated (UTC): `{snapshot.get('updated_utc', '')}`")
        lines.append(f"- Run prefix: `{snapshot.get('run_prefix', '')}`")
        lines.append(f"- Headline: {snapshot.get('headline', '')}")
        metrics = snapshot.get('metrics', {})
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                val = v if v is not None else 'NA'
                lines.append(f"- Metric `{k}`: `{val}`")
        artifacts = snapshot.get('artifacts', {})
        if isinstance(artifacts, dict):
            for k, v in artifacts.items():
                lines.append(f"- Artifact `{k}`: `{v}`")
        lines.append('')

    lines.append('## Update History (Most Recent First)')
    lines.append('| Updated (UTC) | Stage | Headline | Run Prefix |')
    lines.append('|---|---|---|---|')
    if len(history) == 0:
        lines.append('| - | - | - | - |')
    else:
        for row in reversed(history[-20:]):
            lines.append(
                f"| {row.get('updated_utc', '')} | {row.get('stage', '')} | "
                f"{str(row.get('headline', '')).replace('|', '\\|')} | {row.get('run_prefix', '')} |"
            )
    lines.append('')
    lines.append('## Next High-Value Checks')
    lines.append('1. Re-run miscalibration probe when shift definitions or candidate generation logic changes.')
    lines.append('2. Re-check false-safe and over-conservative rates at the deployment budget threshold.')
    lines.append('3. Re-validate controller failure-progress tradeoff after calibration/model updates.')
    lines.append('4. Freeze figure/table exports only after hypotheses H2-H4 are supported or clearly bounded.')
    lines.append('')
    return '\n'.join(lines)


def _enabled(cfg: Optional[Any] = None) -> bool:
    if os.environ.get('PYTEST_CURRENT_TEST'):
        return False
    env_value = os.environ.get('RISK_UQ_LIVING_REPORT_AUTO_UPDATE')
    if env_value is not None:
        return str(env_value).strip().lower() not in {'0', 'false', 'no', 'off'}
    if cfg is None:
        return True
    flag = getattr(cfg, 'risk_uq_report_auto_update', None)
    if flag is None:
        return True
    return bool(flag)


def _persist_state_and_report(
    *,
    state: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> Dict[str, str]:
    report_text = _render_markdown(state)
    _write_json_atomic(paths['state_path'], state)
    _write_text_atomic(paths['report_path'], report_text)
    out = {
        'living_report': str(paths['report_path']),
        'living_report_state': str(paths['state_path']),
    }
    run_state = paths.get('run_state_path')
    run_report = paths.get('run_report_path')
    if run_state is not None:
        _write_json_atomic(run_state, state)
        out['living_report_run_state'] = str(run_state)
    if run_report is not None:
        _write_text_atomic(run_report, report_text)
        out['living_report_run_copy'] = str(run_report)
    return out


def _summarize_miscalibration(
    summary_df: pd.DataFrame,
    per_shift_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
    class_balance_df: pd.DataFrame,
) -> Tuple[str, Dict[str, Any], Dict[str, Tuple[str, str]]]:
    focus_label = 'failure_proxy_h15'
    if not summary_df.empty and ('label' in summary_df.columns):
        labels = summary_df['label'].astype(str).unique().tolist()
        if focus_label not in labels and len(labels) > 0:
            focus_label = str(labels[0])
    elif not per_shift_df.empty and ('label' in per_shift_df.columns):
        labels = per_shift_df['label'].astype(str).unique().tolist()
        if focus_label not in labels and len(labels) > 0:
            focus_label = str(labels[0])

    per_shift = per_shift_df.copy()
    if not per_shift.empty and ('variant' in per_shift.columns):
        per_shift['variant_is_raw'] = per_shift['variant'].astype(str).str.contains('raw', case=False, regex=False)
        per_shift['variant_is_cal'] = per_shift['variant'].astype(str).str.contains('cal|platt', case=False, regex=True)
    else:
        per_shift['variant_is_raw'] = False
        per_shift['variant_is_cal'] = False

    raw_nom_ece = _mean_metric(
        per_shift,
        metric='ece',
        filters={'label': focus_label, 'shift_suite': 'nominal_clean', 'variant_is_raw': True},
    )
    cal_nom_ece = _mean_metric(
        per_shift,
        metric='ece',
        filters={'label': focus_label, 'shift_suite': 'nominal_clean', 'variant_is_cal': True},
    )
    raw_shift_ece = _mean_metric(
        per_shift[~per_shift['shift_suite'].astype(str).eq('nominal_clean')] if ('shift_suite' in per_shift.columns) else pd.DataFrame(),
        metric='ece',
        filters={'label': focus_label, 'variant_is_raw': True},
    )
    cal_shift_ece = _mean_metric(
        per_shift[~per_shift['shift_suite'].astype(str).eq('nominal_clean')] if ('shift_suite' in per_shift.columns) else pd.DataFrame(),
        metric='ece',
        filters={'label': focus_label, 'variant_is_cal': True},
    )

    raw_false_safe = float('nan')
    cal_false_safe = float('nan')
    if not threshold_df.empty and ('variant' in threshold_df.columns):
        work = threshold_df.copy()
        work['variant_is_raw'] = work['variant'].astype(str).str.contains('raw', case=False, regex=False)
        work['variant_is_cal'] = work['variant'].astype(str).str.contains('cal|platt', case=False, regex=True)
        raw_false_safe = _mean_metric(
            work,
            metric='false_safe_rate',
            filters={'label': focus_label, 'variant_is_raw': True},
        )
        cal_false_safe = _mean_metric(
            work,
            metric='false_safe_rate',
            filters={'label': focus_label, 'variant_is_cal': True},
        )

    leakage_pass_rate = float('nan')
    if (not leakage_df.empty) and ('pass' in leakage_df.columns):
        leakage_pass_rate = float(np.mean(pd.to_numeric(leakage_df['pass'], errors='coerce').fillna(0.0).to_numpy(dtype=float)))

    nominal_positive_rate = float('nan')
    if not class_balance_df.empty:
        nominal_positive_rate = _mean_metric(
            class_balance_df,
            metric='positive_rate',
            filters={'label': focus_label, 'shift_suite': 'nominal_clean'},
        )

    metrics = {
        'focus_label': focus_label,
        'raw_nominal_ece': raw_nom_ece,
        'cal_nominal_ece': cal_nom_ece,
        'raw_shift_ece_mean': raw_shift_ece,
        'cal_shift_ece_mean': cal_shift_ece,
        'raw_false_safe_rate': raw_false_safe,
        'cal_false_safe_rate': cal_false_safe,
        'leakage_check_pass_rate': leakage_pass_rate,
        'nominal_positive_rate': nominal_positive_rate,
    }
    headline = (
        f"Miscalibration probe ({focus_label}): nominal ECE raw {_format_float(raw_nom_ece)} -> "
        f"cal {_format_float(cal_nom_ece)}; false-safe raw {_format_float(raw_false_safe)} -> "
        f"cal {_format_float(cal_false_safe)}."
    )

    miscalibration_signal = bool(
        (np.isfinite(raw_nom_ece) and raw_nom_ece >= 0.03)
        or (np.isfinite(raw_false_safe) and raw_false_safe >= 0.02)
        or (np.isfinite(raw_shift_ece) and raw_shift_ece >= 0.04)
    )
    h1_status = 'supported' if miscalibration_signal else 'inconclusive'
    h1_evidence = (
        f"raw_nominal_ece={_format_float(raw_nom_ece)}, raw_false_safe={_format_float(raw_false_safe)}, "
        f"raw_shift_ece_mean={_format_float(raw_shift_ece)}"
    )
    h2_signal = bool(
        np.isfinite(raw_nom_ece)
        and np.isfinite(cal_nom_ece)
        and (cal_nom_ece <= raw_nom_ece + 1e-8)
        and (
            (not np.isfinite(raw_false_safe))
            or (not np.isfinite(cal_false_safe))
            or (cal_false_safe <= raw_false_safe + 1e-8)
        )
    )
    h2_status = 'supported' if h2_signal else 'mixed'
    h2_evidence = (
        f"nominal_ece_raw={_format_float(raw_nom_ece)} vs cal={_format_float(cal_nom_ece)}; "
        f"false_safe_raw={_format_float(raw_false_safe)} vs cal={_format_float(cal_false_safe)}"
    )

    hypothesis_updates = {
        'H1': (h1_status, h1_evidence),
        'H2': (h2_status, h2_evidence),
    }
    return headline, metrics, hypothesis_updates


def _summarize_risk_training(
    train_summary_df: pd.DataFrame,
    calibration_summary_df: pd.DataFrame,
) -> Tuple[str, Dict[str, Any], Dict[str, Tuple[str, str]]]:
    focus_label = 'failure_proxy_h15'
    if not calibration_summary_df.empty and ('label' in calibration_summary_df.columns):
        labels = calibration_summary_df['label'].astype(str).unique().tolist()
        if focus_label not in labels and len(labels) > 0:
            focus_label = str(labels[0])

    raw_nll = _mean_metric(calibration_summary_df, metric='nll', filters={'label': focus_label, 'variant': 'raw'})
    cal_nll = _mean_metric(calibration_summary_df, metric='nll', filters={'label': focus_label, 'variant': 'cal'})
    raw_ece = _mean_metric(calibration_summary_df, metric='ece', filters={'label': focus_label, 'variant': 'raw'})
    cal_ece = _mean_metric(calibration_summary_df, metric='ece', filters={'label': focus_label, 'variant': 'cal'})
    delta_nll = cal_nll - raw_nll if (np.isfinite(raw_nll) and np.isfinite(cal_nll)) else float('nan')
    delta_ece = cal_ece - raw_ece if (np.isfinite(raw_ece) and np.isfinite(cal_ece)) else float('nan')

    mean_best_val_loss = _mean_metric(train_summary_df, metric='best_val_loss', filters={})
    mean_epochs = _mean_metric(train_summary_df, metric='epochs_ran', filters={})
    resumed_fraction = _mean_metric(train_summary_df, metric='resumed_from_checkpoint', filters={})
    n_members = int(train_summary_df['member_index'].nunique()) if ('member_index' in train_summary_df.columns) else 0

    metrics = {
        'focus_label': focus_label,
        'raw_validation_nll': raw_nll,
        'cal_validation_nll': cal_nll,
        'delta_validation_nll_cal_minus_raw': delta_nll,
        'raw_validation_ece': raw_ece,
        'cal_validation_ece': cal_ece,
        'delta_validation_ece_cal_minus_raw': delta_ece,
        'mean_best_val_loss': mean_best_val_loss,
        'mean_epochs_ran': mean_epochs,
        'resumed_fraction': resumed_fraction,
        'ensemble_members': n_members,
    }
    headline = (
        f"Risk training ({focus_label}): val NLL raw {_format_float(raw_nll)} -> cal {_format_float(cal_nll)}, "
        f"val ECE raw {_format_float(raw_ece)} -> cal {_format_float(cal_ece)}."
    )

    h2_signal = bool(
        np.isfinite(raw_nll)
        and np.isfinite(cal_nll)
        and (cal_nll <= raw_nll + 1e-8)
        and ((not np.isfinite(raw_ece)) or (not np.isfinite(cal_ece)) or (cal_ece <= raw_ece + 1e-8))
    )
    h2_status = 'supported' if h2_signal else 'mixed'
    h2_evidence = (
        f"delta_nll={_format_float(delta_nll)}, delta_ece={_format_float(delta_ece)}, "
        f"mean_best_val_loss={_format_float(mean_best_val_loss)}"
    )
    return headline, metrics, {'H2': (h2_status, h2_evidence)}


def _summarize_uq_benchmark(
    summary_df: pd.DataFrame,
    per_shift_df: pd.DataFrame,
    controller_summary_df: pd.DataFrame,
) -> Tuple[str, Dict[str, Any], Dict[str, Tuple[str, str]]]:
    focus_label = 'failure_proxy_h15'
    if not summary_df.empty and ('label' in summary_df.columns):
        labels = summary_df['label'].astype(str).unique().tolist()
        if focus_label not in labels and len(labels) > 0:
            focus_label = str(labels[0])

    raw_nll = _mean_metric(summary_df, metric='nll', filters={'label': focus_label, 'variant': 'raw'})
    cal_nll = _mean_metric(summary_df, metric='nll', filters={'label': focus_label, 'variant': 'cal'})
    raw_ece = _mean_metric(summary_df, metric='ece', filters={'label': focus_label, 'variant': 'raw'})
    cal_ece = _mean_metric(summary_df, metric='ece', filters={'label': focus_label, 'variant': 'cal'})

    non_nominal = per_shift_df.copy()
    if 'shift_suite' in non_nominal.columns:
        non_nominal = non_nominal[~non_nominal['shift_suite'].astype(str).eq('nominal_clean')].copy()

    improved_count = 0
    shift_count = 0
    if (not non_nominal.empty) and {'label', 'variant', 'shift_suite', 'ece'}.issubset(set(non_nominal.columns)):
        for shift_name, grp in non_nominal.groupby('shift_suite', sort=True):
            row_raw = grp[(grp['label'].astype(str).eq(focus_label)) & (grp['variant'].astype(str).eq('raw'))]
            row_cal = grp[(grp['label'].astype(str).eq(focus_label)) & (grp['variant'].astype(str).eq('cal'))]
            if row_raw.empty or row_cal.empty:
                continue
            ece_raw = _safe_float(row_raw['ece'].iloc[0])
            ece_cal = _safe_float(row_cal['ece'].iloc[0])
            if np.isfinite(ece_raw) and np.isfinite(ece_cal):
                shift_count += 1
                if ece_cal <= ece_raw + 1e-8:
                    improved_count += 1

    rel_progress = _mean_metric(controller_summary_df, metric='relative_progress_change', filters={})
    failure_delta = _mean_metric(controller_summary_df, metric='failure_rate_delta', filters={})

    metrics = {
        'focus_label': focus_label,
        'raw_test_nll': raw_nll,
        'cal_test_nll': cal_nll,
        'raw_test_ece': raw_ece,
        'cal_test_ece': cal_ece,
        'shift_suites_with_ece_improvement': int(improved_count),
        'shift_suites_compared': int(shift_count),
        'controller_relative_progress_change': rel_progress,
        'controller_failure_rate_delta': failure_delta,
    }
    headline = (
        f"UQ benchmark ({focus_label}): test ECE raw {_format_float(raw_ece)} -> cal {_format_float(cal_ece)}; "
        f"controller failure delta {_format_float(failure_delta)} with progress change {_format_float(rel_progress)}."
    )

    h2_signal = bool(
        np.isfinite(raw_nll)
        and np.isfinite(cal_nll)
        and (cal_nll <= raw_nll + 1e-8)
        and ((not np.isfinite(raw_ece)) or (not np.isfinite(cal_ece)) or (cal_ece <= raw_ece + 1e-8))
    )
    h3_signal = bool(np.isfinite(failure_delta) and np.isfinite(rel_progress) and (failure_delta < 0.0) and (rel_progress >= -0.05))
    h4_signal = bool(shift_count > 0 and improved_count >= max(1, min(3, shift_count)))

    h2_status = 'supported' if h2_signal else 'mixed'
    h3_status = 'supported' if h3_signal else 'mixed'
    h4_status = 'supported' if h4_signal else 'inconclusive'

    updates = {
        'H2': (h2_status, f"test_delta_nll={_format_float(cal_nll - raw_nll if np.isfinite(raw_nll) and np.isfinite(cal_nll) else float('nan'))}, test_delta_ece={_format_float(cal_ece - raw_ece if np.isfinite(raw_ece) and np.isfinite(cal_ece) else float('nan'))}"),
        'H3': (h3_status, f"failure_rate_delta={_format_float(failure_delta)}, relative_progress_change={_format_float(rel_progress)}"),
        'H4': (h4_status, f"shift_ece_improvement={int(improved_count)}/{int(shift_count)}"),
    }
    return headline, metrics, updates


def _summarize_paper_export(exported_paths: Mapping[str, str]) -> Tuple[str, Dict[str, Any], Dict[str, Tuple[str, str]]]:
    n_items = int(len(dict(exported_paths)))
    has_reliability = int(any('reliability' in str(k).lower() for k in exported_paths.keys()))
    has_coverage = int(any('coverage' in str(k).lower() for k in exported_paths.keys()))
    metrics = {
        'exported_item_count': n_items,
        'has_reliability_outputs': has_reliability,
        'has_coverage_outputs': has_coverage,
    }
    headline = f'Paper export generated {n_items} table/figure outputs.'
    return headline, metrics, {}


def _finalize_update(
    *,
    cfg: Optional[Any],
    stage: str,
    run_prefix: str,
    headline: str,
    metrics: Mapping[str, Any],
    hypothesis_updates: Mapping[str, Tuple[str, str]],
    artifact_paths: Optional[Mapping[str, str]],
    repo_root: Optional[str | Path] = None,
) -> Dict[str, str]:
    if not _enabled(cfg):
        return {}
    now = _utc_now_iso()
    paths = _resolve_paths(run_prefix=run_prefix, repo_root=repo_root)
    state = _load_state(paths)
    _update_stage(
        state,
        stage=stage,
        run_prefix=run_prefix,
        headline=headline,
        metrics=metrics,
        artifacts=artifact_paths,
        now_utc=now,
    )
    for hid, record in hypothesis_updates.items():
        status, evidence = record
        _record_hypothesis(state, hypothesis_id=hid, status=status, evidence=evidence, now_utc=now)
    return _persist_state_and_report(state=state, paths=paths)


def update_living_report_from_miscalibration_probe(
    *,
    cfg: Optional[Any],
    run_prefix: str,
    summary_df: pd.DataFrame,
    per_shift_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
    class_balance_df: pd.DataFrame,
    artifact_paths: Optional[Mapping[str, str]] = None,
    repo_root: Optional[str | Path] = None,
) -> Dict[str, str]:
    headline, metrics, hypothesis_updates = _summarize_miscalibration(
        summary_df,
        per_shift_df,
        threshold_df,
        leakage_df,
        class_balance_df,
    )
    return _finalize_update(
        cfg=cfg,
        stage='miscalibration_probe',
        run_prefix=run_prefix,
        headline=headline,
        metrics=metrics,
        hypothesis_updates=hypothesis_updates,
        artifact_paths=artifact_paths,
        repo_root=repo_root,
    )


def update_living_report_from_risk_training(
    *,
    cfg: Optional[Any],
    run_prefix: str,
    train_summary_df: pd.DataFrame,
    calibration_summary_df: pd.DataFrame,
    artifact_paths: Optional[Mapping[str, str]] = None,
    repo_root: Optional[str | Path] = None,
) -> Dict[str, str]:
    headline, metrics, hypothesis_updates = _summarize_risk_training(
        train_summary_df,
        calibration_summary_df,
    )
    return _finalize_update(
        cfg=cfg,
        stage='risk_training',
        run_prefix=run_prefix,
        headline=headline,
        metrics=metrics,
        hypothesis_updates=hypothesis_updates,
        artifact_paths=artifact_paths,
        repo_root=repo_root,
    )


def update_living_report_from_uq_benchmark(
    *,
    cfg: Optional[Any],
    run_prefix: str,
    summary_df: pd.DataFrame,
    per_shift_df: pd.DataFrame,
    controller_summary_df: pd.DataFrame,
    artifact_paths: Optional[Mapping[str, str]] = None,
    repo_root: Optional[str | Path] = None,
) -> Dict[str, str]:
    headline, metrics, hypothesis_updates = _summarize_uq_benchmark(
        summary_df,
        per_shift_df,
        controller_summary_df,
    )
    return _finalize_update(
        cfg=cfg,
        stage='uq_benchmark',
        run_prefix=run_prefix,
        headline=headline,
        metrics=metrics,
        hypothesis_updates=hypothesis_updates,
        artifact_paths=artifact_paths,
        repo_root=repo_root,
    )


def update_living_report_from_paper_export(
    *,
    cfg: Optional[Any],
    run_prefix: str,
    exported_paths: Mapping[str, str],
    repo_root: Optional[str | Path] = None,
) -> Dict[str, str]:
    headline, metrics, hypothesis_updates = _summarize_paper_export(exported_paths)
    return _finalize_update(
        cfg=cfg,
        stage='paper_export',
        run_prefix=run_prefix,
        headline=headline,
        metrics=metrics,
        hypothesis_updates=hypothesis_updates,
        artifact_paths=exported_paths,
        repo_root=repo_root,
    )

