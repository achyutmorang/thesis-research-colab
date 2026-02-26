from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_definition_heatmap(
    metrics_df: pd.DataFrame,
    score_col: str = "discovery_efficiency",
    figsize: tuple[float, float] = (10.0, 5.0),
):
    required = ["definition_key", "method", score_col]
    missing = [c for c in required if c not in metrics_df.columns]
    if missing:
        raise ValueError(f"Missing columns for heatmap plot: {missing}")
    pivot = (
        metrics_df.pivot_table(index="definition_key", columns="method", values=score_col, aggfunc="mean")
        .sort_index()
    )
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{score_col} by Method and Definition")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(score_col)
    fig.tight_layout()
    return fig


def plot_rank_heatmap(
    metrics_df: pd.DataFrame,
    score_col: str = "discovery_efficiency",
    figsize: tuple[float, float] = (10.0, 5.0),
):
    required = ["definition_key", "method", score_col]
    missing = [c for c in required if c not in metrics_df.columns]
    if missing:
        raise ValueError(f"Missing columns for rank heatmap: {missing}")
    pivot = (
        metrics_df.pivot_table(index="method", columns="definition_key", values=score_col, aggfunc="mean")
        .sort_index()
    )
    if pivot.empty:
        return None
    ranks = pivot.rank(axis=0, ascending=False, method="average")
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(ranks.to_numpy(dtype=float), aspect="auto", vmin=1.0, vmax=float(ranks.max().max()))
    ax.set_xticks(np.arange(len(ranks.columns)))
    ax.set_yticks(np.arange(len(ranks.index)))
    ax.set_xticklabels(ranks.columns, rotation=45, ha="right")
    ax.set_yticklabels(ranks.index)
    ax.set_title(f"Method Rank by Definition ({score_col})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rank (1=best)")
    fig.tight_layout()
    return fig


def plot_method_score_distribution(
    metrics_df: pd.DataFrame,
    score_col: str = "discovery_efficiency",
    figsize: tuple[float, float] = (8.0, 4.0),
):
    required = ["method", score_col]
    missing = [c for c in required if c not in metrics_df.columns]
    if missing:
        raise ValueError(f"Missing columns for score distribution: {missing}")
    if metrics_df.empty:
        return None
    methods = sorted(metrics_df["method"].astype(str).unique().tolist())
    data = [
        pd.to_numeric(metrics_df.loc[metrics_df["method"] == m, score_col], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
        for m in methods
    ]
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, labels=methods, showmeans=True)
    ax.set_title(f"Distribution of {score_col} across Definitions")
    ax.set_ylabel(score_col)
    ax.set_xlabel("Method")
    fig.tight_layout()
    return fig


def plot_time_to_k(
    metrics_df: pd.DataFrame,
    k: int = 10,
    figsize: tuple[float, float] = (7.0, 4.0),
):
    col = f"compute_to_k_{int(k)}"
    required = ["method", col]
    missing = [c for c in required if c not in metrics_df.columns]
    if missing:
        raise ValueError(
            f"Missing columns for time-to-k plot: {missing}. "
            f"Make sure evaluate_discovery_grid was run with k_values including {k}."
        )
    sub = metrics_df[["method", col]].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    agg = sub.groupby("method", as_index=False)[col].mean().sort_values(col, ascending=True)
    if agg.empty:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(agg["method"], agg[col])
    ax.set_title(f"Mean Compute to Discover K={int(k)} Blindspots")
    ax.set_ylabel("Compute units (lower is better)")
    ax.set_xlabel("Method")
    fig.tight_layout()
    return fig


def save_figure(fig, path: str) -> None:
    if fig is None:
        return
    fig.savefig(path, dpi=200, bbox_inches="tight")
