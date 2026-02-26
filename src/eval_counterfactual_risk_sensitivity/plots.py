from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_factor_importance_ci(
    importance_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    figsize: tuple[float, float] = (8.0, 4.0),
):
    if importance_df.empty:
        return None
    required = ["factor_name", "mean_abs_slope"]
    missing = [c for c in required if c not in importance_df.columns]
    if missing:
        raise ValueError(f"Missing columns for factor importance plot: {missing}")

    plot_df = importance_df.copy()
    if not ci_df.empty and {"factor_name", "ci_low", "ci_high"}.issubset(ci_df.columns):
        plot_df = plot_df.merge(ci_df[["factor_name", "ci_low", "ci_high"]], on="factor_name", how="left")
    else:
        plot_df["ci_low"] = np.nan
        plot_df["ci_high"] = np.nan

    y = pd.to_numeric(plot_df["mean_abs_slope"], errors="coerce").to_numpy(dtype=float)
    y_low = pd.to_numeric(plot_df["ci_low"], errors="coerce").to_numpy(dtype=float)
    y_high = pd.to_numeric(plot_df["ci_high"], errors="coerce").to_numpy(dtype=float)
    yerr_low = np.where(np.isfinite(y_low), np.maximum(0.0, y - y_low), 0.0)
    yerr_high = np.where(np.isfinite(y_high), np.maximum(0.0, y_high - y), 0.0)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(plot_df))
    ax.bar(x, y)
    if np.isfinite(y_low).any() or np.isfinite(y_high).any():
        ax.errorbar(x, y, yerr=np.vstack([yerr_low, yerr_high]), fmt="none", capsize=4, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["factor_name"], rotation=45, ha="right")
    ax.set_ylabel("Mean absolute slope")
    ax.set_xlabel("Factor")
    ax.set_title("Counterfactual Factor Importance with Bootstrap CI")
    fig.tight_layout()
    return fig


def plot_factor_slope_distribution(
    atlas_df: pd.DataFrame,
    figsize: tuple[float, float] = (9.0, 4.0),
):
    required = ["factor_name", "slope"]
    missing = [c for c in required if c not in atlas_df.columns]
    if missing:
        raise ValueError(f"Missing columns for slope distribution plot: {missing}")
    if atlas_df.empty:
        return None

    factors = sorted(atlas_df["factor_name"].astype(str).unique().tolist())
    data = []
    for fac in factors:
        vals = pd.to_numeric(atlas_df.loc[atlas_df["factor_name"] == fac, "slope"], errors="coerce").dropna()
        data.append(vals.to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, labels=factors, showmeans=True)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Slope")
    ax.set_xlabel("Factor")
    ax.set_title("Distribution of Scenario-Level Factor Slopes")
    fig.tight_layout()
    return fig


def plot_method_factor_heatmap(
    method_importance_df: pd.DataFrame,
    value_col: str = "mean_abs_slope",
    figsize: tuple[float, float] = (10.0, 4.5),
):
    required = ["method", "factor_name", value_col]
    missing = [c for c in required if c not in method_importance_df.columns]
    if missing:
        raise ValueError(f"Missing columns for method-factor heatmap: {missing}")
    if method_importance_df.empty:
        return None
    pivot = method_importance_df.pivot_table(
        index="method",
        columns="factor_name",
        values=value_col,
        aggfunc="mean",
    ).sort_index()
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Method × Factor Heatmap ({value_col})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)
    fig.tight_layout()
    return fig


def plot_response_profile(
    profile_df: pd.DataFrame,
    factor_name: str,
    figsize: tuple[float, float] = (8.0, 4.0),
):
    if profile_df.empty:
        return None
    required = ["factor_name", "method", "bin_mean", "outcome_rate"]
    missing = [c for c in required if c not in profile_df.columns]
    if missing:
        raise ValueError(f"Missing columns for response profile plot: {missing}")

    sub = profile_df[profile_df["factor_name"] == factor_name].copy()
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    for method, grp in sub.groupby("method", sort=True):
        gg = grp.sort_values("bin_mean")
        ax.plot(gg["bin_mean"], gg["outcome_rate"], marker="o", label=str(method))
    ax.set_title(f"Response Profile: {factor_name}")
    ax.set_xlabel(f"{factor_name} (bin mean)")
    ax.set_ylabel("High-risk outcome rate")
    ax.legend()
    fig.tight_layout()
    return fig


def save_figure(fig, path: str) -> None:
    if fig is None:
        return
    fig.savefig(path, dpi=200, bbox_inches="tight")
