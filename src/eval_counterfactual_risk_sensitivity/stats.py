from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def bootstrap_factor_importance_ci(
    atlas_df: pd.DataFrame,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 17,
) -> pd.DataFrame:
    if atlas_df.empty:
        return pd.DataFrame()
    required = ["factor_name", "abs_slope"]
    missing = [c for c in required if c not in atlas_df.columns]
    if missing:
        raise ValueError(f"Missing columns for bootstrap_factor_importance_ci: {missing}")

    rng = np.random.default_rng(seed)
    rows = []
    for factor, grp in atlas_df.groupby("factor_name", sort=True):
        vals = pd.to_numeric(grp["abs_slope"], errors="coerce")
        vals = vals[np.isfinite(vals)].to_numpy(dtype=float)
        if vals.size == 0:
            rows.append(
                {
                    "factor_name": factor,
                    "n_groups": 0,
                    "mean_abs_slope": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
            )
            continue

        boot = np.empty((int(max(1, n_boot)),), dtype=float)
        n = vals.size
        for i in range(boot.size):
            idx = rng.integers(0, n, size=n)
            boot[i] = float(np.mean(vals[idx]))
        lo = float(np.quantile(boot, alpha / 2.0))
        hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
        rows.append(
            {
                "factor_name": factor,
                "n_groups": int(n),
                "mean_abs_slope": float(np.mean(vals)),
                "ci_low": lo,
                "ci_high": hi,
            }
        )

    return pd.DataFrame(rows).sort_values("mean_abs_slope", ascending=False).reset_index(drop=True)


def permutation_test_factor_nonzero(
    atlas_df: pd.DataFrame,
    factor_name: str,
    n_perm: int = 5000,
    seed: int = 17,
) -> Dict[str, Any]:
    sub = atlas_df[atlas_df["factor_name"] == factor_name].copy()
    if sub.empty:
        return {
            "factor_name": factor_name,
            "n_groups": 0,
            "observed_mean_slope": np.nan,
            "p_value_two_sided": np.nan,
        }

    vals = pd.to_numeric(sub["slope"], errors="coerce")
    vals = vals[np.isfinite(vals)].to_numpy(dtype=float)
    if vals.size == 0:
        return {
            "factor_name": factor_name,
            "n_groups": 0,
            "observed_mean_slope": np.nan,
            "p_value_two_sided": np.nan,
        }

    obs = float(np.mean(vals))
    rng = np.random.default_rng(seed)
    n_perm = int(max(1, n_perm))
    ge = 0
    for _ in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=vals.size, replace=True)
        perm = float(np.mean(vals * signs))
        if abs(perm) >= abs(obs):
            ge += 1
    p = float((ge + 1) / (n_perm + 1))
    return {
        "factor_name": factor_name,
        "n_groups": int(vals.size),
        "observed_mean_slope": obs,
        "p_value_two_sided": p,
    }
