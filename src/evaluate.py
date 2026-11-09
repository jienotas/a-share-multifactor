"""Factor evaluation: forward returns, Rank IC, and quantile spreads.

Spearman (rank) IC is used rather than Pearson IC throughout. Rank correlation
is insensitive to the monotone transformations that preprocessing applies and
is far less driven by a handful of extreme values, which matters a lot in an
A-share cross-section where a single limit-up day can dominate a Pearson fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return from t to t+n, aligned so it sits on date t."""
    return close.shift(-n) / close - 1.0


def tradable_mask(volume: pd.DataFrame, min_periods: int = 1) -> pd.DataFrame:
    """A name is tradable on a date if it actually traded."""
    return volume.rolling(min_periods, min_periods=min_periods).count().astype(bool) & (volume > 0)


def rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 30) -> pd.Series:
    """Cross-sectional Spearman correlation between factor and forward return."""
    common_idx = factor.index.intersection(fwd.index)
    common_cols = factor.columns.intersection(fwd.columns)
    f = factor.loc[common_idx, common_cols]
    r = fwd.loc[common_idx, common_cols]

    ic = f.corrwith(r, axis=1, method="spearman")
    counts = f.notna().mul(r.notna()).sum(axis=1)
    ic = ic.where(counts >= min_names)
    ic.name = "rank_ic"
    return ic.dropna()


def ic_stats(ic: pd.Series, periods_per_year: int = 12) -> dict:
    """Summarise an IC series the way a factor report would.

    ``ic_ir`` is the raw information ratio; ``ic_ir_ann`` annualises it, which
    is the number usually quoted in Chinese sell-side factor reports.
    """
    n = int(ic.notna().sum())
    if n == 0:
        return {"n": 0}
    mean = float(ic.mean())
    sd = float(ic.std(ddof=1))
    ir = mean / sd if sd > 0 else np.nan
    return {
        "n": n,
        "ic_mean": mean,
        "ic_std": sd,
        "ic_ir": ir,
        "ic_ir_ann": ir * np.sqrt(periods_per_year) if sd > 0 else np.nan,
        "ic_positive_ratio": float((ic > 0).mean()),
        "t_stat": ir * np.sqrt(n) if sd > 0 else np.nan,
    }


def quantile_returns(
    factor: pd.DataFrame,
    fwd: pd.DataFrame,
    n_groups: int = 5,
    min_names: int = 30,
) -> pd.DataFrame:
    """Average forward return per quantile bucket, per date.

    Bucket 1 holds the lowest factor values, bucket ``n_groups`` the highest.
    Buckets are formed on the rank percentile so unequal cross-section sizes
    across dates do not distort the cut points.
    """
    common_idx = factor.index.intersection(fwd.index)
    common_cols = factor.columns.intersection(fwd.columns)
    f = factor.loc[common_idx, common_cols]
    r = fwd.loc[common_idx, common_cols]

    records = {}
    for dt in common_idx:
        fv = f.loc[dt]
        rv = r.loc[dt]
        valid = fv.notna() & rv.notna()
        if valid.sum() < min_names:
            continue
        pct = fv[valid].rank(pct=True, method="first")
        bucket = np.ceil(pct * n_groups).clip(1, n_groups).astype(int)
        records[dt] = rv[valid].groupby(bucket).mean()
    if not records:
        return pd.DataFrame()
    out = pd.DataFrame(records).T.sort_index()
    out.columns = [f"Q{int(c)}" for c in out.columns]
    return out


def long_short(quantile_df: pd.DataFrame) -> pd.Series:
    """Top-minus-bottom bucket return, the standard factor spread series."""
    if quantile_df.empty:
        return pd.Series(dtype=float)
    cols = list(quantile_df.columns)
    return quantile_df[cols[-1]] - quantile_df[cols[0]]


def summarise_quantiles(quantile_df: pd.DataFrame, periods_per_year: int = 12) -> pd.DataFrame:
    """Annualised return and hit rate for each bucket."""
    if quantile_df.empty:
        return pd.DataFrame()
    ann = (1.0 + quantile_df.mean()) ** periods_per_year - 1.0
    return pd.DataFrame(
        {
            "mean_period_ret": quantile_df.mean(),
            "ann_ret": ann,
            "std": quantile_df.std(ddof=1),
            "win_rate": (quantile_df > 0).mean(),
        }
    )
