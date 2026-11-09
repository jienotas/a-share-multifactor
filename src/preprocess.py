"""Cross-sectional preprocessing: outlier clipping, standardisation, neutralisation.

The pipeline runs in this order, per cross-section:

    raw factor  ->  MAD winsorise  ->  z-score  ->  industry + size neutralise

Winsorising before standardising matters: a single 30% one-day move otherwise
inflates the standard deviation enough to squash every other name towards zero.
Neutralising last means the residuals are expressed in z-score units, so factor
magnitudes stay comparable across dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# MAD -> standard deviation conversion factor for a normal distribution.
MAD_TO_SD = 1.4826


def mad_winsorize(df: pd.DataFrame, k: float = 5.0) -> pd.DataFrame:
    """Clip each cross-section at median +/- k * 1.4826 * MAD.

    MAD is preferred to a plain 3-sigma rule because the mean and standard
    deviation are themselves dragged around by the outliers being removed.
    """
    med = df.median(axis=1)
    mad = df.sub(med, axis=0).abs().median(axis=1)
    spread = (k * MAD_TO_SD * mad).replace(0.0, np.nan)
    return df.clip(lower=med.sub(spread, axis=0), upper=med.add(spread, axis=0), axis=0)


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional standardisation to zero mean, unit variance."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def _collapse_rare(industry: pd.Series, min_count: int = 5) -> pd.Series:
    """Merge industries with too few names into a single bucket.

    A dummy for a one-stock industry would absorb that stock's residual
    entirely, silently deleting it from the sample.
    """
    counts = industry.value_counts()
    rare = counts[counts < min_count].index
    return industry.where(~industry.isin(rare), "其他")


def neutralize(
    factor: pd.DataFrame,
    industry: pd.Series,
    log_mcap: pd.DataFrame,
    min_stocks: int = 30,
) -> pd.DataFrame:
    """Residualise each cross-section on industry dummies and log float cap.

    Implemented as a per-date least-squares fit. Doing it date by date (rather
    than pooling) is the whole point: "industry-neutral" means neutral *within
    each period*, not on average across the sample.
    """
    industry = _collapse_rare(industry.reindex(factor.columns))
    dummies_all = pd.get_dummies(industry, drop_first=True, dtype=float)

    out = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
    for dt in factor.index:
        y = factor.loc[dt]
        size = log_mcap.loc[dt] if dt in log_mcap.index else pd.Series(np.nan, index=factor.columns)
        valid = y.notna() & size.notna() & industry.notna()
        if valid.sum() < min_stocks:
            continue

        codes = y.index[valid]
        design = [np.ones(len(codes))]
        design.append(size.loc[codes].to_numpy(dtype=float))
        d = dummies_all.reindex(codes).to_numpy(dtype=float)
        # drop dummy columns that are constant within this cross-section
        if d.size:
            keep = d.std(axis=0) > 0
            design.append(d[:, keep])
        X = np.column_stack(design)
        yv = y.loc[codes].to_numpy(dtype=float)

        coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ coef
        out.loc[dt, codes] = resid
    return out


def prepare(
    factor: pd.DataFrame,
    industry: pd.Series,
    log_mcap: pd.DataFrame,
    *,
    winsor_k: float = 5.0,
    do_neutralize: bool = True,
    min_stocks: int = 30,
) -> pd.DataFrame:
    """Full preprocessing chain for one factor, in the documented order."""
    z = zscore(mad_winsorize(factor, k=winsor_k))
    if do_neutralize:
        z = neutralize(z, industry, log_mcap, min_stocks=min_stocks)
    return z
