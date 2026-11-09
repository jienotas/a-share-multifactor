"""Cross-sectional price/volume factors.

Design notes
------------
* Factors are computed on **backward-adjusted** prices (see ``data.py``), so a
  factor value at date t only uses information available at t.
* Every factor is left in its textbook orientation (e.g. short-horizon
  reversal is *not* sign-flipped). The sign of its Rank IC therefore carries
  information rather than being an artefact of the author's preference.
* No fundamental factors. Mixing in reported financials would require mapping
  each ratio to its *announcement* date rather than its report period; doing it
  carelessly is one of the most common sources of look-ahead bias in A-share
  research. This project stays on price/volume and says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Panel = dict[str, pd.DataFrame]


def _div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Divide, treating a zero denominator as missing rather than infinite."""
    return a / b.replace(0.0, np.nan)


# --------------------------------------------------------------------------
# momentum / reversal
# --------------------------------------------------------------------------
def mom_20(p: Panel) -> pd.DataFrame:
    """One-month price momentum."""
    return p["close"] / p["close"].shift(20) - 1.0


def mom_60(p: Panel) -> pd.DataFrame:
    return p["close"] / p["close"].shift(60) - 1.0


def mom_120(p: Panel) -> pd.DataFrame:
    return p["close"] / p["close"].shift(120) - 1.0


def mom_12_1(p: Panel) -> pd.DataFrame:
    """12-1 momentum: eleven months of return, skipping the most recent month.

    Skipping the last month is the standard construction because the most
    recent month is contaminated by short-term reversal.
    """
    return p["close"].shift(20) / p["close"].shift(250) - 1.0


def rev_5(p: Panel) -> pd.DataFrame:
    """One-week reversal."""
    return p["close"] / p["close"].shift(5) - 1.0


def rev_1(p: Panel) -> pd.DataFrame:
    """One-day reversal."""
    return p["close"] / p["close"].shift(1) - 1.0


# --------------------------------------------------------------------------
# volatility / higher moments
# --------------------------------------------------------------------------
def vol_20(p: Panel) -> pd.DataFrame:
    return p["ret"].rolling(20, min_periods=15).std()


def vol_60(p: Panel) -> pd.DataFrame:
    return p["ret"].rolling(60, min_periods=40).std()


def downside_vol_20(p: Panel) -> pd.DataFrame:
    """Downside semi-deviation: sqrt(mean of squared negative returns)).

    Defined as a root-mean-square over the window rather than a standard
    deviation of the negative subset. The subset form returns NaN whenever a
    stock happens to post fewer than ten down days in the window, which
    silently deleted half the cross-section in testing.
    """
    neg_sq = p["ret"].clip(upper=0.0).pow(2)
    return neg_sq.rolling(20, min_periods=15).mean().pow(0.5)


def max_ret_20(p: Panel) -> pd.DataFrame:
    """Largest single-day return over 20 days: the lottery-demand proxy."""
    return p["ret"].rolling(20, min_periods=15).max()


def skew_20(p: Panel) -> pd.DataFrame:
    return p["ret"].rolling(20, min_periods=15).skew()


# --------------------------------------------------------------------------
# liquidity / turnover
# --------------------------------------------------------------------------
def turnover_20(p: Panel) -> pd.DataFrame:
    return p["turnover"].rolling(20, min_periods=15).mean()


def turnover_std_20(p: Panel) -> pd.DataFrame:
    """Turnover volatility: a proxy for speculative attention churn."""
    return p["turnover"].rolling(20, min_periods=15).std()


def amihud_20(p: Panel) -> pd.DataFrame:
    """Amihud illiquidity: mean of |return| per unit of traded value.

    Scaled by 1e8 so the numbers sit in a readable range rather than at 1e-9.
    """
    illiq = p["ret"].abs() * 1e8 / p["amount"].replace(0.0, np.nan)
    return illiq.rolling(20, min_periods=15).mean()


def volume_ratio(p: Panel) -> pd.DataFrame:
    """Short- over long-window volume: a volume shock indicator."""
    return _div(p["volume"].rolling(5, min_periods=4).mean(),
                p["volume"].rolling(60, min_periods=40).mean())


# --------------------------------------------------------------------------
# price pattern
# --------------------------------------------------------------------------
def ma_gap_20(p: Panel) -> pd.DataFrame:
    """Deviation of price from its 20-day moving average."""
    ma = p["close"].rolling(20, min_periods=15).mean()
    return _div(p["close"], ma) - 1.0


def bias_60(p: Panel) -> pd.DataFrame:
    ma = p["close"].rolling(60, min_periods=40).mean()
    return _div(p["close"], ma) - 1.0


def rsi_14(p: Panel) -> pd.DataFrame:
    """Relative strength index, scaled to [0, 1]."""
    delta = p["close"].diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = _div(gain, loss)
    return rs / (1.0 + rs)


def price_pos_60(p: Panel) -> pd.DataFrame:
    """Where price sits inside its own 60-day range, in [0, 1]."""
    lo = p["close"].rolling(60, min_periods=40).min()
    hi = p["close"].rolling(60, min_periods=40).max()
    return _div(p["close"] - lo, hi - lo)


def dist_high_250(p: Panel) -> pd.DataFrame:
    """Drawdown from the 52-week high (negative, closer to zero is stronger)."""
    return _div(p["close"], p["close"].rolling(250, min_periods=180).max()) - 1.0


# --------------------------------------------------------------------------
# risk
# --------------------------------------------------------------------------
def beta_60(p: Panel) -> pd.DataFrame:
    """Rolling market beta against the benchmark, computed pairwise."""
    mkt = p["mkt_ret"]
    win = 60
    cov = p["ret"].rolling(win, min_periods=40).cov(mkt)
    var = mkt.rolling(win, min_periods=40).var()
    return cov.div(var, axis=0)


def idio_vol_60(p: Panel) -> pd.DataFrame:
    """Idiosyncratic volatility from a rolling single-factor regression.

    Uses sigma * sqrt(1 - rho^2), which equals the residual standard deviation
    of a univariate regression, without needing a full rolling OLS.
    """
    mkt = p["mkt_ret"]
    win = 60
    rho = p["ret"].rolling(win, min_periods=40).corr(mkt)
    sd = p["ret"].rolling(win, min_periods=40).std()
    return sd * np.sqrt((1.0 - rho.pow(2)).clip(lower=0.0))


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
FACTORS: dict[str, callable] = {
    "mom_20": mom_20,
    "mom_60": mom_60,
    "mom_120": mom_120,
    "mom_12_1": mom_12_1,
    "rev_5": rev_5,
    "rev_1": rev_1,
    "vol_20": vol_20,
    "vol_60": vol_60,
    "downside_vol_20": downside_vol_20,
    "max_ret_20": max_ret_20,
    "skew_20": skew_20,
    "turnover_20": turnover_20,
    "turnover_std_20": turnover_std_20,
    "amihud_20": amihud_20,
    "volume_ratio": volume_ratio,
    "ma_gap_20": ma_gap_20,
    "bias_60": bias_60,
    "rsi_14": rsi_14,
    "price_pos_60": price_pos_60,
    "dist_high_250": dist_high_250,
    "beta_60": beta_60,
    "idio_vol_60": idio_vol_60,
}


def compute_all(p: Panel, names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Compute every registered factor; returns name -> date x code panel."""
    out: dict[str, pd.DataFrame] = {}
    for name in names or FACTORS:
        fn = FACTORS[name]
        out[name] = fn(p)
    return out
