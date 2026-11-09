"""Portfolio backtest with A-share trading frictions modelled explicitly.

Constraints that actually bind in practice and are therefore implemented:

* **T+1 execution.** A signal formed on the close of day t cannot be acted on
  until day t+1. Executing on the signal bar is the single most common
  look-ahead bug in retail backtests.
* **Limit-up lockout.** A name that closes limit-up on the execution day cannot
  be bought, because there is no offer. Limits are 20% for STAR/ChiNext
  (688/300) and 10% for the main boards.
* **Suspensions.** Zero volume means no trade; such names are dropped from the
  target book rather than silently filled at the previous close.
* **Costs.** One-way cost in basis points is charged on traded notional, so a
  full replacement of the book pays it twice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUAL_TRADING_DAYS = 252


def _limit_threshold(codes: pd.Index) -> np.ndarray:
    """Per-name daily price limit, as a fraction."""
    prefixes = pd.Series(codes, index=codes).astype(str).str[:3]
    return np.where(prefixes.isin(["300", "688"]).to_numpy(), 0.195, 0.095)


def _target_weights(
    score_row: pd.Series,
    pct_row: pd.Series,
    vol_row: pd.Series,
    top_quantile: float,
    max_weight: float,
) -> pd.Series:
    """Pick the long book and size it equal-weight under a per-name cap."""
    valid = score_row.notna() & vol_row.fillna(0).gt(0)
    if valid.sum() == 0:
        return pd.Series(dtype=float)

    ranked = score_row[valid].sort_values(ascending=False)
    n_take = max(1, int(np.ceil(len(ranked) * top_quantile)))
    picked = ranked.head(n_take).index

    # limit-up lockout on the execution bar
    limits = _limit_threshold(picked)
    at_limit = pct_row.reindex(picked).fillna(0.0).to_numpy() >= limits * 100
    picked = picked[~at_limit]
    if len(picked) == 0:
        return pd.Series(dtype=float)

    raw = pd.Series(1.0 / len(picked), index=picked)
    # iterative cap-and-redistribute, so the book stays fully invested
    for _ in range(10):
        over = raw > max_weight
        if not over.any():
            break
        excess = (raw[over] - max_weight).sum()
        raw[over] = max_weight
        free = ~over
        if not free.any():
            break
        raw[free] += excess * raw[free] / raw[free].sum()
    return raw / raw.sum()


def run_backtest(
    score: pd.DataFrame,
    ret: pd.DataFrame,
    pct_chg: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    rebalance_days: int = 20,
    top_quantile: float = 0.2,
    cost_bps: float = 15.0,
    exec_lag: int = 1,
    max_weight: float = 0.05,
    benchmark: pd.Series | None = None,
) -> dict:
    """Simulate the long book and return NAV, holdings log and statistics."""
    dates = score.index.intersection(ret.index)
    dates = dates[dates.isin(pct_chg.index) & dates.isin(volume.index)]
    if len(dates) <= rebalance_days + exec_lag:
        raise ValueError("sample too short for the requested rebalance schedule")

    # signal dates: every `rebalance_days` bars
    signal_idx = list(range(0, len(dates) - rebalance_days - exec_lag, rebalance_days))

    nav = 1.0
    nav_series = {}
    weights = pd.Series(dtype=float)
    turnover_log = []
    holdings_log = []

    for k, si in enumerate(signal_idx):
        sig_dt = dates[si]
        exec_pos = si + exec_lag
        end_pos = signal_idx[k + 1] + exec_lag if k + 1 < len(signal_idx) else len(dates)
        end_pos = min(end_pos, len(dates))

        target = _target_weights(
            score.loc[sig_dt], pct_chg.loc[dates[exec_pos]], volume.loc[dates[exec_pos]],
            top_quantile, max_weight,
        )

        # charge cost on the difference between the drifted book and the target
        prev = weights.reindex(target.index.union(weights.index)).fillna(0.0)
        tgt = target.reindex(prev.index).fillna(0.0)
        traded = float((tgt - prev).abs().sum())
        cost = traded * cost_bps / 1e4
        turnover_log.append({"date": dates[exec_pos], "traded_notional": traded, "cost": cost})
        nav *= 1.0 - cost
        weights = target

        # hold from exec_pos to end_pos, buy-and-hold within the period
        for pos in range(exec_pos, end_pos):
            dt = dates[pos]
            r = ret.loc[dt].reindex(weights.index).fillna(0.0)
            port_ret = float((weights * r).sum())
            nav *= 1.0 + port_ret
            nav_series[dt] = nav
            # weights drift with prices; renormalise so they keep summing to 1
            weights = weights * (1.0 + r)
            total = weights.sum()
            if total > 0:
                weights = weights / total
        holdings_log.append({"signal_date": sig_dt, "exec_date": dates[exec_pos], "n_names": len(target)})

    nav_df = pd.Series(nav_series).sort_index()
    daily_ret = nav_df.pct_change().dropna()

    stats = {
        "total_return": float(nav_df.iloc[-1] - 1.0) if len(nav_df) else np.nan,
        "ann_return": float((nav_df.iloc[-1]) ** (ANNUAL_TRADING_DAYS / len(nav_df)) - 1.0) if len(nav_df) > 1 else np.nan,
        "ann_vol": float(daily_ret.std(ddof=1) * np.sqrt(ANNUAL_TRADING_DAYS)),
        "sharpe": float(daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(ANNUAL_TRADING_DAYS)) if daily_ret.std(ddof=1) > 0 else np.nan,
        "max_drawdown": float((nav_df / nav_df.cummax() - 1.0).min()),
        "avg_names": float(np.mean([h["n_names"] for h in holdings_log])) if holdings_log else np.nan,
        "n_rebalances": len(signal_idx),
        "avg_turnover": float(np.mean([t["traded_notional"] for t in turnover_log])) if turnover_log else np.nan,
        "total_cost": float(sum(t["cost"] for t in turnover_log)),
    }

    if benchmark is not None:
        bench = benchmark.reindex(nav_df.index).ffill().dropna()
        if len(bench) > 2:
            bench_ret = bench.pct_change().dropna()
            stats["bench_ann_return"] = float((bench.iloc[-1] / bench.iloc[0]) ** (ANNUAL_TRADING_DAYS / len(bench)) - 1.0)
            stats["bench_sharpe"] = float(bench_ret.mean() / bench_ret.std(ddof=1) * np.sqrt(ANNUAL_TRADING_DAYS)) if bench_ret.std(ddof=1) > 0 else np.nan
            stats["excess_ann_return"] = stats["ann_return"] - stats["bench_ann_return"]
            nav_df = pd.concat([nav_df.rename("strategy"), (bench / bench.iloc[0]).rename("benchmark")], axis=1)

    return {
        "nav": nav_df,
        "stats": stats,
        "turnover": pd.DataFrame(turnover_log),
        "holdings": pd.DataFrame(holdings_log),
    }


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1.0).min())
