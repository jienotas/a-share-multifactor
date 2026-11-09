# A-Share Cross-Sectional Multi-Factor Model

A from-scratch research framework for building and validating cross-sectional
equity factors on China A-shares: 22 price/volume factors, industry- and
size-neutralised, evaluated with Rank IC and a five-bucket layer test, and
traded through a backtest that models T+1 settlement, limit-up lockouts and
transaction costs.

No factor libraries, no backtesting engines, no vendor SDKs — the data layer,
factor set, neutralisation, evaluation and execution simulator are all
implemented here, which means every number below can be traced to a line of
code.

---

## What this is, and what it is not

**It is** a worked example of the research process: how to build a factor
panel without leaking future information, how to tell a genuine signal from a
data artefact, and what happens to a factor when you move it out of sample.

**It is not** a production trading system, and the absolute return in the
backtest section should not be read as an achievable return. See
[Known limitations](#known-limitations) before quoting any performance number.

---

## Results

Sample: CSI 300 constituents, 2021-01 to 2026-11, 1416 trading days × 300
names, 20-day forward return horizon (roughly monthly rebalancing).

### Composite factor (six strongest factors, equal weight)

Sign alignment uses an **expanding** IC window shifted by one period, so the
direction applied on any date is decided only from IC history *before* that
date.

| | Rank IC | IC IR | annualised IR | t-stat |
|---|---|---|---|---|
| In-sample (2021 – 2023) | 0.039 | 0.269 | 0.93 | 9.91 |
| Out-of-sample (2024 – 2026) | 0.034 | 0.213 | 0.74 | 5.50 |

**Sign stability: 20 of 22 factors keep their IC sign out of sample.**
The three strongest over the full sample — `max_ret_20`, `turnover_std_20`,
`turnover_20` — not only keep the sign but keep most of their magnitude, which
is the behaviour you want to see and the opposite of what a spurious factor
does.

### Top factors by information ratio (full sample, 2021 – 2026)

| Factor | Rank IC | IC IR | IC > 0 | Economic reading |
|---|---|---|---|---|
| `max_ret_20` | −0.033 | −0.270 | 41% | Lottery-demand discount |
| `turnover_std_20` | −0.031 | −0.264 | 41% | Attention churn is penalised |
| `turnover_20` | −0.034 | −0.261 | 42% | Low-turnover premium |
| `idio_vol_60` | −0.034 | −0.229 | 40% | Low-volatility anomaly |
| `vol_20` | −0.034 | −0.223 | 40% | Low-volatility anomaly |
| `bias_60` | −0.030 | −0.220 | 43% | Overextension reverses |
| `amihud_20` | +0.018 | +0.209 | 58% | Illiquidity premium |

These are the standard A-share anomalies and the signs match the published
literature: turnover, volatility and lottery-like payoffs all carry a
*discount*, liquidity carries a *premium*. Momentum is weak-to-negative across
1, 2, 6 and 12 months — again consistent with A-share reversal being stronger
than momentum at these horizons.

### Backtest (long the top quintile of the composite score)

| Metric | Strategy | CSI 300 |
|---|---|---|
| Annualised return | 9.4% | −4.9% |
| Annualised volatility | 15.5% | — |
| Sharpe | 0.66 | −0.18 |
| Max drawdown | −21.6% | — |
| Average holdings | 57 names | — |
| Rebalances | 70 | — |
| Average one-way turnover | 85% | — |
| Cumulative cost paid | 8.9% | — |

⚠️ The 14-point excess return **overstates the edge**, because the roster is
today's index membership applied to 2021. Stocks that were dropped from the
index — or delisted — are invisible here. Treat the backtest as a check that
the machinery works, and the IC out-of-sample result as the evidence.

---

## Four design decisions that matter

### 1. Backward-adjusted prices, not forward-adjusted

Forward adjustment (`fqt=1`) rewrites the *entire* history every time a new
dividend occurs: the "historical" series downloaded today is not the series a
researcher could have observed at the time. That is a subtle form of
look-ahead bias baked into the data. Backward adjustment (`fqt=2`) only
appends to future bars, so historical levels stay frozen.

### 2. Float market cap recovered from turnover

No vendor field, no fundamentals download:

```
float_mcap = amount / (turnover / 100)
```

`amount` is traded value in CNY and `turnover` is the share turnover rate, so
`amount = price × shares` and `turnover = shares / float_shares`; the traded
share term cancels. Verified against Kweichow Moutai: CNY 3.58bn / 0.20%
implies CNY 1.79trn, against an actual float cap of about CNY 1.8trn.

This yields a point-in-time market cap on *every bar*, which is what allows
the size neutralisation to be computed date by date.

### 3. Neutralise per cross-section, not pooled

"Industry neutral" has to mean neutral *within each period*, not on average
across the sample. Each date gets its own least-squares fit of the factor on
industry dummies plus log float cap, and the residuals are kept. Rare
industries (fewer than five names) are pooled into a residual bucket first —
otherwise a dummy for a one-stock industry silently absorbs that stock and
deletes it from the sample.

### 4. A-share frictions are modelled, not assumed away

- **T+1**: a signal formed on the close of day *t* executes on *t+1*. Executing
  on the signal bar is the most common look-ahead bug in retail backtests.
- **Limit-up lockout**: a name that closes limit-up on the execution day cannot
  be bought — there is no offer. Limits are 20% for STAR/ChiNext (688/300) and
  10% for the main boards.
- **Suspensions**: zero volume means no trade; such names are dropped rather
  than filled at the previous close.
- **Costs**: charged on traded notional at 15bp one-way, so replacing the whole
  book costs it twice.

---

## Quick start

```bash
pip install -r requirements.txt

# full run: CSI 300, 2021 to today (downloads once, then caches to parquet)
python scripts/run.py --universe hs300 --start 20210101

# smoke test on 60 names
python scripts/run.py --limit 60 --start 20230101

# re-download, ignore the cache
python scripts/run.py --force
```

Everything lands in `output/`:

| File | Contents |
|---|---|
| `factor_ic.csv` | Rank IC, IR, t-stat per factor |
| `ic_stability.csv` | In-sample vs out-of-sample IC per factor |
| `quantile_spread.csv` | Annualised return by quintile, top factors |
| `backtest_stats.csv` | Return, volatility, Sharpe, drawdown, costs |
| `nav.csv` | Daily NAV, strategy vs benchmark |
| `composite_ic.csv` | Monthly composite Rank IC series |
| `performance.png` | NAV and IC charts |

---

## Project structure

```
config.py              every knob: universe, window, horizons, costs
src/
  data.py              East Money endpoints, bar download, panel assembly
  factors.py           22 price/volume factors + registry
  preprocess.py        MAD winsorise -> z-score -> industry/size neutralise
  evaluate.py          forward returns, Rank IC, quantile spreads
  backtest.py          T+1 execution, limit-up lockout, costs, NAV
scripts/
  run.py               end-to-end pipeline
data/cache/            parquet bar cache and cached rosters (git-ignored)
output/                result tables and charts (committed)
```

---

## Known limitations

Stated plainly, because a factor study that hides its weaknesses is worth
nothing.

1. **Survivorship bias.** The universe is today's CSI 300 roster applied
   retrospectively. Delisted and dropped names are unavailable from the free
   data source, so absolute returns are biased upward and the backtest
   overstates the achievable edge. `--universe cap800` exists to filter on
   point-in-time float cap instead of a fixed roster, but it cannot resurrect
   delisted stocks either.
2. **No fundamental factors.** Value, quality and growth are absent. Adding
   them requires mapping each ratio to its *announcement* date rather than its
   report period — done carelessly, it is one of the largest sources of
   look-ahead bias in A-share research.
3. **Factor selection is in-sample.** The six factors in the composite were
   picked on full-sample IR. The *signs* are aligned out-of-sample, and the
   stability table shows the out-of-sample IC, but the selection itself was
   not rolling. A fully honest version would re-select the factor basket every
   rebalance.
4. **Industry labels are current, not historical.** Reclassifications over the
   sample are not reconstructed.
5. **Turnover is high (85% one-way).** Costs are modelled but not optimised
   away; a real book would penalise turnover explicitly in the position
   construction.
6. **Single execution assumption.** Everything trades at the close of *t+1*.
   Real execution has intraday structure and market impact that a 15bp flat
   cost does not capture.

---

## Data source

Daily bars, board membership and industry labels come from East Money's
public quote endpoints. No API key required. The endpoints are rate-limited;
the downloader retries misses serially and caches to parquet, and the roster
is cached to CSV so it only needs to succeed once.

The data belongs to the exchange and the vendor. This repository contains
code only — no redistributed price data.

## License

MIT. See [LICENSE](LICENSE).
