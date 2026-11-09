"""End-to-end pipeline: download -> factors -> evaluate -> backtest -> report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CFG, CACHE_DIR, OUTPUT_DIR
from src import backtest as bt
from src import data, evaluate, factors, preprocess

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_data(cfg, force: bool = False, limit: int | None = None):
    """Build the roster, download bars, and assemble wide panels."""
    uni = data.build_universe(cfg.universe)
    if limit:
        uni = uni.head(limit)
    codes = uni["code"].tolist()
    industry = dict(zip(uni["code"], uni["industry"].fillna("其他")))

    cache = CACHE_DIR / f"bars_{cfg.universe}_{cfg.start}_{cfg.end}.parquet"
    bars = data.get_bars(codes, cfg.start, cfg.end, cache_path=cache, force=force, n_jobs=cfg.n_jobs)
    panels = data.build_panel(bars, industry)

    # A fixed roster carries survivorship bias. In cap800 mode the investable
    # set is instead re-decided on every date from point-in-time float cap, so
    # a name only enters the cross-section once it is large enough to qualify.
    if cfg.universe == "cap800":
        top_n = 800
        rank = panels["float_mcap"].rank(axis=1, ascending=False, method="first")
        keep = rank <= top_n
        for key in ["open", "close", "high", "low", "volume", "amount",
                    "turnover", "pct_chg", "float_mcap"]:
            panels[key] = panels[key].where(keep)
        panels["ret"] = panels["close"].pct_change(fill_method=None)
        print(f"cap800: kept a median of {int(keep.sum(axis=1).median())} names per date")

    bench = data.fetch_index("hs300_idx", cfg.start, cfg.end)
    panels["bench_close"] = bench.reindex(panels["close"].index).ffill()
    panels["mkt_ret"] = panels["bench_close"].pct_change()
    return uni, panels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=CFG.universe)
    ap.add_argument("--start", default=CFG.start)
    ap.add_argument("--end", default=CFG.end)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N names (smoke test)")
    ap.add_argument("--force", action="store_true", help="ignore the parquet cache")
    args = ap.parse_args()

    cfg = CFG
    cfg.universe, cfg.start, cfg.end = args.universe, args.start, args.end

    print("=" * 70)
    print(f"universe={cfg.universe}  window={cfg.start}..{cfg.end}")
    print("=" * 70)

    uni, panels = load_data(cfg, force=args.force, limit=args.limit)
    close = panels["close"]
    print(f"panel: {close.shape[0]} dates x {close.shape[1]} names")

    # log float cap, used for size neutralisation
    log_mcap = np.log(panels["float_mcap"].replace(0.0, np.nan))
    industry = panels["industry"]

    # ---- factors ---------------------------------------------------------
    raw = factors.compute_all(panels)
    print(f"computed {len(raw)} factors")

    fwd = evaluate.forward_return(close, cfg.fwd_days)
    periods_per_year = 252 // cfg.fwd_days

    rows = []
    prepared = {}
    ic_series = {}
    for name, panel in raw.items():
        z = preprocess.prepare(
            panel, industry, log_mcap,
            winsor_k=cfg.winsor_mad, do_neutralize=cfg.neutralize, min_stocks=cfg.min_obs_per_day,
        )
        prepared[name] = z
        ic = evaluate.rank_ic(z, fwd, min_names=cfg.min_obs_per_day)
        if len(ic) == 0:
            continue
        ic_series[name] = ic
        st = evaluate.ic_stats(ic, periods_per_year=periods_per_year)
        st["factor"] = name
        rows.append(st)

    # rank on |IR|: a strongly negative factor (e.g. the lottery-effect proxy)
    # is every bit as useful as a strongly positive one
    ic_df = pd.DataFrame(rows).set_index("factor")
    ic_df = ic_df.reindex(ic_df["ic_ir"].abs().sort_values(ascending=False).index)
    ic_df = ic_df[["n", "ic_mean", "ic_std", "ic_ir", "ic_ir_ann", "ic_positive_ratio", "t_stat"]]
    print("\n--- Rank IC by factor (sorted by |IR|) ---")
    print(ic_df.round(3).to_string())
    ic_df.to_csv(OUTPUT_DIR / "factor_ic.csv")

    # ---- in-sample vs out-of-sample stability ---------------------------
    split = pd.Timestamp(cfg.oos_split)
    oos_rows = []
    for name, ic in ic_series.items():
        a = evaluate.ic_stats(ic[ic.index < split], periods_per_year)
        b = evaluate.ic_stats(ic[ic.index >= split], periods_per_year)
        if a.get("n", 0) < 12 or b.get("n", 0) < 12:
            continue
        oos_rows.append({
            "factor": name,
            "is_ic": a["ic_mean"], "is_ir": a["ic_ir"],
            "oos_ic": b["ic_mean"], "oos_ir": b["ic_ir"],
            "sign_kept": np.sign(a["ic_mean"]) == np.sign(b["ic_mean"]),
        })
    oos_df = pd.DataFrame(oos_rows).set_index("factor") if oos_rows else pd.DataFrame()
    if not oos_df.empty:
        oos_df = oos_df.sort_values("is_ir", key=abs, ascending=False)
        print(f"\n--- in-sample (<{cfg.oos_split}) vs out-of-sample ---")
        print(oos_df.round(3).to_string())
        oos_df.to_csv(OUTPUT_DIR / "ic_stability.csv")
        kept = int(oos_df["sign_kept"].sum())
        print(f"sign preserved in {kept}/{len(oos_df)} factors")

    # ---- quantile spread for the strongest factors -----------------------
    best = ic_df.index[:5].tolist()
    q_summary = {}
    for name in best:
        q = evaluate.quantile_returns(prepared[name], fwd, cfg.n_groups, cfg.min_obs_per_day)
        if q.empty:
            continue
        q_summary[name] = evaluate.summarise_quantiles(q, periods_per_year)
        q.to_csv(OUTPUT_DIR / f"quantiles_{name}.csv")
    if q_summary:
        spread = pd.DataFrame({k: v["ann_ret"] for k, v in q_summary.items()}).round(4)
        print("\n--- annualised return by quantile (Q1 lowest factor, Q5 highest) ---")
        print(spread.to_string())
        spread.to_csv(OUTPUT_DIR / "quantile_spread.csv")

    # ---- equal-weight composite score -----------------------------------
    # Sign alignment uses an EXPANDING window shifted by one period, so the
    # direction applied on date t is decided only from IC history before t.
    # Using the full-sample IC sign here would be a look-ahead bias: it lets
    # the strategy know in 2021 which way a factor would pay off in 2025.
    top = ic_df.index[:6].tolist()
    comp_index = prepared[top[0]].index
    comp_frames = []
    for n in top:
        sign_t = (
            np.sign(ic_series[n].expanding(min_periods=12).mean().shift(1))
            .reindex(comp_index).ffill().fillna(1.0)
        )
        comp_frames.append(prepared[n].mul(sign_t, axis=0))
    composite = sum(comp_frames) / len(top)
    composite.name = "composite"

    comp_ic = evaluate.rank_ic(composite, fwd, min_names=cfg.min_obs_per_day)
    comp_stats = evaluate.ic_stats(comp_ic, periods_per_year=periods_per_year)
    comp_oos = evaluate.ic_stats(comp_ic[comp_ic.index >= split], periods_per_year)
    print(f"\ncomposite ({len(top)} factors, expanding IC-sign alignment): "
          f"IC={comp_stats['ic_mean']:.4f}  IR={comp_stats['ic_ir']:.3f}  "
          f"IR_ann={comp_stats['ic_ir_ann']:.3f}  t={comp_stats['t_stat']:.2f}")
    if comp_oos.get("n", 0) > 12:
        print(f"composite out-of-sample: IC={comp_oos['ic_mean']:.4f}  IR={comp_oos['ic_ir']:.3f}  "
              f"IR_ann={comp_oos['ic_ir_ann']:.3f}  t={comp_oos['t_stat']:.2f}")

    # ---- backtest --------------------------------------------------------
    res = bt.run_backtest(
        composite, panels["ret"], panels["pct_chg"], panels["volume"],
        rebalance_days=cfg.rebalance_days, top_quantile=cfg.top_quantile,
        cost_bps=cfg.cost_bps, exec_lag=cfg.exec_lag, max_weight=cfg.max_weight,
        benchmark=panels["bench_close"],
    )
    print("\n--- backtest ---")
    for k, v in res["stats"].items():
        print(f"  {k:20s} {v:,.4f}" if isinstance(v, float) else f"  {k:20s} {v}")

    pd.Series(res["stats"]).to_csv(OUTPUT_DIR / "backtest_stats.csv")
    res["nav"].to_csv(OUTPUT_DIR / "nav.csv")
    res["turnover"].to_csv(OUTPUT_DIR / "turnover.csv", index=False)

    # ---- plots -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    nav = res["nav"]
    if isinstance(nav, pd.DataFrame):
        for col in nav.columns:
            axes[0].plot(nav.index, nav[col], label=col, lw=1.4)
        axes[0].legend()
    else:
        axes[0].plot(nav.index, nav, lw=1.4)
    axes[0].set_title("Net asset value (T+1 execution, costs included)")
    axes[0].grid(alpha=0.3)

    ic_ma = comp_ic.rolling(6).mean()
    axes[1].bar(comp_ic.index, comp_ic.values, width=12, alpha=0.35, label="monthly Rank IC")
    axes[1].plot(ic_ma.index, ic_ma.values, color="crimson", lw=1.6, label="6-period MA")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Composite factor Rank IC")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "performance.png", dpi=130)
    print(f"\nsaved -> {OUTPUT_DIR}")

    comp_ic.to_csv(OUTPUT_DIR / "composite_ic.csv")


if __name__ == "__main__":
    main()
