"""Global configuration for the A-share cross-sectional multi-factor study.

Everything that a reader might want to change (universe, sample window, cost
assumptions) lives here rather than being scattered across modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = ROOT / "output"

for _d in (DATA_DIR, CACHE_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    # ---- universe -------------------------------------------------------
    # "hs300"  : current CSI 300 constituents (may carry survivorship bias)
    # "csi500" : CSI 300 excluded, next 500 by float-cap (approximation)
    # "cap800" : point-in-time top 800 by float cap, no fixed roster
    universe: str = "hs300"

    # ---- sample window --------------------------------------------------
    start: str = "20210101"
    end: str = "20500101"

    # ---- factor / evaluation -------------------------------------------
    fwd_days: int = 20            # forward return horizon, in trading days
    n_groups: int = 5             # quantile buckets for the layer test
    min_obs_per_day: int = 50     # drop cross-sections thinner than this
    oos_split: str = "20240101"   # in-sample / out-of-sample boundary

    # ---- backtest -------------------------------------------------------
    rebalance_days: int = 20      # holding period, aligned with fwd_days
    top_quantile: float = 0.2     # long the top 20% of the score
    cost_bps: float = 15.0        # one-way cost in basis points
    exec_lag: int = 1             # signal at T close, execute at T+1 (T+1 rule)
    max_weight: float = 0.05      # single-name cap in the long book

    # ---- neutralisation -------------------------------------------------
    winsor_mad: float = 5.0       # MAD multiplier for outlier clipping
    neutralize: bool = True       # industry dummies + log float cap

    # ---- misc -----------------------------------------------------------
    n_jobs: int = 4               # concurrent download workers (endpoint throttles)

    @property
    def label(self) -> str:
        return f"{self.universe}_{self.start}_{self.end}"


CFG = Config()
