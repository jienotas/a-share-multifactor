"""Data access layer built on East Money's public quote endpoints.

Two design choices worth flagging up front, because both are places where
naive implementations quietly leak future information:

1. **Backward-adjusted prices (fqt=2), not forward-adjusted (fqt=1).**
   Forward adjustment rewrites the entire price history every time a new
   dividend or split occurs: the "historical" series you download today is
   not the series a researcher could have observed at the time. Backward
   adjustment only appends an adjustment to *future* bars, so historical
   levels stay frozen and returns remain comparable across the sample.

2. **Float market cap is recovered from turnover, not from a vendor field.**
   ``amount`` is turnover in CNY and ``turnover`` is the share turnover rate,
   so

        float_mcap = amount / (turnover / 100)

   because amount = price * shares_traded and turnover = shares_traded /
   float_shares, the traded-share term cancels. This yields a genuine
   point-in-time market cap for every bar with no dependence on fundamentals,
   which in turn lets us build the universe by market cap *as of each date*
   instead of borrowing today's index roster.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from config import CACHE_DIR

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

KLINE_COLS = [
    "date", "open", "close", "high", "low",
    "volume", "amount", "amplitude", "pct_chg", "chg", "turnover",
]

# Board codes used by the East Money universe endpoint (fs parameter).
BOARDS = {
    "hs300": "b:BK0500",       # CSI 300 constituent board
    "hs300_csi500": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",  # all A-shares
}

INDEX_SECID = {
    "hs300_idx": "0.399300",   # 沪深300
    "csi500_idx": "1.000905",  # 中证500
    "csi1000_idx": "1.000852", # 中证1000
}

# The quote-listing endpoint throttles per (scheme, host). When the primary
# host starts dropping connections mid-handshake, the delayed feed and the
# plain-HTTP route usually stay open, so the caller rotates through them in
# order rather than failing outright.
LIST_HOSTS = (
    "https://push2.eastmoney.com",
    "https://push2delay.eastmoney.com",
    "http://push2.eastmoney.com",
)


# --------------------------------------------------------------------------
# low level http
# --------------------------------------------------------------------------
def _get_json(url: str, retries: int = 3, timeout: int = 20) -> dict:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"})
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
        except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_err = exc
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"request failed after {retries} tries: {url[:120]}") from last_err


def to_secid(code: str) -> str:
    """Map a 6-digit A-share code to an East Money secid (market prefix)."""
    code = str(code).zfill(6)
    return f"1.{code}" if code[0] in "69" else f"0.{code}"


def _get_with_rotation(path: str, hosts: tuple[str, ...] = LIST_HOSTS,
                       retries_per_host: int = 2, timeout: int = 30) -> dict:
    """Try each host in turn until one answers."""
    last_err: Exception | None = None
    for host in hosts:
        try:
            return _get_json(host + path, retries=retries_per_host, timeout=timeout)
        except RuntimeError as exc:
            last_err = exc
    raise RuntimeError(f"all hosts failed for {path[:80]}") from last_err


# --------------------------------------------------------------------------
# universe
# --------------------------------------------------------------------------
def fetch_board_codes(board_key: str, page_size: int = 500, use_cache: bool = True) -> pd.DataFrame:
    """Pull every member of a board with its industry label and float cap.

    Results are cached to CSV. The listing endpoint is rate-limited much more
    aggressively than the bar endpoint, so a single successful pull is worth
    keeping around: roster membership changes only at semi-annual rebalances.
    """
    cache = CACHE_DIR / f"universe_{board_key}.csv"
    if use_cache and cache.exists():
        df = pd.read_csv(cache, dtype={"code": str})
        df["code"] = df["code"].str.zfill(6)
        print(f"[cache] {cache.name} ({len(df)} names)")
        return df

    path_tpl = (
        "/api/qt/clist/get?pn={pn}&pz={pz}&po=1&np=1&fltt=2&invt=2&fid=f3"
        "&fs={fs}&fields=f12,f14,f20,f21,f100"
    )
    fs = BOARDS[board_key]
    rows: list[dict] = []
    page = 1
    while True:
        path = path_tpl.format(pn=page, pz=page_size, fs=fs)
        data = _get_with_rotation(path).get("data") or {}
        diff = data.get("diff") or []
        if not diff:
            break
        for item in diff:
            rows.append(
                {
                    "code": str(item.get("f12", "")).zfill(6),
                    "name": item.get("f14"),
                    "mcap_total": item.get("f20"),
                    "mcap_float": item.get("f21"),
                    "industry": item.get("f100"),
                }
            )
        total = data.get("total", 0)
        if len(rows) >= total:
            break
        page += 1
        time.sleep(0.4)

    df = pd.DataFrame(rows).drop_duplicates("code").reset_index(drop=True)
    if len(df):
        df.to_csv(cache, index=False)
        print(f"[cache] wrote {cache.name} ({len(df)} names)")
    return df


def build_universe(kind: str = "hs300") -> pd.DataFrame:
    """Return the investable roster for the study.

    ``hs300``  current CSI 300 constituents.  NOTE: the roster is today's, so
               naming stocks that were only added later introduces
               survivorship bias. ``cap800`` exists precisely to avoid this.
    ``csi500`` CSI 300 excluded, next 500 by current float cap (approximation
               of the real index, which rebalances semi-annually).
    ``cap800`` all A-shares, filtered *per date* on point-in-time float cap in
               ``data.build_panel``; no fixed roster is baked in here.
    """
    # Fetch the expensive all-A listing only when the chosen universe needs it.
    if kind == "hs300":
        return fetch_board_codes("hs300")

    if kind == "csi500":
        top300 = set(fetch_board_codes("hs300")["code"])
        rest = fetch_board_codes("hs300_csi500")
        rest = rest[~rest["code"].isin(top300)].copy()
        rest = rest.sort_values("mcap_float", ascending=False).head(500)
        return rest.reset_index(drop=True)

    if kind == "cap800":
        # Roster is decided later, per date. Keep everything with a usable
        # industry label; the panel builder does the point-in-time cut.
        all_a = fetch_board_codes("hs300_csi500")
        return all_a.dropna(subset=["industry"]).reset_index(drop=True)

    raise ValueError(f"unknown universe kind: {kind}")


# --------------------------------------------------------------------------
# daily bars
# --------------------------------------------------------------------------
def fetch_kline(code: str, beg: str, end: str, adjust: int = 2) -> pd.DataFrame | None:
    """Daily bars for one code. ``adjust``: 0 none, 1 forward, 2 backward."""
    url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        f"?secid={to_secid(code)}"
        "&fields1=f1,f2,f3,f4,f5,f6"
        "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
        f"&klt=101&fqt={adjust}&beg={beg}&end={end}"
    )
    try:
        js = _get_json(url)
    except RuntimeError:
        return None
    data = js.get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        return None

    rows = [k.split(",") for k in klines]
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    df["code"] = str(code).zfill(6)
    for col in KLINE_COLS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_index(key: str, beg: str, end: str) -> pd.Series:
    """Close series for a benchmark index, used for excess returns and beta."""
    url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        f"?secid={INDEX_SECID[key]}"
        "&fields1=f1,f2&fields2=f51,f53&klt=101&fqt=1"
        f"&beg={beg}&end={end}"
    )
    data = _get_json(url).get("data") or {}
    klines = data.get("klines") or []
    idx = pd.Series(
        [float(k.split(",")[1]) for k in klines],
        index=pd.to_datetime([k.split(",")[0] for k in klines]),
        name=key,
    )
    return idx.sort_index()


# --------------------------------------------------------------------------
# panel construction
# --------------------------------------------------------------------------
def download_bars(
    codes: Sequence[str],
    beg: str,
    end: str,
    n_jobs: int = 4,
    adjust: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download daily bars concurrently, then re-try the misses serially.

    The endpoint throttles aggressively: at six workers roughly half the
    requests come back empty. Anything that fails in the parallel pass is
    retried one by one, which costs a few seconds and takes the success rate
    to effectively 100%.
    """
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(fetch_kline, c, beg, end, adjust): c for c in codes}
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                res = fut.result()
            except Exception:
                res = None
            if res is None or res.empty:
                failures.append(code)
            else:
                frames.append(res)
            done += 1
            if verbose and done % 50 == 0:
                print(f"  downloaded {done}/{len(codes)}", flush=True)

    if failures:
        if verbose:
            print(f"  retrying {len(failures)} misses serially ...", flush=True)
        still: list[str] = []
        for code in failures:
            try:
                res = fetch_kline(code, beg, end, adjust)
            except Exception:
                res = None
            if res is not None and not res.empty:
                frames.append(res)
            else:
                still.append(code)
        if still and verbose:
            print(f"  {len(still)} codes genuinely unavailable: {still[:10]}")

    if not frames:
        raise RuntimeError("no bars downloaded")
    return pd.concat(frames, ignore_index=True)


def build_panel(df: pd.DataFrame, industry: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """Reshape long bars into date x code wide panels.

    Returns a dict of panels: close/open/high/low/volume/amount/turnover/
    ret/float_mcap/industry (the last one as a static code -> label Series).
    """
    df = df.copy()
    df["amount"] = df["amount"].astype(float)
    df["turnover"] = df["turnover"].astype(float)

    # amount / (turnover/100) cancels the traded-share term and leaves float cap
    with np.errstate(divide="ignore", invalid="ignore"):
        fcap = df["amount"] * 100.0 / df["turnover"].replace(0.0, np.nan)
    df["float_mcap"] = fcap.replace([np.inf, -np.inf], np.nan)

    panels: dict[str, pd.DataFrame] = {}
    for col in ["open", "close", "high", "low", "volume", "amount", "turnover", "float_mcap", "pct_chg"]:
        panels[col] = df.pivot_table(index="date", columns="code", values=col, aggfunc="last").sort_index()

    close = panels["close"]
    panels["ret"] = close.pct_change(fill_method=None)

    if industry:
        panels["industry"] = pd.Series(industry, name="industry")
    return panels


def get_bars(
    codes: Sequence[str],
    beg: str,
    end: str,
    cache_path: Path | None = None,
    force: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Download with a parquet cache so repeated runs stay cheap."""
    if cache_path is not None and cache_path.exists() and not force:
        print(f"[cache] {cache_path.name}")
        return pd.read_parquet(cache_path)

    print(f"[download] {len(codes)} codes, {beg}..{end}")
    bars = download_bars(codes, beg, end, **kwargs)
    if cache_path is not None:
        bars.to_parquet(cache_path, index=False)
        print(f"[cache] wrote {cache_path.name}")
    return bars
