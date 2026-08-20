"""
Data layer for Binance USDT perpetual futures.

Why this venue rather than equities
-----------------------------------
Daily residual reversal on US large caps is dead: gross Sharpe 0.18 since 2021
against 0.23 for a null that destroys serial dependence. The premium was
compensation for supplying liquidity, and automated market makers now supply it
in minutes rather than days.

Crypto perpetuals are where that constraint still binds long enough to be
harvested at a daily horizon: the flow is leveraged and largely non-institutional,
liquidation cascades take hours to clear, and realised volatility is 5-10x that
of equities, which is the parameter that matters, since the equity book died at
a 3.74 bp breakeven cost.

Survivorship, and why it is better handled here than in equities
----------------------------------------------------------------
Two things work in our favour compared with the equity universe:

  exact listing dates   `onboardDate` gives the precise moment a contract began
                        trading, so point-in-time entry is exact rather than
                        reconstructed from a proxy with a two-day median lag.
  delisted contracts    `exchangeInfo` reports SETTLING symbols alongside
                        TRADING ones, recovering 126 contracts that have been or
                        are being delisted, including FTT, the FTX token.

What remains uncorrected: contracts purged from the API entirely (Btcst, for
instance, returns "invalid symbol"). Those are lost, and as in equities the
survivors are the healthier names, so the residual bias again runs in the
favourable direction.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from . import config as C

FAPI = "https://fapi.binance.com/fapi/v1"
_SESSION: requests.Session | None = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({"User-Agent": "quant-research/0.1"})
    return _SESSION


def _get(path: str, params: dict, tries: int = 6) -> object:
    """GET with backoff. Fails loudly rather than returning a truncated history:
    a silently short series would be indistinguishable from a late listing."""
    delay = 1.0
    for _ in range(tries):
        try:
            r = _session().get(f"{FAPI}/{path}", params=params, timeout=30)
        except requests.exceptions.RequestException:
            time.sleep(delay); delay = min(delay * 2, 30); continue
        if r.status_code in (418, 429, 503):
            time.sleep(delay); delay = min(delay * 2, 60); continue
        if r.status_code == 400:
            return None                      # symbol purged from the API
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"persistent failure on {path} {params}")


def perpetual_universe() -> pd.DataFrame:
    """All USDT perpetuals, live and settling, with their exact listing dates."""
    info = _get("exchangeInfo", {})
    rows = []
    for s in info["symbols"]:
        if s["contractType"] != "PERPETUAL" or s["quoteAsset"] != "USDT":
            continue
        if s["status"] not in ("TRADING", "SETTLING"):
            continue
        rows.append({
            "symbol": s["symbol"],
            "status": s["status"],
            "onboard": pd.Timestamp(s["onboardDate"], unit="ms", tz="UTC").tz_localize(None),
            "delivery": pd.Timestamp(s["deliveryDate"], unit="ms", tz="UTC").tz_localize(None),
        })
    df = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    # Binance uses a sentinel far-future delivery date for open-ended contracts
    df.loc[df["delivery"] > pd.Timestamp("2100-01-01"), "delivery"] = pd.NaT
    return df


def fetch_klines(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame | None:
    """Daily Ohlcv plus quote volume, paged. Returns None if the symbol is purged."""
    t0 = int(pd.Timestamp(start).timestamp() * 1000)
    t1 = int(pd.Timestamp(end).timestamp() * 1000)
    out = []
    while True:
        data = _get("klines", {"symbol": symbol, "interval": interval,
                               "startTime": t0, "endTime": t1, "limit": 1500})
        if data is None:
            return None
        if not data:
            break
        out += data
        if len(data) < 1500:
            break
        t0 = data[-1][0] + 1
        time.sleep(0.12)
    if not out:
        return None
    df = pd.DataFrame(out, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"])
    ts = pd.to_datetime(df["open_time"], unit="ms")
    df["date"] = ts.dt.normalize() if interval == "1d" else ts
    return df.set_index("date")[["close", "quote_volume"]].astype(float)


def build_membership(universe: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Point-in-time tradability: True from the listing date to the delivery date.

    This is Exact, not reconstructed. A contract cannot be selected before it
    existed, which is the entry-side look-ahead that the equity universe could
    only approximate to within a couple of days.
    """
    m = pd.DataFrame(False, index=index, columns=universe["symbol"].tolist())
    for _, r in universe.iterrows():
        live = index >= r["onboard"]
        if pd.notna(r["delivery"]):
            live &= index <= r["delivery"]
        m.loc[live, r["symbol"]] = True
    return m


def validate_listing_dates(close: pd.DataFrame, universe: pd.DataFrame,
                           tol_days: int = 5) -> pd.DataFrame:
    """Cross-check the declared listing date against the first observed price.

    Same discipline as the equity ticker-identity check: two independent sources
    for the same fact must agree. A first price long before the declared onboard
    date would mean the symbol was reused for a different contract.
    """
    rows = []
    for _, r in universe.iterrows():
        s = r["symbol"]
        if s not in close.columns:
            rows.append({**r, "first_price": pd.NaT, "gap_days": np.nan, "status_check": "NO_DATA"})
            continue
        obs = close[s].dropna()
        if obs.empty:
            rows.append({**r, "first_price": pd.NaT, "gap_days": np.nan, "status_check": "NO_DATA"})
            continue
        gap = (obs.index[0] - r["onboard"]).days
        rows.append({**r, "first_price": obs.index[0], "gap_days": gap,
                     "status_check": "OK" if gap >= -tol_days else "PRE_ONBOARD_DATA"})
    return pd.DataFrame(rows)
