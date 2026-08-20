"""
Price download and cleaning.

Documented choices (see docs/01_universe_decision.md):
  - TOTAL-RETURN prices (auto_adjust=True): utilities pay 3-4 %/yr in dividends,
    widely dispersed across names. On raw prices the yield difference injects a
    quasi-deterministic drift into the spread and destroys the residual's
    stationarity -> genuinely cointegrated pairs would be rejected.
  - NO forward-fill. The usual justification ("filling manufactures mean
    reversion") is FALSE: verified by simulation (see notebook, section 5),
    ffill biases neither the estimated half-life (+0.2 %) nor the number of
    signals. The real defect is TRADABILITY: on an OU spread with 5 % missing
    days, ~3.8 % of |z|>2 signals land on a day the name did not trade. The
    backtest would open positions at a price that never existed. NaNs are
    therefore kept and handled explicitly, pair by pair.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from . import config as C


def download_prices(tickers: list[str], start: str = C.START, end: str = C.END,
                    force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (adjusted_close, dollar_volume), indexed by date, columns = tickers."""
    px_path = C.DATA_PROC / "prices_adj.parquet"
    dv_path = C.DATA_PROC / "dollar_volume.parquet"
    if px_path.exists() and dv_path.exists() and not force:
        return pd.read_parquet(px_path), pd.read_parquet(dv_path)

    raw = yf.download(
        tickers=sorted(tickers), start=start, end=end,
        auto_adjust=True,      # splits AND dividends -> total-return series
        progress=False, group_by="column", threads=True, actions=False,
    )
    close = raw["Close"].copy()
    volume = raw["Volume"].copy()
    dollar_vol = close * volume     # split adjustments cancel out

    close, dollar_vol = _clean(close, dollar_vol)
    close.to_parquet(px_path)
    dollar_vol.to_parquet(dv_path)
    return close, dollar_vol


def _clean(close: pd.DataFrame, dollar_vol: pd.DataFrame,
           min_coverage: float = 0.80) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Minimal, NON-destructive cleaning.

    - drops entirely empty columns (tickers the vendor no longer serves,
      typically delisted companies -> this is the residual survivorship bias,
      quantified rather than hidden);
    - drops dates where fewer than `min_coverage` of tickers trade: these are
      partial holidays or data glitches, not sessions. We do NOT drop a date
      because a single name is missing: a newly listed name legitimately has a
      shorter history.
    """
    close = close.dropna(axis=1, how="all")
    dollar_vol = dollar_vol.reindex(columns=close.columns)

    listed = close.notna().sum(axis=1)
    # number of names "alive" at each date = those with a price before and after
    alive = close.ffill().notna() & close.bfill().notna()
    n_alive = alive.sum(axis=1).clip(lower=1)
    keep = (listed / n_alive) >= min_coverage

    return close.loc[keep], dollar_vol.loc[keep]


def coverage_report(close: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker diagnostics: first/last quote, gaps, coverage."""
    rows = []
    for t in close.columns:
        s = close[t]
        obs = s.dropna()
        if obs.empty:
            continue
        span = s.loc[obs.index[0]:obs.index[-1]]
        rows.append({
            "ticker": t,
            "first": obs.index[0].date(),
            "last": obs.index[-1].date(),
            "n_obs": int(obs.shape[0]),
            "n_gap": int(span.isna().sum()),
            "coverage": round(obs.shape[0] / max(len(close), 1), 3),
        })
    return pd.DataFrame(rows).set_index("ticker").sort_values("coverage")
