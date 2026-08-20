"""
Phase 7 -- evaluation and attribution.

A single aggregate Sharpe ratio hides everything worth knowing. This module
decomposes: what costs take, when the edge decays, where the drawdown comes
from, and whether the P&L is secretly a bet on interest rates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import TRADING_DAYS, metrics


def cost_sensitivity(gross: pd.Series, turnover: pd.Series,
                     bps_grid=(0, 1, 2, 5, 10, 20)) -> pd.DataFrame:
    """Sharpe as a function of the per-trade cost assumption.

    The breakeven cost -- the level at which the net Sharpe crosses zero -- is a
    more honest headline than any single Sharpe, because it states how much
    execution quality the strategy REQUIRES rather than assuming one number.
    """
    rows = []
    for bps in bps_grid:
        net = gross - turnover * bps / 1e4
        m = metrics(net)
        rows.append({"cost_bps": bps, "sharpe": m["sharpe"],
                     "ann_return": m["ann_return"], "max_dd": m["max_dd"]})
    return pd.DataFrame(rows)


def subperiod_metrics(net: pd.Series, freq: str = "YE") -> pd.DataFrame:
    """Per-period metrics. A strategy whose edge decays shows it here and nowhere
    else -- the aggregate number averages the good years with the dead ones."""
    rows = []
    for period, chunk in net.groupby(pd.Grouper(freq=freq)):
        if len(chunk) < 20:
            continue
        m = metrics(chunk)
        rows.append({"period": period.year if freq.startswith("Y") else str(period.date()),
                     "days": m["days"], "ann_return": m["ann_return"],
                     "sharpe": m["sharpe"], "max_dd": m["max_dd"]})
    return pd.DataFrame(rows)


def drawdown_attribution(net: pd.Series, broken: pd.Series) -> dict:
    """How much of the loss happens while the regime flags are up.

    This is the number that decides whether the Phase 4 machinery earns its
    place. If losses are spread evenly across flagged and unflagged days, the
    detector is measuring nothing.
    """
    b = broken.reindex(net.index).fillna(False).astype(bool)
    loss = net[net < 0]
    loss_flagged = net[(net < 0) & b]
    return {
        "share_days_flagged": float(b.mean()),
        "share_of_total_loss_flagged": float(loss_flagged.sum() / loss.sum())
        if loss.sum() != 0 else np.nan,
        "mean_pnl_flagged": float(net[b].mean()) if b.any() else np.nan,
        "mean_pnl_unflagged": float(net[~b].mean()) if (~b).any() else np.nan,
    }


def rate_exposure(net: pd.Series, yield_series: pd.Series) -> dict:
    """Regress daily P&L on changes in the 10-year yield.

    This answers the standing objection to a utilities pairs book: "your spread
    is just a disguised duration bet". The hedge ratio removes the COMMON rate
    exposure by construction; what must be checked is that the residual -- the
    P&L itself -- is no longer explained by rates. A significant coefficient
    would mean the strategy is a rates position wearing a market-neutral label.
    """
    d_y = yield_series.reindex(net.index).ffill().diff()
    df = pd.concat([net.rename("pnl"), d_y.rename("d_yield")], axis=1).dropna()
    if len(df) < 50:
        return {"n": len(df), "beta": np.nan, "t_stat": np.nan, "r2": np.nan}
    x = df["d_yield"].values
    y = df["pnl"].values
    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (beta * x + alpha)
    se = np.sqrt((resid @ resid) / (len(x) - 2) / ((x - x.mean()) @ (x - x.mean())))
    r = np.corrcoef(x, y)[0, 1]
    return {"n": int(len(df)), "beta": float(beta), "t_stat": float(beta / se),
            "r2": float(r ** 2)}


def summary_table(curves: dict) -> pd.DataFrame:
    """One row per selection method, stitched across all folds."""
    rows = []
    for name, c in curves.items():
        m = metrics(c)
        rows.append({"method": name, **{k: m[k] for k in
                     ("days", "ann_return", "ann_vol", "sharpe", "max_dd", "hit_rate")}})
    return pd.DataFrame(rows)
