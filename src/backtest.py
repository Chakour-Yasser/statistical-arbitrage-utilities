"""
Phase 6 (minimal) -- backtest engine, needed to evaluate Phases 3 and 4.

Two rules make or break the honesty of the result:

  1. The signal at date t uses only information up to t, and the position is
     applied to the return from t to t+1. Trading at the close on which the
     signal was computed is the classic execution leak.
  2. Transaction costs are charged on every change of position, from the first
     backtest rather than added later. Pairs trading has high turnover and the
     gross-versus-net gap is part of the result, not a footnote.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def pair_positions(z: pd.Series, entry: float = 2.0, exit_: float = 0.5,
                   stop: float = 4.0, broken: pd.Series | None = None) -> pd.Series:
    """Target position in the spread: +1 long, -1 short, 0 flat.

    Long the spread when z is very negative (spread cheap), short when very
    positive. Hold until the spread comes back inside `exit_`, or until the stop
    or a regime break forces a flat position.

    State is carried forward day by day, so the position depends only on the past.
    """
    z = z.astype(float)
    brk = (broken.reindex(z.index).fillna(False).values
           if broken is not None else np.zeros(len(z), dtype=bool))
    zv = z.values
    pos = np.zeros(len(z))
    cur = 0.0
    blocked = False          # set after a stop-out or a regime exit
    for i in range(len(z)):
        zi = zv[i]
        if not np.isfinite(zi):
            cur, blocked = 0.0, blocked
        elif brk[i]:
            cur, blocked = 0.0, True        # regime exit overrides everything
        elif cur != 0.0:
            if abs(zi) >= stop:
                # Stopped out. Without the block below the position would
                # immediately re-open on the next bar, since |z| is still beyond
                # the entry threshold -- the strategy would fight the move all
                # the way out, churning at the worst possible moment.
                cur, blocked = 0.0, True
            elif abs(zi) <= exit_:
                cur = 0.0
        else:
            if blocked:
                # Re-entry is allowed only once the spread has come back to
                # normal. This is what makes a stop a stop rather than a pause.
                if abs(zi) <= exit_:
                    blocked = False
            elif zi <= -entry:
                cur = 1.0
            elif zi >= entry:
                cur = -1.0
        pos[i] = cur
    return pd.Series(pos, index=z.index)


def pair_pnl(pos: pd.Series, ra: pd.Series, rb: pd.Series, beta: float,
             cost_bps: float = 5.0) -> pd.DataFrame:
    """Net P&L of one pair, dollar-neutral, one dollar of gross exposure.

    Leg weights 1/(1+|beta|) long A and beta/(1+|beta|) short B, so the gross
    exposure is one dollar whatever the hedge ratio. Without this normalisation a
    pair with beta = 3 would silently carry four times the risk of a beta = 1 pair.

    Execution is lagged: the position decided on the close of t earns the return
    from t to t+1.
    """
    w = 1.0 / (1.0 + abs(beta))
    spread_ret = w * (ra - beta * rb)
    held = pos.shift(1).fillna(0.0)                 # <- the anti-leak shift
    gross = held * spread_ret

    turnover = (pos - pos.shift(1).fillna(0.0)).abs() * (w * (1 + abs(beta)))
    cost = turnover * cost_bps / 1e4
    return pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost,
                         "turnover": turnover, "pos": pos})


def metrics(net: pd.Series, turnover: pd.Series | None = None,
            periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised summary. Sharpe is reported without a risk-free rate, which is
    standard for a self-financing long/short book.

    `periods_per_year` must match the calendar of the underlying: 252 for equity
    sessions, 365 for crypto perpetuals, which trade continuously. Using 252 on a
    365-day series would understate the annualised Sharpe by about 20 percent.
    """
    net = net.dropna()
    if net.empty or net.std() == 0:
        return dict(days=len(net), ann_return=np.nan, ann_vol=np.nan,
                    sharpe=np.nan, max_dd=np.nan, hit_rate=np.nan, turnover=np.nan)
    ann_ret = net.mean() * periods_per_year
    ann_vol = net.std(ddof=1) * np.sqrt(periods_per_year)
    curve = net.cumsum()
    dd = (curve - curve.cummax()).min()
    active = net[net != 0]
    return dict(
        days=int(len(net)),
        ann_return=float(ann_ret),
        ann_vol=float(ann_vol),
        sharpe=float(ann_ret / ann_vol),
        max_dd=float(dd),
        hit_rate=float((active > 0).mean()) if len(active) else np.nan,
        turnover=float(turnover.sum() / (len(net) / periods_per_year)) if turnover is not None else np.nan,
    )
