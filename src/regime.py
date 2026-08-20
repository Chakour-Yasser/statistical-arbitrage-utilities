"""
Phase 4 -- regime awareness.

Cointegration is not a permanent property. A pair selected in 2016 can stop
reverting in 2018 and never resume. A naive strategy keeps averaging down into a
spread that has stopped being a spread, which is where the large losses come from.

Every statistic here is computed on a TRAILING window and is therefore causal:
the flag at date t uses only observations up to t. Detecting a break the day it
happens is impossible; the aim is to detect it before the position is destroyed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from .cointegration import half_life


def rolling_coint_pvalue(la: pd.Series, lb: pd.Series, window: int = 250,
                         step: int = 21) -> pd.Series:
    """Engle-Granger p-value on a trailing window, recomputed every `step` days.

    Recomputing daily costs 250x more for no informational gain: the statistic
    barely moves in one day. The value is held constant between recomputations
    and forward-filled, which is causal -- it only ever repeats a past value.
    """
    idx = la.index
    out = pd.Series(np.nan, index=idx, dtype=float)
    for i in range(window, len(idx), step):
        a = la.iloc[i - window:i].values
        b = lb.iloc[i - window:i].values
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < window * 0.8:
            continue
        try:
            out.iloc[i] = coint(a[m], b[m], trend="c", maxlag=1, autolag=None)[1]
        except Exception:
            continue
    return out.ffill()


def rolling_beta(la: pd.Series, lb: pd.Series, window: int = 250) -> pd.Series:
    """Trailing OLS hedge ratio. Drift away from the frozen value is the most
    direct signature of a relation changing shape."""
    cov = lb.rolling(window).cov(la)
    var = lb.rolling(window).var()
    return cov / var.replace(0.0, np.nan)


def rolling_half_life(spread: pd.Series, window: int = 250,
                      step: int = 21) -> pd.Series:
    """Trailing half-life. An exploding half-life means the pull to the mean is
    weakening -- the spread is drifting toward a random walk."""
    idx = spread.index
    out = pd.Series(np.nan, index=idx, dtype=float)
    for i in range(window, len(idx), step):
        out.iloc[i] = half_life(spread.iloc[i - window:i].values)
    return out.ffill()


def days_since_mean_cross(z: pd.Series, band: float = 0.5) -> pd.Series:
    """Number of sessions since the z-score was last inside +/- band.

    The most model-free break signal there is: a mean-reverting spread returns to
    its mean regularly. One that has not done so for months is not reverting,
    whatever the tests say.
    """
    inside = z.abs() <= band
    grp = inside.cumsum()
    return z.groupby(grp).cumcount().where(~inside, 0).astype(float)


def regime_flags(la: pd.Series, lb: pd.Series, z: pd.Series, beta_is: float,
                 window: int = 500, p_max: float = 0.50,
                 beta_drift_max: float = 0.50, hl_max: float = 60.0,
                 stale_max: int = 120) -> pd.DataFrame:
    """Four causal break signals, plus their disjunction.

    A pair is judged BROKEN when any of the four fires:

      p_break     trailing cointegration p-value above `p_max` -- the statistical
                  evidence for the relation has evaporated
      beta_break  trailing hedge ratio has drifted more than `beta_drift_max`
                  (relative) from the frozen in-sample value -- the relation has
                  changed shape, so the frozen hedge no longer hedges
      hl_break    trailing half-life above `hl_max` -- the pull to the mean has
                  weakened past the point of tradability
      stale_break the spread has not revisited its mean for `stale_max` sessions

    The disjunction is deliberate. These are different failure modes, not four
    measurements of one thing, and the cost of exiting a healthy pair is a missed
    trade while the cost of holding a broken one is unbounded.

    ON THE DEFAULTS -- window 500 rather than 250
    ---------------------------------------------
    A 250-day trailing Engle-Granger test has almost no power (Phase 2, Section
    5.1), so p > 0.10 is the NORMAL state even for a pair that passed a 750-day
    test. With window=250 and p_max=0.10 the p_break flag fired on 68 percent of
    days and the disjunction on 95 percent: the detector was not detecting, it
    was simply keeping the book flat. Doubling the window and only flagging when
    the evidence has genuinely gone (p > 0.5) brings the firing rate to roughly
    40 percent.

    MEASURED RESULT: ON THIS UNIVERSE THESE FLAGS DO NOT DISCRIMINATE
    -----------------------------------------------------------------
    The test that matters is not the firing rate but whether flagged days are
    actually worse. Mean daily P&L on flagged days minus mean on unflagged days,
    over four folds, with the flags used as a label and NOT as an exit:

        window  p_max   flag active   P&L(flagged) - P&L(clean), bp/day
           250   0.10        86.5 %                        +0.48
           500   0.30        46.7 %                        +2.60
           500   0.50        39.0 %                        +2.17
           750   0.50        20.2 %                        +3.03

    The differential is POSITIVE at every setting: flagged days are marginally
    BETTER, not worse. The detector has no predictive value here, and the
    drawdown reduction it produces is purely a leverage effect -- a book that is
    flat most of the time has less of everything, including loss.

    The likely reason is structural rather than a tuning failure. Regime
    detection can only protect a regime in which the strategy works, and the
    gross Sharpe of this book is approximately zero throughout (Phase 7). There
    is no good state to preserve. The mechanism is kept, correctly implemented
    and causal, as a negative result: it is the honest answer to "does your
    break detector actually work", and the answer is no.
    """
    spread = la - beta_is * lb
    p = rolling_coint_pvalue(la, lb, window)
    b = rolling_beta(la, lb, window)
    hl = rolling_half_life(spread, window)
    stale = days_since_mean_cross(z)

    flags = pd.DataFrame(index=la.index)
    flags["p_break"] = (p > p_max).fillna(False)
    flags["beta_break"] = ((b - beta_is).abs() / max(abs(beta_is), 1e-8)
                           > beta_drift_max).fillna(False)
    flags["hl_break"] = ((hl > hl_max) | hl.isna()).fillna(False) & hl.notna()
    flags["stale_break"] = (stale > stale_max).fillna(False)
    flags["broken"] = flags.any(axis=1)
    return flags
