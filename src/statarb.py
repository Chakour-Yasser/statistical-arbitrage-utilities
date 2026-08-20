"""
Cross-sectional statistical arbitrage on PCA factor residuals.

Method: Avellaneda and Lee (2010), "Statistical arbitrage in the US equities
market". This replaces the pairs approach, and the replacement is a response to
a measured failure rather than a new roll of the dice.

WHAT WAS WRONG WITH PAIRS, AND WHAT THIS FIXES
----------------------------------------------
Measured on the utilities book: a pair cointegrated at 5 % over three years had a
10.8 % chance of still being cointegrated the following year -- against 22.3 %
for the pairs the screen had REJECTED. Selecting the lowest p-value out of 400
selects the residual that was luckiest in sample, and luck does not repeat. On
top of that the equilibrium level itself moved by a median of 1.47 in-sample
sigmas between the estimation window and the trading year.

This method removes both failure modes by construction:

  no selection      every name in the universe is traded on its own residual, so
                    there is no "best pair out of 400" and no winner's curse
  no frozen level   factors, betas and the OU parameters are re-estimated on a
                    rolling window, so no equilibrium level goes stale
  breadth           several hundred simultaneous residual bets instead of the
                    2-3 independent bets a concentrated pairs book really holds
  a real mechanism  a residual that dislocates without news is an order-imbalance
                    event -- a fund liquidating, an index rebalance. Taking the
                    other side supplies liquidity, and being paid for that is an
                    economic mechanism rather than a statistical coincidence

Every estimate below uses a trailing window only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Factor model
# --------------------------------------------------------------------------- #
def marchenko_pastur_count(eigenvalues: np.ndarray, n_assets: int, n_obs: int,
                           cap: int = 30) -> int:
    """Number of eigenvalues that stand above the pure-noise bound.

    For a sample correlation matrix built from n_obs observations of n_assets
    series, the eigenvalues of a NOISE matrix fill the Marchenko-Pastur support
    up to (1 + sqrt(N/T))^2. Anything below that is indistinguishable from
    sampling error.

    This matters here because N/T is about 1.7: the sample correlation matrix is
    rank-deficient, and only the first handful of eigenvectors carry signal.
    Measured on this universe, 6 to 9 eigenvalues clear the bound depending on
    the date -- against the 15 factors originally hedged. Hedging a name against
    six noise directions removes genuine residual signal and rebalances a random
    hedge every day, paying turnover for nothing.
    """
    bound = (1.0 + np.sqrt(n_assets / n_obs)) ** 2
    return int(np.clip((eigenvalues > bound).sum(), 1, cap))


def eigen_portfolios(returns: np.ndarray, n_factors: int) -> np.ndarray:
    """Risk-adjusted eigenportfolio weights Q (n_assets x n_factors).

    The PCA is run on the CORRELATION matrix, not the covariance matrix, so that
    a handful of high-volatility names cannot dominate the factor structure.
    The eigenvector is then divided by each asset's volatility, which turns a
    correlation-space direction into a tradable dollar-weight portfolio -- this
    is the Q_ij = v_ij / sigma_i construction of Avellaneda-Lee.
    """
    sd = returns.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, np.nan)
    z = (returns - returns.mean(axis=0)) / sd
    corr = np.corrcoef(z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    vals, vecs = np.linalg.eigh(corr)
    idx = np.argsort(vals)[::-1][:n_factors]
    return vecs[:, idx] / sd[:, None], vals[np.argsort(vals)[::-1]]


def factor_returns(returns: np.ndarray, q: np.ndarray) -> np.ndarray:
    """F = R Q, the realised returns of the eigenportfolios."""
    return returns @ q


def residual_betas(returns: np.ndarray, factors: np.ndarray) -> tuple:
    """OLS of every asset on the factors, in one solve.

    Returns (beta, residuals). Regressing each of 500 assets separately would be
    500 solves; since they share the same design matrix, one pseudo-inverse does
    the whole cross-section at once.
    """
    x = np.column_stack([np.ones(len(factors)), factors])
    beta, *_ = np.linalg.lstsq(x, returns, rcond=None)
    resid = returns - x @ beta
    return beta, resid


# --------------------------------------------------------------------------- #
# OU fit on the cumulative residual
# --------------------------------------------------------------------------- #
def ou_scores(resid: np.ndarray) -> dict:
    """Fit an OU process to the cumulative residual of every asset, vectorised.

    The residual RETURN is close to white noise; its cumulative sum X is the
    quantity with an equilibrium level. Fitting AR(1) to X,

        X_{n+1} = a + b X_n + zeta,

    gives the mean-reversion speed kappa = -log(b) * 252 (per year), the
    equilibrium m = a / (1 - b), and the equilibrium dispersion
    sigma_eq = sqrt(var(zeta) / (1 - b^2)).

    The s-score is the standardised deviation from equilibrium,

        s = (X_last - m) / sigma_eq,

    which plays the role the z-score played for pairs -- but recomputed every day
    from a trailing window, so it cannot go stale.
    """
    x = np.cumsum(resid, axis=0)
    x0, x1 = x[:-1], x[1:]
    n = len(x0)
    mx0 = x0.mean(axis=0)
    mx1 = x1.mean(axis=0)
    cov = ((x0 - mx0) * (x1 - mx1)).sum(axis=0)
    var = ((x0 - mx0) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(var > 0, cov / var, np.nan)
        a = mx1 - b * mx0
        pred = a + b * x0
        zeta_var = ((x1 - pred) ** 2).sum(axis=0) / max(n - 2, 1)
        ok = (b > 0) & (b < 1) & np.isfinite(b)
        kappa = np.where(ok, -np.log(np.where(ok, b, 0.5)) * TRADING_DAYS, np.nan)
        m = np.where(ok, a / (1 - np.where(ok, b, 0.5)), np.nan)
        sigma_eq = np.where(ok, np.sqrt(zeta_var / (1 - np.where(ok, b, 0.5) ** 2)), np.nan)
        s = np.where(sigma_eq > 0, (x[-1] - m) / sigma_eq, np.nan)
    return {"s_score": s, "kappa": kappa, "m": m, "sigma_eq": sigma_eq, "b": b}


# --------------------------------------------------------------------------- #
# Trading rules
# --------------------------------------------------------------------------- #
def target_positions(s: np.ndarray, prev: np.ndarray, tradable: np.ndarray,
                     s_open: float = 1.25, s_close_long: float = 0.50,
                     s_close_short: float = 0.75) -> np.ndarray:
    """Avellaneda-Lee entry and exit bands, applied to the previous state.

    A residual far BELOW equilibrium (s very negative) means the name is cheap
    against its factor exposure, so we go long it and short the factors. The
    asymmetric exit bands are theirs: long positions are closed earlier than
    short ones, which reflects the asymmetry of equity residual distributions.
    """
    pos = prev.copy()
    pos[~tradable] = 0.0
    open_long = tradable & (s < -s_open) & (prev == 0)
    open_short = tradable & (s > s_open) & (prev == 0)
    pos[open_long] = 1.0
    pos[open_short] = -1.0
    close_long = tradable & (prev > 0) & (s > -s_close_long)
    close_short = tradable & (prev < 0) & (s < s_close_short)
    pos[close_long] = 0.0
    pos[close_short] = 0.0
    return pos


def book_weights(pos: np.ndarray, beta: np.ndarray, q: np.ndarray) -> np.ndarray:
    # beta: (n_assets, n_factors)   q: (n_assets, n_factors)   pos: (n_assets,)
    """Net dollar weights of the whole book, hedges included.

    Holding one dollar of name i means shorting beta_ij of each eigenportfolio j,
    and each eigenportfolio is itself a basket over the same names. Netting the
    hedges ACROSS the book before computing turnover matters: summing the
    per-name hedges would overstate traded notional several-fold, since most of
    the factor exposure cancels between longs and shorts.
    """
    if pos.sum() == 0 and np.all(pos == 0):
        return np.zeros_like(pos)
    gross = np.abs(pos).sum()
    p = pos / gross if gross > 0 else pos
    factor_exposure = p @ beta            # (n_factors,) : sum_i p_i beta_ij
    return p - q @ factor_exposure


# --------------------------------------------------------------------------- #
# Walk-forward backtest
# --------------------------------------------------------------------------- #
def run_statarb(close: pd.DataFrame, dollar_volume: pd.DataFrame,
                membership: pd.DataFrame, n_factors: int | str = 15,
                pca_window: int = 252, reg_window: int = 60, pca_step: int = 5,
                kappa_min_days: float = 30.0, min_adv: float = 1e7,
                cost_bps: float = 5.0, s_open: float = 1.25,
                no_trade_band: float = 0.0, beta_step: int = 1,
                risk_scale: bool = False, adv_quantile: tuple | None = None,
                periods_per_year: int = TRADING_DAYS,
                null_permute: bool = False, seed: int = 0) -> pd.DataFrame:
    """Daily cross-sectional statistical arbitrage backtest.

    Everything at date t is estimated on data up to and including t, and the
    resulting weights earn the return from t to t+1.

    `no_trade_band` is expressed as a multiple of the average absolute weight: a
    name is left untouched unless its target weight has moved by more than that.
    Most of the daily turnover is not position changes but HEDGE DRIFT -- the
    betas move a little every day, so the hedge legs are rebalanced every day for
    no informational reason. The band is a cost-control device, not a signal
    parameter: it never changes which names the model wants to hold, only how
    precisely the book tracks them.

    `beta_step` re-estimates the factor loadings every k days instead of daily,
    for the same reason.

    `n_factors="mp"` picks the factor count each day from the Marchenko-Pastur
    noise bound instead of fixing it at 15. This is a correction, not a tuned
    parameter: 15 was arbitrary and demonstrably above the noise threshold.

    `risk_scale` sizes each position by the inverse of its residual volatility
    rather than by equal notional. With equal notional the risk contribution of
    a name is proportional to its residual vol, so a handful of volatile names
    dominate the book's variance while contributing no more information than the
    quiet ones. Equalising risk contributions is the textbook construction.

    `adv_quantile=(lo, hi)` restricts POSITIONS to names whose trailing dollar
    volume falls in that quantile band of the cross-section, while the factor
    model keeps using the whole universe. This isolates the liquidity dimension:
    if the residual reversal premium is compensation for supplying liquidity, it
    should be larger where liquidity is scarcer -- and so should the cost of
    harvesting it.

    `null_permute` randomises the SIGN of every return row: at each date, all
    symbols are multiplied by the same random +1 or -1. This is the null that
    isolates the effect being traded.

      preserved   the cross-sectional covariance at every date, hence the same
                  PCA, the same eigenportfolios, the same betas; the volatility
                  clustering; and -- crucially -- the availability pattern, since
                  a NaN stays a NaN
      destroyed   the sign predictability of the cumulative residual, which is
                  exactly what mean reversion is

    A time-permutation was used first and is wrong on an unbalanced panel: it
    moves the NaNs as well, so on a universe with staggered listings almost no
    symbol retains a complete estimation window and the backtest produces nothing.
    Sign randomisation has no such problem.

    Any Sharpe the strategy earns under this null is machinery, not signal.
    Running it BEFORE reporting a positive result is the discipline this project
    exists to demonstrate.
    """
    rets = close.pct_change()
    if null_permute:
        rng = np.random.default_rng(seed)
        flip = rng.choice([-1.0, 1.0], size=len(rets))[:, None]
        rets = pd.DataFrame(rets.values * flip, index=rets.index, columns=rets.columns)

    dates = close.index
    start = max(pca_window, reg_window) + 1
    q = beta_full = None
    prev_w = pd.Series(0.0, index=close.columns)
    pos_state = pd.Series(0.0, index=close.columns)
    rows = []

    for i in range(start, len(dates) - 1):
        t = dates[i]
        snaps = membership.index[membership.index <= t]
        if len(snaps) == 0:
            continue
        members = membership.columns[membership.loc[snaps[-1]]]

        win = rets.iloc[i - pca_window + 1: i + 1]
        adv = dollar_volume.iloc[i - pca_window + 1: i + 1].median()
        cols = [c for c in members
                if c in win.columns and win[c].notna().all() and adv.get(c, 0) >= min_adv]
        if len(cols) < 50:
            continue
        R = win[cols].values

        if q is None or (i - start) % pca_step == 0 or q.shape[0] != len(cols) \
                or q_cols != cols:
            k = n_factors
            if n_factors == "mp":
                z = (R - R.mean(0)) / np.where(R.std(0, ddof=1) > 0, R.std(0, ddof=1), np.nan)
                ev = np.linalg.eigvalsh(np.nan_to_num(np.corrcoef(z, rowvar=False), nan=0.0))
                k = marchenko_pastur_count(np.sort(ev)[::-1], len(cols), pca_window)
            q, _ = eigen_portfolios(R, k)
            q_cols = cols

        F = factor_returns(R, q)
        if beta_full is None or (i - start) % beta_step == 0 or beta_full.shape[1] != len(cols):
            beta_full, resid = residual_betas(R[-reg_window:], F[-reg_window:])
        else:
            x = np.column_stack([np.ones(reg_window), F[-reg_window:]])
            resid = R[-reg_window:] - x @ beta_full
        beta, resid = beta_full, resid
        sc = ou_scores(resid)

        hl_days = np.log(2) / sc["kappa"] * periods_per_year
        tradable = np.isfinite(sc["s_score"]) & np.isfinite(hl_days) & (hl_days <= kappa_min_days)
        if adv_quantile is not None:
            a = adv.reindex(cols).values
            lo, hi = np.nanquantile(a, adv_quantile[0]), np.nanquantile(a, adv_quantile[1])
            tradable &= (a >= lo) & (a <= hi)

        prev = pos_state.reindex(cols).fillna(0.0).values
        pos = target_positions(sc["s_score"], prev, tradable, s_open=s_open)
        pos_state = pd.Series(0.0, index=close.columns)
        pos_state.loc[cols] = pos

        sized = pos
        if risk_scale:
            sd_resid = resid.std(axis=0, ddof=1)
            inv = np.where(sd_resid > 0, 1.0 / sd_resid, 0.0)
            sized = pos * inv
        w_vec = book_weights(sized, beta[1:].T, q)        # drop the intercept row
        w = pd.Series(0.0, index=close.columns)
        w.loc[cols] = w_vec
        if no_trade_band > 0:
            scale = np.abs(w.values).mean()
            delta = (w - prev_w).abs()
            hold = (delta < no_trade_band * scale) & (delta > 0)
            w[hold] = prev_w[hold]

        r_next = rets.iloc[i + 1].reindex(close.columns).fillna(0.0)
        gross = float((w * r_next).sum())
        turn = float((w - prev_w).abs().sum())
        rows.append({"date": dates[i + 1], "gross": gross, "turnover": turn,
                     "net": gross - turn * cost_bps / 1e4,
                     "n_names": len(cols), "n_pos": int((pos != 0).sum()),
                     "n_factors": q.shape[1],
                     "gross_exposure": float(w.abs().sum())})
        prev_w = w

    return pd.DataFrame(rows).set_index("date")
