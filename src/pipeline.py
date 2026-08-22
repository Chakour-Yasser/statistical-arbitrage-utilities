"""
Walk-forward pipeline: selection (Phase 3) and regime-aware trading (Phase 4).

The single invariant this file exists to enforce: everything is estimated on the
in-sample window and frozen before the out-of-sample window starts. The universe,
the liquidity filter, the cointegration test, the hedge ratio and the z-score
calibration all stop at `selection_date`. Nothing is re-fitted during trading.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .backtest import metrics, pair_pnl, pair_positions
from .cointegration import (PairResult, calibrate_zscore, passes_quality,
                            screen_pairs, spread_from_beta, zscore_frozen)
from .regime import regime_flags
from .selection import Corrections, liquidity_filter, walk_forward_folds

_COLS = ("a", "b", "pvalue", "beta", "alpha", "half_life", "n_obs", "resid_std")


def _universe_at(memb: pd.DataFrame, date: pd.Timestamp) -> list[str]:
    """Point-in-time sector members at the selection date."""
    snaps = memb.index[memb.index <= date]
    if len(snaps) == 0:
        return []
    return sorted(memb.columns[memb.loc[snaps[-1]]])


def run_fold(fold: dict, close: pd.DataFrame, dollar_volume: pd.DataFrame,
             memb: pd.DataFrame, alpha: float = 0.05,
             cost_bps: float = 5.0, use_regime: bool = True,
             methods: tuple = ("naive", "bonferroni", "bh", "by")) -> dict:
    """Screen and freeze on the in-sample window, then trade the out-of-sample year."""
    is_win, oos_win = fold["is"], fold["oos"]
    tickers = _universe_at(memb, fold["selection_date"])
    tickers = liquidity_filter(dollar_volume, is_win, tickers)
    if len(tickers) < 5:
        return {}

    log_px = np.log(close)
    scan = screen_pairs(log_px.loc[is_win], tickers, min_obs=int(0.8 * len(is_win)))
    if scan.empty:
        return {}
    scan["quality"] = [passes_quality(PairResult(**{k: r[k] for k in _COLS}))
                       for _, r in scan.iterrows()]

    # trailing history for the causal regime statistics at the start of the Oos year
    hist_start = oos_win[0] - pd.Timedelta(days=420)
    ext = close.index[(close.index >= hist_start) & (close.index <= oos_win[-1])]
    rets = close.pct_change()

    out = {"fold": fold["fold"], "n_tests": len(scan), "n_tickers": len(tickers)}
    for method in methods:
        keep = Corrections[method](scan["pvalue"].values, alpha) & scan["quality"].values
        sel = scan[keep]
        out[f"n_{method}"] = int(len(sel))
        if sel.empty:
            out[f"res_{method}"] = None
            continue

        legs, brk_legs = [], []
        for _, r in sel.iterrows():
            a, b, beta, al = r["a"], r["b"], r["beta"], r["alpha"]
            la_e, lb_e = log_px[a].reindex(ext), log_px[b].reindex(ext)
            spread_is = spread_from_beta(log_px[a].loc[is_win], log_px[b].loc[is_win], beta, al)
            mu, sd = calibrate_zscore(spread_is)                       # Frozen
            z_ext = zscore_frozen(spread_from_beta(la_e, lb_e, beta, al), mu, sd)

            broken = None
            if use_regime:
                broken = regime_flags(la_e, lb_e, z_ext, beta)["broken"]

            z_oos = z_ext.reindex(oos_win)
            pos = pair_positions(z_oos, broken=(broken.reindex(oos_win)
                                                if broken is not None else None))
            pnl = pair_pnl(pos, rets[a].reindex(oos_win), rets[b].reindex(oos_win),
                           beta, cost_bps)
            legs.append(pnl[["gross", "cost", "net", "turnover"]])
            if broken is not None:
                brk_legs.append(broken.reindex(oos_win).fillna(False).astype(float))

        book = sum(legs) / len(legs)          # equal weight across pairs
        book["broken_share"] = (sum(brk_legs) / len(brk_legs)) if brk_legs else 0.0
        out[f"res_{method}"] = {
            **metrics(book["net"], book["turnover"]),
            "gross_sharpe": metrics(book["gross"])["sharpe"],
            "book": book,
        }
    return out


def run_walk_forward(close: pd.DataFrame, dollar_volume: pd.DataFrame,
                     memb: pd.DataFrame, use_regime: bool = True,
                     cost_bps: float = 5.0, alpha: float = 0.05) -> tuple:
    folds = walk_forward_folds(close.index, C.START, C.END)
    rows, curves = [], {}
    for f in folds:
        r = run_fold(f, close, dollar_volume, memb, alpha=alpha,
                     cost_bps=cost_bps, use_regime=use_regime)
        if not r:
            continue
        row = {k: v for k, v in r.items() if not k.startswith("res_")}
        for m in ("naive", "bonferroni", "bh", "by"):
            res = r.get(f"res_{m}")
            if res:
                row[f"sharpe_{m}"] = res["sharpe"]
                row[f"ret_{m}"] = res["ann_return"]
                row[f"gross_{m}"] = res["gross_sharpe"]
                row[f"turn_{m}"] = res["turnover"]
                curves.setdefault(m, []).append(res["book"])
            else:
                row[f"sharpe_{m}"] = np.nan
                row[f"ret_{m}"] = np.nan
        rows.append(row)
    stitched = {m: pd.concat(v).sort_index() for m, v in curves.items()}  # DataFrames
    return pd.DataFrame(rows), stitched
