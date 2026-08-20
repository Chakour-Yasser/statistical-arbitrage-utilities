"""Tests for Phases 3, 4 and 6. As in Phase 2, the tests that matter are the
ones that fail if a leak is reintroduced."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import pair_pnl, pair_positions
from src.selection import (benjamini_hochberg, benjamini_yekutieli, bonferroni,
                           liquidity_filter, walk_forward_folds)


# --------------------------------------------------------------------------- #
# Multiple-testing corrections
# --------------------------------------------------------------------------- #
def test_bonferroni_threshold_is_alpha_over_n():
    p = np.array([0.004, 0.006, 0.5])
    assert list(bonferroni(p, 0.015)) == [True, False, False]   # alpha/N = 0.005


def test_bh_is_at_least_as_liberal_as_bonferroni():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.random(200) ** 3
        assert benjamini_hochberg(p).sum() >= bonferroni(p).sum()


def test_by_is_more_conservative_than_bh():
    """BY divides the level by the harmonic number, so it can never reject more."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.random(200) ** 3
        assert benjamini_yekutieli(p).sum() <= benjamini_hochberg(p).sum()


def test_bh_step_up_takes_everything_below_the_last_crossing():
    """The step-up rule rejects all p-values below the Largest k that satisfies
    the bound, not only those individually below their own threshold."""
    p = np.array([0.001, 0.02, 0.025, 0.9])          # k*alpha/N = .0125,.025,.0375,.05
    got = benjamini_hochberg(p, 0.05)
    assert list(got) == [True, True, True, False]


def test_corrections_handle_empty_input():
    for f in (bonferroni, benjamini_hochberg, benjamini_yekutieli):
        assert f(np.array([])).shape == (0,)


# --------------------------------------------------------------------------- #
# Walk-forward: the anti-leak structure
# --------------------------------------------------------------------------- #
def test_folds_never_overlap_and_selection_precedes_trading():
    idx = pd.bdate_range("2014-01-01", "2026-06-30")
    folds = walk_forward_folds(idx, "2014-01-01", "2026-06-30")
    assert len(folds) >= 8
    for f in folds:
        assert f["is"][-1] < f["oos"][0]              # no overlap at all
        assert f["selection_date"] >= f["is"][-1]
        assert f["selection_date"] < f["oos"][0]      # decisions precede trading


def test_folds_roll_forward_one_year_at_a_time():
    idx = pd.bdate_range("2014-01-01", "2026-06-30")
    folds = walk_forward_folds(idx, "2014-01-01", "2026-06-30")
    starts = [f["is"][0].year for f in folds]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)


def test_liquidity_filter_ignores_everything_after_the_in_sample_window():
    """A name that is liquid only after the in-sample window must not be picked:
    that would be survivorship bias wearing a technical disguise."""
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    dv = pd.DataFrame(1e6, index=idx, columns=["OLD", "NEW"])
    is_win = idx[idx <= "2020-12-31"]
    dv.loc[is_win, "OLD"] = 5e7
    dv.loc[idx > "2020-12-31", "NEW"] = 5e7          # only liquid out-of-sample
    assert liquidity_filter(dv, is_win, ["OLD", "NEW"], min_adv=2e7) == ["OLD"]


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def test_execution_is_lagged_by_one_day():
    """The position decided on the close of t must earn the return from t to
    t+1. Earning the return of day t itself is the classic execution leak."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    pos = pd.Series([1.0, 1.0, 0.0, 0.0], index=idx)
    ra = pd.Series([0.10, 0.20, 0.30, 0.40], index=idx)
    rb = pd.Series(0.0, index=idx)
    out = pair_pnl(pos, ra, rb, beta=0.0, cost_bps=0.0)
    assert out["gross"].iloc[0] == 0.0               # nothing held on day 0
    assert out["gross"].iloc[1] == pytest.approx(0.20)   # day-0 signal, day-1 return


def test_stop_out_blocks_reentry_until_the_spread_normalises():
    """Without the block the position reopens on the very next bar, while the
    move is still against it. That bug inflated turnover more than tenfold."""
    z = pd.Series([0, 2.5, 4.5, 4.2, 3.0, 2.6, 0.2, 2.4],
                  index=pd.date_range("2020-01-01", periods=8, freq="D"))
    pos = pair_positions(z, entry=2.0, exit_=0.5, stop=4.0)
    assert pos.iloc[1] == -1.0        # short on entry
    assert pos.iloc[2] == 0.0         # stopped out
    assert (pos.iloc[3:6] == 0.0).all()   # stays out although |z| > entry
    assert pos.iloc[7] == -1.0        # re-enters only after |z| came back inside


def test_regime_break_forces_flat_and_blocks_reentry():
    z = pd.Series([-2.5, -2.4, -2.3, -0.1, -2.6],
                  index=pd.date_range("2020-01-01", periods=5, freq="D"))
    brk = pd.Series([False, True, False, False, False], index=z.index)
    pos = pair_positions(z, broken=brk)
    assert pos.iloc[0] == 1.0
    assert pos.iloc[1] == 0.0         # break flattens
    assert pos.iloc[2] == 0.0         # still blocked, |z| still extreme
    assert pos.iloc[4] == 1.0         # allowed again after z came back inside


def test_gross_exposure_is_one_dollar_whatever_the_hedge_ratio():
    """Without this normalisation a beta=3 pair would silently carry four times
    the risk of a beta=1 pair in an equally weighted book."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    pos = pd.Series([0.0, 1.0, 0.0], index=idx)
    for beta in (0.5, 1.0, 3.0):
        out = pair_pnl(pos, pd.Series(0.0, index=idx), pd.Series(0.0, index=idx),
                       beta=beta, cost_bps=10.0)
        assert out["turnover"].iloc[1] == pytest.approx(1.0)
