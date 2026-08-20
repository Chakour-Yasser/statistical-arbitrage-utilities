"""Tests for the Phase 2 cointegration building blocks.

The tests that matter here are not the ones checking a function returns a
number. They are the ones that FAIL if a leak is ever reintroduced, and the ones
that verify an estimator recovers a parameter we planted ourselves.
"""
import numpy as np
import pandas as pd
import pytest

from src.cointegration import (PairResult, calibrate_zscore, engle_granger, half_life,
                               hedge_ratio_ols, hedge_ratio_tls, ou_params,
                               passes_quality, rolling_zscore, spread_from_beta,
                               zscore_frozen)

SEED = 20260819


def _ou(n, hl, sigma=0.01, rng=None):
    """Ornstein-Uhlenbeck path with a KNOWN half-life."""
    phi = np.exp(-np.log(2.0) / hl)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.normal(0, sigma)
    return s


# --------------------------------------------------------------------------- #
# Half-life: does the estimator recover a planted parameter?
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("true_hl", [5.0, 15.0, 40.0])
def test_half_life_recovers_planted_value(true_hl):
    rng = np.random.default_rng(SEED)
    est = [half_life(_ou(3000, true_hl, rng=rng)) for _ in range(30)]
    assert np.nanmean(est) == pytest.approx(true_hl, rel=0.15)


def test_half_life_on_a_random_walk_is_finite_and_large_not_nan():
    """Documents the Dickey-Fuller downward bias, and pins it down.

    The naive expectation is that a random walk yields NaN (phi >= 1). It does
    not: near a unit root the OLS estimate of phi is biased downward, so a pure
    random walk returns a FINITE, large half-life. This test exists so that the
    behaviour is recorded rather than rediscovered, and so that the calibration
    of `passes_quality`'s upper bound stays honest.
    """
    rng = np.random.default_rng(SEED)
    hl = np.array([half_life(np.cumsum(rng.normal(0, .01, 750))) for _ in range(400)])
    assert np.mean(np.isnan(hl)) < 0.10          # NaN is the exception, not the rule
    assert np.nanmedian(hl) > 60                 # and the estimate is large
    # the default hl_max must keep most random walks out
    assert np.mean(np.isfinite(hl) & (hl <= 30)) < 0.06


def test_half_life_of_a_random_walk_grows_with_the_window():
    """The diagnostic that separates a slow reverter from a random walk: with no
    true half-life to recover, the estimate simply tracks the window length."""
    rng = np.random.default_rng(SEED)
    med = {}
    for n in (500, 1500):
        hl = [half_life(np.cumsum(rng.normal(0, .01, n))) for _ in range(200)]
        med[n] = np.nanmedian(hl)
    assert med[1500] > 1.8 * med[500]


def test_ou_params_agree_with_half_life():
    rng = np.random.default_rng(SEED)
    s = _ou(4000, 20.0, rng=rng)
    p = ou_params(s)
    assert p["half_life"] == pytest.approx(half_life(s), rel=1e-9)
    assert p["theta"] > 0


# --------------------------------------------------------------------------- #
# Hedge ratio: OLS is asymmetric, TLS is not
# --------------------------------------------------------------------------- #
def test_tls_is_exactly_symmetric():
    """beta_TLS(A|B) == 1 / beta_TLS(B|A). This is the property that makes TLS
    free of an arbitrary choice of dependent variable."""
    rng = np.random.default_rng(SEED)
    lb = np.cumsum(rng.normal(0, .01, 800))
    la = 1.4 * lb + rng.normal(0, .05, 800)
    b_ab, _ = hedge_ratio_tls(la, lb)
    b_ba, _ = hedge_ratio_tls(lb, la)
    assert b_ab == pytest.approx(1.0 / b_ba, rel=1e-9)


def test_ols_is_not_symmetric_and_the_gap_is_r_squared():
    """beta_OLS(A|B) * beta_OLS(B|A) == R^2. The noisier the relation, the
    further apart the two regressions are -- which is exactly why the choice of
    dependent variable must be a documented convention, never an optimisation."""
    rng = np.random.default_rng(SEED)
    lb = np.cumsum(rng.normal(0, .01, 800))
    la = 1.4 * lb + rng.normal(0, .08, 800)
    b_ab, _ = hedge_ratio_ols(la, lb)
    b_ba, _ = hedge_ratio_ols(lb, la)
    r2 = np.corrcoef(la, lb)[0, 1] ** 2
    assert b_ab * b_ba == pytest.approx(r2, rel=1e-9)
    assert b_ab != pytest.approx(1.0 / b_ba, rel=1e-3)


# --------------------------------------------------------------------------- #
# THE ANTI-LEAK TEST
# --------------------------------------------------------------------------- #
def test_rolling_zscore_is_causal():
    """Changing the FUTURE must never change a past z-score.

    This is the guardrail against the most common leak in pairs trading. If
    anyone ever replaces the rolling window with a full-sample mean/std, or
    centres the window, this test fails immediately.
    """
    rng = np.random.default_rng(SEED)
    s = pd.Series(_ou(500, 15.0, rng=rng))
    k = 300

    z_original = rolling_zscore(s, window=60)

    s_perturbed = s.copy()
    s_perturbed.iloc[k + 1:] += 10.0          # violently rewrite the future
    z_perturbed = rolling_zscore(s_perturbed, window=60)

    pd.testing.assert_series_equal(z_original.iloc[:k + 1],
                                   z_perturbed.iloc[:k + 1])


def test_rolling_zscore_uses_the_current_observation():
    """z_t must use s_t itself: the signal is computed on the close of day t.
    Execution is lagged to t+1 in the backtest, not the signal."""
    s = pd.Series(np.r_[np.zeros(60), 5.0])
    z = rolling_zscore(s, window=60)
    assert np.isfinite(z.iloc[-1]) and z.iloc[-1] > 0


def test_full_sample_zscore_would_leak():
    """Demonstrates what we are avoiding: a full-sample z-score DOES change the
    past when the future changes. Kept as an executable counter-example."""
    rng = np.random.default_rng(SEED)
    s = pd.Series(_ou(500, 15.0, rng=rng))
    k = 300
    naive = lambda x: (x - x.mean()) / x.std()
    s2 = s.copy(); s2.iloc[k + 1:] += 10.0
    assert not np.allclose(naive(s).iloc[:k + 1], naive(s2).iloc[:k + 1])


# --------------------------------------------------------------------------- #
# Engle-Granger
# --------------------------------------------------------------------------- #
def test_engle_granger_detects_a_planted_cointegration():
    rng = np.random.default_rng(SEED)
    n = 750
    lb = np.cumsum(rng.normal(0, .012, n))
    la = 1.3 * lb + _ou(n, 12.0, sigma=.01, rng=rng) + 2.0   # cointegrated by construction
    res = engle_granger(la, lb)
    assert res.pvalue < 0.01
    assert res.beta == pytest.approx(1.3, abs=0.15)
    assert 5.0 < res.half_life < 30.0


def test_engle_granger_keeps_nominal_size_under_the_null():
    """Independent random walks: rejection rate must sit near 5 %, NOT near the
    14.6 % that `adfuller` on the OLS residual produces."""
    rng = np.random.default_rng(SEED)
    rej = 0
    n_sim = 150
    for _ in range(n_sim):
        a = np.cumsum(rng.normal(0, .012, 500))
        b = np.cumsum(rng.normal(0, .012, 500))
        rej += engle_granger(a, b).pvalue < 0.05
    assert rej / n_sim < 0.11        # generous bound: 150 sims, se ~ 1.8 pts


def test_engle_granger_handles_missing_values():
    rng = np.random.default_rng(SEED)
    n = 600
    lb = np.cumsum(rng.normal(0, .012, n))
    la = 1.1 * lb + _ou(n, 10.0, rng=rng)
    la[50:60] = np.nan
    res = engle_granger(la, lb)
    assert res.n_obs == n - 10
    assert np.isfinite(res.pvalue)


# --------------------------------------------------------------------------- #
# Quality filter
# --------------------------------------------------------------------------- #
def _res(hl, n_obs=750, beta=1.0):
    return PairResult("A", "B", 0.01, beta, 0.0, hl, n_obs, 0.05)


@pytest.mark.parametrize("hl,expected", [
    (0.5, False),    # too fast: microstructure noise, turnover kills it
    (2.0, True),
    (15.0, True),
    (30.0, True),
    (60.0, False),   # too slow, and admits ~18 % of random walks
    (120.0, False),  # too slow: capital tied up, relation likely breaks first
    (np.nan, False), # no mean reversion at all
])
def test_quality_filter_bounds(hl, expected):
    assert passes_quality(_res(hl)) is expected


def test_quality_filter_rejects_short_history():
    assert passes_quality(_res(15.0, n_obs=200)) is False


@pytest.mark.parametrize("beta,expected", [
    (-0.41, False),  # long/long: not a hedge at all
    (0.20, False),   # degenerate regression (the NRG pattern)
    (0.50, True),
    (1.00, True),
    (2.50, True),
    (4.12, False),   # extreme leverage on one leg
    (np.nan, False),
])
def test_quality_filter_hedge_ratio_band(beta, expected):
    assert passes_quality(_res(15.0, beta=beta)) is expected


def test_hedge_ratio_band_is_symmetric_under_leg_inversion():
    """The band must not depend on the alphabetical ordering convention: if beta
    passes, 1/beta must pass too."""
    for beta in (0.4, 0.8, 1.0, 1.25, 2.5):
        assert passes_quality(_res(15.0, beta=beta)) == \
               passes_quality(_res(15.0, beta=1 / beta))


# --------------------------------------------------------------------------- #
# Frozen z-score
# --------------------------------------------------------------------------- #
def test_zscore_frozen_is_causal_by_construction():
    """Frozen parameters come from a window that ends before the trading period,
    so rewriting the trading period cannot move any earlier z-score."""
    rng = np.random.default_rng(SEED)
    s = pd.Series(_ou(1200, 15.0, rng=rng))
    mu, sd = calibrate_zscore(s.iloc[:750])
    oos = s.iloc[750:]
    z1 = zscore_frozen(oos, mu, sd)
    oos2 = oos.copy(); oos2.iloc[200:] += 7.0
    z2 = zscore_frozen(oos2, mu, sd)
    pd.testing.assert_series_equal(z1.iloc[:200], z2.iloc[:200])


def test_frozen_zscore_beats_rolling_on_trigger_fidelity():
    """The bias that motivates the design choice: on an autocorrelated spread the
    rolling z-score fires far too often. Frozen parameters roughly halve the
    excess. Neither is exact -- that is why Phase 6 must MEASURE turnover."""
    rng = np.random.default_rng(SEED)
    roll, froz = [], []
    for _ in range(40):
        s = pd.Series(_ou(1750, 30.0, rng=rng))
        mu, sd = calibrate_zscore(s.iloc[:750])
        froz.append((zscore_frozen(s.iloc[750:], mu, sd).abs() > 2).mean())
        roll.append((rolling_zscore(s, 60).iloc[750:].abs() > 2).mean())
    assert np.mean(roll) > np.mean(froz) > 0.0455 * 0.8
    assert np.mean(roll) > 0.09          # rolling fires ~2.5x too often


def test_zscore_frozen_handles_degenerate_sigma():
    s = pd.Series(np.arange(10.0))
    assert zscore_frozen(s, 0.0, 0.0).isna().all()
