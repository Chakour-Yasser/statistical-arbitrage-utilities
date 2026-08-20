"""
Phase 2 -- cointegration building blocks.

Every function here is causal by construction or explicitly documented as an
in-sample estimator. The two traps this module exists to avoid:

1. WRONG CRITICAL VALUES. Running `adfuller` on an OLS residual is the standard
   textbook mistake. The residual is *estimated*: OLS picked beta precisely to
   minimise its variance, so it is "as stationary as possible" by construction.
   The test statistic's distribution shifts and ADF critical values become far
   too permissive. Measured under H0 (independent random walks, n=750):
       adfuller on the OLS residual : 14.6 % rejections at the 5 % level
       coint (Engle-Granger values) :  4.2 % rejections
   A 3.5x inflation of the false-positive rate, before multiple testing even
   enters the picture. We therefore always use `statsmodels.tsa.stattools.coint`.

2. NON-CAUSAL Z-SCORE. Standardising a spread with the full-sample mean and
   standard deviation uses the future at every instant. It is the single most
   common leak in pairs trading. `rolling_zscore` is causal, and
   tests/test_cointegration.py contains a test that fails if it ever stops
   being so.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


# --------------------------------------------------------------------------- #
# Hedge ratio
# --------------------------------------------------------------------------- #
def hedge_ratio_ols(la: np.ndarray, lb: np.ndarray) -> tuple[float, float]:
    """OLS of `la` on `lb`. Returns (beta, alpha).

    ASYMMETRY -- the limitation to state in an interview. OLS minimises
    *vertical* errors, so it assigns all measurement noise to the left-hand
    variable. Consequently beta(A|B) != 1 / beta(B|A): the two regressions
    differ by a factor R^2. The gap widens as the relation gets noisier --
    exactly when the pair matters least, but also exactly when the choice of
    dependent variable matters most.
    """
    beta, alpha = np.polyfit(lb, la, 1)
    return float(beta), float(alpha)


def hedge_ratio_tls(la: np.ndarray, lb: np.ndarray) -> tuple[float, float]:
    """Total least squares (orthogonal regression). Returns (beta, alpha).

    Minimises the PERPENDICULAR distance to the line rather than the vertical
    one, which treats both legs symmetrically: beta(A|B) == 1 / beta(B|A)
    exactly. Computed as the first principal component of the centred data.

    Trade-off vs OLS
    ----------------
    TLS removes the arbitrary choice of dependent variable, which is the honest
    thing to do when neither leg has a privileged status. But it assumes both
    legs carry comparable noise, and it has no closed-form standard error, so
    the Engle-Granger test machinery (built on an OLS first stage) no longer
    applies directly. We therefore use OLS for TESTING and expose TLS to check
    that conclusions do not hinge on the regression direction.
    """
    x = np.column_stack([lb - lb.mean(), la - la.mean()])
    # smallest-singular-value direction is the orthogonal residual direction
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    vx, vy = vt[0]                      # first principal component
    beta = float(vy / vx)
    alpha = float(la.mean() - beta * lb.mean())
    return beta, alpha


def spread_from_beta(la, lb, beta: float, alpha: float):
    """s = log P_A - beta * log P_B - alpha."""
    return la - beta * lb - alpha


# --------------------------------------------------------------------------- #
# Mean-reversion speed
# --------------------------------------------------------------------------- #
def half_life(spread: np.ndarray) -> float:
    """Half-life of mean reversion, in periods (days here).

    Derivation -- know this one by heart
    -----------------------------------
    Regress the change on the level:      ds_t = a + b * s_{t-1} + eps
    which is an AR(1):                    s_t  = a + (1 + b) s_{t-1} + eps
    Write phi = 1 + b. A shock decays as phi^k, so the half-life solves
    phi^h = 1/2, i.e.

        half_life = -ln(2) / ln(phi)

    Link to Ornstein-Uhlenbeck: ds = theta (mu - s) dt + sigma dW discretises to
    phi = exp(-theta dt), hence half_life = ln(2) / theta. Same object.

    Returns NaN when phi is outside (0, 1): phi >= 1 means no mean reversion,
    phi <= 0 means alternating behaviour that the OU model does not describe.

    CRITICAL CAVEAT -- this estimator is biased and is NOT a stationarity test
    -------------------------------------------------------------------------
    Near a unit root the OLS estimate of phi is biased downward (the classic
    Dickey-Fuller bias). A pure random walk therefore does NOT return NaN: it
    returns a finite, large half-life. Measured on 4000 random walks of 750
    observations (our in-sample window length):

        NaN returned          :   4 % of the time
        median estimated h-l  : 111 days
        share passing hl<=60  :  18.4 %
        share passing hl<=30  :   2.7 %

    A useful diagnostic falls out of this: on a genuine random walk the
    estimated half-life GROWS with the estimation window (76 d at n=500, 111 d
    at n=750, 222 d at n=1500), because there is no true half-life to recover --
    the estimate merely tracks the window length. A half-life that is unstable
    across window lengths is a red flag.

    Consequence for the pipeline: the half-life is a TRADABILITY filter applied
    *after* the cointegration test has done the statistical work. It must never
    be used to decide whether a spread mean-reverts.
    """
    s = np.asarray(spread, dtype=float)
    s = s[~np.isnan(s)]
    if s.size < 20:
        return np.nan
    ds = np.diff(s)
    b, _ = np.polyfit(s[:-1], ds, 1)
    phi = 1.0 + b
    if not (0.0 < phi < 1.0):
        return np.nan
    return float(-np.log(2.0) / np.log(phi))


def ou_params(spread: np.ndarray, dt: float = 1.0) -> dict:
    """Fit an Ornstein-Uhlenbeck process. Returns theta, mu, sigma, half_life."""
    s = np.asarray(spread, dtype=float)
    s = s[~np.isnan(s)]
    if s.size < 20:
        return dict(theta=np.nan, mu=np.nan, sigma=np.nan, half_life=np.nan)
    ds = np.diff(s)
    b, a = np.polyfit(s[:-1], ds, 1)
    phi = 1.0 + b
    if not (0.0 < phi < 1.0):
        return dict(theta=np.nan, mu=np.nan, sigma=np.nan, half_life=np.nan)
    theta = -np.log(phi) / dt
    mu = -a / b
    resid = ds - (a + b * s[:-1])
    sigma = float(np.std(resid, ddof=2)) / np.sqrt(dt)
    return dict(theta=float(theta), mu=float(mu), sigma=sigma,
                half_life=float(np.log(2.0) / theta))


# --------------------------------------------------------------------------- #
# Cointegration test
# --------------------------------------------------------------------------- #
@dataclass
class PairResult:
    a: str
    b: str
    pvalue: float
    beta: float
    alpha: float
    half_life: float
    n_obs: int
    resid_std: float

    def as_dict(self) -> dict:
        return asdict(self)


def engle_granger(la: np.ndarray, lb: np.ndarray, name_a: str = "A",
                  name_b: str = "B", maxlag: int = 1) -> PairResult:
    """Engle-Granger cointegration test with the CORRECT critical values.

    `coint` runs the OLS first stage internally and evaluates the ADF statistic
    against MacKinnon's Engle-Granger distribution, which accounts for the fact
    that the residual is estimated. See this module's docstring for the size
    distortion incurred by using `adfuller` instead.

    On `maxlag`: fixed at 1 rather than selected by AIC. AIC selection is
    standard but makes the lag a data-dependent choice, which adds one more
    researcher degree of freedom to a screener that already tests hundreds of
    pairs. A fixed lag is defensible precisely because it is not chosen.
    """
    la = np.asarray(la, dtype=float)
    lb = np.asarray(lb, dtype=float)
    mask = ~(np.isnan(la) | np.isnan(lb))
    la, lb = la[mask], lb[mask]

    _, pvalue, _ = coint(la, lb, trend="c", maxlag=maxlag, autolag=None)
    beta, alpha = hedge_ratio_ols(la, lb)
    resid = spread_from_beta(la, lb, beta, alpha)
    return PairResult(a=name_a, b=name_b, pvalue=float(pvalue), beta=beta,
                      alpha=alpha, half_life=half_life(resid), n_obs=int(la.size),
                      resid_std=float(np.std(resid, ddof=2)))


# --------------------------------------------------------------------------- #
# Causal z-score
# --------------------------------------------------------------------------- #
def rolling_zscore(spread: pd.Series, window: int = 60,
                   min_periods: int | None = None) -> pd.Series:
    """Causal z-score: at date t, uses only observations up to and including t.

    WHY NOT A FULL-SAMPLE Z-SCORE
    -----------------------------
    z_t = (s_t - mean(s)) / std(s) computed over the whole period injects the
    future into every single observation. The strategy would "know" the spread's
    eventual average level, which is precisely the quantity it is supposed to be
    betting on. It is the most common leak in pairs trading and it inflates
    performance dramatically.

    Including t itself is deliberate: t is the close on which the signal is
    computed. Execution is then lagged to t+1 in the backtest (Phase 6). The
    leak would be to execute at t.

    NOT THE DEFAULT FOR THE BACKTEST -- see `zscore_frozen`. On a strongly
    autocorrelated spread this estimator is biased, and the bias INFLATES the
    z-score (numerator shrinks, denominator shrinks more), firing about 2.5x too
    many entries at a 30-day half-life. Kept because it is what most published
    implementations use and because Phase 4 needs a genuinely adaptive
    statistic; documented so the bias is never accidental.
    """
    if min_periods is None:
        min_periods = window
    m = spread.rolling(window, min_periods=min_periods).mean()
    sd = spread.rolling(window, min_periods=min_periods).std(ddof=1)
    return (spread - m) / sd.replace(0.0, np.nan)


def calibrate_zscore(spread_is: pd.Series | np.ndarray) -> tuple[float, float]:
    """Freeze (mu, sigma) on the in-sample window. Use with `zscore_frozen`.

    Called once per fold, on the same in-sample window that produced beta and
    the cointegration p-value. Nothing about the trading window enters.
    """
    s = np.asarray(spread_is, dtype=float)
    s = s[~np.isnan(s)]
    return float(np.mean(s)), float(np.std(s, ddof=1))


def zscore_frozen(spread: pd.Series, mu: float, sigma: float) -> pd.Series:
    """z_t = (s_t - mu) / sigma with mu, sigma FROZEN from the in-sample window.

    WHY THIS IS THE DEFAULT, AND ROLLING IS NOT
    -------------------------------------------
    A rolling z-score looks safer -- it adapts -- but on a strongly
    autocorrelated spread it is badly biased, and the bias inflates the signal
    count rather than damping it.

    For a stationary AR(1) with parameter phi, the variance of a W-point rolling
    mean is Var(s) / W_eff, with the EXACT effective sample size

        W_eff = W^2 / ( W + 2 * sum_{k=1}^{W-1} (W-k) phi^k )

    (the familiar asymptotic form W(1-phi)/(1+phi) is valid only for
    W >> 1/(1-phi) and is off by a factor ~2 in our range -- it gives 0.69 where
    the exact value is 1.51). Effective sample sizes, simulation-verified:

        half-life    phi      W_eff(W=60)    W_eff(in-sample T=750)
             5 d   0.8706         4.72                52.4
            15 d   0.9548         2.09                17.8
            30 d   0.9772         1.51                 9.2

    A 60-day rolling window carries 1.5 effective observations at a 30-day
    half-life: the rolling mean is essentially the spread itself. Measured on
    simulated OU paths with W = 60:

        half-life   num. ratio   den. ratio    net     share |z|>2
             5 d       0.98         0.88       1.12    6.9 % (vs 4.6 %)
            20 d       0.84         0.64       1.31   10.7 % (vs 4.6 %)
            30 d       0.76         0.56       1.36   11.7 % (vs 4.6 %)

    The numerator shrinks because the rolling mean tracks the spread. But the
    rolling standard deviation shrinks MORE, because it measures the LOCAL
    dispersion of a highly autocorrelated series rather than its unconditional
    dispersion. The quotient therefore exceeds 1: the z-score is INFLATED, and
    the strategy fires roughly 2.5x too many entries at a 30-day half-life.

    Freezing (mu, sigma) on the in-sample window raises the effective sample size
    from 1.5 to 9.2 at a 30-day half-life and roughly halves the distortion
    (share of |z|>2 days at h=30: 11.4 % rolling, 8.0 % frozen, 4.6 % true). It
    does NOT eliminate it, because the in-sample window itself carries only ~9
    effective observations at that half-life.

    Two consequences to carry into Phase 6. First, an entry threshold of |z| > 2
    is not a reliable prior on trade frequency -- realised turnover must be
    measured, not assumed. Second, the half-life cap does triple duty: it bounds
    turnover, it keeps random walks out (see `passes_quality`), and it keeps the
    in-sample effective sample size large enough for sigma to be estimable at
    all. On that last count h <= 20 (W_eff = 13.5) is comfortable and h = 30
    (W_eff = 9.2) is the edge; the default cap of 30 is therefore a documented
    compromise, not a comfortable margin.

    What we give up: the frozen z-score cannot absorb a shift in the spread's
    level. That is deliberate. Silently absorbing a level shift is precisely how
    a broken pair keeps generating signals; detecting it is the job of the
    regime machinery in Phase 4, not of the normalisation.
    """
    if sigma <= 0 or not np.isfinite(sigma):
        return pd.Series(np.nan, index=spread.index)
    return (spread - mu) / sigma


# --------------------------------------------------------------------------- #
# Quality filter
# --------------------------------------------------------------------------- #
def passes_quality(res: PairResult, hl_min: float = 2.0, hl_max: float = 30.0,
                   min_obs: int = 500, beta_min: float = 1/3,
                   beta_max: float = 3.0) -> bool:
    """Tradability filter, applied AFTER the statistical test, in-sample only.

    Two bounds, both economic rather than statistical:

    - half-life < hl_min: the spread reverts within a couple of days. The signal
      is dominated by microstructure noise and bid-ask bounce, and the turnover
      it implies makes transaction costs eat the edge. Phase 6 quantifies this.
    - half-life > hl_max: the spread reverts too slowly to be traded. Capital is
      tied up for months, and the longer the horizon the likelier the relation
      breaks before reverting -- which is the whole subject of Phase 4.

    Why hl_max = 30 and not 60
    --------------------------
    Two independent arguments converge on 30 days.

    Economic: the out-of-sample trading window is one year (~250 sessions). A
    60-day half-life allows roughly 4 reversion cycles per year -- too few round
    trips to say anything statistical about the pair, and a long exposure to the
    relation breaking before it reverts.

    Statistical: because of the downward bias documented in `half_life`, a loose
    upper bound admits random walks. Measured on 4000 random walks of 750
    observations, the share passing the filter is 18.4 % at hl_max=60 but only
    2.7 % at hl_max=30. Tightening the bound is the cheapest available defence
    against a spread that does not actually revert.

    Why a hedge-ratio band, and why it is symmetric in 1/beta
    --------------------------------------------------------
    A beta outside a sane band is not a hedge, it is a directional bet on one
    leg with the other leg as decoration.

    - beta <= 0 means going long BOTH legs. The position carries full sector
      exposure and market neutrality -- the entire premise -- is gone. On the
      2016-2018 screen, all five negative-beta pairs involved PCG (collapsing on
      the wildfire liabilities) or SCG (the nuclear-project scandal): a negative
      beta "fits" two names trending in opposite directions, with no economic
      content whatsoever.
    - beta near 0 (or symmetrically, near infinity) means the regression is
      degenerate. This bites when the two legs have very different volatility:
      OLS shrinks beta toward zero and the residual becomes little more than a
      rescaled copy of the dependent variable. On the same screen, NRG -- a
      merchant generator, far more volatile than the regulated names -- appeared
      in 13 of 41 significant pairs (32 %, against 3.2 expected under uniform
      spread), every one of them with beta between 0.13 and 0.30. Those were not
      equilibrium relations; they were near-degenerate regressions.

    The band is symmetric under leg inversion (1/3 and 3), so the filter does
    not depend on the arbitrary alphabetical ordering convention.
    """
    if not np.isfinite(res.half_life) or not np.isfinite(res.beta):
        return False
    return (hl_min <= res.half_life <= hl_max
            and beta_min <= res.beta <= beta_max
            and res.n_obs >= min_obs)


# --------------------------------------------------------------------------- #
# Screener
# --------------------------------------------------------------------------- #
def screen_pairs(log_prices: pd.DataFrame, tickers: list[str],
                 maxlag: int = 1, min_obs: int = 500) -> pd.DataFrame:
    """Run Engle-Granger on every pair of `tickers` over `log_prices`.

    IN-SAMPLE ONLY. The caller is responsible for passing a window that ends
    before the trading period. Nothing in this function knows about time, so
    nothing here protects against handing it out-of-sample data -- that
    discipline lives in the walk-forward loop (Phase 3).

    Pair ordering convention: alphabetical, i.e. `a < b`, with `a` as the
    dependent variable. This is arbitrary but FIXED. Testing both directions and
    keeping the better p-value would double the number of tests and bias the
    selection toward whichever direction happened to look better -- a
    multiple-testing problem disguised as a modelling choice.
    """
    import itertools
    rows = []
    for a, b in itertools.combinations(sorted(tickers), 2):
        if a not in log_prices.columns or b not in log_prices.columns:
            continue
        sub = log_prices[[a, b]].dropna()
        if len(sub) < min_obs:
            continue
        res = engle_granger(sub[a].values, sub[b].values, a, b, maxlag=maxlag)
        rows.append(res.as_dict())
    return pd.DataFrame(rows)


def concentration_report(selected: pd.DataFrame) -> pd.DataFrame:
    """How often each name appears in a selection. A diagnostic, not a filter.

    WHY THIS MATTERS
    ----------------
    A selection of 24 pairs in which one name appears 12 times is not 24 bets:
    it is one bet on that name reverting to the sector, expressed twelve ways.
    Position sizing that treats the pairs as independent would silently
    concentrate risk.

    Measured across the nine walk-forward folds, the dominant name holds between
    16 % and 50 % of the selected pairs in EVERY fold -- and it is a different
    name each time (D, SRE, AEP, AEE, ETR, AWK). So this is not one odd stock,
    it is a property of the method: any name whose idiosyncratic path happens to
    look mean-reverting against the sector in-sample becomes a hub.

    The deeper consequence, which Phase 3 must handle: the pair tests are NOT
    independent. All names share the sector factor, so a hub name induces strong
    positive dependence across its own pairs. The effective number of
    independent tests is far below N(N-1)/2, which is exactly the regime where
    the choice between Bonferroni (valid under any dependence) and
    Benjamini-Hochberg (valid under independence or positive regression
    dependence) stops being cosmetic.
    """
    if selected.empty:
        return pd.DataFrame(columns=["ticker", "n_pairs", "share"])
    counts = pd.concat([selected["a"], selected["b"]]).value_counts()
    return pd.DataFrame({
        "ticker": counts.index,
        "n_pairs": counts.values,
        "share": counts.values / len(selected),
    }).reset_index(drop=True)


def excess_over_chance(scan: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Discoveries against what pure chance would produce -- Phase 3's opening move.

    Under H0 (no pair truly cointegrated) a test of level alpha still declares
    alpha * N pairs significant. Comparing the observed count to alpha * N is
    the cheapest possible sanity check, and on this universe it is sobering:
    across the nine folds the excess ranges from +1 (2015-2017: 20 found, 19
    expected) and 0 (2022-2024: 23 found, 23 expected) up to +75 (2019-2021,
    a window dominated by the COVID shock, where a common shock makes
    everything co-move and manufactures spurious cointegration).

    A fold with zero excess means the screener found nothing but noise. Trading
    its selection is trading noise.
    """
    n = len(scan)
    found = int((scan["pvalue"] < alpha).sum())
    expected = alpha * n
    return dict(n_tests=n, found=found, expected_under_null=expected,
                excess=found - expected,
                ratio=found / expected if expected else np.nan)
