# Cross-sectional statistical arbitrage

A walk-forward study of mean reversion in equity and crypto cross-sections. It starts from
cointegration-based pairs trading, measures why that approach does not work, and replaces it with
PCA residual arbitrage. Every positive result is reported next to a null-hypothesis control run on
the same pipeline.

## Results

S&P 500, daily, PCA residual arbitrage:

| | 2015-2020 | 2021-2026 | Full sample |
|---|---|---|---|
| Gross Sharpe | 1.38 | 0.18 | 0.82 |
| Net Sharpe at 1 bp | 1.15 | -0.04 | 0.60 |
| Net Sharpe at 2 bp | 0.93 | -0.26 | 0.38 |
| Gross return p.a. | 4.46 % | 0.54 % | 2.59 % |
| Null control (gross Sharpe) | | 0.23 | -0.32 |

Turnover is 69 per year and the breakeven execution cost is 3.74 bp. Maximum gross drawdown is
6.3 %. The book holds about 150 simultaneous positions across 434 names. Regressing daily P&L on
10-year yield changes gives t = 1.30, so the returns are not a disguised duration bet.

The strategy works until 2020 and then stops. After 2021 its gross Sharpe of 0.18 sits below the
0.23 produced by a null that preserves the factor structure and destroys serial dependence.

Binance USDT perpetuals, daily, same method: gross Sharpe 0.86 against a null of -0.27 across five
seeds, breakeven cost 7.34 bp. Split by period, 2021-2023 gives 1.11 net at 4 bp and 2024-2026
gives -0.64. The decay pattern repeats about three years later than in equities.

## Data

Index membership is rebuilt point in time from the revision history of the constituents page, so
each selection date sees the page as it stood then. There are 25 semi-annual snapshots with a
median lag of two days. Two things this catches that a current-constituents universe does not:

PCG is out of the index from July 2019 to July 2022 after PG&E filed for Chapter 11. A naive
universe trades it through the bankruptcy knowing that it recovered.

Four tickers in the utilities universe denote different companies today than during their index
membership. NU was Northeast Utilities until 2015 and has been Nu Holdings, a Brazilian neobank,
since December 2021. `validate_identity` requires price history to overlap the membership window
and removes the name otherwise.

Residual survivorship is quantified in `reports/survivorship.md`. The names that drop out left the
index by acquisition, which is the event that breaks a statistical relation most violently, so the
remaining bias runs in the favourable direction.

For crypto the exchange reports listing dates directly, so point-in-time entry is exact rather than
reconstructed, and the `SETTLING` status recovers 126 delisted contracts. Four contracts are purged
from the API and cannot be recovered.

## Why pairs trading fails

Two implementation errors came first, since a negative result only means something if the
implementation is right.

Running ADF on an OLS residual rejects 14.6 % of the time under the null instead of 5 %, because
the residual was estimated by minimising its own variance. Using MacKinnon's Engle-Granger critical
values takes a 465-pair screen from 112 significant pairs down to 47.

The half-life estimator does not return NaN on a random walk. Kendall's small-sample bias produces
a finite value that grows linearly in the sample, roughly 0.173 T. It is a tradability filter and
not a stationarity test.

With those fixed, the diagnosis is straightforward. A pair cointegrated at the 5 % level over three
years has a 10.8 % chance of still being cointegrated the next year. Pairs the screen rejected at
p >= 0.20 come in at 22.3 %. In-sample cointegration has no predictive power for out-of-sample
cointegration on this universe, and the rejected pairs survive twice as often. The equilibrium
level also moves by a median of 1.47 in-sample standard deviations between estimation and trading,
so a frozen z-score starts the year around 1.5 and positions never close.

Net Sharpe comes out at -0.26. Multiple-testing correction does not fix it, since Bonferroni
selects zero pairs in 6 of 10 folds. Regime-break detection does not fix it either: the flags fire
on 18.6 % of days and capture 17.7 % of losses, which is proportional and therefore uninformative.
Johansen baskets give -0.22 for ten times as many tests.

## What replaces it

`src/statarb.py` follows Avellaneda and Lee (2010). Rolling PCA on the correlation matrix of the
full index, 15 eigenportfolios, each name hedged against them, an Ornstein-Uhlenbeck fit to the
cumulative residual, and an s-score band.

This removes the two measured failure modes. There is no pair selection, so no winner's curse.
Factors, betas and OU parameters are re-estimated on a rolling window, so no equilibrium level goes
stale. The book holds around 150 positions rather than the two or three independent bets a
concentrated pairs book really carries. A residual that dislocates without news is an order
imbalance, and taking the other side supplies liquidity, which is a mechanism rather than a
coincidence. The best year is 2020 at +11.9 % and Sharpe 1.99, which fits that reading.

The null control applies one random sign per date to every return series. That leaves the
cross-sectional covariance, the volatility clustering and the availability pattern intact and
removes the sign predictability that mean reversion depends on.

## Learned signal

The OU specification forces the expected return to be linear in the s-score. Binning 415,743
observations shows it is not: the response peaks near |s| = 1.5 at 3 to 4.5 bp with t-statistics
around 3, and flattens in the tails.

An MLP trained walk-forward on the tradable target, alpha plus epsilon, reaches a mean daily
information coefficient of +0.0081 (t = 5.0) against +0.0056 (t = 3.25) for the s-score. Gross
Sharpe rises from 0.86 to 0.96 and the null comes out at -0.70. Turnover rises from 66 to 191, so
breakeven cost falls from 3.51 bp to 1.31 bp and the net result is worse at every realistic cost
level.

Predicting the 5-day cumulative residual instead brings turnover back to 76 but drops gross Sharpe
to 0.25. Reversion is consumed within a day or two, so a longer horizon adds noise without adding
information. The s-score is hard to beat because it is a state variable rather than a forecast: it
is informative and persistent at the same time, whereas a forecast has to buy persistence with
predictive power.

## Limitations

Vendor prices for delisted names are unavailable, so residual survivorship runs in the favourable
direction.

Costs are a flat per-notional assumption with no market-impact term, so capacity is not modelled.
A 3.74 bp breakeven is tight for large-cap US equities and would tighten further with size.

Short borrow is assumed free and always available.

Performance is concentrated. Excluding 2020, the full-sample net Sharpe at 2 bp falls from 0.38 to
0.07.

The pair tests in the diagnosis are strongly dependent, with one name holding 16 to 50 % of the
selected pairs in every fold, so the independence assumption behind Benjamini-Hochberg is not
verified. A permutation null for the pair screen is not implemented.

The no-trade band was picked from four values on the full sample. It affects execution efficiency
rather than holdings: gross Sharpe is 0.80 without it and 0.82 with it.

## Layout

```
src/universe.py       point-in-time membership, ticker identity validation
src/data.py           download, cleaning, coverage diagnostics
src/cointegration.py  Engle-Granger, hedge ratios, half-life, z-scores, quality filters
src/selection.py      Bonferroni, Benjamini-Hochberg, Benjamini-Yekutieli, walk-forward folds
src/regime.py         four causal break signals
src/backtest.py       t+1 execution, costs, metrics
src/analysis.py       cost sensitivity, decay, drawdown attribution, rate exposure
src/johansen.py       basket cointegration
src/statarb.py        PCA residuals, OU s-scores, hedged book construction
src/crypto.py         Binance perpetuals data layer
src/mlsignal.py       learned expected residual return
src/pipeline.py       walk-forward loop
tests/                52 unit tests
```

Four tests exist specifically to fail if a leak is reintroduced:
`test_rolling_zscore_is_causal` rewrites the future and asserts past z-scores are unchanged,
`test_folds_never_overlap_and_selection_precedes_trading`,
`test_execution_is_lagged_by_one_day`, and
`test_liquidity_filter_ignores_everything_after_the_in_sample_window`.

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.universe
.venv/bin/python scripts/fetch_spx.py
.venv/bin/python scripts/run_statarb_final.py
.venv/bin/python -m pytest tests/ -q
```
