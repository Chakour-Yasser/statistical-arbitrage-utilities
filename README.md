# Cross-sectional statistical arbitrage on the S&P 500

A walk-forward study that starts as a cointegration pairs-trading project, measures why that
approach fails, and replaces it with the method that addresses the measured failure.

**Result.** PCA-residual statistical arbitrage on the full S&P 500 earns a **gross Sharpe of 1.38
(net 0.93 after 2 bp) over 2015-2020**, and **decays to nothing after 2021** — gross Sharpe 0.18
against 0.23 for a null that preserves the factor structure and destroys serial dependence.

Every positive number here is reported together with the null-hypothesis control that was run
before it.

---

## Headline numbers

| | 2015-2020 | 2021-2026 | Full sample |
|---|---|---|---|
| Gross Sharpe | **1.38** | 0.18 | 0.82 |
| Net Sharpe @ 1 bp | 1.15 | −0.04 | 0.60 |
| Net Sharpe @ 2 bp | **0.93** | −0.26 | 0.38 |
| Gross return p.a. | 4.46 % | 0.54 % | 2.59 % |
| Null control (gross Sharpe) | — | **0.23** | −0.32 |

Turnover 69/yr, breakeven execution cost **3.74 bp**, max drawdown −6.3 % gross, 434 names and 152
simultaneous positions per day. Regression of P&L on 10-year yield changes: t = 1.30, so the book
is genuinely rate-neutral rather than a disguised duration bet.

**Read the last column carefully.** After 2021 the strategy's gross Sharpe is *below* what pure
noise produces under the same machinery. That is not a weak period, it is the absence of signal.

---

## Part 1 — Data discipline

Nothing downstream is worth anything if the universe is wrong, and universe mistakes never raise
an error. Three checks, all in `src/universe.py`:

**Point-in-time membership.** Index membership is rebuilt from the revision history of the
constituents page: at each selection date the page is read *as it stood then*, which yields
membership and GICS sector as observed at the time. 25 semi-annual snapshots, 2-day median lag.
This catches, for example, that **PCG (PG&E) is out of the index from 2019-07 to 2022-07** after
its Chapter 11. A current-constituents universe trades it straight through the bankruptcy knowing
it came out alive.

**Ticker identity is checked, and the check blocks.** Symbols get recycled: `NU` was Northeast
Utilities until 2015; since December 2021 it is Nu Holdings, a Brazilian neobank. A naive download
injects a fintech into a US equity book with no error raised. The rule is that price history must
overlap the membership window; four tickers fail it in the utilities universe, 187 names across
the full index have no usable vendor history.

**Residual survivorship is quantified and signed** ([report](reports/survivorship.md)). The names
lost are exits by acquisition — the event that breaks a statistical relation most violently — so
the bias runs in the favourable direction, which is the dangerous kind.

## Part 2 — Why pairs trading fails, measured

[Decision record](docs/02_cointegration_decisions.md) · [notebook](notebooks/02_cointegration_foundations.ipynb)

Two implementation errors were found and fixed first, because a negative result is only worth
reporting if the implementation is correct:

- **The standard cointegration test is oversized.** ADF on an OLS residual rejects **14.6 %** of
  the time under the null instead of 5 %, since the residual was estimated by minimising its own
  variance. Using MacKinnon's Engle-Granger values takes a 465-pair screen from 112 "significant"
  pairs to 47.
- **The half-life estimator does not return NaN on a random walk.** Kendall's small-sample bias
  gives a finite phantom half-life growing linearly in the sample, h ≈ 0.173·T. It is a
  tradability filter, never a stationarity test.

With those fixed, the diagnosis is unambiguous:

| | Still cointegrated the following year |
|---|---|
| Pairs **selected** (p < 0.05 in-sample) | **10.8 %** |
| Pairs **rejected** (p ≥ 0.20 in-sample) | **22.3 %** |

In-sample cointegration has no predictive power for out-of-sample cointegration — the rejected
pairs survive twice as often. This is the winner's curse: picking the lowest p-value out of 400
picks the residual that was luckiest, and luck does not repeat. On top of that the equilibrium
level itself moves by a median of **1.47 in-sample sigmas** between estimation and trading, so a
frozen z-score starts the year already at |z| ≈ 1.5 and positions never close.

Result: net Sharpe −0.26. Multiple-testing correction does not fix it (Bonferroni selects zero
pairs in 6 of 10 folds), a regime-break detector does not fix it (flags fire on 18.6 % of days and
capture 17.7 % of losses — proportional, hence uninformative), and Johansen baskets do not fix it
(−0.22 for ten times the number of tests).

## Part 3 — What replaces it

[`src/statarb.py`](src/statarb.py), following Avellaneda and Lee (2010). Rolling PCA on the
correlation matrix of the full index, 15 eigenportfolios, each name hedged against them, an
Ornstein-Uhlenbeck fit to the cumulative residual, and an s-score band.

The replacement is chosen to remove the two measured failure modes:

| Failure mode in pairs | How this removes it |
|---|---|
| Winner's curse from selecting the best pair out of 400 | No selection at all — every name is traded on its own residual |
| Equilibrium level goes stale | Factors, betas and OU parameters re-estimated on a rolling window every day |
| 2-3 genuinely independent bets | ~150 simultaneous positions across ~434 names |
| No economic mechanism forcing reversion | A residual dislocating without news is an order-imbalance event; taking the other side supplies liquidity, and being paid for that is a mechanism rather than a coincidence |

Consistent with that mechanism, the single best year is **2020 (+11.9 %, Sharpe 1.99)** — residual
dislocations pay when liquidity withdraws.

**The null control.** One common random time-permutation is applied to every return series. That
preserves the cross-sectional factor structure exactly and destroys all serial dependence, so it
destroys residual reversion and nothing else. Full sample: **−0.32** against +0.82 for the real
data. Post-2021: **+0.23** against +0.18 — the signal is gone.

**One tuned parameter, disclosed.** The no-trade band was picked from four values tested on the
full sample. It is a cost-control device, not a signal parameter: gross Sharpe is 0.80 without it
and 0.82 with it, so it changes execution efficiency (turnover 109 → 69) and not what the model
wants to hold.

---

## Limitations

- Vendor prices for delisted names are unavailable, so residual survivorship runs favourably.
- Costs are a flat per-notional assumption with no market-impact term, so capacity is not modelled.
  The 3.74 bp breakeven is tight for large-cap US equities and would tighten further with size.
- Short borrow is assumed free and always available.
- Performance is concentrated: excluding 2020, the full-sample net Sharpe at 2 bp falls from 0.38
  to 0.07.
- The pair tests in Part 2 are strongly dependent — one name holds 16-50 % of selected pairs in
  every fold — so Benjamini-Hochberg's PRDS assumption is unverified. A permutation null for the
  pair screen is not implemented.

## Structure

```
src/universe.py       point-in-time membership, ticker identity validation
src/data.py           download, non-destructive cleaning, coverage diagnostics
src/cointegration.py  Engle-Granger, hedge ratios, half-life, z-scores, quality filters
src/selection.py      Bonferroni / BH / BY, walk-forward folds, causal liquidity filter
src/regime.py         four causal break signals
src/backtest.py       t+1 execution, costs, metrics
src/analysis.py       cost sensitivity, decay, drawdown attribution, rate exposure
src/johansen.py       basket cointegration
src/statarb.py        PCA factor residuals, OU s-scores, hedged book construction
src/pipeline.py       the walk-forward loop that keeps estimation frozen in-sample
tests/                52 unit tests
```

The tests that matter are the ones that fail if a leak is reintroduced:
`test_rolling_zscore_is_causal` rewrites the future and asserts past z-scores are bit-identical;
`test_folds_never_overlap_and_selection_precedes_trading`; `test_execution_is_lagged_by_one_day`;
`test_liquidity_filter_ignores_everything_after_the_in_sample_window`.

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.universe              # point-in-time universe (cached revisions)
.venv/bin/python scripts/fetch_spx.py         # prices
.venv/bin/python scripts/run_statarb_final.py # the main result, with its null control
.venv/bin/python -m pytest tests/ -q
```
