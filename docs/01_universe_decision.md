# Phase 1 — Universe decision (FROZEN 2026-08-19, before any backtest)

> This document is written **before** the first line of backtest code and is not revisable in
> light of results. Any later revision would be sector-level data snooping and must be flagged
> as such. French version: `docs/01_universe_decision.md`.

## 1. Universe: S&P 500 Utilities (GICS 55)

**Decision.** Stocks in the GICS *Utilities* sector that belonged to the S&P 500 at any point
between 2014-01-01 and 2026-06-30. Roughly 31 current names, ~35-40 names in union over the
period.

**Why intra-sector.** Cointegration is not a free statistical regularity: it requires two prices
to share the *same* stochastic trend, so that a linear combination is I(0). An economic prior is
therefore necessary. Within a sector it is identifiable: the same exposure to long rates, the same
regulatory regime (allowed return on an asset base), the same input costs. A secondary and
non-negligible benefit: it reduces the number of tests from N(N-1)/2 over ~500 names (124,750) to
~465, hugely lightening the multiple-testing correction load (Phase 3).

**Why utilities specifically.**
- The clearest common factor in the entire US equity market (duration + regulation) → a non-zero
  expected fraction of genuinely cointegrated pairs, which makes the "naive vs corrected
  selection" experiment informative rather than degenerate.
- N = 31 → 465 pairs → under H0 at α = 5 %, **~23 spurious discoveries expected**. A concrete
  number, comparable against the count of pairs actually found significant.
- Three breaks that are datable **and economically explainable**:
  1. COVID (March 2020) — liquidity shock, correlations → 1.
  2. Rate shock (2022) — dispersion of duration sensitivities.
  3. AI / data-centre repricing (2023-2024) — *merchant* generators (VST, CEG, NRG) structurally
     decouple from regulated names (ED, WEC, ...). This is THE cointegration break to tell in an
     interview.

**Anticipated objection.** "Your spread is just a disguised duration bet."
Answer: partly true by construction, and measurable — the hedge ratio absorbs the common rate
exposure; the residual must no longer be explained by rates. To be checked explicitly (Phase 7:
regress P&L on 10-year yield changes).

**Alternatives ruled out, and why.**
| Alternative | Reason for rejection |
|---|---|
| Regional banks (KRE) | Richer multiple testing (~4,000 pairs) and spectacular survivorship (SVB/SBNY/FRC to zero), but delisted prices are not obtainable for free → data plumbing would have eaten the methodology budget. |
| Energy E&P (XOP) | The 2023-24 M&A wave truncates many series; heterogeneous leverage makes cointegration fragile. |
| Semiconductors | Too trending; a real risk of finding no cointegrated pair at all, which would empty V2 of content. |

## 2. Period: 2014-01-01 → 2026-06-30

~12.5 years. Intended walk-forward split: **3 years in-sample (selection) / 1 year out-of-sample
(trading)**, rolling forward one year → **~9 folds**.

Accepted trade-off: going back before 2014 would have added folds but worsened survivorship bias
and tested against a sector structure (pre-renewables boom, pre-CEG spin-off) barely comparable to
today's.

## 3. Prices: total-return (adjusted for splits + dividends)

**Decision.** `auto_adjust=True` (yfinance), i.e. prices adjusted for splits *and* dividends.

**Why this is a decision and not a detail.** Utilities' dividend yield is 3-4 % per year and
**widely dispersed** across names. On raw prices, the yield difference between two stocks injects
a quasi-deterministic drift into the spread: the residual is no longer stationary and genuinely
cointegrated pairs would be rejected. On total-return prices that drift disappears. It is also the
right economic object: a long/short position's P&L collects dividends.

**Accepted residual leak.** Adjusted prices are *retro*-adjusted by future dividends: the 2018
series as downloaded in 2026 is not the one a trader observed in 2018. The effect is second-order
on a long/short spread (both legs adjusted in the same direction) but it exists and is named here.

## 4. Survivorship bias

**Correction applied.** S&P 500 membership reconstructed *point-in-time* from the page's revision
history. At each selection date the screener sees only names that were members **on that date**.
This eliminates the "candidate selection" component of the bias.

**Residual bias, uncorrected, and its direction.** Prices for delisted companies (bankruptcy,
acquisition) are not available for free. Names that disappear are predominantly distressed or
acquired — precisely the cases where a cointegration relation breaks violently (a takeover means
the price instantly decouples toward the offer price). Excluding them **overstates** the
strategy's performance. Magnitude quantified in `reports/survivorship.md`.

**Limit of the reconstruction.** The GICS sector listed on the public source is the *current*
sector; for names that left the index before today, the sector is reassigned manually (a short,
verifiable list versioned in `src/universe.py`).

## 5. Liquidity filter — causal by construction

A median dollar-volume threshold applied **fold by fold, on the in-sample window only**. Computing
this filter on the full sample would be a leak: it would select the names that *will remain*
liquid, which is future information correlated with survival and therefore with performance.

## 6. Single market, single currency

All names trade in the US, same close, same currency. This avoids the cross-market pairs trap
where asynchronous closing times manufacture illusory spread predictability.
