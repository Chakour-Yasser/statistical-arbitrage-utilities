# Statistical arbitrage — honest selection under multiple testing & regime awareness

*French version: [README.md](README.md).*

A pairs-trading backtest on S&P 500 utilities, built around one question: **where, why and when
the strategy degrades** — not around a P&L.

Three strands:
- **V1 — Regime**: cointegration is not stable; detect breaks and exit.
- **V2 — Honest selection**: screening hundreds of pairs is a multiple-testing problem; correct
  for it (Bonferroni / Benjamini-Hochberg) and strictly separate *in-sample* selection from
  *out-of-sample* trading.
- **V3 — Baskets** (extension): multivariate generalisation via Johansen.

## Status

| Phase | Content | Status |
|---|---|---|
| 0 | Setup, seed, structure | done |
| 1 | Data & point-in-time universe | **done** |
| 2 | Cointegration foundations (EG, ADF, half-life, z-score) | **done** |
| 3 | V2 — selection under multiple testing | upcoming |
| 4 | V1 — regime awareness | upcoming |
| 6 | Backtest engine (costs, t+1 execution) | upcoming |
| 7 | Evaluation & attribution | upcoming |
| 8 | Write-up & defence | upcoming |

## Phase 1 — what is established

**Universe frozen before any backtest** ([docs/01_universe_decision.en.md](docs/01_universe_decision.en.md)):
S&P 500 utilities (GICS 55), 2014-01 → 2026-06, total-return prices. The document is dated and not
revisable after the fact — changing universe after seeing a P&L would be sector-level data
snooping.

**Point-in-time membership reconstructed** from Wikipedia's revision history: at each selection
date we read the page *as it existed*. 25 semi-annual snapshots, 2-day median lag. This yields
membership **and** the GICS sector as observed at the time, including for companies that have since
disappeared.

Two findings this construction surfaces and which a "current constituents" universe would hide
entirely:

1. **PCG (PG&E) is out of the index from 2019-07 to 2022-07** — Chapter 11 after the Camp Fire. A
   naive universe would trade it through the bankruptcy, knowing it came out alive.
2. **Four tickers are recycled.** `NU` denoted Northeast Utilities until 2015; since December 2021
   it denotes Nu Holdings, a Brazilian neobank. Same for `POM`, `TE`, `TEG`. A naive per-ticker
   download injects an unrelated company's prices into the sector universe — **silently**. The
   identity check (`validate_identity`) is therefore blocking: price history must overlap the
   membership window.

**Residual survivorship quantified** ([reports/survivorship.en.md](reports/survivorship.en.md)):
33 tradable tickers out of 38 in union; 3.3 % of membership slots lost, concentrated in 2014-2016
and **zero from 2017**. All five lost names are exits **by acquisition** — the event that breaks
cointegration most brutally. Their exclusion therefore **overstates** performance: the bias runs in
the favourable direction, the most dangerous case.

## Phase 2 — what is established

Building blocks in [src/cointegration.py](src/cointegration.py), decisions recorded in
[docs/02_cointegration_decisions.en.md](docs/02_cointegration_decisions.en.md), narrative in
[notebooks/02_cointegration_foundations_EN.ipynb](notebooks/02_cointegration_foundations_EN.ipynb).
37 unit tests.

**A correction to Phase 1.** Phase 1 tested cointegration with `adfuller` on the OLS residual.
That uses the wrong critical values: the residual is *estimated*, so its distribution is shifted.
Measured under H0 (independent random walks, n=750): `adfuller` rejects **14.6 %** of the time at
the 5 % level, against 4.2 % for `coint` with Engle-Granger critical values — a **3.5x inflation**
of false positives before multiple testing enters. On the 2016-2018 screen this takes the count of
"significant" pairs from 112 down to **47**. Phase 2 uses `coint` everywhere; the Phase 1
notebooks are left as written, as a record of what was done.

**Two findings that set up Phase 3.**

1. **Discoveries barely exceed chance, and in two folds not at all.** Across the nine walk-forward
   folds the excess over what a 5 % test produces under H0 ranges from **+75** (2019-2021 and
   2020-2022, both dominated by the COVID shock, where a common shock manufactures spurious
   cointegration) down to **+1** (2015-2017) and **0** (2022-2024). Two folds find nothing but
   noise.
2. **The tests are not independent and the selection is concentrated.** In every fold one name
   holds 16-50 % of the selected pairs — and it is a different name each time (D, SRE, AEP, AEE,
   ETR, AWK). Not one odd stock: a property of the method. The effective number of independent
   tests is far below N(N-1)/2, which is exactly where the Bonferroni / Benjamini-Hochberg choice
   stops being cosmetic.

**Two quality filters, both calibrated on evidence.** Half-life in [2, 30] days — the upper bound
is 30 rather than 60 because the half-life estimator is biased downward near a unit root, so a
pure random walk returns a *finite, large* value: 18.4 % of random walks pass at 60, only 2.7 % at
30. Hedge ratio in [1/3, 3] — outside that band it is not a hedge but a directional bet on one
leg. On the 2016-2018 screen this removed a cluster of 13 pairs all involving **NRG**, a merchant
generator whose much higher volatility shrinks beta toward zero and makes the regression
degenerate.

## Documents

- **[reports/Phase1_decisions_explained.pdf](reports/Phase1_decisions_explained.pdf)** (21 pp.) —
  every Phase 1 decision explained in detail, plus a full **mathematical foundations** section:
  stationarity and integration order, the Granger representation theorem, spurious regression,
  and short proofs that dividend drift destroys stationarity, that forward-fill preserves the
  autocovariance of increments, that survivorship bias is signed positive, and that
  E[false discoveries] = alpha*N under arbitrary dependence.
- **[reports/Phase2_cointegration_explained.pdf](reports/Phase2_cointegration_explained.pdf)**
  (12 pp.) — the Phase 2 estimators with their mathematics: OLS asymmetry and TLS symmetry
  (proved), the AR(1)/Ornstein-Uhlenbeck half-life (derived, including the Ito proof of the OU
  solution), the Dickey-Fuller limiting distribution, Phillips-Ouliaris on estimated residuals,
  Kendall's small-sample bias and the phantom half-life h ~ 0.173*T, and the effective sample
  size of a rolling mean.
- **[notebooks/01_phase1_universe_and_data_EN.ipynb](notebooks/01_phase1_universe_and_data_EN.ipynb)** —
  the executed narrative with charts, including the two times the initial intuition did not survive
  testing (forward-fill, raw prices).

## Two results obtained by testing our own assumptions

1. **Forward-fill does not manufacture mean reversion.** The standard argument is false: on a pure
   random walk the ADF rejection rate is unchanged (6.8 % → 6.5 %). The real defect is
   tradability — ~3.8 % of signals land on a non-trading day.
2. **Raw prices change pair selection.** Across 465 pairs on 2016-2018 the two series disagree on
   11 %. Most of it is p-value noise near the threshold, but the asymmetry of decisive flips —
   **16 pairs manufactured against 1 destroyed** — is robust.

## Methodological guardrails

- Strictly temporal split; selection frozen after the in-sample window.
- No forward-fill of prices: not because it would manufacture mean reversion (claim tested and
  refuted, see notebook §5), but because it generates signals on non-trading days — ~3.8 % of
  entries at a price that never existed.
- Liquidity filter recomputed fold by fold on the in-sample window only.
- Total-return prices: utilities' dividends (3-4 %/yr, widely dispersed) would otherwise inject a
  deterministic drift into the spread.
- Transaction costs included from the first backtest; execution lagged to t+1.
- Fixed seed (`src/config.py`), disk caches for reproducibility.

## Structure

```
src/config.py      frozen constants (universe, period, seed)
src/universe.py    point-in-time membership + ticker identity check
src/data.py        download, non-destructive cleaning, coverage diagnostics
docs/              dated methodological decisions (FR + EN)
reports/           survivorship quantification, identity validation, PDFs (FR + EN)
notebooks/         executed narrative (FR + EN)
scripts/           document and notebook generators
tests/             unit tests
```

## Installation

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.universe        # builds the point-in-time universe
.venv/bin/python -m pytest tests/ -q
```
