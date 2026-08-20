# Statistical arbitrage on S&P 500 utilities — an honest post-mortem

*French version of the Phase 1 material: [README.md](README.md). From Phase 2 onward everything is
in English.*

A walk-forward pairs-trading study built to answer one question: **where, why and when does the
strategy degrade?** Not *how much did it make* — it did not make anything, and the value of the
project is in establishing that credibly rather than in hiding it.

**Headline result.** Over ten non-overlapping out-of-sample years (2017-2026), the book has a
gross Sharpe of **−0.17** and a net Sharpe of **−0.26** after 5 bp of costs. Multiple-testing
correction does not rescue it; a regime-break detector does not rescue it; generalising to Johansen
baskets does not rescue it. Every one of those negative results is quantified below.

---

## What was actually found

| Claim | Evidence |
|---|---|
| **Ticker symbols are not identifiers.** | Four tickers in this universe denote different companies today than during their index membership. `NU` was Northeast Utilities until 2015; since Dec 2021 it is Nu Holdings, a Brazilian neobank. A naive download injects a fintech into a utilities book, silently. |
| **The textbook cointegration test is oversized.** | Running ADF on an OLS residual rejects **14.6 %** of the time under the null instead of 5 %. Correcting it takes the 2016-2018 screen from **112** significant pairs to **47**. |
| **Half the "discoveries" are noise.** | 47 significant pairs against **23** expected under H0. In 2 of 10 folds the excess over chance is zero. |
| **Bonferroni finds nothing.** | 0 pairs selected in **6 of 10** folds; Benjamini-Yekutieli in 8 of 10. |
| **The regime detector does not work.** | Flags fire on 18.6 % of days and capture 17.7 % of losses — i.e. nothing. Measured as a label rather than an exit, flagged days are *marginally better*, not worse. |
| **Baskets are not better than pairs.** | Johansen triplets: Sharpe **−0.22** against **−0.26** for pairs, for 10x the number of tests (2925-4495 triplets vs ~400 pairs) and a trace test whose own null rejection rate is **8.6 %**, not 5 %. |
| **Costs are not the culprit.** | Gross Sharpe is already −0.17. Costs subtract a further 0.09. There is no edge being eaten; there is no edge. |

---

## Phase 1 — A universe you can trust

Universe frozen before any backtest ([decision record](docs/01_universe_decision.en.md)): S&P 500
utilities (GICS 55), 2014-01 → 2026-06, total-return prices.

**Point-in-time membership** rebuilt from the revision history of the constituents page: at each
selection date the page is read *as it stood then*. 25 semi-annual snapshots, 2-day median lag.
Two things this surfaces that a "current constituents" universe hides completely:

- **PCG (PG&E) is out of the index from 2019-07 to 2022-07** — Chapter 11 after the Camp Fire. A
  naive universe trades it straight through the bankruptcy, knowing it came out alive.
- **Four recycled tickers** (`NU`, `POM`, `TE`, `TEG`). The identity check is blocking: price
  history must overlap the membership window.

**Residual survivorship quantified** ([report](reports/survivorship.en.md)): 33 tradable tickers of
38; 3.3 % of membership slots lost, all in 2014-2016 and zero from 2017. All five lost names are
exits **by acquisition** — the event that breaks cointegration most violently — so the bias
**overstates** performance.

## Phase 2 — Estimators that do what they claim

[Decision record](docs/02_cointegration_decisions.en.md) · [notebook](notebooks/02_cointegration_foundations_EN.ipynb) · [PDF with proofs](reports/Phase2_cointegration_explained.pdf)

- **Engle-Granger with MacKinnon critical values**, never ADF on the residual (see size table above).
- **Half-life via AR(1)**, with the Dickey-Fuller downward bias documented: a random walk returns a
  *finite* half-life growing linearly in the sample, h ≈ 0.173·T. So the statistic is a tradability
  filter, never a stationarity test.
- **Z-score frozen on the in-sample window.** A rolling window is causal but biased on an
  autocorrelated spread, and the bias *inflates* the signal: the numerator shrinks (0.76 at a
  30-day half-life) but the rolling standard deviation shrinks faster (0.56), firing ~2.5x too many
  entries.
- **Hedge-ratio band [1/3, 3]**, symmetric under leg inversion. This removed a cluster of 13 pairs
  all containing NRG, a merchant generator whose much higher volatility drives beta toward zero and
  makes the regression degenerate.

## Phase 3 — Honest selection

Walk-forward, 3 years in-sample / 1 year out-of-sample, 10 folds. Universe, liquidity filter,
cointegration test, hedge ratio and z-score calibration are all estimated in-sample and **frozen**.

| Method | Folds with 0 pairs | Pairs total | Net Sharpe | Max DD |
|---|---|---|---|---|
| Naive (p < 0.05) | 0/10 | 338 | −0.26 | −0.069 |
| Bonferroni | **6/10** | 7 | −0.44 | −0.061 |
| Benjamini-Hochberg | 6/10 | 29 | −0.04 | −0.045 |
| Benjamini-Yekutieli | 8/10 | 3 | −0.39 | −0.035 |

The correction does not turn a losing strategy into a winning one. What it does is make the absence
of signal **visible**: on 400 tests Bonferroni's threshold is 1.2×10⁻⁴ and nothing survives it in
most folds.

## Phase 4 — Regime detection, and why it fails here

Four causal break signals: trailing cointegration p-value, hedge-ratio drift, half-life explosion,
time since the spread last revisited its mean.

**First attempt was broken.** With a 250-day window the p-value flag fired on 68 % of days and the
disjunction on 95 %: the detector was not detecting, it was keeping the book flat. A 250-day
Engle-Granger test has almost no power, so p > 0.10 is the normal state even for a genuine pair.

**Second attempt is honest and negative.** With a 500-day window the flags fire on 18.6 % of days
and capture 17.7 % of the losses — proportional, therefore uninformative. Used as a label rather
than an exit, flagged days are *marginally better* than unflagged ones at every threshold tried.

The likely reason is structural, not a tuning failure: **regime detection can only protect a regime
in which the strategy works**, and the gross Sharpe here is approximately zero throughout.

## Phase 5 — Johansen baskets (extension)

Symmetric in the assets, recovers the cointegration rank, estimates all relations jointly. It also
multiplies the multiple-testing problem by an order of magnitude, and its trace test is itself
oversized: measured null rejection rate **8.6 %** at n = 3, T = 750.

Stitched out-of-sample Sharpe: **−0.22**, against −0.26 for pairs. Same answer, more machinery.

## Phase 7 — Attribution

- **Costs**: gross −0.17 → net −0.26 at 5 bp. Breakeven cost does not exist; the signal is already
  negative gross.
- **By year**: positive in 2019, 2021, 2022 and the 2026 stub; negative in 2017, 2018, 2020, 2023,
  2024, 2025. No trend, just noise around zero.
- **Duration bet?** Regressing daily P&L on 10-year yield changes gives **t = +4.03** for the naive
  book — statistically significant, so the standing objection has some merit, though R² is 0.7 % so
  it is economically small. With the regime overlay, t = +1.87 (not significant).

---

## The self-referential caveat

Across the configurations reported here (4 selection rules x with/without regime), **one** is
positive: Benjamini-Hochberg with the regime overlay, at +0.38.

That is exactly what this project is about. With eight configurations tried and a true edge of
zero, finding one at +0.38 is the expected outcome, not a discovery. Reporting it as *the* result
would be the precise error the whole study exists to expose. It is reported here as noise.

## Known limitations

- Prices for delisted names are unavailable, so the residual survivorship bias runs in the
  favourable direction (Phase 1).
- The pair tests are strongly dependent — one name holds 16-50 % of the selected pairs in every
  fold — so BH's independence/PRDS assumption is not verified. A permutation or block-bootstrap
  null would be the honest alternative and is not implemented.
- Transaction costs are a flat 5 bp per trade with no market-impact or borrow-cost model. Short
  borrow is assumed free and always available, which is generous.
- The half-life cap of 30 days is a documented compromise: at that half-life the in-sample window
  carries only ~9 effective observations for estimating sigma.
- Adjusted prices are retro-adjusted by future dividends, a second-order leak that is named but not
  removed.

## Structure

```
src/config.py         frozen constants (universe, period, seed)
src/universe.py       point-in-time membership + ticker identity check
src/data.py           download, non-destructive cleaning, coverage diagnostics
src/cointegration.py  Engle-Granger, hedge ratios, half-life, z-scores, quality filter
src/selection.py      Bonferroni / BH / BY, walk-forward folds, liquidity filter
src/regime.py         four causal break signals
src/backtest.py       t+1 execution, costs, metrics
src/analysis.py       cost sensitivity, decay, drawdown attribution, rate exposure
src/johansen.py       basket cointegration (extension)
src/pipeline.py       the walk-forward loop that keeps everything frozen
scripts/              runners and document generators
docs/ reports/ notebooks/    decision records, results, narratives
tests/                52 unit tests
```

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.universe          # rebuild the point-in-time universe
.venv/bin/python scripts/run_walkforward.py
.venv/bin/python scripts/run_analysis.py
.venv/bin/python -m pytest tests/ -q
```
