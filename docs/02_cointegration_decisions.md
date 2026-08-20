# Cointegration building blocks: decision record

Companion to `src/cointegration.py` and `tests/test_cointegration.py`.

---

## 0. A correction to the earlier pair screen

The first version of the screen tested cointegration by regressing A on B and
running `adfuller` on the residual. That is wrong, and the error is large.

The ADF test assumes it is applied to an *observed* series. Here the residual is
**estimated**: OLS chose β precisely to minimise its variance, so the residual is
"as stationary as possible" by construction. The test statistic's distribution
shifts and ADF critical values become far too permissive. The correct reference
distribution is MacKinnon's Engle-Granger one, which accounts for the estimated
first stage and depends on the number of regressors.

Size measured under H₀ (independent random walks, n = 750, 2000 simulations):

| Method | Rejection rate at the 5 % level |
|---|---|
| `adfuller` on the OLS residual | **14.6 %** |
| `coint` (Engle-Granger critical values) | 4.2 % |

A **3.5× inflation** of the false-positive rate, before multiple testing enters.

Effect on the Phase 1 screen (2016-2018, 465 pairs): 112 "significant" pairs
with `adfuller`, **47** with `coint`. 65 discoveries evaporate on fixing the
test. The expected-false-positive figure quoted in Phase 1 (~23) was right for
`coint` but the screen it was compared against had been run with `adfuller`,
which made the comparison meaningless.

Every Phase 2 function uses `coint`. The Phase 1 notebooks are left as written
(they are a record of what was done) but the README flags the correction.

---

## 1. Hedge ratio: OLS for testing, TLS as a robustness check

Decision. Estimate β by OLS with a fixed alphabetical ordering convention
(`a < b`, `a` the dependent variable). Expose a total-least-squares estimator to
verify conclusions do not hinge on the regression direction.

Why the convention matters. OLS minimises *vertical* errors, so it assigns
all noise to the left-hand variable. Consequently

    β(A|B) · β(B|A) = R²

(verified as a unit test), so the two directions disagree by exactly the factor
R². The noisier the relation, the larger the disagreement.

The three options, and why the middle one is forbidden. | Option | Verdict |
|---|---|
| Fixed convention (alphabetical) | **Chosen.** Arbitrary but frozen, and therefore not a researcher degree of freedom. |
| Test both directions, keep the better p-value | **Forbidden.** Doubles the number of tests and biases selection toward whichever direction happened to look better, a multiple-testing problem disguised as a modelling choice. |
| Total least squares | Symmetric by construction (β(A\|B) = 1/β(B\|A) exactly), which is the honest thing when neither leg has privileged status. But it assumes both legs carry comparable noise, and the Engle-Granger machinery is built on an OLS first stage, so the p-value would no longer be valid. Kept as a robustness check. |

Lag selection. `maxlag=1`, fixed, rather than AIC-selected. AIC is standard
but makes the lag a data-dependent choice, adding one more degree of freedom to
a screener already running hundreds of tests. A fixed lag is defensible
*because* it is not chosen.

---

## 2. Half-life: derivation, and a bias that matters

Derivation. Regress the change on the level:

    Δs_t = a + b·s_{t-1} + ε      ⟺      s_t = a + (1+b)·s_{t-1} + ε

Write φ = 1 + b. A shock decays as φᵏ, so the half-life solves φʰ = ½:

    half_life = -ln 2 / ln φ

Link to Ornstein-Uhlenbeck: ds = θ(μ - s)dt + σ dW discretises to φ = e^(-θΔt),
hence half_life = ln 2 / θ. The same object.

The bias, this is the part that is not in the textbooks' summary. Near a
unit root the OLS estimate of φ is biased *downward* (Dickey-Fuller bias). A
pure random walk therefore does **not** return NaN; it returns a finite, large
half-life. Measured on 4000 random walks of 750 observations:

| Quantity | Value |
|---|---|
| NaN returned | 4 % |
| Median estimated half-life | 111 days |
| Share passing hl ≤ 60 | 18.4 % |
| Share passing hl ≤ 30 | 2.7 % |

A diagnostic falls out of this. On a genuine random walk the estimated
half-life *grows with the estimation window*, 76 d at n=500, 111 d at n=750,
222 d at n=1500, because there is no true half-life to recover; the estimate
merely tracks the window length. A half-life unstable across window lengths is a
red flag.

Consequence. The half-life is a **tradability filter applied after** the
cointegration test has done the statistical work. It must never be used to
decide whether a spread mean-reverts.

---

## 3. Causal z-score

Decision. Rolling window of 60 days, using observations up to **and
including** t.

Why not full-sample. `z_t = (s_t - mean(s)) / std(s)` over the whole period
injects the future into every observation: the strategy would "know" the
spread's eventual average level, which is precisely what it is supposed to be
betting on. It is the single most common leak in pairs trading.

Why t is included. t is the close on which the signal is computed. Execution
is lagged to t+1 in the backtest (Phase 6). The leak would be to execute at t,
not to observe t.

Why 60 days. The window must be several times the half-life. Otherwise the
rolling mean tracks the spread itself, the z-score is systematically pulled
toward zero, and the estimator eats its own signal. Sixty days suits the 2-30
day half-lives the quality filter admits.

Guardrail. `test_rolling_zscore_is_causal` rewrites the series *after* index
k and asserts the z-scores up to k are bit-identical. If anyone ever swaps in a
full-sample or centred window, the test fails immediately. A companion test
keeps the naive version around as an executable counter-example.

---

## 4. Quality filter

Applied **after** the statistical test, **in-sample only**.

### 4.1 Half-life band: 2 ≤ hl ≤ 30 days

- hl < 2. dominated by microstructure noise and bid-ask bounce; the implied
  turnover lets transaction costs eat the edge (Phase 6 quantifies it).
- hl > 30. two independent arguments converge. *Economic*, the
  out-of-sample window is one year (~250 sessions), so a 60-day half-life allows
  only ~4 reversion cycles, too few round trips to say anything statistical, and
  a long exposure to the relation breaking before it reverts. *Statistical* , 
  because of the bias above, a loose bound admits random walks: 18.4 % pass at
  hl_max = 60 against 2.7 % at hl_max = 30.

### 4.2 Hedge-ratio band: 1/3 ≤ β ≤ 3

A β outside a sane band is not a hedge, it is a directional bet on one leg with
the other as decoration.

- **β ≤ 0** means going long *both* legs: full sector exposure, market
  neutrality gone. On the 2016-2018 screen all five negative-β pairs involved
  PCG (collapsing on wildfire liabilities) or SCG (nuclear-project scandal). A
  negative β "fits" two names trending in opposite directions with no economic
  content.
- **β near 0 or near infinity** means a degenerate regression. This bites when
  the legs have very different volatility: OLS shrinks β toward zero and the
  residual becomes a rescaled copy of the dependent variable. On the same
  screen, **NRG**, a merchant generator, far more volatile than the regulated
  names, appeared in **13 of 41** significant pairs (32 %, against 3.2 expected
  under a uniform spread), every one with β between 0.13 and 0.30. Those were
  near-degenerate regressions, not equilibrium relations.

The band is symmetric under leg inversion (1/3 and 3), so the filter does not
depend on the alphabetical convention. Enforced by a unit test.

---

## 5. Two findings that set up Phase 3

### 5.1 Discoveries barely exceed chance, and in two folds, not at all

Under H₀, a test of level α still declares α·N pairs significant. Across the
nine walk-forward folds:

| In-sample window | Pairs tested | Significant | Expected under H₀ | Excess |
|---|---|---|---|---|
| 2014-2016 | 351 | 51 | 18 | +33 |
| 2015-2017 | 378 | 20 | 19 | **+1** |
| 2016-2018 | 406 | 41 | 20 | +21 |
| 2017-2019 | 378 | 28 | 19 | +9 |
| 2018-2020 | 378 | 39 | 19 | +20 |
| 2019-2021 | 378 | 94 | 19 | +75 |
| 2020-2022 | 406 | 95 | 20 | +75 |
| 2021-2023 | 406 | 25 | 20 | +5 |
| 2022-2024 | 465 | 23 | 23 | **0** |

Two folds (2015-2017, 2022-2024) find **nothing beyond pure chance**. Trading
their selection is trading noise. The 2019-2021 and 2020-2022 spikes sit
squarely on the COVID shock, where a common shock makes everything co-move and
manufactures spurious cointegration, a caution, not a discovery.

### 5.2 The tests are not independent, and the selection is concentrated

In **every** fold, one name holds 16 % to 50 % of the selected pairs, and it is
a different name each time (D, SRE, AEP, AEE, ETR, AWK). This is not one odd
stock; it is a property of the method. Any name whose idiosyncratic path happens
to look mean-reverting against the sector in-sample becomes a hub.

Two consequences:

1. **Risk.** A selection of 24 pairs with one name in 12 of them is one bet
   expressed twelve ways. Sizing that treats the pairs as independent silently
   concentrates risk. Phase 6 must handle this.
2. **Statistics.** The N(N-1)/2 tests are far from independent: all names share
   the sector factor, and a hub induces strong positive dependence across its own
   pairs. The effective number of independent tests is far below 406. This is
   exactly the regime where the choice between Bonferroni (valid under *any*
   dependence) and Benjamini-Hochberg (valid under independence or positive
   regression dependence, PRDS) stops being cosmetic, and where a permutation
   or block-bootstrap null becomes the honest alternative.

---

## 6. What Phase 2 deliberately does not do

- **No selection.** `screen_pairs` runs the tests; it does not decide what to
  trade. Selection, multiple-testing correction and the walk-forward loop are
  Phase 3.
- **No time awareness.** Nothing in this module knows about dates, so nothing
  here prevents it being handed out-of-sample data. That discipline lives in the
  walk-forward loop, not in the estimator.
- **No liquidity filter.** Still deferred to Phase 3, for the reason given in
  Phase 1: it must be recomputed fold by fold on the in-sample window only.
