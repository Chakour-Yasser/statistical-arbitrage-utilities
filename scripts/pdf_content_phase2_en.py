# -*- coding: utf-8 -*-
"""Phase 2 explanatory document: cointegration building blocks, with the
mathematics stated and proved wherever the proof is short.

Figures are recomputed from the actual data where possible.
"""
from __future__ import annotations
import pandas as pd
from src import config as C


def blocks() -> list:
    folds = pd.read_parquet(C.DATA_PROC / "folds_phase2.parquet")
    B = []; A = B.append

    # ================================================================== #
    A(("h1", "1. What this document covers", False))
    A(("p", "Phase 1 produced a universe we can trust. Phase 2 produces the "
           "statistical machinery: the cointegration test, the hedge ratio, the "
           "half-life and the z-score. Four small functions, and four ways to be "
           "subtly and invisibly wrong."))
    A(("p", "This document states each estimator precisely, proves what can be proved "
           "in a few lines, and reports what was measured when theory alone was not "
           "enough. It opens with a correction to Phase 1, because that correction "
           "changes the numbers Phase 3 will be built on."))
    A(("key", "Three results that were not anticipated",
       ["**The Phase 1 p-values were wrong.** Running the ADF test on an OLS residual "
        "uses the wrong critical values and rejects 14.6 % of the time under the null "
        "instead of 5 %. Fixing it takes the 2016-2018 screen from 112 significant "
        "pairs to 47.",
        "**The half-life estimator does not return NaN on a random walk.** It returns a "
        "finite, large value, and Kendall's bias formula predicts it: the phantom "
        "half-life grows linearly in the sample size, h ≈ 0.173 T.",
        "**A rolling z-score does not damp the signal, it inflates it.** The numerator "
        "shrinks but the denominator shrinks faster, and the strategy fires roughly "
        "2.5 times too many entries at a 30-day half-life."]))

    # ================================================================== #
    A(("h1", "2. Setting and notation", True))
    A(("p", "Fix two securities A and B. Write a<sub>t</sub> = log P<sub>A,t</sub> and "
           "b<sub>t</sub> = log P<sub>B,t</sub> for their total-return log-prices, both "
           "assumed I(1) in the sense of Phase 1. The *spread* with hedge ratio β and "
           "intercept α is"))
    A(("formula", r"s_t \;=\; a_t-\beta\,b_t-\alpha ."))
    A(("p", "The pair is cointegrated when some (β, α) makes (s<sub>t</sub>) I(0). "
           "Everything in Phase 2 is an answer to one of four questions: how to "
           "estimate β, how to test that s<sub>t</sub> is I(0), how fast s<sub>t</sub> "
           "reverts, and how to turn s<sub>t</sub> into a bounded trading signal "
           "without using the future."))
    A(("p", "Sample moments over an in-sample window of T observations are written "
           "S<sub>aa</sub>, S<sub>bb</sub>, S<sub>ab</sub> for the centred sums of "
           "squares and cross-products. All estimation is in-sample; the walk-forward "
           "discipline that keeps it that way is Phase 3's responsibility, not this "
           "module's."))

    # ================================================================== #
    A(("h1", "3. The hedge ratio", True))
    A(("h2", "3.1 Ordinary least squares and its asymmetry"))
    A(("p", "The OLS estimator minimises the sum of squared **vertical** deviations, "
           "which gives the familiar normal equations and"))
    A(("formula", r"\hat\beta_{A|B}=\frac{S_{ab}}{S_{bb}},\qquad \hat\alpha=\bar a-\hat\beta_{A|B}\,\bar b ."))
    A(("p", "The asymmetry this induces is not a subtlety to be mentioned in passing; it "
           "is exact and quantifiable."))
    A(("thm", "1 (OLS asymmetry)", [
        "For any sample with S<sub>aa</sub>, S<sub>bb</sub> > 0,",
        ("formula", r"\hat\beta_{A|B}\;\cdot\;\hat\beta_{B|A}\;=\;R^2 ,"),
        "where R<sup>2</sup> is the coefficient of determination of either regression. "
        "In particular the two directions agree, in the sense that "
        "β̂<sub>A|B</sub> = 1/β̂<sub>B|A</sub>, if and only if R<sup>2</sup> = 1.",
    ]))
    A(("proof", [
        "Directly from the definitions,",
        ("formula", r"\hat\beta_{A|B}\,\hat\beta_{B|A}=\frac{S_{ab}}{S_{bb}}\cdot\frac{S_{ab}}{S_{aa}}=\frac{S_{ab}^{2}}{S_{aa}S_{bb}}=r^{2}=R^{2},"),
        "r being the sample correlation. The second claim follows because "
        "β̂<sub>A|B</sub> = 1/β̂<sub>B|A</sub> is equivalent to the product being 1.",
    ]))
    A(("p", "The practical consequence is larger than intuition suggests. Measured on "
           "simulated pairs: at R<sup>2</sup> = 0.93 the two directions already differ "
           "by 7 % (1.416 against 1.518); at R<sup>2</sup> = 0.11 they differ by a "
           "factor of nearly ten (1.23 against 11.72). Same data, same pair, and a "
           "hedge ratio you could size a position on either way."))
    A(("h2", "3.2 Total least squares is symmetric"))
    A(("p", "Total least squares minimises **perpendicular** distance instead. For "
           "centred data and a unit direction vector u, the objective is"))
    A(("formula", r"\min_{\|u\|=1}\sum_i \left\| x_i-(x_i'u)\,u \right\|^{2}=\sum_i\|x_i\|^{2}-\max_{\|u\|=1} u'S u ,"))
    A(("p", "with S the 2 × 2 scatter matrix. The minimiser is therefore the leading "
           "eigenvector of S, and the fitted slope is the ratio of its components."))
    A(("thm", "2 (TLS is exactly symmetric)", [
        "Write D = S<sub>aa</sub> − S<sub>bb</sub> and R = (D<sup>2</sup> + 4S<sub>ab</sub><sup>2</sup>)<sup>1/2</sup>. "
        "The total-least-squares slopes are",
        ("formula", r"\hat\beta^{\,\mathrm{TLS}}_{A|B}=\frac{D+R}{2S_{ab}},\qquad \hat\beta^{\,\mathrm{TLS}}_{B|A}=\frac{-D+R}{2S_{ab}},"),
        "and they satisfy β̂<sup>TLS</sup><sub>A|B</sub> · β̂<sup>TLS</sup><sub>B|A</sub> = 1 "
        "identically, so the estimator does not depend on which leg is called dependent.",
    ]))
    A(("proof", [
        "The leading eigenvalue of S is λ = ((S<sub>aa</sub> + S<sub>bb</sub>) + R)/2, "
        "with eigenvector proportional to (S<sub>ab</sub>, λ − S<sub>bb</sub>)′; the "
        "slope is the ratio of the second component to the first, which gives the stated "
        "formula. Exchanging the roles of A and B replaces D by −D and leaves R "
        "unchanged. Hence",
        ("formula", r"\hat\beta^{\,\mathrm{TLS}}_{A|B}\,\hat\beta^{\,\mathrm{TLS}}_{B|A}=\frac{(D+R)(-D+R)}{4S_{ab}^{2}}=\frac{R^{2}-D^{2}}{4S_{ab}^{2}}=\frac{4S_{ab}^{2}}{4S_{ab}^{2}}=1 ."),
    ]))
    A(("h2", "3.3 The decision, and the option that is forbidden"))
    A(("table",
       ["Option", "Verdict"],
       [["Fixed convention (alphabetical, `a` dependent)",
         "**Chosen for testing.** Arbitrary but frozen, therefore not a researcher "
         "degree of freedom."],
        ["Test both directions, keep the better p-value",
         "**Forbidden.** Doubles the number of tests and biases selection toward "
         "whichever direction happened to look better -- a multiple-testing problem "
         "disguised as a modelling choice."],
        ["Total least squares",
         "Symmetric by Theorem 2, which is the honest choice when neither leg has "
         "privileged status. But the Engle-Granger p-value is built on an OLS first "
         "stage; substituting TLS would invalidate the critical values of Section 5. "
         "Kept as a robustness check."]],
       [0.30, 0.70]))

    # ================================================================== #
    A(("h1", "4. Mean reversion: AR(1), Ornstein-Uhlenbeck, half-life", True))
    A(("h2", "4.1 The discrete-time picture"))
    A(("p", "Model the spread as a first-order autoregression around a level μ,"))
    A(("formula", r"s_t-\mu=\varphi\,(s_{t-1}-\mu)+\varepsilon_t,\qquad \varepsilon_t \text{ i.i.d. } (0,\sigma_\varepsilon^2)."))
    A(("p", "The process is stationary if and only if |φ| < 1, in which case its "
           "unconditional variance is σ<sub>ε</sub><sup>2</sup>/(1 − φ<sup>2</sup>). "
           "Mean reversion is the statement φ < 1; the value of φ measures its speed."))
    A(("prop", "1 (Geometric decay of a shock)", [
        "For every k ≥ 0,",
        ("formula", r"\mathbb{E}\left[s_{t+k}\mid s_t\right]=\mu+\varphi^{k}\left(s_t-\mu\right)."),
    ]))
    A(("proof", [
        "By induction. The case k = 0 is trivial. Assuming the statement for k and "
        "using the tower property together with the recursion,",
        ("formula", r"\mathbb{E}[s_{t+k+1}\mid s_t]=\mu+\varphi\left(\mathbb{E}[s_{t+k}\mid s_t]-\mu\right)=\mu+\varphi^{k+1}(s_t-\mu),"),
        "since the innovation is centred and independent of the past.",
    ]))
    A(("defn", "1 (Half-life)", [
        "The *half-life* h is the horizon at which the expected deviation from the mean "
        "has halved: the solution of φ<sup>h</sup> = 1/2.",
    ]))
    A(("prop", "2 (Half-life formula, and how to estimate it)", [
        "For 0 < φ < 1,",
        ("formula", r"h=-\frac{\ln 2}{\ln\varphi}."),
        "In practice one regresses the change on the level,",
        ("formula", r"\Delta s_t=a+b\,s_{t-1}+\varepsilon_t \quad\Longleftrightarrow\quad s_t=a+(1+b)s_{t-1}+\varepsilon_t,"),
        "so that φ = 1 + b, with μ = −a/b.",
    ]))
    A(("proof", [
        "Take logarithms in φ<sup>h</sup> = 1/2: h ln φ = −ln 2, and ln φ < 0 for "
        "φ in (0,1). The regression identity is obtained by adding s<sub>t−1</sub> to "
        "both sides of the first equation; setting the expected change to zero gives "
        "a + bμ = 0.",
    ]))

    A(("h2", "4.2 The continuous-time picture, and why it is the same object"))
    A(("p", "The Ornstein-Uhlenbeck process is the continuous-time analogue,"))
    A(("formula", r"ds_t=\theta\,(\mu-s_t)\,dt+\sigma\,dW_t,\qquad \theta>0 ."))
    A(("thm", "3 (Solution and moments of the OU process)", [
        "The unique strong solution is",
        ("formula", r"s_t=\mu+(s_0-\mu)e^{-\theta t}+\sigma\int_0^{t}e^{-\theta(t-u)}\,dW_u ,"),
        "with conditional moments",
        ("formula", r"\mathbb{E}[s_t\mid s_0]=\mu+(s_0-\mu)e^{-\theta t},\qquad \mathrm{Var}(s_t\mid s_0)=\frac{\sigma^{2}}{2\theta}\left(1-e^{-2\theta t}\right),"),
        "and stationary law N(μ, σ<sup>2</sup>/2θ).",
    ]))
    A(("proof", [
        "Apply Ito's formula to f(t, s) = e<sup>θt</sup>(s − μ):",
        ("formula", r"df=\theta e^{\theta t}(s_t-\mu)\,dt+e^{\theta t}\,ds_t=\theta e^{\theta t}(s_t-\mu)dt+e^{\theta t}\left[\theta(\mu-s_t)dt+\sigma dW_t\right]=\sigma e^{\theta t}dW_t ,"),
        "the drift terms cancelling exactly. Integrating from 0 to t and multiplying by "
        "e<sup>−θt</sup> gives the stated solution. The conditional mean follows because "
        "the stochastic integral is a centred martingale; the conditional variance "
        "follows from the Ito isometry,",
        ("formula", r"\mathrm{Var}=\sigma^{2}\int_0^{t}e^{-2\theta(t-u)}du=\frac{\sigma^{2}}{2\theta}\left(1-e^{-2\theta t}\right),"),
        "and letting t tend to infinity gives the stationary law.",
    ]))
    A(("prop", "3 (Exact discretisation)", [
        "Sampling the OU process at a fixed step Δ yields exactly an AR(1) with",
        ("formula", r"\varphi=e^{-\theta\Delta},\qquad\text{hence}\qquad h=\frac{\ln 2}{\theta}."),
        "The discrete and continuous half-lives are therefore the same number, not "
        "approximations of one another.",
    ]))
    A(("proof", [
        "Set s<sub>0</sub> = s<sub>t</sub> and t = Δ in Theorem 3. The conditional law "
        "of s<sub>t+Δ</sub> given s<sub>t</sub> is Gaussian with mean "
        "μ + (s<sub>t</sub> − μ)e<sup>−θΔ</sup> and a variance not depending on "
        "s<sub>t</sub>, which is precisely an AR(1) recursion with φ = "
        "e<sup>−θΔ</sup>. Substituting into Proposition 2 gives h = ln 2 / θ.",
    ]))

    # ================================================================== #
    A(("h1", "5. Testing: Dickey-Fuller and Engle-Granger", True))
    A(("h2", "5.1 The test, and why its distribution is not standard"))
    A(("p", "Testing whether a series has a unit root means testing ρ = 1 in "
           "X<sub>t</sub> = ρX<sub>t−1</sub> + ε<sub>t</sub>, equivalently b = 0 in "
           "ΔX<sub>t</sub> = a + b X<sub>t−1</sub> + ε<sub>t</sub>. The augmented "
           "version adds lagged differences to absorb serial correlation in the "
           "innovations. The essential point is that the usual t-table is inapplicable."))
    A(("thm", "4 (Dickey-Fuller limiting distribution, 1979)", [
        "Under H<sub>0</sub>: ρ = 1 with i.i.d. centred innovations of finite variance,",
        ("formula", r"T\left(\hat\rho-1\right)\;\overset{d}{\longrightarrow}\;\frac{\int_0^1 W\,dW}{\int_0^1 W^{2}\,du}=\frac{\frac{1}{2}\left(W(1)^{2}-1\right)}{\int_0^1 W^{2}\,du},"),
        "where W is a standard Brownian motion. The limit is **not** Gaussian, and the "
        "normalisation is T rather than the usual square root of T.",
    ]))
    A(("proof", [
        "Sketch. By Donsker's invariance principle the partial-sum process converges in "
        "the Skorokhod topology,",
        ("formula", r"T^{-1/2}X_{\lfloor Tu\rfloor}\;\Rightarrow\;\sigma W(u)."),
        "Writing the OLS error as a ratio and rescaling numerator and denominator "
        "separately,",
        ("formula", r"T^{-1}\sum_t X_{t-1}\varepsilon_t\;\Rightarrow\;\sigma^{2}\!\int_0^1 W\,dW,\qquad T^{-2}\sum_t X_{t-1}^{2}\;\Rightarrow\;\sigma^{2}\!\int_0^1 W^{2}du ,"),
        "and the continuous mapping theorem gives the ratio. The identity for the "
        "numerator is Ito's formula applied to W<sup>2</sup>. The faster T-rate is the "
        "usual super-consistency under a unit root: the regressor is of order square "
        "root of T rather than order one.",
    ]))
    A(("h2", "5.2 Engle-Granger: the residual is estimated, and that changes everything"))
    A(("p", "The two-step procedure regresses a<sub>t</sub> on b<sub>t</sub>, then tests "
           "the residual for a unit root. The temptation is to feed that residual to a "
           "standard ADF routine. **That is the error made in Phase 1.**"))
    A(("thm", "5 (Phillips and Ouliaris 1990 — stated without proof)", [
        "Under the null of no cointegration, the ADF statistic computed on the OLS "
        "residual û<sub>t</sub> converges to a functional of a *demeaned and projected* "
        "Brownian motion whose law depends on the number of regressors. Its "
        "distribution is stochastically smaller (further into the left tail) than the "
        "Dickey-Fuller law of Theorem 4.",
    ]))
    A(("p", "The intuition is worth more than the formal statement. OLS chose β̂ to "
           "minimise the residual variance over an entire one-parameter family. The "
           "residual being tested is therefore the **most stationary-looking** linear "
           "combination available in that sample, not an arbitrary one. Testing it "
           "against critical values designed for an observed series necessarily "
           "over-rejects."))
    A(("h2", "5.3 How large is the error? Measure it."))
    A(("p", "A theorem saying that critical values shift does not tell you whether the "
           "shift matters. Simulate under the null — two independent random walks, no "
           "cointegration by construction — and count rejections. A correct 5 % test "
           "rejects 5 % of the time."))
    A(("code",
       "2000 simulations under H0 (independent random walks, n = 750)\n"
       "\n"
       "  adfuller on the OLS residual : 14.60 % rejections   <- should be 5 %\n"
       "  coint (Engle-Granger values) :  4.15 % rejections\n"
       "\n"
       "  false-positive inflation factor : x3.5"))
    A(("warn", "The effect on the Phase 1 numbers",
       ["On the 2016-2018 screen of 465 pairs, the count of pairs called significant at "
        "the 5 % level falls from **112** with `adfuller` to **47** with `coint`. "
        "Sixty-five discoveries evaporate.",
        "The expected-false-positive figure quoted in Phase 1 (about 23, from "
        "Proposition 7 of that document) was correct for a valid 5 % test — but the "
        "screen it was being compared against was running at an effective level of "
        "14.6 %, where the expected count is 68. The comparison was meaningless.",
        "Every Phase 2 function uses `coint`. The Phase 1 notebooks are left as "
        "written, as a record of what was done, with the correction flagged in the "
        "README."]))

    # ================================================================== #
    A(("h1", "6. The half-life estimator is biased, and the bias is predictable", True))
    A(("p", "The naive expectation is that the half-life estimator returns NaN on a "
           "random walk, since φ ≥ 1 admits no solution to φ<sup>h</sup> = 1/2. It does "
           "not. On 4000 simulated random walks of 750 observations it returned NaN "
           "only 4 % of the time, with a median estimate of 111 days. The reason is a "
           "classical result."))
    A(("thm", "6 (Small-sample bias of the AR(1) coefficient; Kendall 1954, Marriott and Pope 1954)", [
        "For the AR(1) model estimated with an intercept on T observations,",
        ("formula", r"\mathbb{E}\left[\hat\varphi\right]-\varphi\;\approx\;-\frac{1+3\varphi}{T}+O(T^{-2})."),
        "The estimator is biased **downward**, and the bias is largest exactly where it "
        "hurts: near the unit root.",
    ]))
    A(("prop", "4 (The phantom half-life grows linearly in the sample size)", [
        "Applying Theorem 6 at φ = 1 gives φ̂ ≈ 1 − 4/T, whence",
        ("formula", r"h_{\text{phantom}}=-\frac{\ln 2}{\ln\left(1-4/T\right)}\;\approx\;\frac{\ln 2}{4}\,T\;\approx\;0.173\,T ."),
        "So a pure random walk does not merely produce a finite half-life: it produces "
        "one **proportional to the estimation window**.",
    ]))
    A(("proof", [
        "Substitute φ = 1 in Theorem 6 to get the leading bias −4/T. Then expand the "
        "logarithm for large T, ln(1 − 4/T) = −4/T + O(T<sup>−2</sup>), and substitute "
        "into the half-life formula of Proposition 2.",
    ]))
    A(("p", "Theory and measurement can now be compared directly. The agreement is close "
           "enough to confirm the mechanism, and the residual gap has a clear cause: the "
           "sampling distribution of φ̂ is skewed, so the median of h(φ̂) sits below "
           "h(E[φ̂]) by Jensen's inequality."))
    A(("table",
       ["Sample size T", "φ̂ measured", "φ̂ predicted (1 − 4/T)", "h predicted", "h median measured"],
       [["250", "0.97837", "0.98400", "43.0 d", "37.4 d"],
        ["500", "0.98913", "0.99200", "86.3 d", "74.5 d"],
        ["750", "0.99284", "0.99467", "129.6 d", "115.2 d"],
        ["1500", "0.99649", "0.99733", "259.6 d", "227.1 d"]],
       [0.18, 0.18, 0.24, 0.18, 0.22],
       "Kendall's formula predicts the phantom half-life to within about 12 %. "
       "3000 simulations per row."))
    A(("key", "Two consequences",
       ["**A diagnostic.** Proposition 4 says the estimate should scale with T on a "
        "random walk and be stable in T on a genuine mean-reverting spread. A half-life "
        "that changes when you change the estimation window is therefore a red flag, "
        "and a cheap one to check.",
        "**A rule.** The half-life must never be used to decide *whether* a spread "
        "reverts — that is the cointegration test's job. It is a tradability filter, "
        "applied afterwards. And its upper bound must be tight: 18.4 % of random walks "
        "pass a 60-day cap, against 2.7 % at 30 days."]))

    # ================================================================== #
    A(("h1", "7. The z-score: causality, and an inflation nobody warns you about", True))
    A(("h2", "7.1 What causal means, formally"))
    A(("p", "Let (*F*<sub>t</sub>) be the filtration generated by the observable price "
           "history up to and including date t."))
    A(("defn", "2 (Causal signal)", [
        "A signal (z<sub>t</sub>) is *causal* if z<sub>t</sub> is "
        "*F*<sub>t</sub>-measurable for every t.",
    ]))
    A(("p", "The full-sample z-score, z<sub>t</sub> = (s<sub>t</sub> − mean of s) / (sd "
           "of s) with moments taken over the whole period, is *F*<sub>T</sub>-measurable "
           "and not *F*<sub>t</sub>-measurable. It is therefore not a signal at all: it "
           "encodes the spread's eventual average level, which is exactly the quantity "
           "the strategy is supposed to be betting on. This is the single most common "
           "leak in pairs trading."))
    A(("p", "Measurability is not a property one can eyeball, so it is enforced by a "
           "test. `test_rolling_zscore_is_causal` rewrites the series after some index "
           "k and asserts that every z-score up to k is bit-identical. If anyone ever "
           "substitutes a full-sample or centred window, the suite fails immediately. A "
           "companion test keeps the naive estimator as an executable counter-example."))

    A(("h2", "7.2 Why a rolling window is worse than it looks"))
    A(("p", "A rolling window is causal, so the obvious fix is to standardise with a "
           "rolling mean and standard deviation. That is what most published "
           "implementations do. On a strongly autocorrelated spread it is badly biased, "
           "and — contrary to the usual warning — the bias **inflates** the signal "
           "rather than damping it."))
    A(("prop", "5 (Effective sample size of a rolling mean)", [
        "Let (s<sub>t</sub>) be stationary AR(1) with parameter φ and variance γ, and "
        "let m<sub>W</sub> be the mean of W consecutive observations. Then",
        ("formula", r"\mathrm{Var}(m_W)=\frac{\gamma}{W^{2}}\left(W+2\sum_{k=1}^{W-1}(W-k)\varphi^{k}\right)\;\equiv\;\frac{\gamma}{W_{\mathrm{eff}}} ,"),
        "and as W grows large,",
        ("formula", r"W_{\mathrm{eff}}\;\longrightarrow\;W\,\frac{1-\varphi}{1+\varphi}."),
    ]))
    A(("proof", [
        "Expand the double sum and group by lag. The autocovariance at lag k is "
        "γφ<sup>|k|</sup>, and among the W<sup>2</sup> ordered pairs of indices exactly "
        "W have lag 0 and 2(W − k) have lag ±k, giving the stated expression. For the "
        "limit, note that the sum equals Wφ/(1 − φ) + O(1), so",
        ("formula", r"\mathrm{Var}(m_W)\approx\frac{\gamma}{W}\left(1+\frac{2\varphi}{1-\varphi}\right)=\frac{\gamma}{W}\cdot\frac{1+\varphi}{1-\varphi}."),
    ]))
    A(("warn", "The asymptotic form is not usable in our range",
       ["At a 30-day half-life, φ = 0.977 and 1/(1 − φ) = 44, so a 60-day window is "
        "nowhere near the regime W much greater than 1/(1 − φ). The asymptotic formula "
        "gives W_eff = 0.69 where the exact expression gives **1.51** — an error of a "
        "factor two, and in the direction that would have made the problem look worse "
        "than it is. The exact sum was used throughout and checked against simulation."]))
    A(("table",
       ["Half-life", "φ", "W_eff, 60-day window", "W_eff, 750-day in-sample window"],
       [["5 d", "0.8706", "4.72", "52.4"],
        ["10 d", "0.9330", "2.72", "26.5"],
        ["15 d", "0.9548", "2.09", "17.8"],
        ["20 d", "0.9659", "1.79", "13.5"],
        ["30 d", "0.9772", "**1.51**", "**9.2**"],
        ["45 d", "0.9847", "1.33", "6.3"]],
       [0.18, 0.16, 0.30, 0.36],
       "Exact effective sample sizes from Proposition 5, verified by simulation. "
       "A 60-day rolling window carries 1.5 effective observations at a 30-day "
       "half-life: the rolling mean is essentially the spread itself."))
    A(("h2", "7.3 The direction of the bias, measured"))
    A(("p", "With so few effective observations the rolling mean tracks the spread, so "
           "the numerator of the z-score shrinks. The usual warning stops there and "
           "concludes that the signal is damped. It is not: the rolling **standard "
           "deviation** shrinks faster, because it measures the *local* dispersion of a "
           "highly autocorrelated series rather than its unconditional dispersion. The "
           "quotient therefore exceeds one."))
    A(("table",
       ["Half-life", "numerator ratio", "denominator ratio", "net effect on z", "share of days |z| > 2"],
       [["5 d", "0.98", "0.88", "1.12", "6.9 % (true 4.6 %)"],
        ["10 d", "0.94", "0.77", "1.21", "8.8 %"],
        ["20 d", "0.84", "0.64", "1.31", "10.7 %"],
        ["30 d", "0.76", "0.56", "**1.36**", "**11.7 %**"],
        ["45 d", "0.69", "0.49", "1.40", "12.1 %"]],
       [0.16, 0.20, 0.22, 0.18, 0.24],
       "Rolling window W = 60, simulated OU paths. At a 30-day half-life the strategy "
       "fires roughly 2.5 times too many entries."))
    A(("h2", "7.4 The design decision"))
    A(("p", "The remedy is to freeze μ and σ on the in-sample window — the same window "
           "that produced β and the cointegration p-value — and apply them unchanged "
           "through the trading year. This is exactly as causal, has no small-sample "
           "tracking problem, and keeps the entire calibration on one window."))
    A(("formula", r"z_t=\frac{s_t-\hat\mu_{\mathrm{IS}}}{\hat\sigma_{\mathrm{IS}}},\qquad (\hat\mu_{\mathrm{IS}},\hat\sigma_{\mathrm{IS}})\ \text{estimated on the in-sample window only.}"))
    A(("p", "This raises the effective sample size from 1.5 to 9.2 at a 30-day half-life "
           "and roughly halves the distortion: the share of days with |z| > 2 falls from "
           "11.4 % to 8.0 %, against 4.6 % for the ideal. It does **not** eliminate it, "
           "because the in-sample window itself carries only about nine effective "
           "observations at that half-life."))
    A(("key", "Two things this forces us to admit",
       ["**An entry threshold of |z| > 2 is not a reliable prior on trade frequency.** "
        "Realised turnover must be measured in Phase 6, not assumed from the Gaussian "
        "tail probability.",
        "**The half-life cap does triple duty.** It bounds turnover, it keeps random "
        "walks out (Section 6), and it keeps the in-sample effective sample size large "
        "enough for σ to be estimable at all. On that last count h ≤ 20 (W_eff = 13.5) "
        "is comfortable and h = 30 (W_eff = 9.2) is the edge. The default cap of 30 is "
        "a documented compromise, not a comfortable margin — and a sensitivity analysis "
        "on it belongs in Phase 7.",
        "What we give up by freezing: the z-score cannot absorb a shift in the spread's "
        "level. That is deliberate. Silently absorbing a level shift is precisely how a "
        "broken pair keeps generating signals; detecting it is the job of the regime "
        "machinery in Phase 4, not of the normalisation."]))

    # ================================================================== #
    A(("h1", "8. The quality filter", True))
    A(("p", "The cointegration test says a spread is stationary. It does not say the "
           "spread is tradable. Two bands, applied **after** the test and **in-sample "
           "only**, both calibrated on evidence rather than taste."))
    A(("h2", "8.1 Half-life band: 2 to 30 days"))
    A(("ul", [
        "**Below 2 days**: the signal is dominated by microstructure noise and bid-ask "
        "bounce, and the implied turnover lets transaction costs consume the edge. "
        "Phase 6 quantifies this.",
        "**Above 30 days**: three arguments converge, and they are independent of one "
        "another. *Economic* — the out-of-sample window is one year, so a 60-day "
        "half-life allows only about four reversion cycles, too few round trips to say "
        "anything statistical and a long exposure to the relation breaking first. "
        "*Statistical* — by Proposition 4 a loose cap admits random walks: 18.4 % pass "
        "at 60 days against 2.7 % at 30. *Estimation* — by Proposition 5 the in-sample "
        "effective sample size falls to 9.2 at 30 days, which is already marginal for "
        "estimating σ.",
    ]))
    A(("h2", "8.2 Hedge-ratio band: 1/3 to 3"))
    A(("p", "A β outside a sane band is not a hedge; it is a directional bet on one leg "
           "with the other as decoration."))
    A(("ul", [
        "**β ≤ 0** means going long *both* legs. The position carries full sector "
        "exposure and market neutrality — the entire premise — is gone. On the "
        "2016-2018 screen all five negative-β pairs involved PCG (collapsing on "
        "wildfire liabilities) or SCG (the nuclear-project scandal). A negative β fits "
        "two names trending in opposite directions and means nothing economically.",
        "**β near 0, or symmetrically near infinity**, means a degenerate regression. "
        "This bites when the legs have very different volatility: by Theorem 1, "
        "β̂ = S<sub>ab</sub>/S<sub>bb</sub> shrinks toward zero as the regressor's "
        "variance grows, and the residual becomes little more than a rescaled copy of "
        "the dependent variable. On the same screen **NRG** — a merchant generator, far "
        "more volatile than the regulated names — appeared in **13 of 41** significant "
        "pairs (32 %, against 3.2 expected under a uniform spread), every one with β "
        "between 0.13 and 0.30.",
    ]))
    A(("p", "The band is symmetric under leg inversion, 1/3 and 3, so the filter does "
           "not inherit the arbitrariness of the alphabetical convention. A unit test "
           "enforces that symmetry."))

    # ================================================================== #
    A(("h1", "9. Two findings that set up Phase 3", True))
    A(("h2", "9.1 Discoveries barely exceed chance, and twice not at all"))
    A(("p", "Proposition 7 of the Phase 1 document gives the expected number of spurious "
           "discoveries under the null, E[V] = αN, with no independence assumption. "
           "Comparing the observed count against it is the cheapest available sanity "
           "check, and on this universe it is sobering."))
    A(("table",
       ["In-sample window", "Pairs tested", "Significant", "Expected under H0", "Excess"],
       [[r.fold, f"{int(r.n_tests)}", f"{int(r.found)}", f"{r.expected:.0f}",
         ("**" + f"{r.excess:+.0f}" + "**") if abs(r.excess) <= 2 else f"{r.excess:+.0f}"]
        for _, r in folds.iterrows()],
       [0.24, 0.18, 0.18, 0.20, 0.20],
       "Two folds find nothing beyond pure chance. The 2019-2021 and 2020-2022 spikes "
       "sit squarely on the COVID shock, where a common shock makes everything co-move "
       "and manufactures spurious cointegration."))
    A(("p", "Two folds — 2015-2017 and 2022-2024 — find nothing beyond what a 5 % test "
           "produces on pure noise. Selecting pairs there means trading noise, and a "
           "strategy that trades them regardless will show it out of sample. Phase 7 "
           "will have to report those folds separately rather than aggregate them away."))
    A(("h2", "9.2 The tests are not independent, and the selection is concentrated"))
    A(("p", "In **every** fold, one name holds between 16 % and 50 % of the selected "
           "pairs — and it is a different name each time: D, SRE, AEP, AEE, ETR, AWK. "
           "This is not one odd stock. It is a property of the method: any name whose "
           "idiosyncratic path happens to look mean-reverting against the sector "
           "in-sample becomes a hub."))
    A(("table",
       ["Fold", "Pairs kept", "Most frequent name", "Its share"],
       [[r.fold, f"{int(r.kept)}", r.hub, f"{100*r.hub_share:.0f} %"]
        for _, r in folds.iterrows()],
       [0.26, 0.22, 0.26, 0.26]))
    A(("key", "Why this is the hinge between Phase 2 and Phase 3",
       ["**Risk.** A selection of 24 pairs with one name in 12 of them is not 24 bets. "
        "It is one bet expressed twelve ways. Sizing that treats the pairs as "
        "independent silently concentrates risk, which Phase 6 must handle.",
        "**Statistics.** Recall that E[V] = αN holds under arbitrary dependence, but "
        "Var(V) does not. Every name shares the sector factor, and a hub induces strong "
        "positive dependence across all pairs containing it, so the effective number of "
        "independent tests is far below N(N−1)/2.",
        "That is exactly the regime where the Phase 3 choice stops being cosmetic: "
        "**Bonferroni** controls the family-wise error rate under *any* dependence "
        "structure, by the union bound; **Benjamini-Hochberg** controls the false "
        "discovery rate under independence or positive regression dependence, but not "
        "in general. A permutation or block-bootstrap null, which reproduces the "
        "dependence rather than assuming it away, is the honest third option."]))

    # ================================================================== #
    A(("h1", "10. What Phase 2 deliberately does not do", True))
    A(("ul", [
        "**No selection.** `screen_pairs` runs the tests; it does not decide what to "
        "trade. Selection, the multiple-testing correction and the walk-forward loop "
        "are Phase 3.",
        "**No time awareness.** Nothing in `src/cointegration.py` knows about dates, so "
        "nothing there prevents it being handed out-of-sample data. That discipline "
        "lives in the walk-forward loop, not in the estimator — a module that cannot "
        "see dates cannot silently cheat with them.",
        "**No liquidity filter.** Still deferred, for the Phase 1 reason: it must be "
        "recomputed fold by fold on the in-sample window only, or it becomes "
        "survivorship bias dressed as a technical criterion.",
    ]))
    A(("h2", "Interview questions this phase should let you answer"))
    A(("ul", [
        "Derive the half-life from an AR(1). What is its continuous-time counterpart, "
        "and why are they the same number?",
        "Why can you not test cointegration with an ordinary t-statistic? What goes "
        "wrong, and at what rate?",
        "You ran ADF on an OLS residual. What is the size of that test, and why is it "
        "not 5 %?",
        "Your half-life estimator returns 111 days on a random walk. Explain, and give "
        "the scaling in T.",
        "Why is a rolling z-score biased on a mean-reverting spread, and in which "
        "direction?",
        "Your screener finds 47 significant pairs and expects 23 by chance. What do you "
        "conclude, and what would you do next?",
        "Bonferroni is valid under arbitrary dependence and BH is not. Why is that not "
        "an argument for using Bonferroni here?",
    ]))

    # ================================================================== #
    A(("h1", "11. Notation", True))
    A(("table",
       ["Symbol", "Meaning"],
       [["a<sub>t</sub>, b<sub>t</sub>", "Total-return log-prices of the two legs"],
        ["s<sub>t</sub>", "Spread, a<sub>t</sub> − β b<sub>t</sub> − α"],
        ["β", "Hedge ratio; β̂ its estimate"],
        ["φ", "AR(1) autoregressive parameter; φ = 1 is a unit root"],
        ["θ", "OU mean-reversion speed; φ = exp(−θΔ)"],
        ["h", "Half-life, h = −ln 2 / ln φ = ln 2 / θ"],
        ["γ", "Unconditional variance of the spread"],
        ["W_eff", "Effective sample size of a rolling mean (Proposition 5)"],
        ["T", "Number of in-sample observations"],
        ["N", "Number of pairs tested, N = n(n−1)/2"],
        ["V", "Number of false discoveries"],
        ["W(u)", "Standard Brownian motion on [0,1]"],
        ["*F*<sub>t</sub>", "Information available at date t"]],
       [0.22, 0.78]))
    return B
