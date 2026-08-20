# -*- coding: utf-8 -*-
"""Mathematical foundations section for the Phase 1 document."""


def blocks() -> list:
    B = []; A = B.append

    A(("h1", "3. Mathematical foundations", True))
    A(("p", "This section states precisely what the rest of the document uses "
           "informally: what stationarity is, what cointegration is, why a "
           "cointegration *test* is needed rather than an ordinary regression, and "
           "why each of the Phase 1 decisions is forced rather than chosen. Proofs "
           "are given whenever they are short; the two results whose proofs need "
           "functional limit theory are stated with a reference and a sketch."))

    # ------------------------------------------------------------------ 3.1
    A(("h2", "3.1 Stationarity"))
    A(("p", "Throughout, (Ω, *F*, *P*) is a probability space and all processes are "
           "real-valued and indexed by t in **Z**."))
    A(("defn", "1 (Strict stationarity)", [
        "A process (X<sub>t</sub>) is *strictly stationary* if for every k ≥ 1, every "
        "t<sub>1</sub> < … < t<sub>k</sub> and every h in **Z**, the joint laws agree:",
        ("formula", r"(X_{t_1},\ldots,X_{t_k}) \;\overset{d}{=}\; (X_{t_1+h},\ldots,X_{t_k+h})."),
    ]))
    A(("defn", "2 (Weak or covariance stationarity)", [
        "(X<sub>t</sub>) is *covariance stationary* if *E*[X<sub>t</sub><sup>2</sup>] < ∞ "
        "and there exist μ in **R** and γ : **Z** → **R** with",
        ("formula", r"\mathbb{E}[X_t]=\mu \quad\text{and}\quad \mathrm{Cov}(X_t,X_{t+h})=\gamma(h) \quad \text{for all } t,h."),
        "The essential point is that neither the mean nor the autocovariance depends "
        "on *t*: the process has no privileged date.",
    ]))
    A(("p", "Strict stationarity with finite second moment implies covariance "
           "stationarity; the converse fails in general but holds for Gaussian "
           "processes, since a Gaussian law is determined by its first two moments. "
           "All tests used here concern **covariance** stationarity."))

    # ------------------------------------------------------------------ 3.2
    A(("h2", "3.2 Integration order, and why a price is not stationary"))
    A(("defn", "3 (Order of integration)", [
        "(X<sub>t</sub>) is *integrated of order 0*, written I(0), if it is covariance "
        "stationary with a strictly positive spectral density at frequency zero. It is "
        "I(d) for an integer d ≥ 1 if the d-th difference (1 − L)<sup>d</sup>X<sub>t</sub> "
        "is I(0), where L is the lag operator, LX<sub>t</sub> = X<sub>t−1</sub>.",
    ]))
    A(("prop", "1 (A random walk is not covariance stationary)", [
        "Let X<sub>0</sub> = 0 and X<sub>t</sub> = X<sub>t−1</sub> + ε<sub>t</sub> with "
        "(ε<sub>t</sub>) i.i.d., *E*[ε<sub>t</sub>] = 0, Var(ε<sub>t</sub>) = "
        "σ<sup>2</sup> in (0, ∞). Then",
        ("formula", r"\mathrm{Var}(X_t)=t\,\sigma^2,"),
        "which depends on t, so (X<sub>t</sub>) is not covariance stationary. It is I(1).",
    ]))
    A(("proof", [
        "Telescoping the recursion gives X<sub>t</sub> = ε<sub>1</sub> + … + "
        "ε<sub>t</sub>. The increments are independent and centred, so variances add:",
        ("formula", r"\mathrm{Var}(X_t)=\sum_{s=1}^{t}\mathrm{Var}(\varepsilon_s)=t\sigma^2."),
        "Since σ<sup>2</sup> > 0 the map t → Var(X<sub>t</sub>) is strictly "
        "increasing, contradicting Definition 2. Its first difference is "
        "ΔX<sub>t</sub> = ε<sub>t</sub>, which is i.i.d. and hence I(0).",
    ]))
    A(("p", "Log-prices of liquid equities are, to a good approximation, I(1): daily "
           "log-returns are close to serially uncorrelated with finite variance, while "
           "the level wanders without an attracting mean. **This is the whole "
           "difficulty.** A generic linear combination of two I(1) series is again "
           "I(1), so a spread is generically non-stationary and has no mean to revert "
           "to. Pairs trading is only possible in the exceptional case of the next "
           "definition."))

    # ------------------------------------------------------------------ 3.3
    A(("h2", "3.3 Cointegration"))
    A(("defn", "4 (Cointegration, Engle and Granger 1987)", [
        "Let X<sub>t</sub> = (X<sub>t</sub><sup>(1)</sup>, …, X<sub>t</sub><sup>(n)</sup>)′ "
        "with every component I(1). The vector process is *cointegrated* if there exists "
        "β in **R**<sup>n</sup>, β ≠ 0, such that",
        ("formula", r"\beta' X_t \;\text{ is } \; I(0)."),
        "Such a β is a *cointegrating vector*. The set of cointegrating vectors, "
        "together with 0, is a linear subspace of **R**<sup>n</sup>; its dimension r is the "
        "*cointegration rank*.",
    ]))
    A(("p", "Two immediate remarks. First, β is defined only up to scale, since cβ "
           "works for any c ≠ 0; for a pair one therefore normalises "
           "β = (1, −β<sub>2</sub>)′, which is exactly the hedge ratio of Section 2. "
           "Second, for n = 2 the rank is 0 or 1: either the pair is cointegrated with "
           "an essentially unique β, or it is not cointegrated at all. Rank only "
           "becomes an interesting object for baskets, which is the subject of the "
           "Johansen extension."))
    A(("thm", "1 (Granger representation theorem, 1987 — stated without proof)", [
        "If X<sub>t</sub> is I(1) with cointegration rank r ≥ 1, then it admits a "
        "*vector error-correction* representation",
        ("formula", r"\Delta X_t=\alpha\,\beta' X_{t-1}+\sum_{i=1}^{p}\Gamma_i\,\Delta X_{t-i}+\varepsilon_t,"),
        "with α, β of full rank r. Conversely such a representation implies "
        "cointegration.",
    ]))
    A(("key", "Why Theorem 1 is the economic content of this project",
       ["The error-correction term α β′X<sub>t−1</sub> says that **today's deviation "
        "from equilibrium predicts tomorrow's change**. That is not a restatement of "
        "cointegration — it is the assertion that a forecastable component exists, and "
        "α measures how fast the deviation is corrected.",
        "This is the precise sense in which cointegration, unlike correlation, is a "
        "*tradable* property. Correlation describes contemporaneous co-movement and "
        "implies nothing about future returns; the error-correction representation is "
        "a statement about conditional expectations of future increments.",
        "It also locates exactly what Phase 4 is about: a *regime break* is α drifting "
        "toward 0, or β itself moving."]))

    # ------------------------------------------------------------------ 3.4
    A(("h2", "3.4 Why a regression is not a test: spurious regression"))
    A(("p", "A natural but wrong instinct is to regress one log-price on the other and "
           "read off the t-statistic. The following result explains why that procedure "
           "is worthless on I(1) data, and why a dedicated cointegration test is "
           "unavoidable."))
    A(("thm", "2 (Spurious regression; Granger and Newbold 1974, Phillips 1986)", [
        "Let X<sub>t</sub> and Y<sub>t</sub> be **independent** random walks and "
        "consider the OLS regression Y<sub>t</sub> = a + b X<sub>t</sub> + u<sub>t</sub> "
        "on T observations. Then, as T → ∞,",
        ("formula", r"\hat b \;\overset{d}{\longrightarrow}\; \xi \;(\text{a non-degenerate random variable}), \qquad R^2 \;\overset{d}{\longrightarrow}\; \eta \in (0,1),"),
        "and the usual t-statistic diverges,",
        ("formula", r"T^{-1/2}\,t_{\hat b}\;\overset{d}{\longrightarrow}\;\zeta \neq 0, \qquad\text{so}\qquad |t_{\hat b}|\;\overset{\mathbb{P}}{\longrightarrow}\;\infty."),
        "Consequently the test of b = 0 at any fixed level rejects with probability "
        "tending to 1, although the two series are independent by construction.",
    ]))
    A(("p", "The mechanism is that under independence the residual u<sub>t</sub> is "
           "itself I(1). The OLS standard error formula assumes I(0) errors; applied to "
           "an I(1) residual it is inconsistent and understates the true dispersion by "
           "a factor growing with T. Neither β̂ nor its t-statistic converges to "
           "anything usable."))
    A(("key", "The consequence that shapes the whole pipeline",
       ["A high R<sup>2</sup> and a large t-statistic between two log-prices are "
        "**evidence of nothing**. The only admissible procedure is to run the "
        "regression, then test whether the *residual* is I(0) — which is exactly the "
        "Engle-Granger two-step procedure of Phase 2.",
        "It also explains why the critical values must be adjusted: the residual being "
        "tested is not observed but estimated, and it was estimated by minimising its "
        "own variance. Phase 2 quantifies the resulting size distortion at a factor "
        "of 3.5."]))

    # ------------------------------------------------------------------ 3.5
    A(("h2", "3.5 Why dividends force total-return prices"))
    A(("p", "Section 7 argues that raw prices inject a deterministic drift into the "
           "spread. Here is the statement that makes the argument binding rather than "
           "suggestive."))
    A(("prop", "2 (A deterministic trend destroys stationarity)", [
        "Let (y<sub>t</sub>) be covariance stationary with mean μ, let "
        "δ ≠ 0, and set s<sub>t</sub> = y<sub>t</sub> − δt. Then "
        "(s<sub>t</sub>) is not covariance stationary.",
    ]))
    A(("proof", [
        "By linearity of the expectation, *E*[s<sub>t</sub>] = μ − δt. Since δ ≠ 0 this "
        "is a non-constant function of t, contradicting Definition 2.",
    ]))
    A(("p", "Now apply this to prices. Write P<sup>TR</sup> for the total-return price "
           "and P<sup>raw</sup> for the raw price of a stock paying a continuously "
           "compounded dividend yield δ. Reinvesting dividends multiplies wealth by "
           "e<sup>δt</sup> relative to holding the raw share, so"))
    A(("formula", r"\log P^{\mathrm{raw}}_t=\log P^{\mathrm{TR}}_t-\delta t ."))
    A(("prop", "3 (The raw-price spread carries a deterministic drift)", [
        "Let A and B have dividend yields δ<sub>A</sub>, δ<sub>B</sub> and suppose the "
        "total-return log-prices are cointegrated with hedge ratio β, so that "
        "y<sub>t</sub> = log P<sup>TR</sup><sub>A,t</sub> − β log "
        "P<sup>TR</sup><sub>B,t</sub> is I(0). Then the spread computed on raw prices "
        "satisfies",
        ("formula", r"s^{\mathrm{raw}}_t=y_t-(\delta_A-\beta\,\delta_B)\,t ,"),
        "which is I(0) if and only if δ<sub>A</sub> = β δ<sub>B</sub>.",
    ]))
    A(("proof", [
        "Substitute the identity above into the definition of the spread:",
        ("formula", r"s^{\mathrm{raw}}_t=\left(\log P^{\mathrm{TR}}_{A,t}-\delta_A t\right)-\beta\left(\log P^{\mathrm{TR}}_{B,t}-\delta_B t\right)=y_t-(\delta_A-\beta\delta_B)t."),
        "If δ<sub>A</sub> ≠ β δ<sub>B</sub>, Proposition 2 applies and the raw spread "
        "is not stationary; if they are equal the drift term vanishes identically.",
    ]))
    A(("p", "The condition δ<sub>A</sub> = β δ<sub>B</sub> is a knife-edge: it holds on "
           "a set of measure zero in the parameter space. On the actual universe, "
           "measured dividend yields range from 1.07 % (PCG, which suspended its "
           "dividend during the bankruptcy) to 6.39 % (OKE), so the drift is generically "
           "non-zero and of the same order as the spread's own amplitude. **This is why "
           "the choice of price series is a modelling decision and not data "
           "preparation.**"))

    # ------------------------------------------------------------------ 3.6
    A(("h2", "3.6 What forward-fill does, and what it does not do"))
    A(("p", "Section 7.2 reports that forward-fill does not manufacture mean reversion, "
           "contrary to the standard argument. Here is why, in two lines."))
    A(("prop", "4 (Forward-fill preserves the zero autocovariance of increments)", [
        "Let X<sub>t</sub> = X<sub>t−1</sub> + ε<sub>t</sub> be a random walk with "
        "i.i.d. centred increments, and let (G<sub>t</sub>) taking values in {0,1} be a gap indicator "
        "**independent of (ε<sub>t</sub>)**. Define the forward-filled series "
        "X<sup>f</sup><sub>t</sub> = X<sub>τ(t)</sub> with τ(t) = max{ s ≤ t : "
        "G<sub>s</sub> = 0 }. Then for every t,",
        ("formula", r"\mathrm{Cov}\!\left(\Delta X^{f}_{t},\,\Delta X^{f}_{t+1}\right)=0 ."),
    ]))
    A(("proof", [
        "Condition on the whole gap sequence G, which is independent of ε. Given G, "
        "the increment ΔX<sup>f</sup><sub>t</sub> equals 0 if G<sub>t</sub> = 1, and "
        "otherwise equals the sum of the ε<sub>s</sub> over the deterministic index "
        "block (τ(t−1), τ(t)]. Three cases exhaust the possibilities. If "
        "G<sub>t</sub> = 1 then ΔX<sup>f</sup><sub>t</sub> = 0 and the product "
        "vanishes; if G<sub>t+1</sub> = 1 then ΔX<sup>f</sup><sub>t+1</sub> = 0 and it "
        "vanishes likewise; otherwise the two increments are sums of ε over **disjoint** "
        "index blocks, hence uncorrelated. In all cases the conditional covariance is "
        "0, and the tower property gives the unconditional result.",
    ]))
    A(("p", "The intuition the proof formalises: forward-fill **redistributes** "
           "increments across dates without creating or destroying any. The missing day "
           "carries a zero return and the next observed day carries the accumulated "
           "sum. Nothing compensatory happens, so no mean reversion appears. What is "
           "genuinely damaged is tradability — signals land on dates where the security "
           "did not trade — which is an execution issue, not a statistical one."))

    # ------------------------------------------------------------------ 3.7
    A(("h2", "3.7 Survivorship as a conditioning bias"))
    A(("p", "Section 8.1 asserts that survivorship bias is a form of look-ahead whose "
           "sign is known. The following makes both claims precise."))
    A(("prop", "5 (Sign of the survivorship bias)", [
        "Let R be the return of a strategy over the sample, and let S be the event "
        "that a name survives to the end of the sample (no bankruptcy, no acquisition). "
        "A universe built from current constituents estimates *E*[R | S], whereas the "
        "quantity of interest is the unconditional *E*[R]. By the law of total "
        "expectation,",
        ("formula", r"\mathbb{E}[R]=\mathbb{E}[R\mid S]\,\mathbb{P}(S)+\mathbb{E}[R\mid S^{c}]\,\mathbb{P}(S^{c}),"),
        "so that",
        ("formula", r"\mathbb{E}[R\mid S]-\mathbb{E}[R]=\mathbb{P}(S^{c})\left(\mathbb{E}[R\mid S]-\mathbb{E}[R\mid S^{c}]\right)."),
        "Hence the bias is positive if and only if *E*[R | S] > *E*[R | S<sup>c</sup>].",
    ]))
    A(("proof", [
        "The displayed identity is the law of total expectation rearranged; the sign "
        "claim follows because *P*(S<sup>c</sup>) ≥ 0.",
    ]))
    A(("p", "The empirical content is the sign of *E*[R | S] − *E*[R | S<sup>c</sup>]. Every "
           "one of the five names lost from our universe left the index by "
           "**acquisition**, and a takeover breaks a cointegration relation in the most "
           "damaging possible way: the target's price jumps to the offer and then "
           "freezes, so a short leg on the target takes an instantaneous loss that never "
           "reverts. Therefore *E*[R | S<sup>c</sup>] much smaller than *E*[R | S], the bias is positive, "
           "and the naive universe **overstates** performance. A favourable bias is the "
           "dangerous kind: nothing in the results invites you to look for it."))

    # ------------------------------------------------------------------ 3.8
    A(("h2", "3.8 How many false discoveries a screener produces"))
    A(("p", "Section 10.1 quotes an expected number of spurious discoveries. That figure "
           "rests on two elementary facts worth stating, because the second is stronger "
           "than people expect."))
    A(("prop", "6 (Probability integral transform)", [
        "Let T be a test statistic with continuous distribution function F under "
        "H<sub>0</sub>, and let p = 1 − F(T) be the associated p-value. Then p is "
        "uniform on [0, 1] under H<sub>0</sub>, so *P*(p ≤ α) = α for every α in [0,1].",
    ]))
    A(("proof", [
        "For u in [0,1], *P*(p ≤ u) = *P*(1 − F(T) ≤ u) = *P*(F(T) ≥ 1 − u). Since F is "
        "continuous, F(T) is uniform on [0,1], hence this equals 1 − (1 − u) = u.",
    ]))
    A(("prop", "7 (Expected false discoveries, under arbitrary dependence)", [
        "Test N null hypotheses, all true, at level α, and let "
        "V = sum over i of *1*{p<sub>i</sub> ≤ α} be the number of rejections. Then",
        ("formula", r"\mathbb{E}[V]=\alpha N,"),
        "**with no independence assumption whatsoever**.",
    ]))
    A(("proof", [
        "By linearity of the expectation and Proposition 6,",
        ("formula", r"\mathbb{E}[V]=\sum_{i=1}^{N}\mathbb{P}(p_i\leq\alpha)=\alpha N ."),
        "Linearity holds for arbitrarily dependent summands, which is the point.",
    ]))
    A(("warn", "The part that does need independence",
       ["Only the **expectation** of V is dependence-free. Its variance is not:",
        "Var(V) = Nα(1−α) + the sum over i ≠ j of Cov(*1*{p<sub>i</sub> ≤ α}, *1*{p<sub>j</sub> ≤ α}), "
        "and the covariance term dominates when the tests share structure.",
        "In our screener the pairs overlap by construction — every pair containing a "
        "given name is correlated with every other pair containing it — so V is far "
        "more dispersed than a binomial count. Phase 2 measures the consequence: one "
        "name holds 16 % to 50 % of the selected pairs in every fold. This is exactly "
        "why the Bonferroni-versus-Benjamini-Hochberg choice of Phase 3 is substantive "
        "rather than cosmetic: the former is valid under arbitrary dependence, the "
        "latter is not."]))
    A(("p", "With N = 465 pairs and α = 5 %, Proposition 7 gives 23 expected spurious "
           "discoveries. The screener finds 47. Roughly half of what looks like a "
           "discovery is, in expectation, noise — and that is the observation Phase 3 "
           "is built to address."))
    return B
