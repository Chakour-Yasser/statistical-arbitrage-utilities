# -*- coding: utf-8 -*-
"""English content for the Phase 1 explanatory document.

Figures are recomputed from the actual data: the PDF cannot drift from the code.
Mirror of scripts/pdf_content.py (French).
"""
from __future__ import annotations
import pandas as pd
from src import config as C
from src.universe import validate_identity, tradable_universe
from scripts.pdf_math_phase1 import blocks as math_blocks


def _facts():
    memb = pd.read_parquet(C.DATA_PROC / "membership_utilities.parquet")
    close = pd.read_parquet(C.DATA_PROC / "prices_adj.parquet")
    return memb, close, validate_identity(memb, close), tradable_universe(memb, close)


def blocks() -> list:
    memb, close, val, tu = _facts()
    n_union, n_trad = memb.shape[1], tu.shape[1]
    n_max, n_min = int(tu.sum(axis=1).max()), int(tu.sum(axis=1).min())
    pairs_max = n_max * (n_max - 1) // 2
    fp_max = 0.05 * pairs_max
    slots_all, slots_kept = int(memb.values.sum()), int(tu.values.sum())
    lost = sorted(set(memb.columns) - set(tu.columns))
    B = []; A = B.append

    # ================================================================== #
    A(("h1", "1. What this document is for", False))
    A(("p", "This document revisits **every decision** made during Phase 1 of the project "
           "and explains it in detail: what was decided, why, what the alternatives were, "
           "and what each choice costs. It is written so you can defend the project at the "
           "whiteboard without notes."))
    A(("p", "Phase 1 produces no signal, no backtest, no P&L. It produces only a **universe "
           "of securities and a price series**. That is why it looks innocuous — and "
           "exactly why it is dangerous: a mistake made here never raises an error. It "
           "produces a backtest that runs, reports a Sharpe ratio, and is wrong. Every "
           "later phase silently inherits its defects."))
    A(("key", "The rule that structures all of Phase 1",
       ["At date *t*, a backtest may use only information genuinely available at date *t*. "
        "Any violation is a **leak** (look-ahead bias), and a leak inflates performance "
        "without ever announcing itself.",
        "The difficulty is that the worst leaks do not hide in the signal. They hide in how "
        "the data was built — that is, here."]))
    A(("p", "Read in order. Sections 2 and 3 set up vocabulary and the mathematics (skip if you are comfortable "
           "with cointegration). Sections 5 to 10 cover the decisions. Section 11 answers "
           "the two questions posed at the end of Phase 1. Section 12 is a glossary."))

    # ================================================================== #
    A(("h1", "2. The vocabulary, with no shortcuts", True))
    A(("h2", "2.1 What a pair is"))
    A(("p", "A *pair* is a couple of stocks (A, B) traded **together and in opposite "
           "directions**: buy A, short B, or the reverse. You never bet on market "
           "direction, only on the **gap** between the two names."))
    A(("p", "That gap is the **spread**. Its simplest definition:"))
    A(("math", "s(t) = log P<sub>A</sub>(t) − β · log P<sub>B</sub>(t)"))
    A(("p", "The coefficient β is the **hedge ratio**. It answers: for one dollar invested "
           "in A, how much of B must be sold to neutralise the shared risk? If β = 1.3, you "
           "short 1.3 dollars of B for every dollar of A."))
    A(("p", "We work in **log-prices**, not raw prices, for two reasons. First, a difference "
           "of log-prices is dimensionless (a log-ratio), hence comparable across pairs "
           "trading at very different price levels. Second, returns are additive in logs, "
           "which makes the P&L computation direct."))
    A(("h2", "2.2 The strategy in one sentence"))
    A(("p", "If the spread is **stationary** — oscillating around a fixed mean rather than "
           "drifting away — then when it moves abnormally far from that mean, we bet it "
           "will come back. Sell the spread when high, buy it when low, close on reversion."))
    A(("p", "The whole strategy therefore rests on **a single assumption**: spread "
           "stationarity. Phase 2 tests it, Phase 3 corrects for the fact that we test it "
           "hundreds of times, and Phase 4 deals with it ceasing to hold without warning."))
    A(("h2", "2.3 Correlation vs cointegration: the distinction that comes up in interviews"))
    A(("p", "This is **the** foundational question on this project. Two series can be "
           "strongly correlated without being cointegrated, and vice versa."))
    A(("p", "**Correlation** is about *returns* — short-horizon variation. Saying A and B "
           "correlate at 0.9 means their daily moves go the same way. It says **nothing** "
           "about the gap between their levels over the long run: two names can rise and "
           "fall together every single day while drifting apart indefinitely."))
    A(("p", "**Cointegration** is about *levels*. It says there exists a β such that "
           "log P<sub>A</sub> − β log P<sub>B</sub> is stationary, even though "
           "log P<sub>A</sub> and log P<sub>B</sub> individually are not."))
    A(("p", "Formally: a stock price behaves like a series **integrated of order 1**, "
           "written I(1) — its difference (the return) is stationary, but its level drifts "
           "without bound. Cointegration is the remarkable case where a linear combination "
           "of two I(1) series is I(0), i.e. stationary."))
    A(("key", "The intuition to give at the whiteboard",
       ["Two cointegrated I(1) series share the **same stochastic trend**. β is the exact "
        "weight that cancels it by subtraction: what remains is stationary noise around the "
        "equilibrium relation.",
        "The classic image: a drunk and their dog. Each walks randomly (each path is I(1), "
        "unpredictable). But the leash bounds the distance between them, and that distance "
        "is stationary. Correlation describes whether they step at the same instant; "
        "cointegration describes the existence of the leash."]))
    A(("p", "Direct consequence for this project: cointegration is not a statistical "
           "coincidence, it requires an **economic mechanism** playing the role of the "
           "leash. That argument, and only that argument, justifies the decision in "
           "section 5."))

    # ================================================================== #
    B.extend(math_blocks())

    # ================================================================== #
    A(("h1", "4. Why Phase 1 deserves a whole document", True))
    A(("p", "Pairs trading has a poor reputation among quantitative funds, for good reason: "
           "it is the exercise where producing a spectacular and entirely false backtest is "
           "easiest. Three mechanisms suffice, and all three play out in Phase 1 or just "
           "after."))
    A(("ol", [
        "**Survivorship bias.** You build the universe from companies that exist today, so "
        "you only ever trade survivors.",
        "**Multiple testing.** You test hundreds of pairs and keep the best, without "
        "correcting for the fact that if you test enough, you always find something.",
        "**Execution or normalisation look-ahead.** You use, to decide at date *t*, a "
        "statistic computed on data after *t*.",
    ]))
    A(("p", "Point 2 is handled in Phase 3, point 3 in Phases 2 and 6. **Point 1 is settled "
           "entirely in Phase 1**, and it is the only one that cannot be fixed afterwards: "
           "if the universe is wrong, everything downstream is contaminated."))
    A(("warn", "What makes these mistakes different",
       ["A normal software bug announces itself: the program stops, an exception is raised, "
        "a test fails. Phase 1 mistakes do none of that.",
        "They produce a backtest that runs normally, smooth curves and a presentable Sharpe "
        "ratio. The only way to catch them is to look for them deliberately, by writing "
        "checks whose sole purpose is to fail."]))

    # ================================================================== #
    A(("h1", "5. Decision 1 — The universe: S&P 500 utilities", True))
    A(("h2", "5.1 First sub-decision: stay inside a single sector"))
    A(("p", "We could have formed pairs across the whole S&P 500. We chose to restrict to a "
           "single GICS sector. Two justifications, of different kinds."))
    A(("h3", "Economic justification"))
    A(("p", "Section 2.3 established that cointegration requires a shared stochastic trend. "
           "Within a sector you can name it: for utilities, it is exposure to long rates "
           "(these are very high-duration assets, bought for yield), the regulatory regime "
           "(tariffs set by commissions on an asset base), and input costs (gas, fuels). "
           "Across sectors that mechanism does not exist — cointegration between a bank and "
           "a semiconductor maker would be a sample artefact."))
    A(("h3", "Statistical justification"))
    A(("p", "The number of pairs grows as N². Across 500 names that is "
           "500 × 499 / 2 = 124,750 tests to correct for. Under a Bonferroni correction the "
           "threshold becomes α/124750 ≈ 4 × 10<sup>-7</sup>: no pair would ever pass, and Phase 3 "
           f"would have nothing to show. Restricting the sector brings the problem to a "
           f"scale where correction is **binding but not destructive** — {pairs_max} tests."))
    A(("p", "This matters for interviews: the sector restriction is not a shortcut to reduce "
           "workload. It is a **reduction of the search space grounded in a prior**, which "
           "is precisely what a serious quantitative researcher does before running a "
           "screener."))
    A(("h2", "5.2 Second sub-decision: which of the eleven sectors"))
    A(("p", "Four criteria were fixed **before** looking at any data, which is the decisive "
           "element: picking the sector after seeing which one gives the best P&L would be "
           "multiple testing at the sector level, and would destroy the project's own "
           "argument."))
    A(("table",
       ["Criterion", "What it requires", "Why"],
       [["Identifiable common factor",
         "Being able to name the economic mechanism linking the names",
         "Without it, any cointegration found is a sample artefact"],
        ["Sufficient size",
         "At least ~25 names, i.e. ≥ 300 pairs",
         "Below that, the multiple-testing demonstration carries no weight"],
        ["Datable, explainable break",
         "At least one episode where the relation breaks, with a known cause",
         "V1 must explain *why* it breaks, not merely observe that it did"],
        ["Execution homogeneity",
         "One market, one currency, one closing time",
         "Avoids illusory predictability from asynchronous closes"]],
       [0.24, 0.36, 0.40],
       "The four sector-selection criteria, fixed before any examination of the data."))
    A(("p", "Utilities satisfy all four. The common factor is the clearest in the entire US "
           f"equity market. The sector holds between {n_min} and {n_max} names depending on "
           "the date. And it offers not one but **three tellable breaks**:"))
    A(("ul", [
        "**March 2020, COVID.** A generalised liquidity shock: every correlation converges "
        "to 1 and long-run relations are temporarily drowned out.",
        "**2022, rate shock.** The rapid rise in rates hits utilities through their "
        "duration, but **unevenly**, depending on leverage and debt structure. The "
        "dispersion of sensitivities breaks relations that had held for years.",
        "**2023-2024, AI repricing.** Data-centre electricity demand abruptly revalues "
        "*merchant* generators — those selling at market prices (VST, CEG, NRG) — while "
        "regulated names (ED, WEC) stay capped by their tariffs. This break is "
        "**structural**: the two groups permanently stop sharing a common trend.",
    ]))
    A(("key", "The 2023-2024 break is the narrative core of V1",
       ["It has everything an interviewer wants: a date, a precise economic mechanism, a "
        "verifiable statistical consequence (β drifts, half-life explodes, ADF p-value "
        "rises), and a **permanent** character — this is not a temporary shock to wait out, "
        "it is a relation that no longer exists and must be exited."]))
    A(("h2", "5.3 What was ruled out, and why"))
    A(("table",
       ["Sector", "What it offered", "Why ruled out"],
       [["Regional banks (KRE)",
         "~90-140 names, so ~4,000 pairs: far more spectacular multiple testing. And March "
         "2023 (SVB, Signature, First Republic) makes survivorship bias the central subject "
         "rather than a footnote.",
         "Prices for bankrupt companies are not obtainable for free. The budget would have "
         "gone into data plumbing instead of methodology."],
        ["Energy E&P (XOP)",
         "Obvious common factor (the oil price) and a spectacular 2020 break, with WTI "
         "going negative in April.",
         "The 2023-2024 M&A wave truncates many series. Heterogeneous leverage makes "
         "cointegration fragile."],
        ["Semiconductors",
         "The 2023 AI break is extremely visible.",
         "Too trending: a real risk of finding almost no cointegrated pair at all, which "
         "would empty Phase 3 of content."]],
       [0.20, 0.42, 0.38]))
    A(("h2", "5.4 The objection you will get"))
    A(("warn", "\"Your spread is just a disguised duration bet\"",
       ["The objection is serious and partly right: if every name reacts to rates, a "
        "long/short between two of them still carries the differential rate sensitivity.",
        "**The right answer is not to deny it, it is to measure it.** The hedge ratio "
        "absorbs the *common* exposure by construction. What must be checked is that the "
        "**residual** — i.e. the P&L — is no longer explained by rates. The test is "
        "explicit: regress daily strategy returns on changes in the US 10-year yield and "
        "show the coefficient is not significant. That is scheduled for Phase 7.",
        "Answering \"I don't know, but here is the test I would run\" is worth far more than "
        "denying the objection."]))

    # ================================================================== #
    A(("h1", "6. Decision 2 — Sample period and time split", True))
    A(("h2", "6.1 Why 2014 → 2026"))
    A(("p", "The period results from a trade-off between two opposing forces."))
    A(("p", "**Lengthening** history increases the number of independent sub-periods, hence "
           "the reliability of conclusions: with three years of data you cannot say "
           "anything about a strategy's stability. **Shortening** history improves data "
           "quality (survivorship worsens as you go back) and relevance (the sector's "
           "structure in 2010 — before the renewables boom, before the Constellation "
           "spin-off — bears little resemblance to today's)."))
    A(("p", "Twelve and a half years puts the cursor where the two effects balance: enough "
           "sub-periods to discuss signal decay, not so many that you test a strategy on a "
           "sector that no longer exists."))
    A(("h2", "6.2 Walk-forward: three years of selection, one year of trading"))
    A(("p", "This is the protocol guaranteeing that pair selection and pair trading never "
           "happen on the same data."))
    A(("code",
       "fold 1 :  SELECTION 2014-2016 (3 yrs)  ->  TRADING 2017 (1 yr)\n"
       "fold 2 :  SELECTION 2015-2017          ->  TRADING 2018\n"
       "fold 3 :  SELECTION 2016-2018          ->  TRADING 2019\n"
       "...\n"
       "fold 9 :  SELECTION 2022-2024          ->  TRADING 2025\n"
       "\n"
       "Each fold: point-in-time universe recomputed, liquidity filter\n"
       "recomputed, cointegration retested, pairs selected -- then FROZEN.\n"
       "The trading window NEVER influences the selection preceding it."))
    A(("p", "Why three years of selection? A cointegration test needs enough points to "
           "separate a genuine equilibrium relation from chance: roughly 750 trading days. "
           "Over one year the augmented Dickey-Fuller test has almost no power — it rarely "
           "rejects non-stationarity even when it is false."))
    A(("p", "Why one year of trading? Beyond that you implicitly assume a relation selected "
           "in 2016 still holds in 2020. That is exactly the assumption V1 disputes. One "
           "year balances re-selection frequency against turnover cost."))
    A(("warn", "The trap not to fall into",
       ["Once in the trading window it is tempting to retest cointegration and keep only "
        "the pairs that \"still work\". That is a **direct reintroduction of look-ahead**: "
        "you would use information from the trading period to decide what to trade during "
        "that same period.",
        "The rule is absolute: **selection is frozen at the end of the in-sample window.** "
        "If a pair degrades during trading, it is the regime-detection mechanism (V1) that "
        "must exit it — and that mechanism, too, looks only at the past."]))

    # ================================================================== #
    A(("h1", "7. Decision 3 — Which prices, exactly", True))
    A(("h2", "7.1 Raw, split-adjusted, or total-return?"))
    A(("p", "Three different price series exist for the same stock, and the choice among "
           "them is not cosmetic."))
    A(("ul", [
        "The **raw price** is what shows on screen. It drops mechanically on a split or an "
        "ex-dividend date.",
        "The **split-adjusted price** corrects share splits.",
        "The **total-return price** corrects splits *and* dividends: it represents the value "
        "of an investment that reinvests its dividends.",
    ]))
    A(("p", "**We chose total-return.** Here is the argument, because it is the kind of "
           "point that immediately separates a serious candidate."))
    A(("h3", "Why it is decisive specifically for utilities"))
    A(("p", "Utilities pay 3 to 4 % dividend yield per year, and crucially that yield is "
           "**widely dispersed** across names. Write δ<sub>A</sub> and δ<sub>B</sub> for the "
           "annual dividend yields of two names. The raw price separates from the "
           "total-return price by a factor growing with time:"))
    A(("math", "log P<sup>raw</sup>(t) ≈ log P<sup>TR</sup>(t) − δ · t"))
    A(("p", "A spread computed on raw prices therefore carries a term "
           "−(δ<sub>A</sub> − β δ<sub>B</sub>) · t: **a deterministic linear drift**. A "
           "series that drifts linearly is not stationary. The ADF test will reject it — and "
           "we would discard **genuinely cointegrated** pairs merely because their dividend "
           "policies differ."))
    A(("p", "Magnitude measured on the actual universe: implied dividend yield ranges from "
           "**1.07 % (PCG) to 6.39 % (OKE)**, a 5.3-point spread between two names in the "
           "same sector. A 2-point gap already injects about 6 % of drift over a three-year "
           "selection window — comparable to the amplitude of the spread itself. The effect "
           "is not marginal."))
    A(("h3", "The experiment, and what it actually licenses us to conclude"))
    A(("p", "We tested all 465 pairs on a representative in-sample window (2016-2018), once "
           "on total-return prices and once on raw prices, and compared the resulting "
           "**selections**:"))
    A(("code",
       "465 pairs tested on 2016-2018 (Engle-Granger, 5 % threshold)\n"
       "\n"
       "  significant on TOTAL-RETURN : 112\n"
       "  significant on RAW PRICES   : 137\n"
       "  disagreement : 51 pairs (11 %)  --  13 lost, 38 created\n"
       "\n"
       "  but most of it is p-value noise near the threshold:\n"
       "     lost    : 11/13 marginal (85 %)\n"
       "     created : 26/38 marginal (68 %)\n"
       "\n"
       "  DECISIVE FLIPS (p > 0.15 one side, < 0.05 the other)\n"
       "     pairs MANUFACTURED by raw prices : 16\n"
       "     pairs DESTROYED    by raw prices :  1"))
    A(("p", "**This result must be read carefully**, and that is an important methodological "
           "point. The tempting conclusion — \"raw prices manufacture 38 false pairs\" — is "
           "not defensible: 68 % of those pairs have a p-value in [0.03, 0.08]. They flip "
           "because a p-value near the threshold is noisy, not because dividend drift "
           "changed anything substantial. Counting them as selection errors would be "
           "dishonest."))
    A(("p", "**What survives is the asymmetry of decisive flips: 16 against 1.** Sixteen "
           "pairs with no equilibrium relation on the economically correct prices become "
           "decisively significant once raw prices are used; only one flips the other way. "
           "The correlation between drift magnitude and p-value gap is 0.29 — the mechanism "
           "is there, but it explains only part of the phenomenon."))
    A(("p", "**The mechanism, for the decisive cases.** If two names diverge economically but "
           "the one rising faster also pays the larger dividend, the drift **cancels** the "
           "divergence on raw prices. The spread becomes artificially flat, hence "
           "\"stationary\" as far as the test is concerned."))
    A(("key", "The formulation to keep",
       ["The two price series produce selections differing on 11 % of pairs. Saying \"38 "
        "false pairs\" would overstate it — most is threshold noise. But the **16 against "
        "1** asymmetry is robust: raw prices manufacture apparent cointegration far more "
        "often than they destroy it.",
        "The choice of price series is not a data-preparation detail: **it is a decision "
        "that changes the selection itself.**",
        "Being able to separate \"what I would like to conclude\" from \"what my data "
        "licenses\" is precisely what a fund probes for in interviews."]))
    A(("p", "Total-return is also the right **economic** object: a long/short position "
           "genuinely collects the dividend on the long leg and pays it on the short leg. "
           "The simulated P&L must reflect that."))
    A(("warn", "The residual leak you must be able to name",
       ["Adjusted prices are **retro-adjusted**. Every new dividend paid retroactively "
        "modifies the entire historical series. Put differently: the 2018 series as I "
        "download it in 2026 is not the one a trader observed in 2018.",
        "That is formally a leak: the series embeds later information. Its effect is "
        "second-order here, since both legs are adjusted in the same direction and the "
        "position is long/short. But it exists, and **naming it spontaneously** is exactly "
        "what separates someone who has thought about it from someone who followed a "
        "tutorial."]))
    A(("h2", "7.2 Why missing prices are never filled"))
    A(("p", "When a series has a gap — partial holiday, trading halt, vendor failure — the "
           "universal reflex is *forward-fill*: carry the last known price. We do not. But "
           "the reason is not the one you read everywhere, and this section deserves close "
           "attention because it contains **a mistake made and then corrected**."))
    A(("h3", "The usual argument — and why it is wrong"))
    A(("p", "The argument found in most tutorials, and which I advanced myself before "
           "testing it, runs: carrying a price forward creates a zero return followed by a "
           "catch-up, hence the sequence \"no move, then compensating move\", which is the "
           "signature of mean reversion. Forward-fill would therefore manufacture the very "
           "signal the strategy looks for."))
    A(("p", "**That is false, and simulation shows it unambiguously.** Simulate a pure "
           "random walk — hence no mean reversion whatsoever — introduce 5 % missing days, "
           "forward-fill, and test:"))
    A(("code",
       "Pure random walk, 400 simulations, 5 % missing days\n"
       "\n"
       "  ADF rejection rate at 5 %   without ffill :   6.8 %\n"
       "  ADF rejection rate at 5 %   with ffill    :   6.5 %\n"
       "  lag-1 autocorr. of diffs    without ffill : -0.0016\n"
       "  lag-1 autocorr. of diffs    with ffill    : -0.0017"))
    A(("p", "No effect. And on a genuinely mean-reverting spread (Ornstein-Uhlenbeck with a "
           "15-day half-life), forward-fill biases neither the estimated half-life (+0.2 %) "
           "nor the number of signals generated (67.6 versus 67.8)."))
    A(("p", "The reason is simple once seen: forward-fill **redistributes** the increments "
           "without modifying them. The missing day carries a zero return, the next day "
           "carries the sum of both increments. The total is unchanged and, since increments "
           "are independent, the covariance between consecutive returns stays zero. There is "
           "no \"compensating catch-up\": there is a deferral."))
    A(("warn", "What this means for interviews",
       ["If you advance the \"manufactured mean reversion\" argument to someone who has run "
        "the simulation, you lose all credibility on the rest of the project.",
        "The correct argument is simpler and more solid — and being able to say \"I believed "
        "it, I tested it, it was false, here is what is true\" is worth more than any "
        "memorised justification."]))
    A(("h3", "The correct argument: tradability"))
    A(("p", "The real problem with forward-fill is not statistical, it is **operational**. On "
           "an Ornstein-Uhlenbeck spread with 5 % missing days:"))
    A(("code",
       "OU spread (15-day half-life), 5 % missing days, 300 simulations\n"
       "\n"
       "  estimated half-life  without ffill : 13.95 d\n"
       "  estimated half-life  with ffill    : 13.98 d     (bias +0.2 %)\n"
       "  signals |z| > 2      without ffill : 67.8\n"
       "  signals |z| > 2      with ffill    : 67.6\n"
       "\n"
       "  signals landing on a NON-TRADING day : 2.6 per series\n"
       "  i.e. 3.8 % of signals, at a price that never existed"))
    A(("key", "The justification to keep",
       ["Forward-fill does not corrupt the spread's statistics: it corrupts **execution**. "
        "About 4 % of entry signals land on a day the name did not trade. The backtest then "
        "opens a position at a price that never existed, and books P&L on an impossible "
        "trade.",
        "This is an execution leak, of the same family as trading at the same day's close on "
        "which the signal was computed — the subject of Phase 6.",
        "**Decision: no forward-fill.** Missing values stay NaN and are handled explicitly, "
        "pair by pair, at test time."]))
    A(("h2", "7.3 Cleaning, and why it is deliberately minimal"))
    A(("p", "Cleaning reduces to two rules, chosen to be **non-destructive** — aggressive "
           "cleaning is itself a source of bias."))
    A(("ol", [
        "**Drop entirely empty columns.** These are tickers the vendor no longer serves, "
        "typically delisted companies. They are not erased from the project's memory: they "
        "are accounted for in the survivorship report (section 10).",
        "**Drop dates where fewer than 80 % of *live* names trade.** These are partial "
        "holidays or data incidents, not sessions. The subtle point is the denominator: we "
        "compare against the number of names **alive on that date**, not the total column "
        "count. Otherwise a name listed in 2022 would fail the criterion on every session "
        "from 2014 to 2021 and we would delete eight years of valid data.",
    ]))

    # ================================================================== #
    A(("h1", "8. Decision 4 — The point-in-time universe", True))
    A(("h2", "8.1 The problem: why survivorship is a leak"))
    A(("p", "Taking an index's **current** constituents and pulling their history is the "
           "method used in the vast majority of student projects. It is wrong, and the "
           "reason runs deeper than \"a bias\"."))
    A(("p", "Selecting today the companies in the S&P 500 means using information dated "
           "today — **the fact that these companies still exist and are large enough to be "
           "in the index** — to decide what to trade in 2015. That is structurally identical "
           "to look-ahead on the signal. The only difference is that the leaking variable is "
           "not a future price: it is **survival**."))
    A(("p", "The effect is systematic and always in the same direction: you exclude exactly "
           "those companies that went bankrupt, were acquired, or fell apart — the cases "
           "where a cointegration relation breaks violently. You measure performance on a "
           "sample stripped of its worst scenarios."))
    A(("h2", "8.2 Two possible methods, and why we switched"))
    A(("p", "**Method A — reconstruct from changes.** Start from today's list and walk back "
           "in time inverting each addition and removal recorded in a historical table. That "
           "was the original plan."))
    A(("p", "That table no longer exists on the public source, so we switched to a "
           "**Method B, which turns out to be strictly superior**: read the page's revision "
           "history. The page listing S&P 500 constituents has been continuously edited for "
           "more than fifteen years, and every past version remains accessible. We therefore "
           "retrieve the page **as it existed** on each selection date."))
    A(("table",
       ["", "Method A — change table", "Method B — revisions (chosen)"],
       [["Principle", "Invert changes backwards from today",
         "Read the page as it stood on the target date"],
        ["GICS sector", "Lost for exited companies: the table records only the ticker, the "
         "sector would have to be reassigned by hand",
         "**Directly observed**, as classified at the time"],
        ["Reclassifications", "Invisible: a name moved between sectors is not an index change",
         "**Captured automatically** (the ONEOK case, section 9.4)"],
        ["Robustness", "One error in the table propagates to every earlier date",
         "Each date is read independently: an error stays local"]],
       [0.16, 0.42, 0.42],
       "Switching to Method B was not a fallback: it yields membership *and* point-in-time "
       "sector, both directly observed."))
    A(("h2", "8.3 The technical detail that guarantees no leak"))
    A(("p", "The entire no-leak property rests on one request parameter:"))
    A(("code",
       'params = {\n'
       '    "rvstart": timestamp,   # the selection date\n'
       '    "rvdir":   "older",     # <-- walk BACK in time from that date\n'
       '    "rvlimit": 1,\n'
       '}'))
    A(("p", "`rvdir=\"older\"` asks for the last revision published **before** the selection "
           "date. With `\"newer\"` you would get the first revision published *after* — a "
           "page that may already reflect index changes later than the decision. **One word "
           "separates a correct reconstruction from a leak.**"))
    A(("p", "The observed gap between selection date and the revision used is **2 days "
           "median, 19 days maximum**. That gap always runs conservative: the page used is "
           "slightly *behind* reality, never ahead."))
    A(("h2", "8.4 Three implementation sub-decisions"))
    A(("h3", "Semi-annual cadence"))
    A(("p", "We snapshot the index every six months. Since walk-forward re-selection is "
           "annual, each selection date has a snapshot at most six months old. Going monthly "
           "would multiply requests sixfold for precision the strategy cannot use: it "
           "rebalances once a year."))
    A(("h3", "Systematic disk cache"))
    A(("p", "Every downloaded revision is written to disk. This is not an optimisation, it "
           "is a **reproducibility requirement**: without a cache, universe construction "
           "depends on a web page editable at any moment, and even retroactively. Two runs a "
           "month apart could yield two different universes, hence two different backtests, "
           "with no code change whatsoever."))
    A(("h3", "Failing loudly on a failed request"))
    A(("p", "If a request fails repeatedly the program raises and stops. The temptation "
           "would be to skip the missing snapshot and continue. That would be exactly the "
           "kind of silent degradation the whole of Phase 1 is built against: a universe "
           "truncated over one period, with nothing flagging it."))

    # ================================================================== #
    A(("h1", "9. Decision 5 — Ticker identity checking", True))
    A(("h2", "9.1 What we found"))
    A(("p", "Once the point-in-time universe was built, we downloaded prices for the "
           f"{n_union} tickers. Inspecting the start dates revealed a problem we had not "
           "anticipated and which would have invalidated the entire project."))
    A(("table",
       ["Ticker", "Company in the index", "Exit", "Data served from", "What it actually is"],
       [["`NU`", "Northeast Utilities", "2015", "9 December 2021",
         "**Nu Holdings** — Brazilian neobank, IPO late 2021"],
        ["`POM`", "Pepco Holdings", "2016 (acquired by Exelon)", "8 October 2025",
         "An unrelated company, listed since 2025"],
        ["`TE`", "TECO Energy", "2016 (acquired by Emera)", "10 January 2020",
         "An unrelated company"],
        ["`TEG`", "Integrys Energy", "2015 (acquired by WEC)", "22 December 2015",
         "An unrelated company, ticker reused months later"]],
       [0.09, 0.24, 0.20, 0.20, 0.27],
       "The four ticker-recycling cases detected in the universe."))
    A(("p", "Tickers are not durable identifiers. When a company disappears its symbol "
           "returns to the pool and can be reassigned. Three letters do not identify a "
           "company — they identify a slot on a market at a point in time."))
    A(("warn", "What would have happened without this check",
       ["The backtest would have treated the prices of a **Brazilian neobank** as those of a "
        "regulated US utility. It would have tested cointegration between Nu Holdings and "
        "Duke Energy, possibly found it significant over some window, and opened positions.",
        "No exception raised. No test failed. The P&L curve perfectly smooth. And in the "
        "interview you would have confidently defended an entirely false result.",
        "This is the exact illustration of section 4's thesis: **the mistakes that matter do "
        "not raise errors.**"]))
    A(("h2", "9.2 The detection rule"))
    A(("p", "The idea: we hold two independent pieces of information about each ticker. On "
           "one side its **membership window**, from the point-in-time reconstruction. On "
           "the other its **quotation period**, from the price vendor. If the ticker denotes "
           "the same company in both sources, the two intervals must overlap."))
    A(("math", "overlap = |[first quote, last quote] ∩ [index entry, index exit]| "
               "/ |[index entry, index exit]|"))
    A(("ul", [
        "**near-zero** overlap → the ticker denotes another company: `RECYCLED`",
        "**partial** overlap (< 50 %) → truncated history, manual review: `SUSPECT`",
        "**substantial** overlap → identity consistent: `OK`",
    ]))
    A(("p", "The check is **blocking**: tickers flagged `RECYCLED`, `SUSPECT` or `NO_DATA` "
           "are removed from the tradable universe, and the exclusion list is written to an "
           "auditable report."))
    A(("h2", "9.3 The two bugs found while writing this check"))
    A(("p", "They are worth telling, because they show that even an anti-error check must "
           "itself be verified."))
    A(("h3", "Bug 1 — the ONEOK false positive"))
    A(("p", "The first version flagged `OKE` as recycled. But ONEOK is the same company, "
           "with a complete history. The error came from the membership-window definition: "
           "`[first member snapshot, last member snapshot]`. ONEOK appears in only **one** "
           "snapshot — GICS reclassified it from *Utilities* to *Energy* in early 2014 — so "
           "its window had zero length and the overlap divided by zero."))
    A(("p", "Fix: being a member at date *t* means being one at least until the next "
           "snapshot, so the window is widened by six months on the right. The ONEOK case is "
           "interesting in itself: it is a **sector reclassification**, which Method B "
           "captures automatically while Method A would have missed it entirely."))
    A(("h3", "Bug 2 — the exact-zero threshold"))
    A(("p", "The initial rule flagged as recycled an overlap of **exactly** zero. `TEG` has "
           "an overlap of 0.014: Integrys was acquired in June 2015 and the ticker "
           "reassigned by December, so the two intervals overlap by ten days. With a "
           "zero threshold, TEG slipped through."))
    A(("p", "Fix: threshold at 5 %. The general lesson is that a binary test resting on "
           "exact equality is fragile — you need a margin, and you need to have looked at "
           "the data to calibrate it."))
    A(("h2", "9.4 The blind spot that remains"))
    A(("p", "The check compares two intervals. It would therefore poorly detect a recycling "
           "occurring **during** the membership window — the overlap would stay high and the "
           "status would read `OK`. That case is not observed here, and is unlikely in "
           "practice (a ticker is not reassigned instantly), but it is documented rather "
           "than passed over."))
    A(("p", "A genuinely robust check would use a durable identifier — CUSIP, SEDOL, or "
           "CRSP's PERMNO — which does not change when the ticker does. That is exactly what "
           "professional databases provide, and one reason they are expensive."))

    # ================================================================== #
    A(("h1", "10. Phase 1 results", True))
    A(("h2", "10.1 The numbers"))
    A(("table",
       ["Quantity", "Value", "What it is for"],
       [["Tickers in raw union", f"{n_union}", "Every name that belonged to the sector over the period"],
        ["Tradable tickers", f"**{n_trad}**", "After the identity check"],
        ["Sector size by date", f"{n_min} to {n_max}", "Varies with index entries and exits"],
        ["Maximum testable pairs", f"**{pairs_max}**", "N(N−1)/2 — Phase 3's multiple-testing load"],
        ["Expected false positives at α = 5 %", f"**≈ {fp_max:.0f}**", "Phase 3's opening figure"],
        ["Trading sessions", f"{len(close)}", f"{close.index[0]:%d/%m/%Y} → {close.index[-1]:%d/%m/%Y}"],
        ["Index snapshots", f"{len(memb)}", "Semi-annual, 2-day median lag"]],
       [0.36, 0.16, 0.48]))
    A(("key", "The figure to remember for Phase 3",
       [f"With {n_max} names you test {pairs_max} pairs. Under the null — no pair is truly "
        f"cointegrated — a 5 % test would still declare **≈ {fp_max:.0f} significant**, by "
        "pure chance.",
        "Phase 3's entire argument consists of comparing that number to the pairs actually "
        "found significant. If you find 30, you have found almost nothing."]))
    A(("h2", "10.2 The PG&E case: what point-in-time actually buys"))
    A(("p", "The reconstruction shows `PCG` is **out of the index from July 2019 to July "
           "2022**: PG&E filed for Chapter 11 in January 2019 after the California "
           "wildfires, was removed from the S&P 500, then readmitted after emerging."))
    A(("p", "Since PCG is a member **today**, a universe built on current constituents would "
           "include it throughout, and would therefore trade it **during its bankruptcy** — "
           "implicitly knowing it came out alive and its price eventually recovered. That is "
           "look-ahead in its purest form, on one of the sector's most violent episodes."))
    A(("p", "Point-in-time reconstruction excludes it automatically, with no rule written "
           "for this case. That is the sign we fixed the cause and not the symptom."))
    A(("h2", "10.3 Residual survivorship, quantified and signed"))
    A(("p", f"Of {n_union} tickers in the union, **{len(lost)} are excluded**: "
           f"{', '.join('`'+t+'`' for t in lost)}."))
    A(("table",
       ["Ticker", "Status", "What actually happened"],
       [["`GAS`", "No data", "AGL Resources, acquired by Southern Company (2016). The vendor serves nothing: outright loss."],
        ["`NU`", "Recycled", "Northeast Utilities, **renamed Eversource (ES)** in 2015. Only an apparent loss: the `ES` series goes back to 2014 and carries the full history."],
        ["`POM`", "Recycled", "Pepco Holdings, acquired by Exelon (2016)."],
        ["`TE`", "Recycled", "TECO Energy, acquired by Emera (2016)."],
        ["`TEG`", "Recycled", "Integrys Energy, acquired by WEC (2015)."]],
       [0.10, 0.14, 0.76]))
    A(("p", f"By volume: {slots_all} membership slots (snapshot × ticker) in the raw "
           f"universe, {slots_kept} retained, i.e. "
           f"**{100*(slots_all-slots_kept)/slots_all:.1f} % lost**. Losses are **entirely "
           "concentrated in 2014-2016** and **zero from January 2017 onward**. At the worst "
           "snapshot, 5 names are missing out of 31, roughly 30 % of potential pairs "
           "untestable."))
    A(("h3", "The direction of the bias — the most important point in this section"))
    A(("p", "All five lost names left the index **by acquisition**, none by bankruptcy. And "
           "a takeover is precisely the event that breaks a cointegration relation most "
           "brutally: the target's price jumps to the offer price, then freezes, permanently "
           "decorrelated from its sector. A short position on the target at announcement "
           "takes an instant loss that **never** reverts."))
    A(("key", "The formulation to keep",
       ["Excluding these five names removes extreme, irreversible loss scenarios from the "
        "sample. **The residual bias therefore overstates the strategy's performance.**",
        "This is the most dangerous case: a favourable bias never announces itself. You have "
        "no reason to look for it, because the results look good.",
        "Practical consequence: the first fold (selection 2014-2016) is the most exposed. "
        "Honesty means reporting its result **separately** rather than silently aggregating "
        "it with the others."]))
    A(("h2", "10.4 Two grey areas, acknowledged"))
    A(("ul", [
        "**EVRG.** Evergy was formed in 2018 from the Westar / Great Plains merger. The "
        "vendor serves history back to 2014: those are the predecessor's prices. Defensible "
        "— the predecessor is the relevant economic object — but worth knowing if an "
        "interviewer digs.",
        "**SCG.** SCANA shows an overlap of 0.69: served history starts in July 2015 while "
        "the company was a member from 2014. History truncated, identity consistent. Kept, "
        "with less data than the others.",
    ]))

    # ================================================================== #
    A(("h1", "11. What was deliberately deferred", True))
    A(("h2", "11.1 The liquidity filter"))
    A(("p", "The plan calls for a universe of **liquid** stocks. Yet no liquidity filter was "
           "applied in Phase 1. That is deliberate, and the reason is a leak."))
    A(("p", "Filtering now would mean computing median volume over the **whole** 2014-2026 "
           "period and keeping names above a threshold. You would then be using, to decide "
           "what to trade in 2015, the information \"this name will still be liquid in "
           "2025\". That information is strongly correlated with the firm's survival and "
           "success: it is survivorship bias dressed up as a technical criterion."))
    A(("p", "**The filter must be recomputed at each fold, on the in-sample window only**, "
           "exactly like pair selection. It therefore arrives in Phase 3, where selection "
           "itself is built. That is why Phase 1 ends on an unfiltered universe — it is not "
           "an oversight."))
    A(("h2", "11.2 Guardrails currently in place"))
    A(("table",
       ["Guardrail", "Against what", "Where"],
       [["Point-in-time membership", "Survivorship in the composition", "`src/universe.py`"],
        ["Blocking identity check", "Ticker recycling", "`validate_identity`"],
        ["Total-return prices", "Dividend drift in the spread", "`src/data.py`"],
        ["No forward-fill", "Signals on non-trading days", "`src/data.py`"],
        ["\"Live names\" denominator", "Deleting valid sessions", "`_clean`"],
        ["Disk cache", "Non-reproducible universe", "`data/raw/wiki/`"],
        ["Loud failure on failed request", "Silently truncated universe", "`_get`"],
        ["Dated universe decision", "Sector-level data snooping", "`docs/01_universe_decision.md`"],
        ["Deferred liquidity filter", "Survivorship disguised as a technical criterion", "Phase 3"]],
       [0.30, 0.42, 0.28]))

    # ================================================================== #
    A(("h1", "12. The two questions — detailed answers", True))
    A(("h2", "Question 1 — The identity check's blind spot"))
    A(("p", "*\"Which recycling case would slip through the rule? And by what concrete "
           "mechanism would the P&L become wrong without any test catching it?\"*"))
    A(("h3", "The case that slips through"))
    A(("p", "A recycling occurring **during** the membership window. The rule measures "
           "overlap between quotation period and membership window; if company X is a member "
           "from 2014 to 2020 and disappears in 2017, a ticker reassigned in 2018 to a "
           "company Y would produce a continuous series from 2014 to 2020, perfectly "
           "covering the window. Status: `OK`."))
    A(("h3", "How the P&L gets contaminated"))
    A(("p", "The concatenated series would show a **level discontinuity** at the switch "
           "date: the price jumps from X's to Y's, with no economic link. Three cascading "
           "consequences:"))
    A(("ol", [
        "The in-sample cointegration test would see a spread containing a jump. Depending on "
        "where the jump sits, the test would **reject** wrongly (the jump resembles a unit "
        "root) or **accept** wrongly if the jump is followed by a stable regime resembling "
        "mean reversion.",
        "If the pair gets selected, the z-score at the jump would reach an extreme value — "
        "typically |z| > 5. The strategy would open a **maximum** position on what it reads "
        "as an exceptional opportunity.",
        "That position never unwinds, because the jump is not a temporary deviation but a "
        "change of object. The P&L books a massive, purely fictitious gain or loss.",
    ]))
    A(("p", "**Why no test catches it:** every Phase 7 diagnostic (Sharpe, drawdown, hit "
           "rate, turnover) is an *aggregate* statistic computed **on that same corrupted "
           "series**. They are mutually consistent and consistent with the data. Nothing is "
           "incoherent — the data is simply wrong. No amount of internal checking can detect "
           "an error that affects the system's input."))
    A(("p", "**What would catch it:** a durable identifier (CUSIP, SEDOL, PERMNO), or "
           "failing that a **return-discontinuity check** — a daily return of several hundred "
           "percent not matched by a known split is the signature of an entity change. That "
           "is a cheap check and would be reasonable to add."))
    A(("h2", "Question 2 — The PG&E objection"))
    A(("p", "*\"A real fund is not constrained by S&P 500 membership; it could have traded "
           "PG&E during the bankruptcy. By excluding it you have not corrected a bias, you "
           "have introduced another.\"*"))
    A(("p", "The objection is right on the facts and wrong in its conclusion. Concede the "
           "first point before answering."))
    A(("h3", "What index membership actually stands in for"))
    A(("p", "We do not use S&P 500 membership because a mandate requires it. We use it as an "
           "**observable, point-in-time proxy** for a set of properties we genuinely want to "
           "impose and which are hard to measure directly: sufficient market cap, sufficient "
           "free float, liquidity allowing entry and exit without impact, and the practical "
           "ability to **sell short** — that is, the existence of a borrowable inventory at "
           "a reasonable cost."))
    A(("p", "That last point is decisive. Pairs trading requires a short leg. During a "
           "bankruptcy proceeding a stock becomes extremely hard and expensive to borrow: "
           "borrow costs can reach tens of percent annualised, and the lender can recall at "
           "any time, forcing a buy-in at the worst moment (a short squeeze). A backtest "
           "that ignores those costs and prices them at zero **massively overstates** "
           "performance."))
    A(("h3", "The answer in three steps"))
    A(("ol", [
        "**Concede.** Yes, index membership is not a fund's real constraint. It is an "
        "approximation.",
        "**Turn the argument around.** It is a **conservative** approximation. It excludes "
        "precisely the periods where execution — especially stock borrow — is costliest and "
        "least certain. Including PG&E during its bankruptcy with normal assumed financing "
        "costs would not be more realistic: it would be **more optimistic**, pricing the "
        "market's most expensive risk at zero.",
        "**Name the real fix.** The rigorous solution is not to include PG&E carelessly, but "
        "to model borrow cost and availability explicitly. That requires securities-lending "
        "data, which is paid. Absent it, exclusion is the conservative choice, and I "
        "document it.",
    ]))
    A(("key", "The general principle behind this answer",
       ["Faced with a well-founded methodological objection, the bad answer is to deny; the "
        "mediocre answer is to concede; **the good answer is to show which way the choice "
        "biases the result, and to systematically prefer the direction unfavourable to the "
        "strategy.**",
        "That is the guiding line of all of Phase 1, and the message the project as a whole "
        "must convey."]))

    # ================================================================== #
    A(("h1", "13. Glossary", True))
    A(("table",
       ["Term", "Definition"],
       [["Spread", "Gap between two log-prices weighted by the hedge ratio: log P_A − β log P_B."],
        ["Hedge ratio (β)", "Weight of the second leg. Answers: how much of B to sell per dollar of A."],
        ["I(0) / I(1)", "Stationary series / series whose first difference alone is stationary. A stock price is typically I(1)."],
        ["Stationarity", "Property of a series whose law does not depend on time: it oscillates around a fixed mean rather than drifting."],
        ["Cointegration", "Existence of an I(0) linear combination of I(1) series. Formalises a long-run equilibrium relation."],
        ["ADF", "Augmented Dickey-Fuller test. Tests for a unit root; applied to the residual to test its stationarity."],
        ["Look-ahead", "Using, to decide at date t, information unavailable at t. Inflates performance without announcing itself."],
        ["Survivorship bias", "A form of look-ahead on existence: keeping only companies that survived to today."],
        ["Point-in-time", "Data reconstructed as it was observable at the date considered, not as we know it today."],
        ["Walk-forward", "Validation protocol where selection and evaluation follow each other in time, never overlapping."],
        ["In-sample / Out-of-sample", "Selection and estimation period / test period, later and strictly disjoint."],
        ["Total-return", "Price series adjusted for splits and reinvested dividends."],
        ["Forward-fill", "Filling a missing value with the last known value. Rejected here (section 7.2)."],
        ["Merchant generator", "Power producer selling at market prices, as opposed to a regulated utility with set tariffs."],
        ["GICS", "Standard sector taxonomy. Utilities is the level-1 sector used here."],
        ["Ticker recycling", "Reassignment of a market symbol to another company after the first one disappears."]],
       [0.22, 0.78]))
    return B
