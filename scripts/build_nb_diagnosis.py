# -*- coding: utf-8 -*-
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "notebooks" / "01_why_pairs_trading_fails.ipynb"
OUT.parent.mkdir(exist_ok=True)
C=[]; M=lambda s: C.append(new_markdown_cell(s.strip())); K=lambda s: C.append(new_code_cell(s.strip()))

M(r"""
# Why pairs trading fails on this universe

A cointegration pairs book on S&P 500 utilities returns a net Sharpe of -0.26 across ten
out-of-sample years. This notebook locates the failure rather than just reporting it.

The short version: the selection rule has no predictive power. A pair cointegrated at the 5 %
level over three years has a 10.8 % chance of still being cointegrated the next year, and pairs
the screen rejected come in at 22.3 %. Everything downstream inherits that, which is why
multiple-testing correction, regime detection and Johansen baskets all fail the same way.

| Section | Content |
|---|---|
| 1 | Discoveries against what chance produces |
| 2 | Performance by selection rule |
| 3 | Regime detection, and why it does not discriminate |
| 4 | The persistence measurement |
| 5 | The equilibrium level moves too |
| 6 | What this rules out |
""")

K(r"""
import sys, warnings, pickle, itertools
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd()/"src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT)); warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from src import config as C
from src.universe import tradable_universe
from src.cointegration import screen_pairs, passes_quality, PairResult, spread_from_beta
from src.selection import walk_forward_folds, liquidity_filter
from src.pipeline import _universe_at, _COLS
from src.backtest import metrics

BLUE, ORANGE, RED, GREY = "#2a78d6", "#eb6834", "#e34948", "#5b6470"
INK, GRID = "#16191d", "#dfe3e8"
plt.rcParams.update({
    "figure.figsize": (11, 3.8), "figure.dpi": 110,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": GRID, "axes.labelcolor": GREY, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 12, "xtick.color": GREY, "ytick.color": GREY,
    "grid.color": GRID, "axes.grid": True, "axes.grid.axis": "y",
    "legend.frameon": False, "font.size": 10, "lines.linewidth": 2,
})

close = pd.read_parquet(C.DATA_PROC/"prices_adj.parquet")
dv    = pd.read_parquet(C.DATA_PROC/"dollar_volume.parquet")
memb  = tradable_universe(pd.read_parquet(C.DATA_PROC/"membership_utilities.parquet"), close)
wf    = pickle.load(open(C.DATA_PROC/"walkforward.pkl","rb"))
lp, rets = np.log(close), close.pct_change()
folds = walk_forward_folds(close.index, C.START, C.END)
print(f"{int(memb.any().sum())} tradable names, {len(folds)} walk-forward folds, "
      f"{close.index[0]:%Y-%m} to {close.index[-1]:%Y-%m}")
""")

M(r"""
## 1. Discoveries against what chance produces

Under the null that no pair is cointegrated, a test at level alpha still declares alpha times N
pairs significant. That is true whatever the dependence between the tests, because expectation is
linear. Comparing the count found to that number is the cheapest available sanity check.
""")

K(r"""
tbl = wf["no_regime"]["table"]
x = np.arange(len(tbl)); w = 0.38
fig, ax = plt.subplots()
ax.bar(x - w/2, tbl.found if "found" in tbl else tbl.n_naive, w, color=BLUE, zorder=3,
       label="pairs found significant")
exp = 0.05 * tbl.n_tests
ax.bar(x + w/2, exp, w, color=GREY, zorder=3, label="expected under the null")
ax.set_xticks(x); ax.set_xticklabels(tbl.fold, rotation=30, ha="right")
ax.set_ylabel("pairs"); ax.set_title("Significant pairs against chance alone, by in-sample window")
for i, r in tbl.reset_index().iterrows():
    if (r.n_naive - 0.05*r.n_tests) <= 2:
        ax.annotate("nothing\nbut noise", xy=(i, max(r.n_naive, 0.05*r.n_tests)+3),
                    ha="center", fontsize=8.5, color=RED, fontweight="bold")
ax.legend(loc="upper left", ncol=2)
plt.tight_layout(); plt.show()
print(tbl.assign(expected=exp.round(0), excess=(tbl.n_naive-exp).round(0))
      [["fold","n_tests","n_naive","expected","excess"]].to_string(index=False))
""")

M(r"""
Two folds find nothing beyond chance. The 2019-2021 and 2020-2022 spikes sit on the COVID shock,
where a common shock makes everything co-move and manufactures spurious cointegration, so they are
a caution rather than a discovery. Across the folds the ratio of found to expected rarely exceeds
two, meaning roughly half of what looks like a discovery is noise in expectation.
""")

M(r"""
## 2. Performance by selection rule

Bonferroni controls the probability of any false discovery and is valid under arbitrary
dependence. Benjamini-Hochberg controls the expected proportion of false discoveries and assumes
independence or positive regression dependence. Benjamini-Yekutieli restores validity under
arbitrary dependence at the cost of dividing the level by the harmonic number.
""")

K(r"""
rows = []
for tag in ("no_regime", "regime"):
    for m in ("naive", "bonferroni", "bh", "by"):
        cs = wf[tag]["curves"].get(m)
        if cs is None: continue
        mm = metrics(cs["net"])
        z = wf[tag]["table"][f"n_{m}"]
        rows.append({"regime exit": tag == "regime", "rule": m,
                     "folds with 0 pairs": int((z == 0).sum()), "pairs": int(z.sum()),
                     "net Sharpe": round(mm["sharpe"], 2), "max DD": round(mm["max_dd"], 3)})
res = pd.DataFrame(rows)
print(res[~res["regime exit"]].drop(columns="regime exit").to_string(index=False))

fig, ax = plt.subplots(figsize=(7.5, 3.4))
sub = res[~res["regime exit"]]
ax.bar(sub["rule"], sub["net Sharpe"], color=[BLUE if v > -0.2 else RED for v in sub["net Sharpe"]],
       width=.55, zorder=3)
ax.axhline(0, color=GREY, lw=1)
ax.set_ylabel("net Sharpe, 5 bp"); ax.set_title("No selection rule produces a positive Sharpe")
for i, v in enumerate(sub["net Sharpe"]):
    ax.text(i, v - 0.05 if v < 0 else v + 0.02, f"{v:+.2f}", ha="center", fontsize=10, color=INK)
plt.tight_layout(); plt.show()
""")

M(r"""
The correction does not turn a losing book into a winning one. What it does is make the absence of
signal visible: on roughly 400 tests the Bonferroni threshold is 1.2e-4 and admits nothing in six
of the ten folds.
""")

M(r"""
## 3. Regime detection, and why it does not discriminate

Four trailing statistics were built to detect a relation breaking down: the rolling cointegration
p-value, drift in the hedge ratio, an exploding half-life, and time since the spread last visited
its mean. All are computed on backward windows, so the flag at date t uses only data up to t.

The first version used a 250-day window and fired on 95 % of days. A 250-day Engle-Granger test has
almost no power, so a p-value above 0.10 is the normal state even for a genuine pair. With a
500-day window the flags fire on 18.6 % of days.

The question is not the firing rate but whether flagged days are actually worse.
""")

K(r"""
bk = wf["regime"]["curves"]["naive"]
flagged = bk["broken_share"] > 0.5
net = bk["net"]
share_days = flagged.mean()
loss = net[net < 0]
share_loss = net[(net < 0) & flagged].sum() / loss.sum()

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar(["days flagged", "share of losses\non flagged days"], [100*share_days, 100*share_loss],
       color=[BLUE, ORANGE], width=.5, zorder=3)
for i, v in enumerate([100*share_days, 100*share_loss]):
    ax.text(i, v + 0.6, f"{v:.1f} %", ha="center", fontsize=11, color=INK)
ax.set_ylabel("%"); ax.set_ylim(0, 26)
ax.set_title("A useful detector would capture more loss than its share of days")
plt.tight_layout(); plt.show()
print(f"flags fire on {100*share_days:.1f} % of days and capture {100*share_loss:.1f} % of losses")
print("proportional, so the flags carry no information about when losses happen")
""")

M(r"""
The two bars are the same height. Using the flags as an exit rather than a label reduces drawdown,
but that is a leverage effect: a book that is flat most of the time has less of everything,
including loss. Measured across thresholds with the flags used only as a label, flagged days are
marginally better than clean days, not worse.

The likely reason is structural. Regime detection can only protect a regime in which the strategy
works, and the gross Sharpe here is approximately zero throughout.
""")

M(r"""
## 4. The persistence measurement

Every mechanism above fails the same way, and the cause sits upstream of all of them. This is the
test: take the pairs the screen selected in sample, take the pairs it rejected, and retest both on
the following year.
""")

K(r"""
sel_p, rej_p = [], []
for f in folds:
    tk = liquidity_filter(dv, f["is"], _universe_at(memb, f["selection_date"]))
    sc = screen_pairs(lp.loc[f["is"]], tk, min_obs=600)
    if sc.empty: continue
    sc["q"] = [passes_quality(PairResult(**{k: r[k] for k in _COLS})) for _, r in sc.iterrows()]
    sel = sc[(sc.pvalue < .05) & sc.q]
    rej = sc[sc.pvalue >= .20].head(len(sel))
    for df, bag in ((sel, sel_p), (rej, rej_p)):
        for _, r in df.iterrows():
            a, b = r["a"], r["b"]
            la, lb = lp[a].loc[f["oos"]].values, lp[b].loc[f["oos"]].values
            m = ~(np.isnan(la) | np.isnan(lb))
            if m.sum() < 200: continue
            bag.append(coint(la[m], lb[m], trend="c", maxlag=1, autolag=None)[1])
sel_p, rej_p = np.array(sel_p), np.array(rej_p)
s_rate, r_rate = 100*np.mean(sel_p < .05), 100*np.mean(rej_p < .05)

fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.bar(["selected in sample\n(p < 0.05)", "rejected in sample\n(p >= 0.20)"],
       [s_rate, r_rate], color=[BLUE, ORANGE], width=.5, zorder=3)
ax.axhline(5, color=GREY, ls="--", lw=1.2)
ax.text(1.45, 5.7, "5 %, what a valid test gives under the null", fontsize=8.5,
        color=GREY, ha="right")
for i, v in enumerate([s_rate, r_rate]):
    ax.text(i, v + 0.5, f"{v:.1f} %", ha="center", fontsize=12, color=INK)
ax.set_ylabel("still cointegrated the next year, %")
ax.set_title("The screen selects pairs that survive less often than the ones it rejects")
plt.tight_layout(); plt.show()
print(f"selected: {s_rate:.1f} %  (n={len(sel_p)})     rejected: {r_rate:.1f} %  (n={len(rej_p)})")
""")

M(r"""
The ranking is inverted. This is what selection on a noisy statistic does. Write the observed
statistic as a true component plus noise. Selecting the most extreme value selects jointly on both,
so the observed extremeness is shared between something real and something lucky. Only the real
part survives into the next sample. On a three-year window the noise dominates: a p-value below
0.05 is reached mostly by pairs whose residual happened to be quiet, and quiet does not repeat.

The rejected pairs face no such selection pressure, so they regress less.
""")

M(r"""
## 5. The equilibrium level moves too

Even for pairs that keep co-moving, the level they revert around shifts between the estimation
window and the trading year.
""")

K(r"""
drift = []
for f in folds:
    tk = liquidity_filter(dv, f["is"], _universe_at(memb, f["selection_date"]))
    sc = screen_pairs(lp.loc[f["is"]], tk, min_obs=600)
    if sc.empty: continue
    sc["q"] = [passes_quality(PairResult(**{k: r[k] for k in _COLS})) for _, r in sc.iterrows()]
    for _, r in sc[(sc.pvalue < .05) & sc.q].iterrows():
        s_is  = spread_from_beta(lp[r.a].loc[f["is"]],  lp[r.b].loc[f["is"]],  r.beta, r.alpha)
        s_oos = spread_from_beta(lp[r.a].loc[f["oos"]], lp[r.b].loc[f["oos"]], r.beta, r.alpha)
        drift.append((s_oos.mean() - s_is.mean()) / s_is.std())
drift = np.array(drift)

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.hist(np.clip(np.abs(drift), 0, 5), bins=40, color=BLUE, zorder=3)
ax.axvline(np.median(np.abs(drift)), color=RED, lw=2)
ax.text(np.median(np.abs(drift)) + .08, ax.get_ylim()[1]*.85,
        f"median {np.median(np.abs(drift)):.2f} sigma", color=RED, fontsize=10)
ax.set_xlabel("shift in the spread mean, in in-sample standard deviations")
ax.set_ylabel("pairs")
ax.set_title("Where the spread reverts to is not where it was estimated")
plt.tight_layout(); plt.show()
print(f"median |shift| {np.median(np.abs(drift)):.2f} sigma | "
      f"beyond 1 sigma: {100*np.mean(np.abs(drift)>1):.0f} % | "
      f"beyond 2 sigma: {100*np.mean(np.abs(drift)>2):.0f} %")
""")

M(r"""
A z-score frozen on the in-sample mean therefore starts the trading year at about 1.5 in absolute
value. The position opens immediately and never closes: the book sits in position 87 % of the time
against 65 % under ideal calibration.

Recalibrating causally does not rescue this. Rolling windows of 60, 125 and 250 days and an
expanding window all give a stitched Sharpe between -0.2 and -0.5. Calibrating on the trading
year's own mean gives +2.57, but the same procedure applied to independent random walks gives a
median of +1.68 and an aggregated book at +21.97. De-meaning a series with its own future makes
almost anything look mean-reverting.
""")

M(r"""
## 6. What this rules out

The pairs approach fails here, and the failure is located rather than observed. The selection
statistic does not persist, and the equilibrium level moves. Multiple-testing correction removes
candidates from a pool with no signal in it. Regime detection has no good regime to protect.
Johansen baskets give -0.22 against -0.26 for pairs, for ten times as many tests and a trace
statistic whose own rejection rate under the null is 8.6 % rather than 5 %.

None of these is a tuning failure. No parameter work changes the persistence number.

The only lever with a measurable effect is requiring stability rather than significance. Demanding
cointegration on both disjoint halves of the in-sample window lifts persistence from 10.8 % to
21.1 %, but leaves 19 pairs across ten folds, about two per fold, which is not a book.

The replacement developed in the companion notebook keeps the residual idea and drops the pair
selection that causes the problem.
""")

nb = new_notebook(cells=C, metadata={"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                                     "language_info": {"name":"python"}})
nbf.write(nb, OUT); print("written ->", OUT, f"({len(C)} cells)")
