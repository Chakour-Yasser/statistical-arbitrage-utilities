"""
Phase 3, honest selection under multiple testing.

The problem: N(N-1)/2 pairs are screened, so at level alpha a fraction alpha of
them look significant by chance alone. On this universe the screener finds ~47
significant pairs against ~23 expected under the null: roughly half the
"discoveries" are noise.

Three corrections are implemented, and the choice between them is not cosmetic
here because the tests are strongly dependent (Phase 2 showed one name holding
16-50 percent of the selected pairs in every fold).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Multiple-testing corrections
# --------------------------------------------------------------------------- #
def bonferroni(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Control the family-wise error rate: reject p_i <= alpha / N.

    FWER = P(at least one false discovery) <= alpha, by the union bound, and the
    union bound needs NO independence assumption. That is Bonferroni's strength
    and the reason it is the safe default under unknown dependence.

    Its cost is power. With N = 406 the threshold is 1.2e-4, so a pair must be
    overwhelmingly significant to survive. Controlling the probability of even
    one false positive is the wrong objective when the goal is to assemble a
    basket of candidates and a few bad ones are tolerable.
    """
    return np.asarray(pvals) <= alpha / max(len(pvals), 1)


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Control the false discovery rate: E[V / max(R,1)] <= alpha.

    Step-up procedure: sort p_(1) <= ... <= p_(N), find the largest k with
    p_(k) <= k*alpha/N, reject the k smallest.

    FDR is the right target for a screener: it bounds the Expected Proportion of
    selected pairs that are spurious, rather than the probability of any error.
    Accepting that 5 percent of a 40-pair basket is junk is a sane trade; demanding
    a 95 percent chance of zero junk is not.

    Validity requires independence or positive regression dependence (PRDS).
    Pair tests sharing a leg are positively dependent, which is suggestive but
    not a proof that PRDS holds here, hence `benjamini_yekutieli` below.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * np.arange(1, n + 1) / n
    below = np.where(ranked <= thresh)[0]
    out = np.zeros(n, dtype=bool)
    if below.size:
        out[order[: below[-1] + 1]] = True
    return out


def benjamini_yekutieli(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """FDR control valid under Arbitrary dependence.

    Same step-up as BH with alpha replaced by alpha / H_N, where
    H_N = sum_{i=1}^{N} 1/i ~ ln N + 0.577. At N = 406, H_N = 6.6, so the
    effective level is alpha / 6.6, a heavy price, but it is the honest option
    when the dependence structure is unknown and demonstrably strong.
    """
    n = len(pvals)
    if n == 0:
        return np.zeros(0, dtype=bool)
    h_n = np.sum(1.0 / np.arange(1, n + 1))
    return benjamini_hochberg(pvals, alpha / h_n)


CORRECTIONS = {
    "naive": lambda p, a: np.asarray(p) <= a,
    "bonferroni": bonferroni,
    "bh": benjamini_hochberg,
    "by": benjamini_yekutieli,
}


def n_discoveries(pvals: np.ndarray, alpha: float = 0.05) -> pd.Series:
    """Discovery counts under each rule, plus the null expectation."""
    p = np.asarray(pvals)
    return pd.Series({
        "n_tests": len(p),
        "expected_under_null": alpha * len(p),
        **{k: int(f(p, alpha).sum()) for k, f in CORRECTIONS.items()},
    })


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
def walk_forward_folds(index: pd.DatetimeIndex, start: str, end: str,
                       is_years: int = 3, oos_years: int = 1) -> list[dict]:
    """Successive (in-sample, out-of-sample) windows that never overlap.

    The in-sample window ends strictly before the out-of-sample window begins.
    Everything, universe, liquidity filter, cointegration test, hedge ratio,
    z-score calibration, is estimated on the in-sample window and then Frozen.
    Re-testing during the trading window would reintroduce look-ahead.
    """
    folds = []
    y0 = pd.Timestamp(start).year
    y_end = pd.Timestamp(end).year
    y = y0
    while y + is_years + oos_years - 1 <= y_end:
        is_lo = pd.Timestamp(f"{y}-01-01")
        is_hi = pd.Timestamp(f"{y + is_years - 1}-12-31")
        oos_lo = pd.Timestamp(f"{y + is_years}-01-01")
        oos_hi = pd.Timestamp(f"{y + is_years + oos_years - 1}-12-31")
        folds.append({
            "fold": f"{y}-{y + is_years - 1}/{y + is_years}",
            "is": index[(index >= is_lo) & (index <= is_hi)],
            "oos": index[(index >= oos_lo) & (index <= oos_hi)],
            "selection_date": is_hi,
        })
        y += oos_years
    return [f for f in folds if len(f["is"]) > 500 and len(f["oos"]) > 100]


def liquidity_filter(dollar_volume: pd.DataFrame, is_window: pd.DatetimeIndex,
                     tickers: list[str], min_adv: float = 2e7) -> list[str]:
    """Median dollar volume over the IN-Sample window only.

    Computing this on the full sample would select names that Remain liquid --
    information correlated with survival, hence survivorship bias wearing a
    technical disguise.
    """
    sub = dollar_volume.reindex(index=is_window, columns=tickers)
    med = sub.median(axis=0, skipna=True)
    return sorted(med[med >= min_adv].dropna().index)
