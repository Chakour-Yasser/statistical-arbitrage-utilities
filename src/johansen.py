"""
Phase 5 (extension) -- cointegrated baskets via Johansen.

Why Johansen rather than Engle-Granger
--------------------------------------
Engle-Granger needs a dependent variable, so it is asymmetric (Phase 2,
Theorem 1) and it can only ever find ONE relation. Johansen tests the whole
system at once: it is symmetric in the assets, it recovers the cointegration
RANK -- how many independent stationary combinations exist -- and it estimates
all of them jointly by maximum likelihood on the vector error-correction model.
For n = 2 the rank is 0 or 1 and the two approaches answer the same question;
the generalisation only becomes interesting for baskets.

The honest catch
----------------
The combinatorics explode. With 29 names there are 3654 triplets and 23751
quadruplets, against 406 pairs. Every multiple-testing problem from Phase 3 gets
an order of magnitude worse, and the trace test has no closed-form p-value in
statsmodels -- only critical values at 90/95/99 percent. The null rejection rate
is therefore measured by simulation here rather than assumed.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# column index of the 95 percent critical value in statsmodels' cvt array
CV95 = 1


def johansen_rank(log_px: pd.DataFrame, k_ar_diff: int = 1,
                  det_order: int = 0) -> dict:
    """Trace test for every rank, returning the estimated cointegration rank.

    The trace statistic tests H0: rank <= r against rank > r, walking r upward
    and stopping at the first non-rejection. `det_order=0` puts a constant in
    the cointegrating relation, which matches the Engle-Granger specification
    used in Phase 2.
    """
    x = log_px.dropna()
    n = x.shape[1]
    if len(x) < 100 or x.shape[1] < 2:
        return {"rank": 0, "trace": None, "cv95": None, "evec": None, "n_obs": len(x)}
    res = coint_johansen(x.values, det_order, k_ar_diff)
    trace, cv = res.lr1, res.cvt[:, CV95]
    rank = 0
    for r in range(n):
        if trace[r] > cv[r]:
            rank = r + 1
        else:
            break
    return {"rank": int(rank), "trace": trace, "cv95": cv, "evec": res.evec,
            "eig": res.eig, "n_obs": len(x), "names": list(x.columns)}


def basket_weights(jres: dict, normalise: str = "gross") -> pd.Series | None:
    """First cointegrating vector as portfolio weights.

    The eigenvector associated with the largest eigenvalue is the most strongly
    mean-reverting combination. Normalising by gross exposure (sum of absolute
    weights) rather than by the first component keeps baskets comparable to the
    pairs book, where gross exposure was also fixed at one dollar.
    """
    if jres.get("rank", 0) < 1 or jres.get("evec") is None:
        return None
    w = np.asarray(jres["evec"])[:, 0]
    if normalise == "gross":
        denom = np.abs(w).sum()
    else:
        denom = w[0]
    if denom == 0 or not np.isfinite(denom):
        return None
    return pd.Series(w / denom, index=jres["names"])


def basket_spread(log_px: pd.DataFrame, w: pd.Series) -> pd.Series:
    return (log_px[w.index] * w).sum(axis=1)


def screen_baskets(log_px: pd.DataFrame, tickers: list[str], size: int = 3,
                   k_ar_diff: int = 1, max_groups: int | None = None,
                   seed: int = 0) -> pd.DataFrame:
    """Johansen on every group of `size` names. IN-SAMPLE ONLY.

    `max_groups` subsamples the groups when the full enumeration is too large.
    Subsampling is done with a fixed seed and the count is reported, because the
    number of groups tried is exactly the quantity a multiple-testing correction
    needs -- and it is the number candidates most often forget to disclose.
    """
    groups = list(itertools.combinations(sorted(tickers), size))
    n_total = len(groups)
    if max_groups is not None and n_total > max_groups:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=max_groups, replace=False)
        groups = [groups[i] for i in sorted(idx)]
    rows = []
    for g in groups:
        sub = log_px[list(g)].dropna()
        if len(sub) < 500:
            continue
        try:
            jr = johansen_rank(sub, k_ar_diff=k_ar_diff)
        except Exception:
            continue
        if jr["trace"] is None:
            continue
        w = basket_weights(jr)
        rows.append({
            "names": "/".join(g), "rank": jr["rank"],
            "trace0": float(jr["trace"][0]), "cv95_0": float(jr["cv95"][0]),
            "excess": float(jr["trace"][0] - jr["cv95"][0]),
            "weights": None if w is None else w.round(4).to_dict(),
            "n_obs": jr["n_obs"],
        })
    out = pd.DataFrame(rows)
    out.attrs["n_groups_total"] = n_total
    out.attrs["n_groups_tested"] = len(groups)
    return out


def null_rejection_rate(n_series: int = 3, n_obs: int = 750, n_sim: int = 300,
                        k_ar_diff: int = 1, seed: int = 0) -> float:
    """Share of INDEPENDENT random-walk systems the trace test calls cointegrated.

    statsmodels gives critical values, not p-values, so the effective level of
    the test has to be measured rather than assumed -- exactly the discipline
    that caught the 14.6 percent Engle-Granger size distortion in Phase 2.
    """
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        x = np.cumsum(rng.normal(0, .012, size=(n_obs, n_series)), axis=0)
        try:
            res = coint_johansen(x, 0, k_ar_diff)
        except Exception:
            continue
        hits += int(res.lr1[0] > res.cvt[0, CV95])
    return hits / n_sim
