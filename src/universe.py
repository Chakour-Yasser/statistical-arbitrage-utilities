"""
Point-in-time construction of the sector universe.

Principle
---------
Reconstructing index membership from a list of *current* constituents introduces
survivorship bias: we would only ever see companies that survived until today.
So we use the revision history of the Wikipedia page listing S&P 500
constituents, and read the page *as it existed* on each selection date. This
gives us membership AND the point-in-time GICS sector by direct observation --
including for companies that have since disappeared.

Accepted limitations (documented in docs/01_universe_decision.en.md):
  - Wikipedia is edited with a lag of a few days after an index change. No
    consequence here: re-selection is annual.
  - *Prices* for delisted companies remain unavailable for free. The residual
    bias is therefore in the data, no longer in the composition.
"""
from __future__ import annotations

import io
import json
import time
from typing import Iterable

import pandas as pd
import requests

from . import config as C

_SESSION: requests.Session | None = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({"User-Agent": C.USER_AGENT})
    return _SESSION


# --------------------------------------------------------------------------- #
# Raw retrieval (with a disk cache: the build must stay reproducible even if the
# Wikipedia page changes tomorrow)
# --------------------------------------------------------------------------- #
def _get(params: dict, tries: int = 8) -> dict:
    """GET with exponential backoff: the Wikipedia API returns 429s in bursts.

    A backtest whose universe construction silently fails on a missing snapshot
    would produce a truncated universe -- so we fail loudly instead.
    """
    delay = 5.0
    for k in range(tries):
        try:
            r = _session().get(C.WIKI_API, params=params, timeout=30)
        except requests.exceptions.RequestException:
            time.sleep(delay); delay = min(delay * 2, 120); continue
        if r.status_code == 429:
            time.sleep(delay)
            delay = min(delay * 2, 120)
            continue
        r.raise_for_status()
        time.sleep(1.5)  # politeness: smooth out the request rate
        return r.json()
    raise RuntimeError(f"persistent 429 after {tries} attempts: {params}")


def revision_at(timestamp: str) -> dict:
    """Last revision of the page published BEFORE `timestamp` (ISO 8601, UTC).

    `rvdir=older` + `rvstart=ts`: walk back in time from ts. This is the crucial
    no-leak property -- we must never read a revision later than the selection
    date.
    """
    # persistent cache of revision lookups -> reruns are incremental
    cache_f = C.DATA_RAW / "wiki" / "revision_index.json"
    cache_f.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    if timestamp in cache:
        return cache[timestamp]

    params = {
        "action": "query", "prop": "revisions", "titles": C.WIKI_PAGE,
        "rvlimit": 1, "rvstart": timestamp, "rvdir": "older",
        "rvprop": "ids|timestamp", "format": "json", "formatversion": 2,
    }
    revs = _get(params)["query"]["pages"][0]["revisions"]
    if not revs:
        raise RuntimeError(f"no revision before {timestamp}")
    cache[timestamp] = revs[0]
    cache_f.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return revs[0]


def _revision_html(revid: int) -> str:
    cache = C.DATA_RAW / "wiki" / f"rev_{revid}.html"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    # The REST endpoint renders the HTML of a specific revision with far more
    # generous rate limits than action=parse (which returns 429s in bursts).
    url = ("https://en.wikipedia.org/api/rest_v1/page/html/"
           "List_of_S%26P_500_companies/" + str(revid))
    delay = 2.0
    for _ in range(8):
        try:
            r = _session().get(url, timeout=90)
        except requests.exceptions.RequestException:
            time.sleep(delay); delay = min(delay * 2, 60); continue
        if r.status_code == 429:
            time.sleep(delay); delay = min(delay * 2, 60); continue
        r.raise_for_status()
        cache.write_text(r.text, encoding="utf-8")
        time.sleep(0.3)
        return r.text
    raise RuntimeError(f"persistent failure on revision {revid}")


# --------------------------------------------------------------------------- #
# Robust parsing: column headers changed several times between 2013 and 2026
# ("Ticker symbol" -> "Symbol", "Company" -> "Security", ...).
# --------------------------------------------------------------------------- #
_TICKER_COLS = ("symbol", "ticker symbol", "ticker")
_SECTOR_COLS = ("gics sector", "gics  sector", "sector")
_SUBIND_COLS = ("gics sub-industry", "gics sub industry", "sub-industry")
_NAME_COLS = ("security", "company", "name")


def _pick(cols: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    low = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def parse_constituents(html: str) -> pd.DataFrame:
    """Extract (ticker, name, sector, sub_industry) from the constituents table."""
    tables = pd.read_html(io.StringIO(html))
    best = None
    for t in tables:
        tcol = _pick(t.columns, _TICKER_COLS)
        scol = _pick(t.columns, _SECTOR_COLS)
        if tcol is not None and scol is not None and len(t) > 300:
            best = (t, tcol, scol)
            break
    if best is None:
        raise RuntimeError("constituents table not found in this revision")
    t, tcol, scol = best
    ncol = _pick(t.columns, _NAME_COLS)
    icol = _pick(t.columns, _SUBIND_COLS)
    out = pd.DataFrame({
        "ticker": t[tcol].astype(str).str.strip(),
        "name": t[ncol].astype(str).str.strip() if ncol else "",
        "sector": t[scol].astype(str).str.strip(),
        "sub_industry": t[icol].astype(str).str.strip() if icol else "",
    })
    out["ticker"] = out["ticker"].map(normalize_ticker)
    return out[out["ticker"].str.len().between(1, 6)].reset_index(drop=True)


def normalize_ticker(t: str) -> str:
    """Wikipedia writes 'BRK.B', Yahoo Finance expects 'BRK-B'.

    A failed conversion silently drops a name from the universe *without raising
    an error* -- exactly the kind of invisible bug a backtest never reports.
    Hence a centralised, unit-tested normalisation.
    """
    t = str(t).strip().upper()
    t = t.split("[")[0].strip()      # wiki footnote markers
    return t.replace(".", "-").replace(" ", "")


# --------------------------------------------------------------------------- #
# Building the membership matrix
# --------------------------------------------------------------------------- #
def snapshot_dates(start: str = C.START, end: str = C.END,
                   months: int = C.SNAPSHOT_FREQ_MONTHS) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq=f"{months}MS")


def build_membership(sector: str = C.SECTOR, force: bool = False) -> pd.DataFrame:
    """Boolean matrix (snapshot dates x tickers): True = member of the sector.

    Also writes a detailed snapshot log to data/processed/.
    """
    out_path = C.DATA_PROC / f"membership_{sector.lower()}.parquet"
    meta_path = C.DATA_PROC / f"membership_{sector.lower()}_meta.json"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    rows, meta = {}, []
    for d in snapshot_dates():
        ts = d.strftime("%Y-%m-%dT00:00:00Z")
        rev = revision_at(ts)
        html = _revision_html(rev["revid"])
        cons = parse_constituents(html)
        sel = cons[cons["sector"].str.casefold() == sector.casefold()]
        rows[d] = {tk: True for tk in sel["ticker"]}
        meta.append({
            "snapshot": d.strftime("%Y-%m-%d"),
            "revid": rev["revid"],
            "rev_timestamp": rev["timestamp"],
            "lag_days": (d - pd.Timestamp(rev["timestamp"]).tz_localize(None)).days,
            "n_index": int(len(cons)),
            "n_sector": int(len(sel)),
            "tickers": sorted(sel["ticker"].tolist()),
        })
        print(f"  {d:%Y-%m-%d}  rev {rev['revid']}  ({rev['timestamp'][:10]}, "
              f"lag {meta[-1]['lag_days']}j)  index={len(cons)}  {sector}={len(sel)}")

    memb = pd.DataFrame.from_dict(rows, orient="index").fillna(False).astype(bool)
    memb = memb.reindex(sorted(memb.columns), axis=1)
    memb.index.name = "snapshot"
    memb.to_parquet(out_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return memb


if __name__ == "__main__":
    m = build_membership(force=True)
    print(f"\nMembership matrix: {m.shape[0]} snapshots x {m.shape[1]} tickers "
          f"(union over the period)")


# --------------------------------------------------------------------------- #
# Identity check: detecting TICKER RECYCLING
# --------------------------------------------------------------------------- #
def membership_windows(memb: pd.DataFrame) -> pd.DataFrame:
    """For each ticker: first and last snapshot where it is a sector member."""
    rows = []
    for t in memb.columns:
        s = memb[t]
        if not s.any():
            continue
        idx = s[s].index
        rows.append({"ticker": t, "memb_first": idx[0], "memb_last": idx[-1],
                     "n_snapshots": int(s.sum())})
    return pd.DataFrame(rows).set_index("ticker")


def validate_identity(memb: pd.DataFrame, close: pd.DataFrame,
                      min_overlap_frac: float = 0.50) -> pd.DataFrame:
    """Check that the price history overlaps the membership window.

    WHY this is a blocking check and not a warning
    ----------------------------------------------
    Tickers get recycled. 'NU' denoted Northeast Utilities until 2015; since
    December 2021 it denotes Nu Holdings, a Brazilian neobank. A naive
    per-ticker download therefore injects an unrelated company's prices into the
    sector universe -- silently, without raising an exception. The backtest then
    produces a perfectly plausible and perfectly false P&L.

    Test: the quotation period must cover at least `min_overlap_frac` of the
    index membership window. Zero overlap is the signature of recycling; partial
    overlap, that of a history truncated by the vendor.
    """
    win = membership_windows(memb)
    rows = []
    for t, w in win.iterrows():
        if t not in close.columns or close[t].dropna().empty:
            rows.append({**w, "px_first": pd.NaT, "px_last": pd.NaT,
                         "overlap_frac": 0.0, "status": "NO_DATA"})
            continue
        obs = close[t].dropna().index
        px_first, px_last = obs[0], obs[-1]
        # Membership window widened by one snapshot on the right: being a member
        # at date t means being one at least until t+6 months. Without this, a
        # name present in a single snapshot (e.g. OKE, reclassified from
        # Utilities to Energy in early 2014) would have a zero-length window and
        # be wrongly declared recycled.
        lo = w["memb_first"]
        hi = w["memb_last"] + pd.DateOffset(months=C.SNAPSHOT_FREQ_MONTHS)
        span = (hi - lo).days
        ov = (min(px_last, hi) - max(px_first, lo)).days
        frac = max(ov, 0) / span
        if frac < 0.05:
            # Near-zero overlap: the ticker denotes a different company. The
            # threshold is not exactly 0 because a ticker recycled shortly after
            # the acquisition (TEG/Integrys) overlaps the window by a few days.
            status = "RECYCLED"
        elif frac < min_overlap_frac:
            status = "SUSPECT"           # truncated history: manual review
        else:
            status = "OK"
        rows.append({**w, "px_first": px_first, "px_last": px_last,
                     "overlap_frac": round(frac, 3), "status": status})
    out = pd.DataFrame(rows, index=win.index)
    return out.sort_values(["status", "ticker"])


def tradable_universe(memb: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Membership matrix with invalid-identity tickers removed.

    Removed names are NOT forgotten: they constitute the residual survivorship
    bias, quantified in reports/survivorship.en.md.
    """
    val = validate_identity(memb, close)
    bad = val.index[val["status"].isin(["RECYCLED", "NO_DATA", "SUSPECT"])]
    keep = [c for c in memb.columns if c not in bad]
    return memb[keep]
