"""Global project constants. Frozen in Phase 1 (see docs/01_universe_decision.md)."""
from pathlib import Path

Seed = 20260819

Root = Path(__file__).resolve().parents[1]
DATA_RAW = Root / "data" / "raw"
DATA_PROC = Root / "data" / "processed"
Reports = Root / "reports"
for _p in (DATA_RAW, DATA_PROC, Reports):
    _p.mkdir(parents=True, exist_ok=True)

# --- Universe (frozen) ---
Sector = "Utilities"          # GICS level-1 label
Start = "2014-01-01"
End = "2026-06-30"

# Cadence of point-in-time index snapshots. Semi-annual: twice as fine as the
# annual walk-forward re-selection, hence never more than 6 months stale.
SNAPSHOT_FREQ_MONTHS = 6

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_PAGE = "List of S&P 500 companies"
USER_AGENT = "pairs-trading-research/0.1 (academic project; contact via GitHub)"
