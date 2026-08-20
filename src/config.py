"""Global project constants. Frozen in Phase 1 (see docs/01_universe_decision.en.md)."""
from pathlib import Path

SEED = 20260819

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
for _p in (DATA_RAW, DATA_PROC, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# --- Universe (frozen) ---
SECTOR = "Utilities"          # GICS level-1 label
START = "2014-01-01"
END = "2026-06-30"

# Cadence of point-in-time index snapshots. Semi-annual: twice as fine as the
# annual walk-forward re-selection, hence never more than 6 months stale.
SNAPSHOT_FREQ_MONTHS = 6

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_PAGE = "List of S&P 500 companies"
USER_AGENT = "pairs-trading-research/0.1 (academic project; contact via GitHub)"
