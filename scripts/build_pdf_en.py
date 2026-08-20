import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pdf_render import build
from scripts.pdf_content_en import blocks

OUT = Path(__file__).resolve().parents[1] / "reports" / "Phase1_decisions_explained.pdf"
build(
    blocks(), OUT,
    title="Phase 1 — Every Decision Explained",
    subtitle="Statistical arbitrage: universe, data, and the absence of leaks",
    meta=["Project *regime-aware stat-arb & honest selection*",
          "S&P 500 Utilities — 2014-2026",
          "Reference document for the oral defence"],
)
print("PDF written ->", OUT, f"({OUT.stat().st_size/1024:.0f} KB)")
