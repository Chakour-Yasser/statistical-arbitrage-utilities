import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pdf_render import build
from scripts.pdf_content_phase2_en import blocks

OUT = Path(__file__).resolve().parents[1] / "reports" / "Phase2_cointegration_explained.pdf"
build(blocks(), OUT,
      title="Phase 2 — Cointegration Building Blocks",
      subtitle="Estimators, theorems, and the three results that contradicted us",
      meta=["Project *regime-aware stat-arb & honest selection*",
            "S&P 500 Utilities — 2014-2026",
            "Mathematics stated and proved where the proof is short"],
      tag="Phase 2")
print("PDF written ->", OUT, f"({OUT.stat().st_size/1024:.0f} KB)")
