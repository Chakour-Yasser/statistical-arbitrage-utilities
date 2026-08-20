import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pdf_render import build
from scripts.pdf_content import blocks

OUT = Path(__file__).resolve().parents[1] / "reports" / "Phase1_decisions_expliquees.pdf"
build(
    blocks(), OUT,
    title="Phase 1 — Chaque décision expliquée",
    subtitle="Statistical arbitrage : univers, données et absence de fuite",
    meta=["Projet *stat-arb régime-aware & sélection honnête*",
          "Utilities du S&P 500 — 2014-2026",
          "Document de référence pour la défense orale"],
)
print("PDF ecrit ->", OUT, f"({OUT.stat().st_size/1024:.0f} Ko)")
