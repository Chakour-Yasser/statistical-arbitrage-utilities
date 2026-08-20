"""Moteur de rendu PDF (reportlab / Platypus).

Le contenu vit dans scripts/pdf_content.py sous forme de blocs declaratifs.
Polices : Times New Roman (corps, couverture WGL4 -> grec + fleches + operateurs
mathematiques), Arial Bold (titres), Courier New (code). Les polices integrees de
reportlab (Helvetica) ne contiennent pas ces glyphes et les rendraient en carres noirs.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, KeepTogether,
                                PageBreak, HRFlowable, CondPageBreak)

SUP = Path("/System/Library/Fonts/Supplemental")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("Body", SUP / "Times New Roman.ttf"))
    pdfmetrics.registerFont(TTFont("Body-B", SUP / "Times New Roman Bold.ttf"))
    pdfmetrics.registerFont(TTFont("Body-I", SUP / "Times New Roman Italic.ttf"))
    pdfmetrics.registerFont(TTFont("Body-BI", SUP / "Times New Roman Bold Italic.ttf"))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-B",
                                  italic="Body-I", boldItalic="Body-BI")
    pdfmetrics.registerFont(TTFont("Head", SUP / "Arial Bold.ttf"))
    pdfmetrics.registerFont(TTFont("Head-R", SUP / "Arial.ttf"))
    pdfmetrics.registerFont(TTFont("Mono", SUP / "Courier New.ttf"))
    pdfmetrics.registerFont(TTFont("Mono-B", SUP / "Courier New Bold.ttf"))
    pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-B")


INK = colors.HexColor("#16191d")
MUTED = colors.HexColor("#5b6470")
ACCENT = colors.HexColor("#0b4f6c")
RULE = colors.HexColor("#c9d1d9")
WARN_BG = colors.HexColor("#fdf3e7")
WARN_ED = colors.HexColor("#d98324")
KEY_BG = colors.HexColor("#eef4f7")
KEY_ED = colors.HexColor("#0b4f6c")
CODE_BG = colors.HexColor("#f4f6f8")
THM_BG = colors.HexColor("#f2f0f7")
THM_ED = colors.HexColor("#4a3aa7")


def styles() -> dict:
    ss = getSampleStyleSheet()
    s = {}
    s["h1"] = ParagraphStyle("h1", parent=ss["Normal"], fontName="Head", fontSize=19,
                             leading=23, textColor=ACCENT, spaceBefore=0, spaceAfter=4)
    s["h2"] = ParagraphStyle("h2", parent=ss["Normal"], fontName="Head", fontSize=13,
                             leading=16, textColor=INK, spaceBefore=16, spaceAfter=5)
    s["h3"] = ParagraphStyle("h3", parent=ss["Normal"], fontName="Head", fontSize=10.5,
                             leading=13, textColor=MUTED, spaceBefore=11, spaceAfter=3)
    s["p"] = ParagraphStyle("p", parent=ss["Normal"], fontName="Body", fontSize=10.2,
                            leading=15.2, alignment=TA_JUSTIFY, textColor=INK,
                            spaceAfter=7)
    s["li"] = ParagraphStyle("li", parent=s["p"], leftIndent=11*mm, bulletIndent=5*mm,
                             spaceAfter=4)
    s["code"] = ParagraphStyle("code", parent=ss["Normal"], fontName="Mono", fontSize=8.1,
                               leading=11.2, textColor=INK, leftIndent=3*mm,
                               rightIndent=2*mm, spaceBefore=2, spaceAfter=2)
    s["cap"] = ParagraphStyle("cap", parent=s["p"], fontSize=8.6, leading=11.5,
                              textColor=MUTED, alignment=TA_JUSTIFY, spaceAfter=9)
    s["cell"] = ParagraphStyle("cell", parent=s["p"], fontSize=8.6, leading=11.2,
                               alignment=0, spaceAfter=0)
    s["cellh"] = ParagraphStyle("cellh", parent=s["cell"], fontName="Head",
                                fontSize=8.2, textColor=colors.white)
    s["box"] = ParagraphStyle("box", parent=s["p"], fontSize=9.6, leading=14,
                              spaceAfter=4)
    s["boxh"] = ParagraphStyle("boxh", parent=s["box"], fontName="Head", fontSize=8.6,
                               spaceAfter=3)
    s["title"] = ParagraphStyle("title", parent=ss["Normal"], fontName="Head",
                                fontSize=26, leading=31, textColor=ACCENT,
                                alignment=TA_CENTER, spaceAfter=8)
    s["sub"] = ParagraphStyle("sub", parent=ss["Normal"], fontName="Body-I",
                              fontSize=12.5, leading=17, textColor=MUTED,
                              alignment=TA_CENTER, spaceAfter=6)
    s["thmh"] = ParagraphStyle("thmh", parent=ss["Normal"], fontName="Head",
                               fontSize=8.8, leading=12, textColor=THM_ED, spaceAfter=4)
    s["thmb"] = ParagraphStyle("thmb", parent=ss["Normal"], fontName="Body",
                               fontSize=9.8, leading=14.2, alignment=TA_JUSTIFY,
                               textColor=INK, spaceAfter=4)
    s["proofh"] = ParagraphStyle("proofh", parent=ss["Normal"], fontName="Body-I",
                                 fontSize=9.6, leading=13, textColor=MUTED,
                                 leftIndent=4*mm, spaceBefore=2, spaceAfter=2)
    s["proofb"] = ParagraphStyle("proofb", parent=ss["Normal"], fontName="Body",
                                 fontSize=9.6, leading=13.8, alignment=TA_JUSTIFY,
                                 textColor=INK, leftIndent=4*mm, spaceAfter=3)
    s["math"] = ParagraphStyle("math", parent=s["p"], fontName="Body-I", fontSize=10.6,
                               alignment=TA_CENTER, spaceBefore=5, spaceAfter=8)
    return s


def esc(t: str) -> str:
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def md(t: str) -> str:
    """Mini-markdown -> markup reportlab : **gras**, *italique*, `code`."""
    import re
    t = esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\*\w])\*([^*]+?)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r'<font face="Mono" size="9">\1</font>', t)
    # esc() a echappe TOUT le markup, y compris les balises typographiques
    # volontaires. On restaure la liste blanche : indices et exposants.
    for tag in ("sub", "sup", "br/"):
        t = t.replace(f"&lt;{tag}&gt;", f"<{tag}>").replace(f"&lt;/{tag}&gt;", f"</{tag}>")
    return t


class Doc(BaseDocTemplate):
    def __init__(self, path, title, tag="Phase 1", **kw):
        super().__init__(str(path), pagesize=A4, title=title,
                         author="Projet stat-arb", **kw)
        fr = Frame(20*mm, 18*mm, A4[0] - 40*mm, A4[1] - 36*mm, id="body",
                   leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[fr]),
            PageTemplate(id="main", frames=[fr], onPage=self._chrome),
        ])
        self.doc_title = title
        self.tag = tag

    def _chrome(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Head-R", 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(20*mm, A4[1] - 12*mm, self.doc_title)
        canvas.drawRightString(A4[0] - 20*mm, A4[1] - 12*mm, self.tag)
        canvas.setStrokeColor(RULE); canvas.setLineWidth(0.4)
        canvas.line(20*mm, A4[1] - 14*mm, A4[0] - 20*mm, A4[1] - 14*mm)
        canvas.drawCentredString(A4[0] / 2, 11*mm, str(doc.page))
        canvas.restoreState()


# --------------------------- constructeurs de blocs ------------------------- #
def _collect_formulas(blocks: list) -> list[str]:
    out = []
    for b in blocks:
        if b[0] == "formula":
            out.append(b[1])
        elif b[0] in ("thm", "defn", "prop"):
            out += [i[1] for i in b[2] if isinstance(i, tuple) and i[0] == "formula"]
        elif b[0] == "proof":
            out += [i[1] for i in b[1] if isinstance(i, tuple) and i[0] == "formula"]
    return out


def validate_formulas(blocks: list) -> None:
    """Render every formula once up front and report ALL failures together.

    matplotlib's mathtext accepts only a subset of LaTeX (no \\xrightarrow,
    \\tfrac, \\stackrel, \\big, \\displaystyle, \\underbrace). Discovering that
    one formula at a time costs a build per error; this fails once with the
    full list.
    """
    bad = []
    for tex in _collect_formulas(blocks):
        try:
            _math_png(tex, 11.0, INK)
        except Exception as e:
            msg = str(e).strip().split("\n")[-1][:100]
            bad.append((tex, msg))
    if bad:
        lines = "\n".join(f"  {t}\n      -> {m}" for t, m in bad)
        raise ValueError(f"{len(bad)} formula(s) failed to render:\n{lines}")


def _walk_text(blocks: list):
    """Yield every user-visible string in a block list."""
    for b in blocks:
        for item in b[1:]:
            if isinstance(item, str):
                yield item
            elif isinstance(item, (list, tuple)):
                for sub in item:
                    if isinstance(sub, str):
                        yield sub
                    elif isinstance(sub, (list, tuple)):
                        for s2 in sub:
                            if isinstance(s2, str):
                                yield s2


def validate_glyphs(blocks: list) -> None:
    """Fail if body text uses a character Times New Roman does not have.

    reportlab renders a missing glyph as a solid black box -- silently. Symbols
    like the set-membership sign, blackboard-bold letters and the double arrow
    are absent from Times New Roman, so they must live inside rendered LaTeX
    formulas, never in body text. This check makes that failure loud.
    """
    from fontTools.ttLib import TTFont
    cmap = TTFont(str(SUP / "Times New Roman.ttf")).getBestCmap()
    bad: dict[str, str] = {}
    for text in _walk_text(blocks):
        for ch in text:
            if ord(ch) > 127 and ord(ch) not in cmap and ch not in bad:
                bad[ch] = text[:70]
    if bad:
        lines = "\n".join(f"  {c!r} (U+{ord(c):04X})  in: {ctx}..." for c, ctx in bad.items())
        raise ValueError(
            f"{len(bad)} character(s) missing from Times New Roman -- they would "
            f"render as black boxes:\n{lines}")


def build(blocks: list, path: Path, title: str, subtitle: str, meta: list[str],
          tag: str = "Phase 1"):
    register_fonts()
    validate_formulas(blocks)
    validate_glyphs(blocks)
    S = styles()
    doc = Doc(path, title, tag=tag)
    story = []

    story.append(Spacer(1, 52*mm))
    story.append(Paragraph(esc(title), S["title"]))
    story.append(Paragraph(esc(subtitle), S["sub"]))
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="42%", color=RULE, hAlign="CENTER"))
    story.append(Spacer(1, 6*mm))
    for m in meta:
        story.append(Paragraph(md(m), ParagraphStyle(
            "m", parent=S["sub"], fontName="Body", fontSize=9.6, leading=14,
            spaceAfter=2)))
    story.append(PageBreak())
    story.append(_NextTemplate("main"))

    W = A4[0] - 40*mm
    for b in blocks:
        k = b[0]
        if k == "h1":
            # CondPageBreak plutot que PageBreak : on ne change de page que s'il
            # ne reste pas assez de place, ce qui evite les demi-pages blanches
            # d'un saut systematique et les titres orphelins en bas de page.
            story.append(CondPageBreak(58*mm) if b[2] else Spacer(1, 6))
            story.append(Paragraph(esc(b[1]), S["h1"]))
            story.append(HRFlowable(width="100%", color=ACCENT, thickness=1.1,
                                    spaceBefore=3, spaceAfter=9))
        elif k == "h2":
            story.append(Paragraph(md(b[1]), S["h2"]))
        elif k == "h3":
            story.append(Paragraph(md(b[1]), S["h3"]))
        elif k == "p":
            story.append(Paragraph(md(b[1]), S["p"]))
        elif k == "math":
            story.append(Paragraph(md(b[1]), S["math"]))
        elif k == "ul":
            for it in b[1]:
                story.append(Paragraph(md(it), S["li"], bulletText="•"))
            story.append(Spacer(1, 4))
        elif k == "ol":
            for i, it in enumerate(b[1], 1):
                story.append(Paragraph(md(it), S["li"], bulletText=f"{i}."))
            story.append(Spacer(1, 4))
        elif k == "code":
            # Paragraph collapse les espaces multiples comme du HTML : on les
            # protege pour conserver l'alignement en colonnes du code.
            body = [Paragraph(
                f'<font face="Mono">{esc(l).replace(" ", "&nbsp;") or "&nbsp;"}</font>',
                S["code"]) for l in b[1].split("\n")]
            t = Table([[body]], colWidths=[W])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
                ("BOX", (0, 0), (-1, -1), 0.5, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(t); story.append(Spacer(1, 8))
        elif k in ("warn", "key"):
            bg, ed = (WARN_BG, WARN_ED) if k == "warn" else (KEY_BG, KEY_ED)
            inner = [Paragraph(md(b[1]).upper() if False else md(b[1]), S["boxh"])]
            for para in b[2]:
                inner.append(Paragraph(md(para), S["box"]))
            t = Table([[inner]], colWidths=[W])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("LINEBEFORE", (0, 0), (0, -1), 2.4, ed),
                ("BOX", (0, 0), (-1, -1), 0.4, ed),
                ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(KeepTogether(t)); story.append(Spacer(1, 9))
        elif k == "table":
            head, rows, widths, caption = b[1], b[2], b[3], (b[4] if len(b) > 4 else None)
            cw = [W * w for w in widths]
            data = [[Paragraph(md(c), S["cellh"]) for c in head]]
            data += [[Paragraph(md(str(c)), S["cell"]) for c in r] for r in rows]
            t = Table(data, colWidths=cw, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
                ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t)
            story.append(Paragraph(md(caption), S["cap"]) if caption else Spacer(1, 9))
        elif k == "formula":
            story.append(Spacer(1, 4))
            story.append(math_flowable(b[1], W, b[2] if len(b) > 2 else 12.5))
            story.append(Spacer(1, 8))
        elif k in ("thm", "defn", "prop"):
            label = {"thm": "Theorem", "defn": "Definition", "prop": "Proposition"}[k]
            head = f"{label} {b[1]}" if b[1] else label
            inner = [Paragraph(md(head), S["thmh"])]
            for item in b[2]:
                if isinstance(item, tuple) and item[0] == "formula":
                    inner.append(_centered(math_flowable(item[1], W - 30, 11.5), W - 22))
                else:
                    inner.append(Paragraph(md(item), S["thmb"]))
            t = Table([[inner]], colWidths=[W])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), THM_BG),
                ("LINEBEFORE", (0, 0), (0, -1), 2.4, THM_ED),
                ("BOX", (0, 0), (-1, -1), 0.4, THM_ED),
                ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]))
            story.append(KeepTogether(t)); story.append(Spacer(1, 9))
        elif k == "proof":
            story.append(Paragraph("<i>Proof.</i>", S["proofh"]))
            for item in b[1]:
                if isinstance(item, tuple) and item[0] == "formula":
                    story.append(_centered(math_flowable(item[1], W - 30, 11.0), W))
                else:
                    story.append(Paragraph(md(item), S["proofb"]))
            story.append(Paragraph("<para alignment=\"right\">&#9632;</para>", S["proofb"]))
            story.append(Spacer(1, 8))
        elif k == "space":
            story.append(Spacer(1, b[1]))
    doc.build(story)


from reportlab.platypus import NextPageTemplate as _NextTemplate  # noqa: E402


# --------------------------------------------------------------------------- #
# Mathematics: LaTeX -> PNG via matplotlib mathtext -> reportlab Image
#
# reportlab has no LaTeX. Unicode-plus-<sup>/<sub> breaks down as soon as you need
# fractions, integrals or hats. matplotlib's mathtext renders a usable LaTeX
# subset; the `stix` fontset is Times-compatible, so formulas blend with the
# Times New Roman body text instead of looking pasted in.
# --------------------------------------------------------------------------- #
from io import BytesIO                                            # noqa: E402
from reportlab.platypus import Image as _RLImage                   # noqa: E402

_MATH_CACHE: dict = {}


def _to_hex(color) -> str:
    """reportlab Color -> '#rrggbb' (matplotlib rejects reportlab Color objects)."""
    if isinstance(color, str):
        return color
    return "#" + color.hexval()[2:].zfill(6)


def _math_png(latex: str, fontsize: float, color) -> tuple[bytes, float, float]:
    color = _to_hex(color)
    key = (latex, fontsize, color)
    if key in _MATH_CACHE:
        return _MATH_CACHE[key]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"

    dpi = 400
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=fontsize, color=color)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                pad_inches=0.01, transparent=True)
    plt.close(fig)
    data = buf.getvalue()

    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(BytesIO(data)).getSize()
    w_pt, h_pt = iw * 72.0 / dpi, ih * 72.0 / dpi
    _MATH_CACHE[key] = (data, w_pt, h_pt)
    return _MATH_CACHE[key]


def _centered(flow, width: float):
    """Centre a flowable inside a fixed width (hAlign is ignored in table cells)."""
    t = Table([[flow]], colWidths=[width])
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def math_flowable(latex: str, max_width: float, fontsize: float = 12.0,
                  color=INK) -> _RLImage:
    data, w, h = _math_png(latex, fontsize, color)
    if w > max_width:                      # never let a formula overflow the frame
        h *= max_width / w
        w = max_width
    img = _RLImage(BytesIO(data), width=w, height=h)
    img.hAlign = "CENTER"
    return img
