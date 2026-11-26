"""Export Maureen's compiled Markdown report to a styled, paginated PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
SOURCE = REPORT_DIR / "maureen_sections.md"
OUTPUT = REPORT_DIR / "maureen_sections.pdf"


def inline_markup(text: str) -> str:
    """Convert the small inline Markdown subset used by the report to ReportLab XML."""
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="7.2">\1</font>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", lambda m: f'<link href="{html.escape(m.group(2), quote=True)}" color="#246b4e"><u>{m.group(1)}</u></link>', text)
    return text


def table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("ReportTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=21,
                                leading=25, textColor=colors.HexColor("#174b3b"), alignment=TA_LEFT, spaceAfter=10),
        "h1": ParagraphStyle("Section", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=15,
                              leading=18, textColor=colors.HexColor("#174b3b"), spaceBefore=9, spaceAfter=6,
                              keepWithNext=True),
        "h2": ParagraphStyle("Subsection", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10.5,
                              leading=13, textColor=colors.HexColor("#246b4e"), spaceBefore=7, spaceAfter=4,
                              keepWithNext=True),
        "body": ParagraphStyle("BodyTextSmall", parent=base["BodyText"], fontName="Times-Roman", fontSize=8.5,
                               leading=11.2, textColor=colors.HexColor("#24332b"), spaceAfter=5, alignment=TA_LEFT),
        "caption": ParagraphStyle("CaptionSmall", parent=base["BodyText"], fontName="Times-Italic", fontSize=7.2,
                                  leading=9, textColor=colors.HexColor("#65746b"), alignment=TA_CENTER, spaceAfter=7),
        "table": ParagraphStyle("TableCell", parent=base["BodyText"], fontName="Helvetica", fontSize=6.6,
                                leading=8, textColor=colors.HexColor("#26362e"), spaceAfter=0),
        "tablehead": ParagraphStyle("TableHead", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=6.7,
                                    leading=8, textColor=colors.HexColor("#174b3b"), spaceAfter=0),
        "bullet": ParagraphStyle("BulletSmall", parent=base["BodyText"], fontName="Times-Roman", fontSize=8.5,
                                 leading=11, leftIndent=12, firstLineIndent=-8, textColor=colors.HexColor("#24332b"), spaceAfter=3),
        "quote": ParagraphStyle("Takeaway", parent=base["BodyText"], fontName="Helvetica", fontSize=8.3,
                                leading=10.8, textColor=colors.HexColor("#174b3b"), spaceAfter=0),
    }


def image_flowable(relative: str, max_width: float, max_height: float):
    path = (REPORT_DIR / relative).resolve()
    if not path.is_file():
        return Paragraph(f"[Figure unavailable: {html.escape(relative)}]", styles()["caption"])
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height, 1.0)
    return Image(str(path), width=width * scale, height=height * scale, hAlign="CENTER")


def table_flowable(lines: list[str], doc_width: float, style_map: dict):
    raw_rows = [table_cells(line) for line in lines]
    if len(raw_rows) > 1 and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in raw_rows[1]):
        raw_rows.pop(1)
    if not raw_rows:
        return Spacer(1, 1)
    count = len(raw_rows[0])
    if count == 2:
        fractions = [0.34, 0.66]
    elif count == 3:
        fractions = [0.56, 0.16, 0.28]
    elif count == 5:
        fractions = [0.58, 0.105, 0.105, 0.105, 0.13]
    else:
        fractions = [1 / count] * count
    data = []
    for row_index, row in enumerate(raw_rows):
        converted = []
        for cell in row:
            converted.append(Paragraph(inline_markup(cell), style_map["tablehead"] if row_index == 0 else style_map["table"]))
        data.append(converted)
    table = Table(data, colWidths=[doc_width * fraction for fraction in fractions], repeatRows=1,
                  hAlign="LEFT", splitByRow=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f2ea")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cad8ce")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return table


def draw_page(canvas, doc):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(colors.HexColor("#d7e3d9"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 10 * mm, width - doc.rightMargin, 10 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#66766e"))
    canvas.drawString(doc.leftMargin, 6.5 * mm, "PlantGuard AI | Maureen's project sections")
    canvas.drawRightString(width - doc.rightMargin, 6.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Build the Markdown report first: {SOURCE}")
    style_map = styles()
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=12 * mm, leftMargin=12 * mm,
                            topMargin=13 * mm, bottomMargin=15 * mm,
                            title="PlantGuard AI: Maureen's Project Sections",
                            author="PlantGuard AI project team")
    story = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line == "---":
            index += 1
            continue
        if line.startswith("!["):
            match = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line)
            if match:
                story.append(Spacer(1, 2))
                story.append(image_flowable(match.group(2), doc.width, 82 * mm))
                story.append(Spacer(1, 2))
            index += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            story.append(table_flowable(table_lines, doc.width, style_map))
            story.append(Spacer(1, 5))
            continue
        if line.startswith("# "):
            title = line[2:].strip()
            if title.startswith("PlantGuard AI:"):
                story.append(Paragraph(inline_markup(title), style_map["title"]))
            else:
                story.append(Paragraph(inline_markup(title), style_map["h1"]))
            index += 1
            continue
        if line.startswith("## ") or line.startswith("### "):
            story.append(Paragraph(inline_markup(line.lstrip("# ")), style_map["h2"]))
            index += 1
            continue
        if line.startswith("> "):
            quote = line[2:].strip()
            quote_box = Table([[Paragraph(inline_markup(quote), style_map["quote"])]], colWidths=[doc.width])
            quote_box.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef6f0")),
                ("LINEBEFORE", (0, 0), (0, -1), 2.5, colors.HexColor("#4e8b68")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(quote_box)
            story.append(Spacer(1, 6))
            index += 1
            continue
        if line.startswith("- "):
            story.append(Paragraph("- " + inline_markup(line[2:]), style_map["bullet"]))
            index += 1
            continue
        if line.startswith("*") and line.endswith("*"):
            story.append(Paragraph(inline_markup(line[1:-1]), style_map["caption"]))
            index += 1
            continue
        if line.startswith("```"):
            index += 1
            code_lines = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(html.escape(lines[index]))
                index += 1
            story.append(Paragraph("<br/>".join(code_lines), style_map["body"]))
            index += 1
            continue
        paragraph_lines = [line]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if (not candidate or candidate == "---" or candidate.startswith(("#", "> ", "- ", "|", "![", "```"))):
                break
            paragraph_lines.append(candidate)
            index += 1
        paragraph = inline_markup(" ".join(paragraph_lines))
        story.append(Paragraph(paragraph, style_map["body"]))
    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return OUTPUT


if __name__ == "__main__":
    print(build())
