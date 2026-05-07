"""
report_generator.py
Generates a one-page audit PDF for exam coordinators after each
liveness session. Includes candidate info, verdict, signal summary,
tamper events, and the session timeline graph.
Requires: reportlab
Install:  pip install reportlab
"""

import os
import time
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image as RLImage,
                                    HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

REPORTS_DIR = "session_logs"

# ── Colour palette ────────────────────────────────────────────────────────────
C_BG = colors.HexColor("#030810")
C_PANEL = colors.HexColor("#07131f")
C_BORDER = colors.HexColor("#0e2a44")
C_CYAN = colors.HexColor("#00e5ff")
C_GREEN = colors.HexColor("#00e676")
C_RED = colors.HexColor("#ff3d57")
C_AMBER = colors.HexColor("#ffab40")
C_TEXT = colors.HexColor("#aac8e8")
C_MUTED = colors.HexColor("#305878")
C_WHITE = colors.white


def generate_report(candidate_name: str,
                    candidate_id: str,
                    session_summary: dict,
                    graph_path: str = None) -> str:
    """
    Generate audit PDF. Returns path to saved PDF.
    candidate_name  — candidate's full name
    candidate_id    — roll number / seat number
    session_summary — dict from SessionLogger.get_summary()
    graph_path      — path to session timeline PNG (optional)
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab not installed. Run: pip install reportlab")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    verdict = session_summary.get("verdict", "UNKNOWN")
    safe_id = candidate_id.replace(
        "/", "-").replace("\\", "-").replace(" ", "_")
    pdf_path = f"{REPORTS_DIR}/audit_{ts}_{safe_id}_{verdict}.pdf"

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    story = []

    # ── Style definitions ─────────────────────────────────────────────
    def sty(name, **kwargs):
        return ParagraphStyle(name, **kwargs)

    title_sty = sty("title",   fontSize=20, textColor=C_CYAN,
                    fontName="Helvetica-Bold", alignment=TA_CENTER,
                    spaceAfter=4)
    sub_sty = sty("sub",     fontSize=9,  textColor=C_MUTED,
                  fontName="Helvetica",    alignment=TA_CENTER,
                  spaceAfter=16)
    heading_sty = sty("heading", fontSize=11, textColor=C_CYAN,
                      fontName="Helvetica-Bold", spaceBefore=14,
                      spaceAfter=6)
    body_sty = sty("body",    fontSize=9,  textColor=C_TEXT,
                   fontName="Helvetica",    spaceAfter=4)
    verdict_sty = sty("verdict", fontSize=28, textColor=C_GREEN
                      if verdict == "LIVE" else C_RED,
                      fontName="Helvetica-Bold", alignment=TA_CENTER,
                      spaceBefore=8, spaceAfter=8)
    reason_sty = sty("reason",  fontSize=9,  textColor=C_AMBER,
                     fontName="Helvetica-Oblique", alignment=TA_CENTER,
                     spaceAfter=12)

    # ── Header ────────────────────────────────────────────────────────
    story.append(Paragraph("LIVENESS DETECTION AUDIT REPORT", title_sty))
    story.append(Paragraph(
        "Face Liveness Detection System v2  |  REVA University", sub_sty))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=C_BORDER, spaceAfter=12))

    # ── Verdict ───────────────────────────────────────────────────────
    story.append(Paragraph(verdict, verdict_sty))
    spoof_reason = session_summary.get("spoof_reason")
    if spoof_reason:
        story.append(Paragraph(f"Reason: {spoof_reason}", reason_sty))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=C_BORDER, spaceAfter=10))

    # ── Candidate info table ──────────────────────────────────────────
    story.append(Paragraph("Candidate Information", heading_sty))
    now_str = datetime.now().strftime("%d %B %Y  %H:%M:%S")
    cand_data = [
        ["Full Name",       candidate_name],
        ["Candidate ID",    candidate_id],
        ["Date & Time",     now_str],
        ["Session Duration",
         f"{session_summary.get('duration_s', 0):.1f} seconds"],
        ["Total Frames Analysed",
         str(session_summary.get("total_frames", 0))],
    ]
    cand_table = Table(cand_data, colWidths=[5*cm, 12*cm])
    cand_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), C_PANEL),
        ("BACKGROUND",  (1, 0), (1, -1), C_BG),
        ("TEXTCOLOR",   (0, 0), (0, -1), C_MUTED),
        ("TEXTCOLOR",   (1, 0), (1, -1), C_TEXT),
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_PANEL, C_BG]),
        ("GRID",        (0, 0), (-1, -1), 0.5, C_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(cand_table)

    # ── Signal scores table ───────────────────────────────────────────
    story.append(Paragraph("Signal Score Summary", heading_sty))
    avg_c = session_summary.get("avg_confidence", 0)
    min_c = session_summary.get("min_confidence", 0)
    max_c = session_summary.get("max_confidence", 0)
    sig_data = [
        ["Metric",              "Value",    "Status"],
        ["Average Confidence",  f"{avg_c:.4f}",
         "PASS" if avg_c >= 0.48 else "FAIL"],
        ["Minimum Confidence",  f"{min_c:.4f}",
         "PASS" if min_c >= 0.20 else "FAIL"],
        ["Maximum Confidence",  f"{max_c:.4f}", "—"],
        ["Tamper Events",
         str(len(session_summary.get("tamper_events", []))),
         "CLEAN" if not session_summary.get("tamper_events") else "FLAGGED"],
    ]
    sig_table = Table(sig_data, colWidths=[8*cm, 4*cm, 5*cm])
    sig_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  C_PANEL),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  C_CYAN),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("BACKGROUND",   (0, 1), (-1, -1), C_BG),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_BG, C_PANEL]),
        ("TEXTCOLOR",    (0, 1), (-1, -1), C_TEXT),
        ("GRID",         (0, 0), (-1, -1), 0.5, C_BORDER),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(sig_table)

    # ── Tamper events table ───────────────────────────────────────────
    tamper_events = session_summary.get("tamper_events", [])
    story.append(Paragraph("Tamper Event Log", heading_sty))
    if tamper_events:
        t_data = [["Time (s)", "Event Type", "Detail"]]
        for evt in tamper_events:
            t_data.append([
                f"{evt.get('elapsed', 0):.2f}s",
                evt.get("type",   "—").replace("_", " "),
                evt.get("detail", "—"),
            ])
        t_table = Table(t_data, colWidths=[3*cm, 6*cm, 8*cm])
        t_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0),  C_PANEL),
            ("TEXTCOLOR",    (0, 0), (-1, 0),  C_RED),
            ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("BACKGROUND",   (0, 1), (-1, -1), C_BG),
            ("TEXTCOLOR",    (0, 1), (-1, -1), C_AMBER),
            ("GRID",         (0, 0), (-1, -1), 0.5, C_BORDER),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t_table)
    else:
        story.append(Paragraph("No tamper events detected.", body_sty))

    # ── Session timeline graph ────────────────────────────────────────
    if graph_path and os.path.exists(graph_path):
        story.append(Paragraph("Session Signal Timeline", heading_sty))
        story.append(Spacer(1, 0.3*cm))
        available_w = A4[0] - 4*cm
        story.append(RLImage(graph_path,
                             width=available_w,
                             height=available_w * 0.72))

    # ── Footer ────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=C_BORDER, spaceAfter=6))
    story.append(Paragraph(
        f"Generated by Face Liveness Detection System v2  |  "
        f"REVA University  |  {now_str}  |  "
        f"This report is system-generated and digitally timestamped.",
        sty("footer", fontSize=7, textColor=C_MUTED,
            fontName="Helvetica", alignment=TA_CENTER)))

    doc.build(story)
    return pdf_path
