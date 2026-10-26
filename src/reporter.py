import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY


def generate_pdf_report(doc_data: dict, qa_history: list[dict] = None) -> bytes:
    """Generate a clean PDF document intelligence report using ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocuTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1E293B"),
        alignment=TA_LEFT
    )

    tagline_style = ParagraphStyle(
        'DocuTagline',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#6366F1")
    )

    section_heading = ParagraphStyle(
        'DocuSection',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#4F46E5"),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'DocuBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        alignment=TA_JUSTIFY
    )

    bullet_style = ParagraphStyle(
        'DocuBullet',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
        leftIndent=15,
        spaceBefore=3,
        spaceAfter=3
    )

    evidence_style = ParagraphStyle(
        'DocuEvidence',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
        leftIndent=20
    )

    story = []

    # Title Banner
    story.append(Paragraph("DOCUMIND AI", title_style))
    story.append(Paragraph("Turn documents into knowledge.", tagline_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#6366F1"), spaceBefore=2, spaceAfter=12))

    # Metadata Table
    filename = doc_data.get("filename", "Document Report")
    file_type = doc_data.get("file_type", "DOC")
    page_count = doc_data.get("page_count", 1)
    word_count = doc_data.get("word_count", 0)
    now_str = datetime.now().strftime("%B %d, %Y")

    meta_data = [
        [Paragraph(f"<b>Document:</b> {filename}", body_style), Paragraph(f"<b>Analyzed:</b> {now_str}", body_style)],
        [Paragraph(f"<b>Format:</b> {file_type}", body_style), Paragraph(f"<b>Stats:</b> {page_count} Pages | {word_count:,} Words", body_style)]
    ]
    t_meta = Table(meta_data, colWidths=[260, 260])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#F1F5F9")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 14))

    # Executive Summary
    summaries = doc_data.get("summaries", {})
    exec_summary = summaries.get("executive", "N/A")
    story.append(Paragraph("Executive Summary", section_heading))
    story.append(Paragraph(exec_summary, body_style))
    story.append(Spacer(1, 12))

    # Key Points
    key_points = summaries.get("key_points", [])
    if key_points:
        story.append(Paragraph("Key Takeaways & Points", section_heading))
        for kp in key_points[:8]:
            story.append(Paragraph(f"• {kp}", bullet_style))
        story.append(Spacer(1, 12))

    # Key Topics
    topics = doc_data.get("topics", [])
    if topics:
        story.append(Paragraph("Major Key Topics", section_heading))
        topic_texts = [f"<b>{t['topic']}</b> ({t['importance']}%)" for t in topics[:8]]
        story.append(Paragraph(" &nbsp;|&nbsp; ".join(topic_texts), body_style))
        story.append(Spacer(1, 12))

    # Entities & Structured Info
    entities_data = doc_data.get("entities", {})
    counts = entities_data.get("counts", {})
    important_info = entities_data.get("important_info", {})

    if counts:
        story.append(Paragraph("Document Entity Breakdown", section_heading))
        ent_matrix = [
            [Paragraph(f"<b>{k}:</b> {v}", body_style) for k, v in list(counts.items())[:3]],
            [Paragraph(f"<b>{k}:</b> {v}", body_style) for k, v in list(counts.items())[3:6]]
        ]
        t_ent = Table(ent_matrix, colWidths=[173, 173, 174])
        t_ent.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EEF2FF")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#C7D2FE")),
            ('PADDING', (0,0), (-1,-1), 6)
        ]))
        story.append(t_ent)
        story.append(Spacer(1, 10))

    if important_info:
        dates = important_info.get("dates", [])
        numbers = important_info.get("numbers", [])
        measurements = important_info.get("measurements", [])
        if dates or numbers or measurements:
            story.append(Paragraph("Extracted Important Data & Metrics", section_heading))
            if dates:
                story.append(Paragraph(f"<b>Key Dates:</b> {', '.join(dates[:6])}", bullet_style))
            if numbers:
                story.append(Paragraph(f"<b>Numbers & Percentages:</b> {', '.join(numbers[:6])}", bullet_style))
            if measurements:
                story.append(Paragraph(f"<b>Measurements & Params:</b> {', '.join(measurements[:6])}", bullet_style))
            story.append(Spacer(1, 12))

    # Document Structure Outline
    structure = doc_data.get("structure", [])
    if structure:
        story.append(Paragraph("Document Structure & Table of Contents", section_heading))
        for sec in structure[:10]:
            story.append(Paragraph(f"<b>Page {sec.get('page', 1)}:</b> {sec.get('heading', 'Section')} ({sec.get('word_count', 0)} words)", bullet_style))
        story.append(Spacer(1, 12))

    # Q&A History
    if qa_history:
        story.append(Paragraph("Grounded Q&A & Evidence", section_heading))
        for item in qa_history[:4]:
            story.append(Paragraph(f"<b>Q: {item['question']}</b>", ParagraphStyle('QStyle', parent=body_style, fontName='Helvetica-Bold', textColor=colors.HexColor("#1E1B4B"))))
            story.append(Paragraph(f"<b>A:</b> {item['answer']}", body_style))
            if item.get("evidence"):
                ev = item["evidence"][0]
                story.append(Paragraph(f"<i>Source Evidence (Page {ev.get('page', 1)}): \"{ev.get('text', '')[:180]}...\"</i>", evidence_style))
            story.append(Spacer(1, 6))

    doc.build(story)
    return buffer.getvalue()
