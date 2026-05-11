"""
Servicio de generación de reportes PDF académicos.
Usa ReportLab para producir un documento formateado con los resultados
del análisis de esteganografía.
"""

import io
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

from backend.domain.analysis_result import AnalysisResult
from backend.core.config import REPORTS_DIR, APP_TITLE
from backend.core.exceptions import ReportGenerationError


def generate_pdf(result: AnalysisResult) -> Path:
    """
    Genera un reporte PDF académico para el análisis dado.

    Retorna la ruta al archivo PDF generado en REPORTS_DIR.
    Lanza ReportGenerationError si algo falla.
    """
    try:
        pdf_path = REPORTS_DIR / f"reporte_{result.id}.pdf"
        _build_pdf(result, pdf_path)
        return pdf_path
    except Exception as exc:
        raise ReportGenerationError(f"Error generando el reporte PDF: {exc}") from exc


# ── Construcción del PDF ──────────────────────────────────────────────────────

def _build_pdf(result: AnalysisResult, output_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2.5 * cm,
        leftMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )

    styles = _build_styles()
    story = []

    # ── Encabezado ────────────────────────────────────────────────────────────
    story.append(Paragraph(APP_TITLE.upper(), styles["title"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Reporte de Análisis de Esteganografía", styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a56db")))
    story.append(Spacer(1, 0.5 * cm))

    # ── Información del análisis ──────────────────────────────────────────────
    story.append(Paragraph("Información del Análisis", styles["section_header"]))
    story.append(Spacer(1, 0.2 * cm))

    info_data = [
        ["ID del Análisis:", result.id],
        ["Archivo analizado:", result.filename],
        ["Fecha y hora (UTC):", _format_date(result.created_at)],
        ["Versión del modelo:", result.model_version],
        ["Modo de operación:", "Demostración (mock)" if result.mock_mode else "Modelo real"],
    ]
    story.append(_build_table(info_data, styles))
    story.append(Spacer(1, 0.5 * cm))

    # ── Resultado principal ───────────────────────────────────────────────────
    story.append(Paragraph("Resultado del Análisis", styles["section_header"]))
    story.append(Spacer(1, 0.2 * cm))

    result_color = colors.HexColor("#dc2626") if result.has_hidden_message else colors.HexColor("#16a34a")
    result_style = ParagraphStyle(
        "result_label",
        parent=styles["body"],
        textColor=result_color,
        fontSize=14,
        fontName="Helvetica-Bold",
    )
    story.append(Paragraph(f"▶ {result.label}", result_style))
    story.append(Spacer(1, 0.3 * cm))

    result_data = [
        ["Predicción:", result.label],
        ["Probabilidad:", f"{result.probability_percent}%"],
        ["Nivel de confianza:", result.confidence],
    ]
    story.append(_build_table(result_data, styles))
    story.append(Spacer(1, 0.5 * cm))

    # ── Explicación técnica ───────────────────────────────────────────────────
    story.append(Paragraph("Explicación Técnica", styles["section_header"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(result.explanation, styles["body_justified"]))
    story.append(Spacer(1, 0.5 * cm))

    # ── Nota académica ────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#9ca3af")))
    story.append(Spacer(1, 0.3 * cm))
    nota = (
        "<b>Nota académica:</b> El resultado generado por este sistema es probabilístico "
        "y debe interpretarse como apoyo técnico, no como prueba absoluta. "
        "La precisión del análisis depende directamente del modelo entrenado y del "
        "conjunto de datos utilizado para su entrenamiento."
    )
    story.append(Paragraph(nota, styles["note"]))

    if result.mock_mode:
        story.append(Spacer(1, 0.2 * cm))
        mock_nota = (
            "<b>Advertencia:</b> Este reporte fue generado en <b>modo demostración</b>. "
            "No existe un modelo entrenado cargado; los resultados son simulados "
            "y no tienen validez científica."
        )
        story.append(Paragraph(mock_nota, styles["warning"]))

    # ── Pie de página ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        f"Generado el {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        styles["footer"]
    ))

    doc.build(story)


# ── Estilos ───────────────────────────────────────────────────────────────────

def _build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"],
            fontSize=14, fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1a56db"),
            alignment=TA_CENTER, spaceAfter=6
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"],
            fontSize=11, textColor=colors.HexColor("#374151"),
            alignment=TA_CENTER, spaceAfter=8
        ),
        "section_header": ParagraphStyle(
            "section_header", parent=base["Heading2"],
            fontSize=12, fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1e40af"),
            spaceBefore=6, spaceAfter=4
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontSize=10, leading=14
        ),
        "body_justified": ParagraphStyle(
            "body_justified", parent=base["Normal"],
            fontSize=10, leading=14, alignment=TA_JUSTIFY
        ),
        "note": ParagraphStyle(
            "note", parent=base["Normal"],
            fontSize=9, leading=13,
            textColor=colors.HexColor("#374151"),
            backColor=colors.HexColor("#f3f4f6"),
            borderPadding=(6, 6, 6, 6),
        ),
        "warning": ParagraphStyle(
            "warning", parent=base["Normal"],
            fontSize=9, leading=13,
            textColor=colors.HexColor("#92400e"),
            backColor=colors.HexColor("#fef3c7"),
            borderPadding=(6, 6, 6, 6),
        ),
        "footer": ParagraphStyle(
            "footer", parent=base["Normal"],
            fontSize=8, textColor=colors.HexColor("#9ca3af"),
            alignment=TA_CENTER
        ),
    }


def _build_table(data: list, styles: dict):
    table = Table(data, colWidths=[5 * cm, 11 * cm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#374151")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#111827")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.HexColor("#f9fafb"), colors.white]),
    ]))
    return table


def _format_date(iso_string: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso_string
