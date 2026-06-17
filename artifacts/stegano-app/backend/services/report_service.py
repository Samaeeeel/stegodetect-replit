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
        ["Predicción ML:", result.label],
        ["Puntaje ML de esteganografía:", f"{result.probability_percent}%"],
        ["Fiabilidad ML (dentro del dominio):", result.confidence],
    ]
    story.append(_build_table(result_data, styles))
    story.append(Spacer(1, 0.5 * cm))

    # ── Explicación técnica ───────────────────────────────────────────────────
    story.append(Paragraph("Explicación Técnica", styles["section_header"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(result.explanation, styles["body_justified"]))
    story.append(Spacer(1, 0.5 * cm))

    # ── Alcance del modelo ML (bloque obligatorio honesto) ────────────────────
    story.append(Paragraph("Alcance del Modelo ML", styles["section_header"]))
    story.append(Spacer(1, 0.2 * cm))
    alcance = (
        "El modelo SRNet-lite fue <b>fine-tuned</b> con imágenes <b>BOSSBase</b> "
        "(PNG ~512×512, baja saturación, payloads p005/p010/p020) <b>más un "
        "dataset externo</b> (wallpapers, fotos de teléfono, imágenes web, "
        "capturas y material comprimido). Threshold calibrado: <b>0.46</b>. "
        "El resultado ML es <b>evidencia probabilística</b>, no una certeza absoluta. "
        "La extracción de payload sólo es concluyente cuando existe cabecera "
        "<b>StegoDetect</b> y SHA-256 válido."
    )
    story.append(Paragraph(alcance, styles["body_justified"]))
    story.append(Spacer(1, 0.5 * cm))

    # ── Nota académica ────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#9ca3af")))
    story.append(Spacer(1, 0.3 * cm))
    nota = (
        "<b>Nota académica:</b> Este reporte refleja exclusivamente el puntaje "
        "del modelo ML. Para una decisión integrada que combine evidencia ML, "
        "análisis LSB y extracción de payload, use el endpoint <i>/stego/full-analysis</i> "
        "desde la pestaña 'Analizar imagen'. El puntaje del modelo no debe "
        "interpretarse como una probabilidad absoluta universal — su calibración "
        "es válida sólo dentro del dominio de entrenamiento."
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


def generate_integrated_pdf(record: dict, output_path: Path) -> Path:
    """
    Genera un PDF basado en la decisión integrada de /stego/full-analysis.

    A diferencia de generate_pdf() (que usa solo el resultado ML del endpoint
    /analyze), este reporte prioriza la extracción LSB cuando existe payload
    válido, exactamente como lo hace la pantalla. Garantiza que el PDF y la UI
    siempre cuenten la misma historia.
    """
    try:
        _build_integrated_pdf(record, output_path)
        return output_path
    except Exception as exc:
        raise ReportGenerationError(f"Error generando reporte integrado: {exc}") from exc


def _build_integrated_pdf(record: dict, output_path: Path) -> None:
    decision      = record.get("final_decision",      {}) or {}
    ml            = record.get("ml_detection",        {}) or {}
    applicability = record.get("model_applicability", {}) or {}
    reliability   = record.get("reliability",         {}) or {}
    extraction    = record.get("payload_extraction",  {}) or {}
    lsb           = record.get("lsb_analysis",        {}) or {}

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        rightMargin=2.5*cm, leftMargin=2.5*cm,
        topMargin=2.5*cm,   bottomMargin=2.5*cm,
    )
    styles = _build_styles()
    story  = []

    # ── Encabezado ────────────────────────────────────────────────────────────
    story.append(Paragraph("REPORTE INTEGRADO DE ESTEGANOGRAFÍA Y ESTEGOANÁLISIS",
                           styles["title"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Decisión combinada: extracción LSB StegoDetect + modelo SRNet-lite + aplicabilidad de dominio",
        styles["subtitle"]
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a56db")))
    story.append(Spacer(1, 0.5*cm))

    # ── Información del análisis ──────────────────────────────────────────────
    story.append(Paragraph("Información del Análisis", styles["section_header"]))
    story.append(Spacer(1, 0.2*cm))
    story.append(_build_table([
        ["ID del Análisis:",      record.get("id", "—")],
        ["Archivo analizado:",    record.get("filename", "—")],
        ["Fecha y hora (UTC):",   _format_date(record.get("created_at", ""))],
        ["Versión del modelo:",   ml.get("model_version", "—")],
        ["Modo de operación:",    "Demostración (mock)" if ml.get("mock_mode") else "Modelo real"],
    ], styles))
    story.append(Spacer(1, 0.5*cm))

    # ── SECCIÓN 1: Decisión general del sistema ───────────────────────────────
    story.append(Paragraph("1. Decisión general del sistema", styles["section_header"]))
    story.append(Spacer(1, 0.2*cm))

    status_colors = {
        "payload_found":             colors.HexColor("#7c3aed"),  # lila
        "ml_suspicious":             colors.HexColor("#dc2626"),  # rojo
        "ml_suspicious_unverified":  colors.HexColor("#d97706"),  # ámbar (compat. PDF antiguo)
        "inconclusive_low_ml":       colors.HexColor("#d97706"),  # ámbar
        "no_evidence":               colors.HexColor("#16a34a"),  # verde (compat. PDF antiguo)
        "no_stego_evidence":         colors.HexColor("#16a34a"),  # verde
    }
    status   = decision.get("status", "no_evidence")
    dec_col  = status_colors.get(status, colors.HexColor("#374151"))
    dec_style = ParagraphStyle(
        "dec", parent=styles["body"], textColor=dec_col,
        fontSize=14, fontName="Helvetica-Bold",
    )
    story.append(Paragraph(f"▶ {decision.get('title', '—')}", dec_style))
    story.append(Spacer(1, 0.2*cm))

    evidence_labels = {
        "lsb_extraction":             "Extracción LSB StegoDetect (evidencia directa)",
        "ml_detection":               "Modelo ML — evidencia probabilística",
        "ml_detection_out_of_domain": "Modelo ML (fuera del dominio — no concluyente)",
        "none":                       "Sin evidencia detectable",
    }
    story.append(_build_table([
        ["Fuente principal:",  evidence_labels.get(decision.get("evidence_source"),
                                                   decision.get("evidence_source", "—"))],
        ["Confiabilidad:",     _reliability_human(decision.get("reliability", "—"))],
    ], styles))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(decision.get("summary", ""), styles["body_justified"]))
    story.append(Spacer(1, 0.5*cm))

    # ── SECCIÓN 2: Extracción LSB ─────────────────────────────────────────────
    story.append(Paragraph("2. Resultado de extracción LSB", styles["section_header"]))
    story.append(Spacer(1, 0.2*cm))

    has_header   = lsb.get("has_system_header", extraction.get("payload_found", False))
    found        = bool(extraction.get("payload_found"))
    sha_ok       = bool(extraction.get("sha256_valid"))
    ext_rows = [
        ["Cabecera StegoDetect:",  "Detectada" if has_header else "No detectada"],
        ["Payload encontrado:",    "Sí" if found else "No"],
        ["SHA-256 válido:",        "Sí" if sha_ok else ("No" if found else "—")],
    ]
    if found:
        ext_rows += [
            ["Tipo de payload:",  str(extraction.get("payload_type", "—"))],
            ["Nombre archivo:",   str(extraction.get("filename") or extraction.get("extracted_filename") or "—")],
            ["Tamaño (bytes):",   str(extraction.get("payload_size", "—"))],
            ["MIME:",             str(extraction.get("mime_type", "—"))],
            ["Algoritmo:",        str(extraction.get("algorithm", "LSB STEGODETECTv1"))],
        ]
        msg = extraction.get("message_text")
        if msg:
            preview = msg if len(msg) <= 200 else msg[:200] + "…"
            ext_rows.append(["Mensaje extraído:", preview])
    story.append(_build_table(ext_rows, styles))
    story.append(Spacer(1, 0.5*cm))

    # ── SECCIÓN 3: Modelo ML (título cambia si hay payload) ──────────────────
    # Cuando hay payload LSB validado, el ML es complementario, no protagonista.
    is_payload = (decision.get("status") == "payload_found") and bool(extraction.get("sha256_valid"))
    ml_section_title = (
        "3. Resultado ML complementario" if is_payload
        else "3. Resultado del modelo ML"
    )
    story.append(Paragraph(ml_section_title, styles["section_header"]))
    story.append(Spacer(1, 0.2*cm))
    prob = ml.get("probability", 0.0) or 0.0
    thr  = ml.get("threshold")   or 0.0404
    story.append(_build_table([
        ["Puntaje ML de esteganografía:", f"{prob*100:.1f}%"],
        ["Threshold del modelo:",         f"{thr*100:.2f}%"],
        ["Predicción interna ML:",        ml.get("label", "—")],
        ["Versión del modelo:",           ml.get("model_version", "—")],
        ["Fiabilidad de interpretación:", reliability.get("label", "—")],
    ], styles))
    story.append(Spacer(1, 0.3*cm))
    if is_payload:
        story.append(Paragraph(
            "<i>El modelo ML asignó un puntaje bajo; sin embargo, la decisión final "
            "se basa en la extracción LSB validada. Puntajes ML bajos son comunes "
            "en payloads pequeños que no alteran significativamente la distribución "
            "estadística.</i>",
            styles["body_justified"]
        ))
        story.append(Spacer(1, 0.3*cm))
    elif found and prob < thr:
        story.append(Paragraph(
            "<i>Nota: el puntaje ML es bajo, pero la decisión final se basa en la "
            "extracción LSB validada.</i>",
            styles["body_justified"]
        ))
        story.append(Spacer(1, 0.3*cm))

    # ── SECCIÓN 4: Aplicabilidad del modelo ───────────────────────────────────
    story.append(Paragraph("4. Aplicabilidad del modelo (dominio)", styles["section_header"]))
    story.append(Spacer(1, 0.2*cm))
    domain_labels = {
        "in_domain":              "Dentro del dominio del modelo",
        "possibly_out_of_domain": "Parcialmente fuera del dominio",
        "out_of_domain":          "Fuera del dominio del modelo",
    }
    story.append(_build_table([
        ["Estado de dominio:",       domain_labels.get(applicability.get("domain_status"),
                                                       applicability.get("domain_status", "—"))],
        ["Compatibilidad:",          f"{applicability.get('compatibility_score', '—')}/100"],
        ["Formato:",                 str(applicability.get("image_format", "—"))],
        ["Dimensiones:",             "×".join(map(str, applicability.get("image_size", []))) or "—"],
        ["Modo de color:",           str(applicability.get("image_mode", "—"))],
        ["Saturación de color:",     str(applicability.get("color_saturation", "—"))],
    ], styles))
    reasons = applicability.get("reasons", [])
    if reasons:
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("<b>Razones:</b>", styles["body"]))
        for r in reasons:
            story.append(Paragraph(f"• {r}", styles["body"]))
    story.append(Spacer(1, 0.5*cm))

    # ── SECCIÓN 5: Conclusión técnica ─────────────────────────────────────────
    story.append(Paragraph("5. Conclusión técnica", styles["section_header"]))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(_build_conclusion(decision, extraction, ml, applicability),
                           styles["body_justified"]))
    story.append(Spacer(1, 0.5*cm))

    # ── Alcance del modelo (bloque honesto obligatorio) ───────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#9ca3af")))
    story.append(Spacer(1, 0.3*cm))
    alcance = (
        "<b>Alcance del modelo ML:</b> SRNet-lite <b>fine-tuned</b> con BOSSBase "
        "+ dataset externo (wallpapers, fotos de teléfono, imágenes web, capturas, "
        "material comprimido). Threshold calibrado: <b>0.46</b>. El puntaje es "
        "<b>evidencia probabilística</b>; en imágenes muy alejadas de los datos de "
        "entrenamiento debe considerarse <b>orientativo, no concluyente</b>. La "
        "extracción de payload sólo es concluyente cuando existe cabecera "
        "<b>StegoDetect</b> y SHA-256 válido."
    )
    story.append(Paragraph(alcance, styles["note"]))

    if ml.get("mock_mode"):
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            "<b>Advertencia:</b> Reporte generado en <b>modo demostración</b>. "
            "El puntaje ML es simulado.",
            styles["warning"]
        ))

    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph(
        f"Generado el {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        styles["footer"]
    ))

    doc.build(story)


def _reliability_human(r: str) -> str:
    return {"high": "Alta", "medium": "Media", "low": "Baja"}.get(r, r)


def _build_conclusion(decision: dict, extraction: dict, ml: dict, applicability: dict) -> str:
    """
    Construye la conclusión técnica en lenguaje natural.
    Soporta los cuatro status del nuevo esquema de decisión:
      payload_found | ml_suspicious | inconclusive_low_ml | no_stego_evidence
    Mantiene compatibilidad con status antiguos (ml_suspicious_unverified, no_evidence).
    """
    status = decision.get("status")
    prob   = (ml.get("probability") or 0.0) * 100
    thr    = (ml.get("threshold") or 0.46) * 100

    if status == "payload_found":
        ptype  = extraction.get("payload_type", "desconocido")
        size   = extraction.get("payload_size", "—")
        fname  = extraction.get("filename") or extraction.get("extracted_filename") or "—"
        alg    = extraction.get("algorithm_detected") or extraction.get("algorithm", "LSB STEGODETECTv1")
        bpc    = extraction.get("bits_per_channel_detected") or extraction.get("bits_per_channel", "—")
        chs    = extraction.get("channels_detected") or extraction.get("channels", [])
        chs_str = ", ".join(chs) if chs else "—"
        return (
            f"<b>Mensaje oculto encontrado.</b> El sistema recuperó y validó un "
            f"payload <b>{ptype}</b> ({size} bytes, archivo «{fname}») mediante "
            f"extracción LSB con cabecera STEGODETECTv1 y SHA-256 válido. "
            f"Parámetros de inserción detectados: algoritmo={alg}, "
            f"bits/canal={bpc}, canales={chs_str}. "
            f"El modelo ML asignó un puntaje de {prob:.1f}%; sin embargo, la "
            f"evidencia directa de extracción tiene prioridad y es "
            f"matemáticamente verificable."
        )
    if status == "ml_suspicious":
        return (
            f"<b>Posible esteganografía detectada.</b> El modelo ML "
            f"asignó un puntaje de {prob:.1f}%, superando el umbral calibrado "
            f"({thr:.1f}%). No se recuperó payload StegoDetect: el posible payload "
            f"podría haber sido insertado con otra herramienta, estar cifrado, "
            f"o tratarse de una falsa alarma estadística. Se recomienda validación "
            f"adicional con otras herramientas forenses."
        )
    if status in ("inconclusive_low_ml",):
        return (
            f"<b>Sin evidencia concluyente.</b> El modelo ML asignó un puntaje de "
            f"{prob:.1f}%, en la banda intermedia (30–{thr:.0f}%). "
            f"Hay cierta variación estadística en los LSBs, pero insuficiente para "
            f"concluir presencia de esteganografía según el umbral calibrado del "
            f"modelo. No se encontró payload StegoDetect recuperable."
        )
    if status in ("no_stego_evidence",):
        return (
            f"<b>Sin evidencia esteganográfica detectable.</b> El puntaje ML fue "
            f"de {prob:.1f}%, por debajo del umbral de baja evidencia (30%). "
            f"No se encontró cabecera StegoDetect ni payload recuperable. Según "
            f"los métodos aplicados, la imagen no presenta evidencia esteganográfica "
            f"detectable. Esto no descarta técnicas externas, cifradas o incompatibles "
            f"con el protocolo StegoDetect."
        )
    # Compatibilidad con status antiguos
    if status == "ml_suspicious_unverified":
        return (
            f"<b>Resultado ML no concluyente.</b> El modelo asignó un puntaje de "
            f"{prob:.1f}%, pero la imagen se encuentra fuera del dominio de "
            f"entrenamiento del modelo. En estas condiciones el puntaje debe "
            f"interpretarse como evidencia probabilística, no como detección "
            f"concluyente. Tampoco se encontró payload StegoDetect recuperable."
        )
    # Fallback genérico (no_evidence y desconocidos)
    return (
        f"<b>Sin evidencia detectable.</b> No se encontró payload compatible con "
        f"el formato StegoDetect y el puntaje ML fue de {prob:.1f}%, por debajo "
        f"del umbral del modelo. Esto indica baja evidencia detectable dentro del "
        f"alcance evaluado, pero no descarta técnicas externas, cifradas o no "
        f"compatibles con el sistema."
    )


def _format_date(iso_string: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso_string
