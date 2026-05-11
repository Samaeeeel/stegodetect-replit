"""
Rutas de la API REST.
Define los endpoints que expone el sistema:
  GET  /           → Frontend HTML
  POST /analyze    → Análisis de imagen
  GET  /report/{id}→ Descarga de reporte PDF
  GET  /health     → Estado del sistema
"""

import json
import uuid
import shutil
import logging
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from backend.core.config import UPLOADS_DIR, REPORTS_DIR, RESULTS_FILE
from backend.core.exceptions import (
    ImageValidationError, ModelInferenceError, ReportGenerationError, AnalysisNotFoundError
)
from backend.domain.analysis_result import AnalysisResult
from backend.services import image_validator, model_service, report_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Ruta al frontend compilado (relativa al directorio raíz del proyecto)
_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


# ── GET / ─────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Sirve la interfaz web principal."""
    index_path = _FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend no encontrado.")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


# ── POST /analyze ─────────────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Recibe una imagen, la valida, ejecuta la inferencia y devuelve el resultado.

    Flujo:
      1. Leer contenido del archivo
      2. Validar (extensión, tamaño, integridad, dimensiones)
      3. Guardar en uploads/
      4. Preprocesar e inferir
      5. Construir AnalysisResult
      6. Persistir en results.json
      7. Devolver resultado como JSON
    """
    # 1. Leer contenido
    content = await file.read()
    filename = file.filename or "unknown.png"

    # 2. Validar imagen
    try:
        image_validator.validate_image(filename, content)
    except ImageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # 3. Guardar archivo en uploads/
    analysis_id = str(uuid.uuid4())
    safe_filename = f"{analysis_id}_{Path(filename).name}"
    upload_path = UPLOADS_DIR / safe_filename

    try:
        upload_path.write_bytes(content)
    except Exception as exc:
        logger.error("Error guardando imagen: %s", exc)
        raise HTTPException(status_code=500, detail="Error guardando la imagen.")

    # 4. Inferencia
    try:
        prediction, probability = model_service.predict(upload_path)
    except ModelInferenceError as exc:
        logger.error("Error en inferencia: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error en análisis: {exc}")

    # 5. Construir resultado
    confidence = model_service._get_confidence(probability)
    explanation = model_service._get_explanation(
        prediction, probability, model_service.is_mock_mode()
    )

    result = AnalysisResult(
        id=analysis_id,
        filename=filename,
        prediction=prediction,
        probability=probability,
        confidence=confidence,
        explanation=explanation,
        model_version=model_service.get_model_version(),
        mock_mode=model_service.is_mock_mode(),
    )

    # 6. Persistir resultado
    _save_result(result)

    logger.info("Análisis completado: id=%s prediction=%s mock=%s",
                analysis_id, prediction, result.mock_mode)

    # 7. Responder
    return JSONResponse(content=result.to_dict())


# ── GET /report/{analysis_id} ─────────────────────────────────────────────────

@router.get("/report/{analysis_id}")
async def download_report(analysis_id: str):
    """
    Genera (o reutiliza) el reporte PDF para el análisis indicado
    y lo devuelve como descarga.
    """
    # Buscar el análisis en el registro
    try:
        result = _load_result(analysis_id)
    except AnalysisNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró el análisis con ID: {analysis_id}"
        )

    # Si el PDF ya existe, devolverlo directamente
    pdf_path = REPORTS_DIR / f"reporte_{analysis_id}.pdf"
    if not pdf_path.exists():
        try:
            pdf_path = report_service.generate_pdf(result)
        except ReportGenerationError as exc:
            logger.error("Error generando PDF: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"reporte_esteganografia_{analysis_id[:8]}.pdf",
    )


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """Devuelve el estado del sistema e indica si el modelo real está cargado."""
    return {
        "status": "ok",
        "mock_mode": model_service.is_mock_mode(),
        "model_version": model_service.get_model_version(),
        "message": (
            "Sistema funcionando en modo demostración. "
            "Coloca el checkpoint en ml/checkpoints/ para activar el modelo real."
        ) if model_service.is_mock_mode() else (
            "Modelo real cargado y listo para inferencia."
        ),
    }


# ── Persistencia de resultados ────────────────────────────────────────────────

def _save_result(result: AnalysisResult) -> None:
    """Guarda el resultado en results.json (appended como lista JSON)."""
    data: list = []
    if RESULTS_FILE.exists():
        try:
            data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = []
    data.append(result.to_dict())
    RESULTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_result(analysis_id: str) -> AnalysisResult:
    """Carga un resultado por su ID desde results.json."""
    if not RESULTS_FILE.exists():
        raise AnalysisNotFoundError(analysis_id)
    try:
        data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        raise AnalysisNotFoundError(analysis_id)
    for entry in data:
        if entry.get("id") == analysis_id:
            return AnalysisResult.from_dict(entry)
    raise AnalysisNotFoundError(analysis_id)
