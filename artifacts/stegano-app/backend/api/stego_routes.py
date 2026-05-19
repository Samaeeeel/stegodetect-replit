"""
Endpoints de esteganografía LSB.

Rutas nuevas (no rompen las existentes):
  POST /stego/embed/text     → Ocultar texto en imagen
  POST /stego/embed/file     → Ocultar archivo en imagen
  POST /stego/extract        → Extraer payload del sistema
  POST /stego/full-analysis  → ML + LSB + extracción combinados
  GET  /stego/download/{id}/{type} → Descargar artefactos generados
"""

import mimetypes
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from backend.core.config import UPLOADS_DIR, STEGO_ARTIFACTS_DIR
from backend.services.steganography_service import (
    LSBSteganographyService,
    StegoCapacityError,
)
from backend.services import model_service

stego_router = APIRouter(prefix="/stego", tags=["steganography"])

# Instancia única del servicio (stateless — seguro para compartir)
_svc = LSBSteganographyService()


# ── POST /stego/embed/text ────────────────────────────────────────────────────

@stego_router.post("/embed/text", summary="Ocultar texto en imagen")
async def embed_text(
    cover_image:      UploadFile = File(..., description="Imagen cover PNG/JPG"),
    message:          str        = Form(..., description="Texto a ocultar"),
    bits_per_channel: int        = Form(1,   description="Bits LSB por canal (1–2)"),
    channels:         str        = Form("RGB", description="Canales: RGB, RG, R, etc."),
):
    """
    Oculta un mensaje de texto dentro de una imagen usando LSB.
    La imagen de salida siempre es PNG para preservar los LSBs.
    """
    cover_path = await _save_upload(cover_image)
    ch_tuple   = _parse_channels(channels)

    try:
        result = _svc.embed_payload(
            cover_image_path  = cover_path,
            payload_bytes     = message.encode("utf-8"),
            payload_type      = "text",
            original_filename = "message.txt",
            mime_type         = "text/plain",
            bits_per_channel  = bits_per_channel,
            channels          = ch_tuple,
        )
    except StegoCapacityError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en inserción LSB: {exc}")

    aid = result["artifact_id"]
    return JSONResponse({
        "success":           True,
        "artifact_id":       aid,
        "stego_image_url":   f"/stego/download/{aid}/image",
        "download_url":      f"/stego/download/{aid}/image",
        "csv_url":           f"/stego/download/{aid}/csv",
        "map_url":           f"/stego/download/{aid}/map",
        "capacity":          result["capacity"],
        "payload":           result["payload"],
        "positions_summary": result["positions_summary"],
        "first_positions":   result["first_positions"],
        "technical":         result["technical"],
    })


# ── POST /stego/embed/file ────────────────────────────────────────────────────

@stego_router.post("/embed/file", summary="Ocultar archivo en imagen")
async def embed_file(
    cover_image:      UploadFile = File(..., description="Imagen cover PNG/JPG"),
    payload_file:     UploadFile = File(..., description="Archivo a ocultar (PDF, TXT, PNG, JPG, binario)"),
    bits_per_channel: int        = Form(1,     description="Bits LSB por canal (1–2)"),
    channels:         str        = Form("RGB", description="Canales: RGB, RG, R, etc."),
):
    """
    Oculta un archivo (PDF, TXT, imagen, binario) dentro de una imagen cover.
    Soporta payloads hasta 2 MB. La salida siempre es PNG.
    """
    cover_path    = await _save_upload(cover_image)
    payload_bytes = await payload_file.read()
    filename      = payload_file.filename or "archivo.bin"
    mime_type     = (
        payload_file.content_type
        or mimetypes.guess_type(filename)[0]
        or "application/octet-stream"
    )

    payload_type = {
        "application/pdf": "pdf",
        "text/plain":      "text",
        "image/png":       "image",
        "image/jpeg":      "image",
        "image/jpg":       "image",
    }.get(mime_type, "binary")

    try:
        result = _svc.embed_payload(
            cover_image_path  = cover_path,
            payload_bytes     = payload_bytes,
            payload_type      = payload_type,
            original_filename = filename,
            mime_type         = mime_type,
            bits_per_channel  = bits_per_channel,
            channels          = _parse_channels(channels),
        )
    except StegoCapacityError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en inserción LSB: {exc}")

    aid = result["artifact_id"]
    return JSONResponse({
        "success":           True,
        "artifact_id":       aid,
        "stego_image_url":   f"/stego/download/{aid}/image",
        "download_url":      f"/stego/download/{aid}/image",
        "csv_url":           f"/stego/download/{aid}/csv",
        "map_url":           f"/stego/download/{aid}/map",
        "capacity":          result["capacity"],
        "payload":           result["payload"],
        "positions_summary": result["positions_summary"],
        "first_positions":   result["first_positions"],
        "technical":         result["technical"],
    })


# ── POST /stego/extract ───────────────────────────────────────────────────────

@stego_router.post("/extract", summary="Extraer payload del sistema")
async def extract_payload(
    stego_image:      UploadFile = File(..., description="Imagen stego PNG generada por StegoDetect"),
    bits_per_channel: int        = Form(1,     description="Bits LSB por canal usados al insertar"),
    channels:         str        = Form("RGB", description="Canales usados al insertar"),
):
    """
    Intenta extraer un payload del sistema desde la imagen.

    Solo puede recuperar payloads generados por StegoDetect (tienen el magic header STEGODETECTv1).
    Si la imagen no fue generada por el sistema, informa claramente que no hay payload compatible.
    """
    stego_path = await _save_upload(stego_image)

    try:
        result = _svc.extract_payload(
            stego_image_path = stego_path,
            bits_per_channel = bits_per_channel,
            channels         = _parse_channels(channels),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en extracción: {exc}")

    response = {"success": True, **result}

    # Si hay un archivo extraído, agregar la URL de descarga
    if result.get("payload_found") and result.get("extracted_filename"):
        aid = result.get("artifact_id", "")
        response["download_payload_url"] = f"/stego/download/{aid}/payload"

    return JSONResponse(response)


# ── POST /stego/full-analysis ─────────────────────────────────────────────────

@stego_router.post("/full-analysis", summary="Análisis completo: ML + LSB + extracción")
async def full_analysis(
    image: UploadFile = File(..., description="Imagen a analizar completamente"),
):
    """
    Análisis combinado:
    1. Detector ML (SRNet-lite) → probabilidad de esteganografía
    2. Análisis LSB técnico → distribución de bits, entropía, capacidad
    3. Intento de extracción → payload del sistema si existe
    4. Resumen técnico integrado
    """
    image_path = await _save_upload(image)

    # 1. Detección ML
    try:
        prediction, probability = model_service.predict(image_path)
        ml_result = {
            "prediction":    prediction,
            "probability":   probability,
            "label":         "Con mensaje oculto" if prediction == "stego" else "Sin mensaje oculto",
            "threshold":     model_service.get_threshold(),
            "model_version": model_service.get_model_version(),
            "mock_mode":     model_service.is_mock_mode(),
        }
    except Exception as exc:
        ml_result = {"error": str(exc)}

    # 2. Análisis LSB
    try:
        lsb_analysis = _svc.analyze_lsb_structure(image_path)
    except Exception as exc:
        lsb_analysis = {"error": str(exc)}

    # 3. Intento de extracción
    try:
        extraction = _svc.extract_payload(image_path)
        if extraction.get("payload_found") and extraction.get("extracted_filename"):
            aid = extraction.get("artifact_id", "")
            extraction["download_payload_url"] = f"/stego/download/{aid}/payload"
    except Exception as exc:
        extraction = {"payload_found": False, "error": str(exc)}

    # 4. Capacidad de la imagen
    try:
        capacity = _svc.calculate_capacity(image_path)
    except Exception as exc:
        capacity = {"error": str(exc)}

    final_decision = _build_final_decision(ml_result, lsb_analysis, extraction)

    return JSONResponse({
        "final_decision":     final_decision,
        "ml_detection":       ml_result,
        "lsb_analysis":       lsb_analysis,
        "payload_extraction": extraction,
        "capacity":           capacity,
        "technical_summary":  _build_summary(ml_result, lsb_analysis, extraction),
    })


# ── GET /stego/download/{artifact_id}/{artifact_type} ─────────────────────────

@stego_router.get(
    "/download/{artifact_id}/{artifact_type}",
    summary="Descargar artefacto generado",
)
async def download_artifact(artifact_id: str, artifact_type: str):
    """
    Descarga artefactos generados por el sistema:
    - image   → imagen stego PNG
    - csv     → CSV de posiciones de bits
    - map     → mapa LSB PNG (rojo = píxeles modificados)
    - payload → archivo extraído del payload
    """
    if artifact_type == "image":
        path = STEGO_ARTIFACTS_DIR / f"stego_{artifact_id}.png"
        if not path.exists():
            raise HTTPException(404, "Imagen stego no encontrada.")
        return FileResponse(
            str(path), media_type="image/png",
            filename=f"stego_{artifact_id[:8]}.png"
        )

    if artifact_type == "csv":
        path = STEGO_ARTIFACTS_DIR / f"positions_{artifact_id}.csv"
        if not path.exists():
            raise HTTPException(404, "CSV de posiciones no encontrado.")
        return FileResponse(
            str(path), media_type="text/csv",
            filename=f"positions_{artifact_id[:8]}.csv"
        )

    if artifact_type == "map":
        path = STEGO_ARTIFACTS_DIR / f"lsb_map_{artifact_id}.png"
        if not path.exists():
            raise HTTPException(404, "Mapa LSB no encontrado.")
        return FileResponse(
            str(path), media_type="image/png",
            filename=f"lsb_map_{artifact_id[:8]}.png"
        )

    if artifact_type == "payload":
        matches = list(STEGO_ARTIFACTS_DIR.glob(f"extracted_{artifact_id}*"))
        if not matches:
            raise HTTPException(404, "Payload extraído no encontrado.")
        path = matches[0]
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return FileResponse(str(path), media_type=mime, filename=path.name)

    raise HTTPException(400, f"Tipo de artefacto desconocido: '{artifact_type}'. "
                             "Usa: image, csv, map, payload")


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _save_upload(upload: UploadFile) -> Path:
    """Guarda el archivo subido en UPLOADS_DIR con nombre único."""
    content  = await upload.read()
    filename = upload.filename or "upload.bin"
    path     = UPLOADS_DIR / f"{uuid.uuid4()}_{Path(filename).name}"
    path.write_bytes(content)
    return path


def _parse_channels(channels_str: str) -> tuple:
    """Convierte string 'RGB' / 'RG' / 'R' etc. a tupla ('R','G','B')."""
    valid = {"R", "G", "B"}
    result = []
    seen   = set()
    for c in channels_str.upper():
        if c in valid and c not in seen:
            result.append(c)
            seen.add(c)
    return tuple(result) if result else ("R", "G", "B")


def _build_final_decision(ml: dict, lsb: dict, extraction: dict) -> dict:
    """
    Determina la decisión final integrando las dos fuentes de evidencia:
    1. Extracción LSB con cabecera STEGODETECTv1 (evidencia directa — prioridad alta)
    2. Detector ML SRNet-lite (evidencia probabilística — secundaria)

    Casos:
      A. payload_found + sha256_valid → "Mensaje oculto encontrado" (LSB gana)
      B. ML stego/sospechoso, sin cabecera → "Posible mensaje oculto detectado"
      C. Sin evidencia en ninguno → "Sin evidencia detectable"
    """
    # Caso A: extracción LSB exitosa con integridad verificada — máxima prioridad
    if extraction.get("payload_found") and extraction.get("sha256_valid"):
        return {
            "status":          "payload_found",
            "title":           "Mensaje oculto encontrado",
            "summary":         (
                "Se encontró y validó un payload mediante extracción LSB del sistema "
                "(cabecera STEGODETECTv1 detectada, SHA-256 verificado)."
            ),
            "evidence_source": "lsb_extraction",
        }

    # Caso B: ML detecta algo pero sin cabecera del sistema
    pred = ml.get("prediction", "cover")
    prob = ml.get("probability", 0.0)
    thr  = ml.get("threshold") or 0.0404

    if pred == "stego" or prob >= thr:
        return {
            "status":          "ml_suspicious",
            "title":           "Posible mensaje oculto detectado",
            "summary":         (
                "El modelo ML detectó patrones estadísticos compatibles con esteganografía, "
                "pero no se encontró una cabecera StegoDetect compatible. "
                "El contenido puede usar otro algoritmo, formato o clave."
            ),
            "evidence_source": "ml_detection",
        }

    # Caso C: sin evidencia en ninguna fuente
    return {
        "status":          "no_evidence",
        "title":           "Sin evidencia detectable",
        "summary":         (
            "No se encontró payload compatible con el formato del sistema "
            "y el modelo ML no detectó patrones estadísticos significativos."
        ),
        "evidence_source": "none",
    }


def _build_summary(ml: dict, lsb: dict, extraction: dict) -> str:
    """Construye un resumen técnico legible integrando los tres módulos."""
    parts = []

    if "prediction" in ml:
        pred = ml["prediction"]
        prob = ml.get("probability", 0)
        parts.append(
            f"ML ({ml.get('model_version', 'N/D')}): "
            f"{'STEGO' if pred == 'stego' else 'COVER'} ({prob * 100:.1f}%)"
        )

    if "has_system_header" in lsb:
        parts.append(
            "Cabecera StegoDetect: detectada." if lsb["has_system_header"]
            else "Cabecera StegoDetect: no encontrada."
        )
        if "randomness_note" in lsb:
            parts.append(lsb["randomness_note"])

    if extraction.get("payload_found"):
        ptype = extraction.get("payload_type", "desconocido")
        size  = extraction.get("payload_size", 0)
        ok    = "OK" if extraction.get("sha256_valid") else "FALLO"
        parts.append(
            f"Payload extraído: tipo={ptype}, {size} bytes, SHA-256={ok}."
        )
    elif extraction.get("payload_found") is False:
        parts.append("Payload: no encontrado o formato incompatible.")

    return " | ".join(parts) if parts else "Análisis completado."
