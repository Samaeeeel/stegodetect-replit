"""
Servicio de validación de imágenes.
Aplica múltiples capas de verificación antes de pasar la imagen al modelo,
garantizando que solo imágenes válidas y seguras lleguen a la inferencia.
"""

import io
from pathlib import Path
from PIL import Image

from backend.core.config import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MIN_IMAGE_WIDTH,
    MIN_IMAGE_HEIGHT,
)
from backend.core.exceptions import ImageValidationError


def validate_image(filename: str, content: bytes) -> None:
    """
    Valida una imagen cargada por el usuario.

    Parámetros:
        filename -- Nombre original del archivo
        content  -- Contenido binario del archivo

    Lanza:
        ImageValidationError si alguna validación falla
    """
    _check_extension(filename)
    _check_file_size(content)
    _check_image_integrity(content)


# ── Validaciones internas ─────────────────────────────────────────────────────

def _check_extension(filename: str) -> None:
    """Verifica que la extensión del archivo esté en la lista permitida."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Tipo de archivo no permitido: '{ext}'. "
            f"Se aceptan: {', '.join(ALLOWED_EXTENSIONS)}"
        )


def _check_file_size(content: bytes) -> None:
    """Verifica que el archivo no supere el tamaño máximo configurado."""
    size = len(content)
    if size > MAX_FILE_SIZE_BYTES:
        raise ImageValidationError(
            f"El archivo es demasiado grande ({size / 1024 / 1024:.1f} MB). "
            f"El máximo permitido es {MAX_FILE_SIZE_MB} MB."
        )
    if size == 0:
        raise ImageValidationError("El archivo está vacío.")


def _check_image_integrity(content: bytes) -> None:
    """
    Verifica que Pillow pueda abrir y procesar la imagen correctamente.
    También comprueba las dimensiones mínimas recomendadas.
    """
    try:
        with Image.open(io.BytesIO(content)) as img:
            img.verify()   # Detecta archivos corruptos o no válidos
    except Exception:
        raise ImageValidationError(
            "El archivo no es una imagen válida o está dañado."
        )

    # Re-abrimos porque verify() consume el stream
    try:
        with Image.open(io.BytesIO(content)) as img:
            width, height = img.size
    except Exception:
        raise ImageValidationError("No se pudo leer las dimensiones de la imagen.")

    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        raise ImageValidationError(
            f"La imagen es demasiado pequeña ({width}x{height} px). "
            f"Se recomienda un mínimo de {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT} px."
        )
