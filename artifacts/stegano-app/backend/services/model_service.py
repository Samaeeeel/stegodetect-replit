"""
Servicio de inferencia del modelo de estegoanálisis.

Arquitectura de carga:
  1. Al iniciar, busca el checkpoint en ml/checkpoints/ (ver config.py).
  2. Si lo encuentra, carga el modelo PyTorch y activa modo real.
  3. Si no existe ningún checkpoint, activa el modo mock/fallback.

Modo mock:
  Genera predicciones simuladas pero coherentes basadas en propiedades
  de la imagen. NO representa un análisis real; sirve solo para probar
  el flujo completo de la aplicación.

Integración del modelo real:
  Cuando tengas el checkpoint entrenado en Colab, colócalo en:
    ml/checkpoints/srnet_lite_best.pt   (nombre preferido)
    ml/checkpoints/model.pt             (nombre alternativo)

  Si tu arquitectura SRNet-lite requiere una clase personalizada, defínela
  en la sección marcada con  # ── DEFINE TU ARQUITECTURA AQUÍ ──  y
  reemplaza `_DummyModel` por tu clase real en `_load_real_model()`.
"""

import io
import random
import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from backend.core.config import (
    CHECKPOINTS_DIR,
    MODEL_CHECKPOINT_NAMES,
    MODEL_INPUT_SIZE,
    MODEL_VERSION_MOCK,
    MODEL_VERSION_REAL,
)
from backend.core.exceptions import ModelInferenceError

logger = logging.getLogger(__name__)

# ── Estado global del servicio ────────────────────────────────────────────────
_model = None               # Modelo PyTorch cargado (None en modo mock)
_mock_mode: bool = True     # True mientras no exista checkpoint


# ── DEFINE TU ARQUITECTURA AQUÍ ──────────────────────────────────────────────
# Cuando tengas el checkpoint, reemplaza _DummyModel por tu clase SRNet-lite.
# Ejemplo:
#
# import torch
# import torch.nn as nn
#
# class SRNetLite(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # ... capas de la arquitectura ...
#
#     def forward(self, x):
#         # ... lógica de forward pass ...
#         return x
#
# Luego en _load_real_model() cambia:
#   model = _DummyModel()
# por:
#   model = SRNetLite()
# ─────────────────────────────────────────────────────────────────────────────


def initialize() -> None:
    """
    Inicializa el servicio de modelo al arrancar la aplicación.
    Llama a esta función en el evento startup de FastAPI.
    """
    global _model, _mock_mode

    checkpoint_path = _find_checkpoint()
    if checkpoint_path:
        try:
            _model = _load_real_model(checkpoint_path)
            _mock_mode = False
            logger.info("Modelo cargado desde: %s", checkpoint_path)
        except Exception as exc:
            logger.warning(
                "No se pudo cargar el modelo (%s). Activando modo mock.", exc
            )
            _model = None
            _mock_mode = True
    else:
        logger.info(
            "No se encontró checkpoint en %s. Modo mock activado.", CHECKPOINTS_DIR
        )
        _mock_mode = True


def is_mock_mode() -> bool:
    return _mock_mode


def get_model_version() -> str:
    return MODEL_VERSION_MOCK if _mock_mode else MODEL_VERSION_REAL


def predict(image_path: Path) -> Tuple[str, float]:
    """
    Analiza la imagen y devuelve (prediction, probability).

    prediction  -- "stego" (con mensaje oculto) o "cover" (sin mensaje)
    probability -- Probabilidad de la predicción [0.0, 1.0]
    """
    # En modo mock no necesitamos PyTorch — la predicción es basada en el hash del archivo
    if _mock_mode:
        return _mock_predict(image_path)

    # Solo preprocesar (requiere torch) cuando hay un modelo real cargado
    try:
        img_tensor = _preprocess(image_path)
    except Exception as exc:
        raise ModelInferenceError(f"Error preprocesando la imagen: {exc}") from exc

    try:
        return _real_predict(img_tensor)
    except Exception as exc:
        raise ModelInferenceError(f"Error durante la inferencia: {exc}") from exc


# ── Funciones privadas ────────────────────────────────────────────────────────

def _find_checkpoint() -> Optional[Path]:
    """Busca el primer checkpoint disponible en el orden configurado."""
    for name in MODEL_CHECKPOINT_NAMES:
        path = CHECKPOINTS_DIR / name
        if path.exists():
            return path
    return None


def _preprocess(image_path: Path):
    """
    Preprocesamiento estándar de imagen para el modelo.
    Pasos:
      1. Abrir con Pillow
      2. Convertir a RGB (descarta canal alpha si existe)
      3. Redimensionar a MODEL_INPUT_SIZE x MODEL_INPUT_SIZE (crop centrado)
      4. Normalizar a [0, 1]
    """
    import torch
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize(MODEL_INPUT_SIZE),
        T.CenterCrop(MODEL_INPUT_SIZE),
        T.ToTensor(),                          # [0, 255] → [0.0, 1.0]
        T.Normalize(mean=[0.5, 0.5, 0.5],     # Normalización estándar
                    std=[0.5, 0.5, 0.5]),
    ])

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        tensor = transform(img)               # Shape: (3, 128, 128)
        return tensor.unsqueeze(0)            # Añadir dimensión batch: (1, 3, 128, 128)


def _load_real_model(checkpoint_path: Path):
    """
    Carga el modelo PyTorch desde el checkpoint.
    ADAPTA esta función cuando tengas tu arquitectura SRNet-lite definida.
    """
    import torch

    # ── Reemplaza _DummyModel con tu clase real ──
    # model = SRNetLite()
    # model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    # model.eval()
    # return model

    # Placeholder hasta tener la arquitectura real
    raise NotImplementedError(
        "Define la arquitectura del modelo en model_service.py "
        "y reemplaza este placeholder."
    )


def _real_predict(img_tensor) -> Tuple[str, float]:
    """Inferencia real con el modelo PyTorch."""
    import torch

    with torch.no_grad():
        output = _model(img_tensor)           # Shape esperado: (1, 2) o (1, 1)

        # Si la salida tiene 2 clases (softmax)
        if output.shape[-1] == 2:
            probs = torch.softmax(output, dim=1)
            stego_prob = probs[0][1].item()   # Probabilidad de clase "stego"
        else:
            # Si la salida es un único valor (sigmoid)
            stego_prob = torch.sigmoid(output)[0][0].item()

    prediction = "stego" if stego_prob >= 0.5 else "cover"
    return prediction, stego_prob


def _mock_predict(image_path: Path) -> Tuple[str, float]:
    """
    Predicción simulada determinista basada en el hash del archivo.
    Usar el hash garantiza que la misma imagen siempre produzca el mismo resultado,
    lo que es útil para pruebas reproducibles.

    NOTA: Este resultado NO tiene valor científico; es solo para demostración.
    """
    with open(image_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    # Convertir los primeros bytes del hash a un número entre 0 y 1
    hash_value = int(file_hash[:8], 16) / 0xFFFFFFFF

    # Rango de probabilidad: 0.51 – 0.95 para que el resultado sea claro
    if hash_value >= 0.5:
        probability = 0.51 + (hash_value - 0.5) * 0.88
        prediction = "stego"
    else:
        probability = 0.51 + (0.5 - hash_value) * 0.88
        prediction = "cover"

    return prediction, round(probability, 4)


def _get_confidence(probability: float) -> str:
    """Convierte la probabilidad en una etiqueta de confianza comprensible."""
    if probability >= 0.85:
        return "Alta"
    elif probability >= 0.65:
        return "Media"
    else:
        return "Baja"


def _get_explanation(prediction: str, probability: float, mock: bool) -> str:
    """Genera una explicación textual del resultado para el usuario."""
    if mock:
        prefix = "[MODO DEMOSTRACIÓN] "
    else:
        prefix = ""

    pct = round(probability * 100, 1)
    if prediction == "stego":
        return (
            f"{prefix}El análisis sugiere con un {pct}% de probabilidad que la imagen "
            "contiene información oculta mediante técnicas de esteganografía LSB. "
            "Se detectaron patrones estadísticos inusuales en los bits menos significativos."
        )
    else:
        return (
            f"{prefix}El análisis indica con un {pct}% de probabilidad que la imagen "
            "no contiene mensajes ocultos detectables. Los patrones de bits son "
            "consistentes con una imagen digital normal."
        )
