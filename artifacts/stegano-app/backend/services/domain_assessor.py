"""
Evaluador de aplicabilidad del modelo SRNet-lite sobre una imagen.

Justificación académica
-----------------------
El modelo SRNet-lite (fine-tuned) fue entrenado con:
  - BOSSBase v1.01: PNG sin compresión, ~512×512, baja saturación, payloads p005/p010/p020.
  - Dataset externo: wallpapers, fotos de teléfono, imágenes web, capturas de pantalla
    y material comprimido para redes sociales.
  - Threshold calibrado = 0.46 sobre conjunto de validación mixto.

Aunque el fine-tuning amplió el dominio, imágenes muy alejadas de los datos de
entrenamiento (resoluciones extremas, formatos inusuales, alta compresión JPEG)
aún pueden producir puntajes fuera de distribución (OOD). En esos casos el
puntaje debe interpretarse como evidencia probabilística, no como certeza.
Reportarlo sin esta advertencia sería académicamente deshonesto.

Esta función NO ejecuta inferencia ML adicional: usa heurísticas
deterministas sobre metadatos de la imagen (formato, dimensiones,
saturación) para clasificar la entrada en uno de tres niveles de
compatibilidad con el dominio de entrenamiento.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


# ── Constantes del dominio de entrenamiento ───────────────────────────────────
TRAIN_FORMAT      = "PNG"
TRAIN_SIDE_MIN    = 256      # límite inferior razonable (BOSSBase = 512)
TRAIN_SIDE_MAX    = 1024     # límite superior razonable
SATURATION_THRESH = 60       # Δ(max-min) promedio por píxel — > 60 = muy saturado


def assess_model_applicability(image_path: Path) -> Dict:
    """
    Evalúa qué tan compatible es la imagen con el dominio de entrenamiento.

    Retorna un dict con la forma:
      {
        "domain_status":        "in_domain" | "possibly_out_of_domain" | "out_of_domain",
        "ml_score_reliability": "high"      | "medium"                  | "low",
        "compatibility_score":  int,        # 0–100 (heurístico, solo orientativo)
        "reasons":              [str, ...], # explicaciones legibles
        "image_format":         str,        # "PNG" / "JPEG" / ...
        "image_size":           [w, h],
        "image_mode":           str,        # "L" / "RGB" / "RGBA"
        "color_saturation":     float,      # Δ promedio (0 = grayscale)
      }
    """
    try:
        img = Image.open(image_path)
        img.load()
    except Exception as exc:
        return {
            "domain_status":        "out_of_domain",
            "ml_score_reliability": "low",
            "compatibility_score":  0,
            "reasons":              [f"No se pudo abrir la imagen: {exc}"],
            "image_format":         "unknown",
            "image_size":           [0, 0],
            "image_mode":           "unknown",
            "color_saturation":     0.0,
        }

    fmt  = (img.format or "UNKNOWN").upper()
    w, h = img.size
    mode = img.mode

    reasons: List[str] = []
    score  = 100  # comienza en in-domain perfecto, va restando

    # 1. Formato del archivo
    # ─────────────────────────────────────────────────────────────────
    # JPG aplica compresión con pérdida (DCT + cuantización) que destruye
    # o altera los LSB originales. Cualquier puntaje del modelo sobre JPG
    # es en realidad inferencia sobre artefactos JPEG, no sobre LSB.
    if fmt in ("JPEG", "JPG"):
        reasons.append(
            "Formato JPG/JPEG con compresión con pérdida — los LSB originales "
            "pueden haberse destruido o alterado por el codec JPEG. "
            "El modelo fue entrenado solo con PNG sin compresión destructiva."
        )
        score -= 50
    elif fmt == "PNG":
        reasons.append("Formato PNG sin compresión destructiva (compatible con entrenamiento).")
    else:
        reasons.append(f"Formato {fmt} no representado en el conjunto de entrenamiento.")
        score -= 25

    # 2. Dimensiones (resolución y aspect ratio)
    # ─────────────────────────────────────────────────────────────────
    # BOSSBase es cuadrado 512×512. Una imagen 4K/wallpaper NUNCA es del
    # dominio del modelo, aunque sea grayscale o PNG. Aplicamos penalización
    # fuerte para que ningún wallpaper 2K/4K quede "in_domain".
    max_side = max(w, h)
    if max_side > 2000:
        reasons.append(
            f"Resolución {w}×{h} muy superior al rango de entrenamiento "
            f"({TRAIN_SIDE_MIN}–{TRAIN_SIDE_MAX} px por lado). "
            "Tamaño típico de wallpaper/foto profesional — no representado en entrenamiento."
        )
        score -= 45
    elif max_side > 1280:
        reasons.append(
            f"Resolución {w}×{h} superior al rango de entrenamiento "
            f"({TRAIN_SIDE_MIN}–{TRAIN_SIDE_MAX} px por lado)."
        )
        score -= 25
    elif max_side < 128:
        reasons.append(f"Imagen muy pequeña ({w}×{h}) — fuera del rango de entrenamiento.")
        score -= 20
    elif TRAIN_SIDE_MIN <= max_side <= TRAIN_SIDE_MAX:
        reasons.append(f"Resolución {w}×{h} compatible con el rango de entrenamiento.")
    else:
        reasons.append(f"Resolución {w}×{h} cercana al rango de entrenamiento.")
        score -= 8

    # Aspect ratio — BOSSBase es 1:1. Wide screens (16:9, 16:10) son OOD
    # incluso si la resolución cae en el rango.
    aspect = max(w, h) / max(1, min(w, h))
    if aspect > 1.5:
        reasons.append(
            f"Relación de aspecto {aspect:.2f}:1 — el entrenamiento usó imágenes "
            "cuadradas (1:1). Formatos panorámicos no están representados."
        )
        score -= 15

    # 3. Modo de color y saturación
    # ─────────────────────────────────────────────────────────────────
    # BOSSBase es esencialmente grayscale. Una imagen muy saturada
    # (foto a color, wallpaper) tiene una distribución de canales muy
    # distinta y el modelo no la ha visto en entrenamiento.
    saturation = 0.0
    if mode == "L":
        reasons.append("Imagen grayscale — compatible directa con BOSSBase original.")
    elif mode in ("RGB", "RGBA"):
        try:
            arr = np.asarray(img.convert("RGB"), dtype=np.int16)
            saturation = float(
                (arr.max(axis=2) - arr.min(axis=2)).mean()
            )
            if saturation > SATURATION_THRESH:
                reasons.append(
                    f"Alta saturación de color (Δ={saturation:.0f} promedio entre canales) "
                    "— distribución de color muy distinta al dominio grayscale/casi-gris "
                    "de entrenamiento BOSSBase."
                )
                score -= 20
            else:
                reasons.append(
                    f"Saturación de color baja (Δ={saturation:.0f}) "
                    "— canales RGB similares, compatible con dominio BOSSBase."
                )
        except Exception:
            reasons.append("No se pudo calcular saturación de color.")
    else:
        reasons.append(f"Modo de color '{mode}' inusual — fuera del dominio típico.")
        score -= 10

    # 4. Clasificación final
    # ─────────────────────────────────────────────────────────────────
    score = max(0, min(100, score))
    if score >= 75:
        domain_status = "in_domain"
        reliability   = "high"
    elif score >= 40:
        domain_status = "possibly_out_of_domain"
        reliability   = "medium"
    else:
        domain_status = "out_of_domain"
        reliability   = "low"

    return {
        "domain_status":        domain_status,
        "ml_score_reliability": reliability,
        "compatibility_score":  score,
        "reasons":              reasons,
        "image_format":         fmt,
        "image_size":           [w, h],
        "image_mode":           mode,
        "color_saturation":     round(saturation, 2),
    }


def interpret_reliability(
    ml_probability: float,
    threshold:      float,
    domain_status:  str,
) -> Dict[str, str]:
    """
    Calcula la etiqueta de fiabilidad mostrada al usuario combinando
    el puntaje ML con la compatibilidad de dominio.

    Reemplaza la antigua noción de "Nivel de confianza" (que confundía
    score 0% con baja confianza). Ahora la fiabilidad depende DE LA
    APLICABILIDAD del modelo, no del valor numérico del puntaje.

    Retorna {"label": str, "level": "high"|"medium"|"low", "tooltip": str}.
    """
    # Si la imagen está fuera del dominio, la fiabilidad SIEMPRE es baja,
    # independientemente de si el score es 0% o 100%.
    if domain_status == "out_of_domain":
        return {
            "label":   "Baja — imagen fuera del dominio del modelo",
            "level":   "low",
            "tooltip": "El modelo no fue entrenado con imágenes de este tipo "
                       "(formato, resolución o saturación incompatibles). "
                       "El puntaje no debe interpretarse como probabilidad calibrada.",
        }

    if domain_status == "possibly_out_of_domain":
        return {
            "label":   "Media — imagen parcialmente compatible",
            "level":   "medium",
            "tooltip": "La imagen comparte algunas características con el dominio "
                       "de entrenamiento, pero no todas. Resultado orientativo.",
        }

    # in_domain — la fiabilidad depende del puntaje
    pct = ml_probability * 100
    if pct < threshold * 100 * 0.5:  # claramente cover
        return {
            "label":   "Alta — baja evidencia de estego (dominio compatible)",
            "level":   "high",
            "tooltip": "Imagen dentro del dominio de entrenamiento y puntaje "
                       "claramente bajo. El modelo descarta esteganografía con confianza.",
        }
    if pct >= 80:  # claramente stego
        return {
            "label":   "Alta — posible estego (dominio compatible)",
            "level":   "high",
            "tooltip": "Imagen dentro del dominio de entrenamiento y puntaje "
                       "claramente alto. Patrón compatible con esteganografía LSB.",
        }
    # zona intermedia
    return {
        "label":   "Media — resultado ambiguo",
        "level":   "medium",
        "tooltip": "El puntaje cae en zona intermedia entre cover y estego. "
                   "Se recomienda análisis técnico complementario (LSB, extracción).",
    }
