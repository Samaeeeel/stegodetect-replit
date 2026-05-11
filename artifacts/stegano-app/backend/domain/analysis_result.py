"""
Entidad principal del dominio: AnalysisResult.
Representa el resultado de analizar una imagen para detectar esteganografía.
Usar una dataclass o modelo Pydantic asegura tipado estricto y serialización limpia.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import json


@dataclass
class AnalysisResult:
    """
    Resultado del análisis de esteganografía para una imagen dada.

    Campos:
        id            -- Identificador único del análisis (UUID)
        filename      -- Nombre original del archivo subido
        prediction    -- "stego" (con mensaje oculto) o "cover" (sin mensaje)
        probability   -- Probabilidad del resultado [0.0, 1.0]
        confidence    -- Nivel de confianza: "Alta", "Media" o "Baja"
        explanation   -- Explicación textual breve del resultado
        model_version -- Versión del modelo usado
        created_at    -- Timestamp ISO 8601 del momento del análisis
        mock_mode     -- True si el modelo no está cargado (modo demostración)
    """
    id: str
    filename: str
    prediction: str                         # "stego" | "cover"
    probability: float                      # 0.0 – 1.0
    confidence: str                         # "Alta" | "Media" | "Baja"
    explanation: str
    model_version: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    mock_mode: bool = False

    # ── Propiedades de conveniencia ───────────────────────────────────────────

    @property
    def has_hidden_message(self) -> bool:
        return self.prediction == "stego"

    @property
    def label(self) -> str:
        return "Con mensaje oculto" if self.has_hidden_message else "Sin mensaje oculto"

    @property
    def probability_percent(self) -> float:
        return round(self.probability * 100, 2)

    # ── Serialización ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label
        d["probability_percent"] = self.probability_percent
        d["has_hidden_message"] = self.has_hidden_message
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisResult":
        """Reconstruye un AnalysisResult desde un diccionario (ej: results.json)."""
        keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in keys}
        return cls(**filtered)
