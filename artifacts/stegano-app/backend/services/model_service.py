"""
Servicio de inferencia del modelo de estegoanálisis.

Arquitectura de carga:
  1. Al iniciar, busca el checkpoint en ml/checkpoints/ (ver config.py).
  2. Si lo encuentra, carga SRNetLite y lee el threshold desde model_metadata.json.
  3. Si no existe ningún checkpoint, activa el modo mock/fallback.

Modo mock:
  Genera predicciones simuladas pero coherentes basadas en el hash del archivo.
  NO representa un análisis real; sirve solo para probar el flujo de la app.

Integración del modelo real (Colab → Replit):
  1. Entrena el modelo en Colab con 02_model_training_colab.py.
  2. Descarga desde Drive:
       srnet_lite_best.pt        → ml/checkpoints/srnet_lite_best.pt
       model_metadata.json       → ml/checkpoints/model_metadata.json
  3. Reinicia el servidor — el sistema los detecta automáticamente.
  4. Verifica: GET /health debe devolver { "mock_mode": false }

Preprocesamiento en inferencia (debe coincidir con el entrenamiento):
  - Convertir a RGB
  - CenterCrop 128×128
  - ToTensor: [0,255] → [0.0, 1.0]
  - Normalize: (x - 0.5) / 0.5  →  [-1.0, 1.0]
"""

import json
import hashlib
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from backend.core.config import (
    CHECKPOINTS_DIR,
    MODEL_CHECKPOINT_NAMES,
    MODEL_INPUT_SIZE,
    MODEL_VERSION_MOCK,
    MODEL_VERSION_REAL,
    BASE_DIR,
)
from backend.core.exceptions import ModelInferenceError

logger = logging.getLogger(__name__)

# ── Estado global del servicio ────────────────────────────────────────────────
_model              = None          # Modelo PyTorch cargado (None en modo mock)
_mock_mode: bool    = True          # True mientras no exista checkpoint válido
_threshold: float   = 0.5          # Threshold óptimo leído de model_metadata.json
_model_version: str = MODEL_VERSION_MOCK
_checkpoint_loaded: str = ""        # Nombre del archivo de checkpoint cargado


def initialize() -> None:
    """
    Inicializa el servicio de modelo al arrancar la aplicación.
    Llama a esta función en el evento startup de FastAPI.
    """
    global _model, _mock_mode, _threshold, _model_version, _checkpoint_loaded

    import os

    # ── Logs de diagnóstico al arrancar ──────────────────────────────────────
    cwd = os.getcwd()
    logger.info("=" * 60)
    logger.info("  INICIALIZANDO SERVICIO DE MODELO")
    logger.info("  CWD:         %s", cwd)
    logger.info("  BASE_DIR:    %s", BASE_DIR)
    logger.info("  CHECKPOINTS: %s", CHECKPOINTS_DIR)

    if CHECKPOINTS_DIR.exists():
        files_found = [f.name for f in CHECKPOINTS_DIR.iterdir() if f.is_file()]
        logger.info("  Archivos en checkpoints: %s", files_found)
    else:
        logger.warning("  Directorio checkpoints NO existe: %s", CHECKPOINTS_DIR)

    logger.info("  Orden de búsqueda: %s", MODEL_CHECKPOINT_NAMES)
    logger.info("=" * 60)

    # ── Buscar y cargar checkpoint ────────────────────────────────────────────
    checkpoint_path = _find_checkpoint()

    if checkpoint_path:
        logger.info("  Intentando cargar: %s", checkpoint_path)
        try:
            _model             = _load_srnet_lite(checkpoint_path)
            _threshold         = _load_threshold()
            _mock_mode         = False
            _model_version     = MODEL_VERSION_REAL
            _checkpoint_loaded = checkpoint_path.name
            logger.info("  [OK] Checkpoint cargado: %s", checkpoint_path.name)
            logger.info("  [OK] Threshold:          %.4f", _threshold)
            logger.info("  [OK] mock_mode:          False  ← modelo real activo")
        except Exception as exc:
            logger.warning("  [ERROR] No se pudo cargar el modelo: %s", exc)
            logger.warning("  [FALLBACK] Activando modo mock.")
            _model             = None
            _mock_mode         = True
            _model_version     = MODEL_VERSION_MOCK
            _checkpoint_loaded = ""
    else:
        logger.info(
            "  [INFO] No se encontró ningún checkpoint en %s.", CHECKPOINTS_DIR
        )
        logger.info("  [INFO] mock_mode: True  ← modo demostración activo")
        _mock_mode         = True
        _model_version     = MODEL_VERSION_MOCK
        _checkpoint_loaded = ""

    logger.info("=" * 60)


def is_mock_mode() -> bool:
    return _mock_mode


def get_model_version() -> str:
    return _model_version


def get_checkpoint_loaded() -> str:
    """Nombre del archivo de checkpoint cargado, o '' si está en modo mock."""
    return _checkpoint_loaded


def get_threshold() -> float:
    """Threshold de decisión activo (leído de model_metadata.json)."""
    return _threshold


def predict(image_path: Path) -> Tuple[str, float]:
    """
    Analiza la imagen y devuelve (prediction, probability).

    prediction  -- "stego" (con mensaje oculto) o "cover" (sin mensaje)
    probability -- Probabilidad de que la imagen sea stego [0.0, 1.0]
    """
    # En modo mock no necesitamos PyTorch
    if _mock_mode:
        return _mock_predict(image_path)

    # Modelo real: preprocesar y ejecutar inferencia
    try:
        img_tensor = _preprocess(image_path)
    except Exception as exc:
        raise ModelInferenceError(f"Error preprocesando la imagen: {exc}") from exc

    try:
        return _real_predict(img_tensor)
    except Exception as exc:
        raise ModelInferenceError(f"Error durante la inferencia: {exc}") from exc


# ── Carga del modelo SRNet-lite ───────────────────────────────────────────────

def _find_checkpoint() -> Optional[Path]:
    """Busca el primer checkpoint disponible en el orden configurado."""
    for name in MODEL_CHECKPOINT_NAMES:
        path = CHECKPOINTS_DIR / name
        if path.exists():
            return path
    return None


def _load_threshold() -> float:
    """
    Lee el threshold óptimo desde model_metadata.json.
    Si no existe, usa 0.5 como valor conservador.
    """
    metadata_path = CHECKPOINTS_DIR / "model_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                meta = json.load(f)
            threshold = float(meta.get("threshold", 0.5))
            logger.info("Threshold cargado desde metadata: %.4f", threshold)
            return threshold
        except Exception as exc:
            logger.warning("No se pudo leer model_metadata.json: %s. Usando 0.5", exc)
    return 0.5


def _load_srnet_lite(checkpoint_path: Path):
    """
    Construye y carga el modelo SRNet-lite desde el checkpoint.

    Importa la arquitectura desde ml/src/models/srnet_lite.py para garantizar
    que el modelo de inferencia es idéntico al que se entrenó en Colab.
    Si el módulo no está disponible, usa la definición embebida de emergencia.
    """
    import torch

    # Añadir la raíz del proyecto al PYTHONPATH para importar ml.src
    project_root = str(BASE_DIR)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Intentar importar la arquitectura desde ml/src
    try:
        from ml.src.models.srnet_lite import SRNetLite
        logger.info("Arquitectura SRNet-lite importada desde ml/src/models/srnet_lite.py")
    except ImportError:
        logger.warning(
            "No se pudo importar ml.src.models.srnet_lite. "
            "Usando definición embebida de emergencia."
        )
        SRNetLite = _build_embedded_srnet_lite()

    # Construir el modelo
    model = SRNetLite()

    # Cargar pesos
    # weights_only=False necesario porque el checkpoint contiene numpy scalars
    # (guardado desde Colab con torch.save que embebe numpy._core.multiarray.scalar).
    # El checkpoint proviene de una fuente de confianza (entrenamiento propio en Colab).
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
        logger.info("Checkpoint completo cargado (con estado del optimizador).")
    else:
        # Es un state_dict puro (srnet_lite_best_state_dict.pt)
        model.load_state_dict(state)
        logger.info("State dict cargado directamente.")

    model.eval()
    return model


def _build_embedded_srnet_lite():
    """
    Definición embebida de SRNetLite como fallback de emergencia.
    Idéntica a ml/src/models/srnet_lite.py — se usa si el módulo no está en PYTHONPATH.
    """
    import torch
    import torch.nn as nn
    import numpy as np

    def _srm_kernels():
        k1 = np.array([[0,0,0,0,0],[0,0,0,0,0],[-1,2,-2,2,-1],[0,0,0,0,0],[0,0,0,0,0]],
                      dtype=np.float32) / 4.0
        k2 = k1.T.copy()
        k3 = np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],
                      dtype=np.float32) / 4.0
        return np.stack([k1, k2, k3], axis=0)[:, np.newaxis]

    class _SRM(nn.Module):
        # ATENCIÓN: el atributo se llama 'srm' (NO 'conv') para que las
        # claves del state_dict coincidan con ml/src/models/blocks.py SRMLayer.
        # La clave en el checkpoint será: "srm.srm.weight"
        def __init__(self):
            super().__init__()
            self.srm = nn.Conv2d(3, 9, 5, padding=2, bias=False)
            k = torch.from_numpy(_srm_kernels())
            w = torch.zeros(9, 3, 5, 5)
            for i in range(3):
                for j in range(3):
                    w[i*3+j, j] = k[i, 0]
            with torch.no_grad():
                self.srm.weight.copy_(w)
            for p in self.srm.parameters():
                p.requires_grad = False
        def forward(self, x): return self.srm(x)

    class _Res(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.b = nn.Sequential(
                nn.Conv2d(c,c,3,padding=1,bias=False), nn.BatchNorm2d(c), nn.ReLU(True),
                nn.Conv2d(c,c,3,padding=1,bias=False), nn.BatchNorm2d(c))
            self.relu = nn.ReLU(True)
        def forward(self, x): return self.relu(x + self.b(x))

    class _Down(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.m = nn.Sequential(
                nn.Conv2d(ci,co,3,stride=2,padding=1,bias=False), nn.BatchNorm2d(co), nn.ReLU(True),
                nn.Conv2d(co,co,3,padding=1,bias=False), nn.BatchNorm2d(co))
            self.s = nn.Sequential(nn.Conv2d(ci,co,1,stride=2,bias=False), nn.BatchNorm2d(co))
            self.relu = nn.ReLU(True)
        def forward(self, x): return self.relu(self.m(x) + self.s(x))

    class _Attn(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(c,c//4), nn.ReLU(True), nn.Linear(c//4,c), nn.Sigmoid())
        def forward(self, x): return x * self.se(x).unsqueeze(-1).unsqueeze(-1)

    class _SRNetLite(nn.Module):
        def __init__(self):
            super().__init__()
            self.srm    = _SRM()
            self.stem   = nn.Sequential(nn.Conv2d(9,16,3,padding=1,bias=False), nn.BatchNorm2d(16), nn.ReLU(True))
            self.stage1 = nn.Sequential(_Res(16), _Res(16))
            self.down1  = _Down(16, 32)
            self.stage2 = nn.Sequential(_Res(32), _Res(32))
            self.attn2  = _Attn(32)
            self.down2  = _Down(32, 64)
            self.stage3 = nn.Sequential(_Res(64), _Res(64))
            self.attn3  = _Attn(64)
            self.down3  = _Down(64, 128)
            self.stage4 = nn.Sequential(_Res(128), _Res(128))
            self.attn4  = _Attn(128)
            # ATENCIÓN: el atributo se llama 'classifier' (NO 'clf') para que las
            # claves del state_dict coincidan con ml/src/models/srnet_lite.py.
            # Las claves en el checkpoint serán: "classifier.2.weight", etc.
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Dropout(0.5), nn.Linear(128,64), nn.ReLU(True),
                nn.Dropout(0.25), nn.Linear(64,1))
        def forward(self, x):
            x = self.srm(x); x = self.stem(x); x = self.stage1(x)
            x = self.attn2(self.stage2(self.down1(x)))
            x = self.attn3(self.stage3(self.down2(x)))
            x = self.attn4(self.stage4(self.down3(x)))
            return self.classifier(x)

    return _SRNetLite


# ── Preprocesamiento ──────────────────────────────────────────────────────────

def _preprocess(image_path: Path):
    """
    Preprocesamiento idéntico al usado durante el entrenamiento en Colab:
      1. Abrir con Pillow
      2. Convertir a RGB
      3. CenterCrop 128×128 (sin interpolación)
      4. ToTensor: [0,255] → [0.0, 1.0]
      5. Normalize: mean=0.5, std=0.5 → [-1.0, 1.0]
    """
    import torch
    import torchvision.transforms as T

    transform = T.Compose([
        T.CenterCrop(MODEL_INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    with Image.open(image_path) as img:
        # Convertir a RGB antes del crop (maneja escala de grises y RGBA)
        img_rgb = img.convert("RGB")
        # Si la imagen es más pequeña que el crop, hacer resize mínimo primero
        w, h = img_rgb.size
        if w < MODEL_INPUT_SIZE or h < MODEL_INPUT_SIZE:
            img_rgb = img_rgb.resize(
                (max(w, MODEL_INPUT_SIZE), max(h, MODEL_INPUT_SIZE)),
                resample=Image.BILINEAR
            )
        tensor = transform(img_rgb)            # [3, 128, 128]
        return tensor.unsqueeze(0)             # [1, 3, 128, 128]


# ── Inferencia real ───────────────────────────────────────────────────────────

def _real_predict(img_tensor) -> Tuple[str, float]:
    """
    Inferencia con el modelo SRNetLite.

    El modelo devuelve un logit [B, 1].
    Aplicamos sigmoid para obtener probabilidad de clase stego.
    Usamos _threshold (calculado en validación) en lugar de 0.5 fijo.
    """
    import torch

    with torch.no_grad():
        logit     = _model(img_tensor)        # [1, 1]
        stego_prob = torch.sigmoid(logit).squeeze().item()  # escalar

    prediction = "stego" if stego_prob >= _threshold else "cover"
    return prediction, round(stego_prob, 4)


# ── Predicción mock ───────────────────────────────────────────────────────────

def _mock_predict(image_path: Path) -> Tuple[str, float]:
    """
    Predicción simulada determinista basada en el hash MD5 del archivo.
    La misma imagen siempre produce el mismo resultado (reproducible para tests).

    NOTA: Este resultado NO tiene validez científica.
    """
    with open(image_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    hash_value = int(file_hash[:8], 16) / 0xFFFFFFFF

    if hash_value >= 0.5:
        probability = 0.51 + (hash_value - 0.5) * 0.88
        prediction  = "stego"
    else:
        probability = 0.51 + (0.5 - hash_value) * 0.88
        prediction  = "cover"

    return prediction, round(probability, 4)


# ── Utilidades públicas ───────────────────────────────────────────────────────

def _get_confidence(probability: float) -> str:
    """Convierte la probabilidad en una etiqueta de confianza comprensible."""
    if probability >= 0.85:
        return "Alta"
    elif probability >= 0.65:
        return "Media"
    else:
        return "Baja"


def _get_explanation(prediction: str, probability: float, mock: bool) -> str:
    """
    Genera una explicación textual del resultado ML para el usuario.

    Nota: este texto se refiere SOLO al puntaje del modelo SRNet-lite,
    sin considerar la aplicabilidad de dominio. La interpretación
    integrada (que sí considera el dominio) se construye en
    `stego_routes._build_final_decision`.
    """
    prefix = "[MODO DEMOSTRACIÓN] " if mock else ""
    pct    = round(probability * 100, 1)

    if prediction == "stego":
        return (
            f"{prefix}Puntaje ML de esteganografía: {pct}%. El modelo detectó "
            "patrones estadísticos en los bits menos significativos compatibles "
            "con esteganografía LSB. Modelo fine-tuned con BOSSBase e imágenes "
            "externas. El resultado ML es probabilístico. "
            "Validar con el análisis técnico LSB antes de concluir."
        )
    else:
        return (
            f"{prefix}Puntaje ML de esteganografía: {pct}%. El modelo no detectó "
            "patrones estadísticos compatibles con esteganografía LSB dentro de su "
            "del modelo. Esto representa baja evidencia ML, pero no "
            "descarta técnicas externas, cifradas o fuera del alcance del modelo. "
            "Modelo fine-tuned con BOSSBase e imágenes externas — resultado probabilístico."
        )
