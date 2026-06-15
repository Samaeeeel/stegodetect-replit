"""
Configuración central del sistema.
Centraliza todas las constantes y rutas del proyecto para facilitar el mantenimiento.
"""

import os
from pathlib import Path

# Directorio raíz del proyecto (stegano-app/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ── Rutas de almacenamiento ───────────────────────────────────────────────────
UPLOADS_DIR         = BASE_DIR / "backend" / "storage" / "uploads"
REPORTS_DIR         = BASE_DIR / "backend" / "storage" / "reports"
STEGO_ARTIFACTS_DIR = BASE_DIR / "backend" / "storage" / "stego_artifacts"
RESULTS_FILE        = BASE_DIR / "backend" / "storage" / "results.json"
INTEGRATED_RESULTS_FILE = BASE_DIR / "backend" / "storage" / "integrated_results.json"

# ── Checkpoint del modelo ─────────────────────────────────────────────────────
CHECKPOINTS_DIR = BASE_DIR / "ml" / "checkpoints"
# El sistema busca estos archivos en orden; usa el primero que encuentre
MODEL_CHECKPOINT_NAMES = ["srnet_lite_best_state_dict.pt", "srnet_lite_best.pt", "model.pt"]

# ── Validación de imágenes ────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE_MB = 10                      # Tamaño máximo en MB
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MIN_IMAGE_WIDTH = 64                       # Ancho mínimo recomendado en píxeles
MIN_IMAGE_HEIGHT = 64                      # Alto mínimo recomendado en píxeles

# ── Preprocesamiento del modelo ───────────────────────────────────────────────
MODEL_INPUT_SIZE = 128                     # El modelo espera imágenes 128x128
MODEL_VERSION_MOCK = "mock-v0.1"
MODEL_VERSION_REAL = "srnet-lite-finetuned-v1"  # Fine-tuned con BOSSBase + dataset externo

# ── Configuración del servidor ────────────────────────────────────────────────
APP_TITLE = "Sistema inteligente de detección de esteganografía"
APP_VERSION = "1.0.0"
PORT = int(os.environ.get("PORT", 8000))

# Crear directorios si no existen al importar la configuración
for _dir in (UPLOADS_DIR, REPORTS_DIR, STEGO_ARTIFACTS_DIR, CHECKPOINTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
