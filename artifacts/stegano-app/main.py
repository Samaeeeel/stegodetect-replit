"""
Punto de entrada de la aplicación FastAPI.

Responsabilidades:
  - Crear la instancia de FastAPI
  - Montar archivos estáticos del frontend
  - Registrar las rutas de la API
  - Inicializar el servicio del modelo al arrancar

Ejecutar con:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import APP_TITLE, APP_VERSION, PORT
from backend.api.routes import router
from backend.api.stego_routes import stego_router
from backend.services import model_service

# Configuración de logging para que los mensajes sean legibles en Replit
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Crear aplicación ──────────────────────────────────────────────────────────
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="Sistema inteligente para la detección de mensajes ocultos en imágenes mediante esteganografía y Machine Learning.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (necesario para desarrollo y Replit proxy) ───────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Archivos estáticos del frontend ──────────────────────────────────────────
_STATIC_DIR = Path(__file__).parent / "frontend" / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ── Rutas de la API ───────────────────────────────────────────────────────────
app.include_router(router)
app.include_router(stego_router)

# ── Evento de inicio ──────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """
    Inicializa el modelo al arrancar el servidor.
    Si no hay checkpoint disponible, activa el modo mock automáticamente.
    """
    logger.info("Iniciando %s v%s...", APP_TITLE, APP_VERSION)
    model_service.initialize()
    if model_service.is_mock_mode():
        logger.warning(
            "MODO MOCK ACTIVO: No se encontró ningún checkpoint en ml/checkpoints/. "
            "La aplicación funciona en modo demostración."
        )
    else:
        logger.info("Modelo real cargado. Versión: %s", model_service.get_model_version())


# ── Ejecución directa ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
