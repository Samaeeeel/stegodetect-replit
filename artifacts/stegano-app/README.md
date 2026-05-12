# StegoDetect — Sistema Inteligente de Detección de Esteganografía

Sistema web para detectar mensajes ocultos en imágenes mediante Machine Learning.
Desarrollado como prototipo académico de tesis usando FastAPI + Python.

---

## Descripción del proyecto

Este sistema implementa un flujo completo de **estegoanálisis basado en Machine Learning**:

1. El usuario sube una imagen (PNG, JPG, JPEG)
2. El backend valida la imagen (tipo, tamaño, integridad, dimensiones)
3. Se ejecuta inferencia con el modelo SRNet-lite (o modo mock si no existe checkpoint)
4. Se muestra el resultado con probabilidad, nivel de confianza y explicación
5. El usuario puede descargar un reporte PDF académico

---

## Estructura de carpetas

```
artifacts/stegano-app/
├── main.py                              # Punto de entrada FastAPI
├── requirements.txt                     # Dependencias Python
├── backend/
│   ├── core/
│   │   ├── config.py                    # Configuración centralizada
│   │   └── exceptions.py               # Excepciones personalizadas
│   ├── domain/
│   │   └── analysis_result.py          # Entidad principal de dominio
│   ├── services/
│   │   ├── image_validator.py          # Validación de imágenes
│   │   ├── model_service.py            # Inferencia SRNet-lite + modo mock
│   │   └── report_service.py          # Generación de PDF (ReportLab)
│   ├── api/
│   │   └── routes.py                   # Endpoints FastAPI
│   └── storage/
│       ├── uploads/                    # Imágenes subidas (temporales)
│       ├── reports/                    # Reportes PDF generados
│       └── results.json               # Registro de análisis
├── frontend/
│   ├── index.html                      # Interfaz web
│   └── static/
│       ├── css/main.css               # Estilos propios
│       └── js/app.js                  # Lógica del frontend
└── ml/
    ├── notebooks/
    │   ├── 01_dataset_pipeline_colab.py   # Pipeline de dataset (ejecutar en Colab)
    │   └── 02_model_training_colab.py     # Entrenamiento del modelo (ejecutar en Colab)
    ├── src/
    │   ├── dataset/
    │   │   ├── generate_dataset_v2.py     # Generador LSB multi-payload
    │   │   ├── dataset_validator.py       # Validación del dataset
    │   │   └── drive_paths.py            # Rutas centralizadas de Drive
    │   ├── models/
    │   │   ├── blocks.py                  # Bloques SRM, residuales, atención
    │   │   └── srnet_lite.py             # Arquitectura SRNet-lite completa
    │   ├── training/
    │   │   ├── trainer.py                # Loop de entrenamiento
    │   │   ├── losses.py                 # Funciones de pérdida
    │   │   └── utils.py                  # EarlyStopping, checkpoints, etc.
    │   └── evaluation/
    │       ├── metrics.py                # AUC-ROC, F1, threshold óptimo
    │       └── plots.py                  # Curva ROC, matriz de confusión
    └── checkpoints/
        ├── srnet_lite_best.pt            # ← Coloca aquí el checkpoint entrenado
        └── model_metadata.json          # ← Coloca aquí los metadatos del modelo
```

---

## Cómo ejecutar en Replit

El sistema se ejecuta automáticamente a través del workflow **StegaDetect**.

Si necesitas iniciarlo manualmente:

```bash
cd artifacts/stegano-app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Cómo probar la aplicación

1. Abre la aplicación en el navegador (preview de Replit)
2. Arrastra o selecciona una imagen PNG, JPG o JPEG
3. Haz clic en **"Analizar imagen"**
4. Observa el resultado: predicción, probabilidad, confianza y explicación
5. Haz clic en **"Descargar reporte PDF"** para obtener el reporte académico

También puedes usar la API directamente:

```bash
# Verificar estado del sistema
curl http://localhost:8000/health

# Analizar una imagen
curl -X POST http://localhost:8000/analyze \
  -F "file=@/ruta/a/imagen.png"

# Descargar reporte PDF (reemplaza {id} con el ID del análisis)
curl http://localhost:8000/report/{analysis_id} -o reporte.pdf
```

---

## Endpoints disponibles

| Método | Ruta                   | Descripción                          |
|--------|------------------------|--------------------------------------|
| GET    | `/`                    | Interfaz web principal               |
| POST   | `/analyze`             | Analiza una imagen subida            |
| GET    | `/report/{id}`         | Descarga el reporte PDF del análisis |
| GET    | `/health`              | Estado del sistema y modo de modelo  |
| GET    | `/docs`                | Documentación interactiva (Swagger)  |
| GET    | `/redoc`               | Documentación alternativa (ReDoc)    |

---

## Entrenamiento del modelo en Google Colab

El modelo SRNet-lite se entrena en Google Colab (NO en Replit) usando BOSSBase v1.01.
Sigue estos pasos en orden:

### Paso 1 — Generar el dataset

Abre `ml/notebooks/01_dataset_pipeline_colab.py` como notebook en Google Colab:

1. Activa GPU: Runtime → Change runtime type → GPU T4
2. Monta Google Drive (celda 2)
3. Ejecuta todas las celdas en orden
4. El dataset queda guardado en:
   ```
   /content/drive/MyDrive/stegadetect_replit/
   ├── cover/               ← 10.000 imágenes PNG limpias
   ├── stego/p005/          ← Stego con payload 5%
   ├── stego/p010/          ← Stego con payload 10%
   ├── stego/p020/          ← Stego con payload 20%
   └── processed/
       ├── train_manifest.csv
       ├── val_manifest.csv
       └── test_manifest.csv
   ```

### Paso 2 — Entrenar el modelo

Abre `ml/notebooks/02_model_training_colab.py` como notebook en Google Colab:

1. Ejecuta todas las celdas en orden (2–4 horas con GPU T4)
2. El checkpoint se guarda automáticamente en Drive:
   ```
   /content/drive/MyDrive/stegadetect_replit/checkpoints/
   ├── srnet_lite_best.pt           ← Checkpoint completo (para reanudar)
   ├── srnet_lite_best_state_dict.pt← Solo pesos (más ligero)
   └── model_metadata.json          ← Threshold óptimo y metadatos
   ```

### Paso 3 — Descargar el checkpoint

Desde la última celda del notebook de entrenamiento:

```python
from google.colab import files
files.download("/content/drive/MyDrive/stegadetect_replit/checkpoints/srnet_lite_best.pt")
files.download("/content/drive/MyDrive/stegadetect_replit/checkpoints/model_metadata.json")
```

### Paso 4 — Subir a Replit

Coloca los dos archivos descargados en:

```
ml/checkpoints/srnet_lite_best.pt       ← checkpoint del modelo
ml/checkpoints/model_metadata.json      ← threshold óptimo y metadatos
```

### Paso 5 — Reiniciar la aplicación

Reinicia el workflow **StegaDetect** en Replit (menú de workflows o botón Run).

### Paso 6 — Verificar que el modo mock está desactivado

```bash
curl http://localhost:8000/health
```

Respuesta esperada:
```json
{
  "status": "ok",
  "mock_mode": false,
  "model_version": "srnet-lite-v1.0",
  "message": "Modelo real cargado y listo para inferencia."
}
```

---

## Arquitectura del modelo SRNet-lite

```
Input RGB [B, 3, 128, 128]
    ↓
SRM HPF (filtros fijos, no entrenables)   → [B, 9, 128, 128]
    ↓                                        ← Extraer señal de alta frecuencia
Conv inicial + BN + ReLU                  → [B, 16, 128, 128]
    ↓
2× ResidualBlock                          → [B, 16, 128, 128]
    ↓
Downsample (stride=2)                     → [B, 32, 64, 64]
    ↓
2× ResidualBlock + Atención SE            → [B, 32, 64, 64]
    ↓
Downsample (stride=2)                     → [B, 64, 32, 32]
    ↓
2× ResidualBlock + Atención SE            → [B, 64, 32, 32]
    ↓
Downsample (stride=2)                     → [B, 128, 16, 16]
    ↓
2× ResidualBlock + Atención SE            → [B, 128, 16, 16]
    ↓
Global Average Pooling                    → [B, 128]
    ↓
Dropout + Linear(128→64) + ReLU
    ↓
Linear(64→1) → logit → sigmoid → P(stego)
```

---

## Explicación del modo mock

Si no existe ningún checkpoint en `ml/checkpoints/`, la aplicación activa el **modo demostración (mock)**:

- `mock_mode = True` en todos los resultados
- La predicción se basa en el hash MD5 del archivo (determinista y reproducible)
- El frontend muestra una advertencia clara indicando que los resultados son simulados
- El endpoint `/health` indica explícitamente el modo activo

El modo mock **no tiene validez científica**. Sirve únicamente para verificar que el flujo completo funciona antes de tener el modelo entrenado.

---

## Stack tecnológico

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **Validación**: Pydantic, Pillow
- **ML**: PyTorch (SRNet-lite), scikit-learn (métricas)
- **PDF**: ReportLab
- **Frontend**: HTML5, Bootstrap 5.3, JavaScript Vanilla
- **Entrenamiento**: Google Colab + Google Drive

---

## Limitaciones actuales

1. El modelo se entrena en Colab — Replit solo hace inferencia
2. Las imágenes subidas se almacenan localmente (sin limpieza automática)
3. `results.json` es un archivo plano — no apto para producción con alto volumen
4. El checkpoint se carga en CPU (suficiente para Replit, lento para producción)
5. No hay autenticación — es un prototipo académico

---

## Próximos pasos

1. **Entrenar el modelo final en Colab** con BOSSBase v1.01 + LSB multi-payload
2. **Cargar el checkpoint real** y verificar que `mock_mode = false`
3. **Mejorar métricas**: payloads más bajos (p=0.02), arquitectura SRNet completa
4. **Análisis de error**: inspeccionar los FN y FP del modelo en el test set
5. **Persistencia real**: migrar `results.json` a SQLite para producción
