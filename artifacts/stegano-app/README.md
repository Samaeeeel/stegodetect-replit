# StegaDetect — Sistema Inteligente de Detección de Esteganografía

Sistema web para detectar mensajes ocultos en imágenes mediante Machine Learning.
Desarrollado como prototipo académico de tesis usando FastAPI + Python.

---

## Descripción del proyecto

Este sistema implementa un flujo completo de **estegoanálisis basado en Machine Learning**:

1. El usuario sube una imagen (PNG, JPG, JPEG)
2. El backend valida la imagen (tipo, tamaño, integridad, dimensiones)
3. Se ejecuta inferencia con el modelo PyTorch (o modo mock si no existe checkpoint)
4. Se muestra el resultado con probabilidad, nivel de confianza y explicación
5. El usuario puede descargar un reporte PDF académico

---

## Estructura de carpetas

```
artifacts/stegano-app/
├── main.py                         # Punto de entrada FastAPI
├── requirements.txt                # Dependencias Python
├── README.md                       # Este archivo
├── backend/
│   ├── core/
│   │   ├── config.py               # Configuración centralizada
│   │   └── exceptions.py           # Excepciones personalizadas
│   ├── domain/
│   │   └── analysis_result.py      # Entidad principal de dominio
│   ├── services/
│   │   ├── image_validator.py      # Validación de imágenes
│   │   ├── model_service.py        # Inferencia PyTorch + modo mock
│   │   └── report_service.py       # Generación de PDF (ReportLab)
│   ├── api/
│   │   └── routes.py               # Endpoints FastAPI
│   └── storage/
│       ├── uploads/                # Imágenes subidas (temporales)
│       ├── reports/                # Reportes PDF generados
│       └── results.json            # Registro de análisis
├── frontend/
│   ├── index.html                  # Interfaz web
│   └── static/
│       ├── css/main.css            # Estilos propios
│       └── js/app.js               # Lógica del frontend
└── ml/
    └── checkpoints/
        └── README.md               # Instrucciones del checkpoint
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

## Dónde colocar el modelo entrenado

Cuando tengas el checkpoint entrenado en Google Colab, colócalo aquí:

```
ml/checkpoints/srnet_lite_best.pt    ← nombre preferido
ml/checkpoints/model.pt              ← nombre alternativo
```

El sistema detecta el archivo automáticamente al reiniciar el servidor.

---

## Explicación del modo mock

Si no existe ningún checkpoint en `ml/checkpoints/`, la aplicación activa el **modo demostración (mock)**:

- `mock_mode = True` en todos los resultados
- La predicción se basa en el hash MD5 del archivo (determinista y reproducible)
- El frontend muestra una advertencia clara indicando que los resultados son simulados
- El endpoint `/health` indica explícitamente el modo activo

El modo mock **no tiene validez científica**. Sirve únicamente para verificar que el flujo completo de la aplicación funciona antes de tener el modelo entrenado.

---

## Cómo integrar el modelo real (pasos detallados)

### 1. Entrenar en Google Colab

```python
# Al finalizar el entrenamiento en Colab:
torch.save(model.state_dict(), "srnet_lite_best.pt")
```

### 2. Subir el checkpoint a Replit

Coloca `srnet_lite_best.pt` en `artifacts/stegano-app/ml/checkpoints/`.

### 3. Definir la arquitectura en el código

Abre `backend/services/model_service.py` y busca la sección:

```python
# ── DEFINE TU ARQUITECTURA AQUÍ ──
```

Define tu clase `SRNetLite` y actualiza `_load_real_model()`.

### 4. Reiniciar el servidor

El sistema detectará el checkpoint y cambiará `mock_mode = False`.

---

## Stack tecnológico

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **Validación**: Pydantic, Pillow
- **ML**: PyTorch (opcional, requerido solo para modelo real)
- **PDF**: ReportLab
- **Frontend**: HTML5, Bootstrap 5.3, JavaScript Vanilla

---

## Limitaciones actuales

1. El modelo aún no está entrenado — el sistema funciona en modo mock
2. Las imágenes subidas se almacenan localmente en `backend/storage/uploads/` (no hay limpieza automática)
3. `results.json` es un archivo plano — no es adecuado para producción con alto volumen
4. El checkpoint de PyTorch se carga en CPU (suficiente para Replit)
5. No hay autenticación — es un prototipo académico

---

## Próximos pasos

1. **Entrenar el modelo final en Colab** con BOSSBase v1.01 + imágenes stego LSB
2. **Cargar el checkpoint real** (`srnet_lite_best.pt`) y verificar que `mock_mode = False`
3. **Mejorar métricas**: experimentar con tasa de bits, aumentos de datos y arquitectura SRNet-lite completa
4. **Limpiar archivos temporales**: agregar tarea periódica para borrar uploads antiguos
5. **Persistencia real**: migrar `results.json` a SQLite o PostgreSQL para producción
6. **Pasar a despliegue más robusto**: containerizar con Docker + servicio cloud para inferencia en GPU
