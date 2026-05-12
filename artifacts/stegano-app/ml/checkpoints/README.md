# Checkpoint del modelo SRNet-lite

Esta carpeta debe contener los archivos del modelo entrenado para que la aplicación
salga del modo mock y realice inferencia real.

---

## Archivos requeridos

| Archivo | Descripción | Tamaño aprox. |
|---------|-------------|---------------|
| `srnet_lite_best.pt` | Pesos del modelo (state_dict de PyTorch) | ~15–50 MB |
| `model_metadata.json` | Threshold óptimo, métricas, configuración | < 5 KB |

Sin estos archivos, la aplicación funciona en **modo mock** (predicciones simuladas).

---

## Paso 1 — Entrenar el modelo en Google Colab

Sigue la guía completa en `ml/COLAB_EXECUTION_GUIDE.md`.

En resumen:
1. Ejecutar `ml/notebooks/01_dataset_pipeline_colab.py` → genera el dataset
2. Ejecutar `ml/notebooks/02_model_training_colab.py` → entrena el modelo
3. Al finalizar, los archivos quedan en Google Drive:
   ```
   /MyDrive/stego_project/checkpoints/srnet_lite_best.pt
   /MyDrive/stego_project/checkpoints/model_metadata.json
   ```

---

## Paso 2 — Descargar desde Google Drive

1. Abre [drive.google.com](https://drive.google.com)
2. Navega a `Mi unidad → stego_project → checkpoints`
3. Descarga **ambos archivos** a tu máquina local

---

## Paso 3 — Subir a Replit

Coloca los archivos exactamente en esta carpeta:

```
artifacts/stegano-app/ml/checkpoints/
├── srnet_lite_best.pt        ← aquí
└── model_metadata.json       ← aquí
```

Para subir en Replit:
- En el explorador de archivos de Replit, navega a `artifacts/stegano-app/ml/checkpoints/`
- Haz clic derecho sobre la carpeta → **Upload file**
- Sube primero `model_metadata.json`, luego `srnet_lite_best.pt`

---

## Paso 4 — Reiniciar la aplicación

En Replit, abre el panel de **Workflows** y reinicia el workflow **StegaDetect**.

En los logs del workflow deberías ver:

```
INFO: Modelo SRNet-lite cargado desde: ml/checkpoints/srnet_lite_best.pt (threshold=0.XXXX)
INFO: Application startup complete.
```

Si ves ese mensaje, el modelo está activo.

---

## Paso 5 — Verificar que salió del modo mock

```bash
curl http://localhost:8000/health
```

**Respuesta esperada (modo real):**
```json
{
  "status": "ok",
  "mock_mode": false,
  "model_version": "srnet-lite-v1",
  "message": "Modelo SRNet-lite activo. Threshold: 0.XXXX"
}
```

**Respuesta si sigue en mock:**
```json
{
  "status": "ok",
  "mock_mode": true,
  "model_version": "mock-v0.1",
  "message": "Sistema funcionando en modo demostración..."
}
```

---

## Solución de problemas — mock_mode sigue siendo true

### 1. Los archivos no están en la ruta correcta
Verifica con el script de diagnóstico:
```bash
cd artifacts/stegano-app
python ml/verify_replit_model.py
```

### 2. El workflow no se reinició
El servidor carga el checkpoint solo al arrancar. Si subiste los archivos con el
servidor ya corriendo, debes reiniciarlo desde el panel de Workflows.

### 3. Error al cargar el modelo
Revisa los logs del workflow "StegaDetect". Si ves:
```
WARNING: No se pudo cargar el modelo (...). Activando modo mock.
```

| Error en los logs | Causa | Solución |
|-------------------|-------|----------|
| `unexpected key(s) in state_dict` | Checkpoint de arquitectura diferente | Usa el notebook 02 sin modificar |
| `Error(s) in loading state_dict` | Nombres de capas no coinciden | No renombres atributos en el modelo |
| `EOFError` o `UnpicklingError` | Archivo corrupto o descarga incompleta | Descarga el `.pt` de Drive de nuevo |
| `FileNotFoundError: model_metadata.json` | Falta el archivo de metadata | Sube también `model_metadata.json` |

---

## Nombres de capas en el state_dict

Las claves del checkpoint generado por el notebook 02 son:

```
srm.srm.weight                  ← filtros SRM fijos (no entrenados)
stem.0.weight / stem.1.weight   ← proyección inicial
stage1.0.block.0.weight ...     ← bloques residuales
down1.main.0.weight ...         ← downsample
stage2 / attn2 / down2 ...      ← etapas 2, 3, 4
classifier.3.weight             ← cabeza clasificadora (Linear 128→64)
classifier.6.weight             ← cabeza clasificadora (Linear 64→1)
```

Estos nombres DEBEN coincidir exactamente con la arquitectura en
`ml/src/models/srnet_lite.py`. No modifiques los nombres de atributos.

---

## Formato de model_metadata.json

```json
{
  "model_name": "SRNetLite",
  "input_size": 128,
  "threshold": 0.4821,
  "normalization": {
    "mean": [0.5, 0.5, 0.5],
    "std":  [0.5, 0.5, 0.5]
  },
  "trained_on": "BOSSBase-1.01 + LSB p005/p010/p020",
  "test_auc": 0.75,
  "test_f1":  0.71,
  "trained_at": "2025-XX-XX",
  "epochs_trained": 35
}
```

El campo `threshold` es el valor crítico: determina a partir de qué probabilidad
el sistema clasifica una imagen como esteganografiada.
