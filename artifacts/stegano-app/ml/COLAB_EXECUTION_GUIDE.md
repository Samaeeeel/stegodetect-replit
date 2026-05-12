# Guía de Ejecución en Google Colab

Guía completa para ejecutar el pipeline de entrenamiento del detector de esteganografía.
Sigue los pasos en orden. No saltes celdas.

---

## Requisitos previos

| Requisito | Detalle |
|-----------|---------|
| Cuenta Google | Con acceso a Google Drive y Google Colab |
| Espacio en Drive | **~20 GB libres** (ver desglose abajo) |
| GPU en Colab | T4 (gratis) o A100 (Colab Pro) |
| Tiempo total | ~6–8 horas (primera vez) |

### Desglose de espacio en Drive

| Carpeta | Contenido | Tamaño aproximado |
|---------|-----------|-------------------|
| `raw/` | BOSSBase tar.gz | 685 MB |
| `cover/` | 10.000 PNG 512×512 | ~4.5 GB |
| `stego/p005/` | 10.000 PNG stego | ~4.5 GB |
| `stego/p010/` | 10.000 PNG stego | ~4.5 GB |
| `stego/p020/` | 10.000 PNG stego | ~4.5 GB |
| `processed/` | Manifests CSV | < 5 MB |
| `checkpoints/` | Checkpoint + metadata | ~50 MB |
| **Total** | | **~19 GB** |

---

## Paso 1 — Abrir el notebook 01 en Google Colab

Tienes dos opciones:

### Opción A — Subir el archivo directamente
1. Ve a [colab.research.google.com](https://colab.research.google.com)
2. Menú **Archivo → Subir notebook**
3. Selecciona `ml/notebooks/01_dataset_pipeline_colab.py`
4. Colab convierte el archivo `.py` a notebook automáticamente

### Opción B — Desde Google Drive
1. Sube el archivo `01_dataset_pipeline_colab.py` a tu Drive
2. Haz doble clic sobre el archivo en Drive
3. Selecciona **Abrir con → Google Colaboratory**

### Verificar que tienes GPU
- Menú **Runtime → Change runtime type → T4 GPU**
- Si no tienes GPU disponible en el plan gratuito, espera unos minutos e intenta de nuevo

---

## Paso 2 — Ejecutar el Notebook 01 (Dataset)

**Tiempo estimado: 3–5 horas**

Ejecuta las celdas en orden con **Shift+Enter** o usa **Runtime → Run all**.

### Celda 1 — Instalaciones
```
pip install pillow tqdm
```
Tiempo: ~30 segundos. No debe dar errores.

### Celda 2 — Montar Google Drive ⚠️ REVISAR
```python
from google.colab import drive
drive.mount('/content/drive')
```
- Colab abrirá un popup pidiendo autorización
- Haz clic en el enlace, elige tu cuenta Google, copia el código de verificación
- Pégalo en el campo que aparece en la celda
- **Salida correcta:** `Mounted at /content/drive`
- **Error común:** Si ves `OSError: [Errno 107]`, ejecuta la celda de nuevo

### Celda 3 — Configuración de rutas
- Crea la estructura de carpetas en `/content/drive/MyDrive/stegadetect_replit/`
- **Salida correcta:**
  ```
  Rutas configuradas:
    Base:       /content/drive/MyDrive/stegadetect_replit
    Cover:      /content/drive/MyDrive/stegadetect_replit/cover
    Stego p005: /content/drive/MyDrive/stegadetect_replit/stego/p005
    Processed:  /content/drive/MyDrive/stegadetect_replit/processed
    Reports:    /content/drive/MyDrive/stegadetect_replit/reports
  ```

### Celda 4 — Descargar BOSSBase ⚠️ REVISAR (puede tardar 15-30 min)
- Descarga `BossBase-1.01-cover.tar.gz` (~685 MB) desde el servidor de la CTU
- Si ya lo descargaste antes, la celda lo detecta y lo salta
- **Salida correcta:**
  ```
  Descargando BOSSBase desde: http://agents.fel.cvut.cz/stegodata/BossBase-1.01-cover.tar.gz
  10.0% (68 MB / 685 MB)
  ...
  Descargado: 685.3 MB
  ```
- **Error: URL no accesible** → El servidor de la CTU puede estar caído. Espera y reintenta, o usa una copia alternativa de BOSSBase. Si tienes el archivo localmente, súbelo manualmente a la carpeta `raw/` de Drive y salta esta celda.

### Celda 5 — Descomprimir BOSSBase (10–20 min)
- Extrae los 10.000 archivos `.pgm` en `/content/cache/`
- **Salida correcta:**
  ```
  Descomprimido: 10000 archivos PGM
  [OK] 10000 imágenes PGM disponibles
  ```
- **Error: menos de 9000 PGM** → El archivo se descargó incompleto. Borra el `.tar.gz` de Drive y descarga de nuevo.

### Celda 6 — Convertir PGM → PNG (20–40 min)
- Convierte los 10.000 PGM a PNG RGB
- Muestra una barra de progreso con `tqdm`
- **Salida correcta:**
  ```
  [OK] 10000 cover PNG generados en .../cover/
  ```
- **Nota:** Si la sesión de Colab se interrumpe, la celda reanuda desde donde quedó (detecta PNGs existentes).

### Celda 7 — Generar imágenes stego (45–90 min) ⚠️ LA MÁS LARGA
- Genera 10.000 imágenes estego para cada uno de los 3 payloads (p005, p010, p020)
- Total: 30.000 imágenes nuevas
- **Salida correcta:**
  ```
  Generando stego p005 (payload=0.05)...
  100%|████████| 10000/10000 [xx:xx<00:00]
  [OK] p005: 10000 imágenes generadas
  Generando stego p010 (payload=0.10)...
  ...
  ```

### Celdas 8–11 — Splits y manifests (5–10 min)
- Crea los splits train/val/test sin leakage
- **Salida correcta al final:**
  ```
  [OK] train_manifest.csv: 28000 muestras (14000 cover + 14000 stego)
  [OK] val_manifest.csv: 6000 muestras
  [OK] test_manifest.csv: 6000 muestras
  [OK] Sin leakage entre splits
  dataset_validation.json guardado
  ```
- **Error: leakage detectado** → No deberías ver esto; si ocurre, reporta el error completo.

---

## Paso 3 — Ejecutar el Notebook 02 (Entrenamiento)

**Tiempo estimado: 3–5 horas con T4**

### Antes de empezar
- El notebook 01 debe haber terminado correctamente
- Puedes usar la misma sesión de Colab o una nueva
- Si es sesión nueva: vuelve a correr la **Celda 2** (montar Drive) del notebook 01, o ejecuta:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

### Celda 1 — Instalaciones
```
pip install torch torchvision scikit-learn matplotlib tqdm
```
Tiempo: ~2 minutos

### Celda 2 — Imports y configuración
- Define `CONFIG` con todos los hiperparámetros
- **Salida correcta:** `[OK] Usando dispositivo: cuda` (si tienes T4)
- **Si ves `cpu`:** La sesión no tiene GPU. Ve a Runtime → Change runtime type

### Celda 3 — Rutas de Drive
- Verifica que los manifests del notebook 01 existen
- **Salida correcta:**
  ```
  [OK] train_manifest.csv: 28000 filas
  [OK] val_manifest.csv: 6000 filas
  [OK] test_manifest.csv: 6000 filas
  ```
- **Error: archivo no encontrado** → El notebook 01 no terminó correctamente. Vuelve a ejecutarlo.

### Celda 6 — Sanity check del primer batch ⚠️ REVISAR
- Carga el primer batch y verifica dimensiones y normalización
- **Salida correcta:**
  ```
  [SanityCheck] Primer batch de entrenamiento:
    images.shape: torch.Size([32, 3, 128, 128])   ← esperado
    images.dtype: torch.float32                   ← esperado
    images.min/max: [-1.xxx, 1.xxx]               ← esperado: [-1, 1]
    labels.unique: [0, 1]
    balance: ~50.0% stego
  [OK] Batch verificado. Listo para entrenar.
  ```
- **Error: AssertionError en shape** → Revisa que los manifests tienen la ruta correcta a las imágenes.
- **Error: NaN en imágenes** → Alguna imagen está corrupta. El dataset loader la reemplaza con negro, pero verifica con la celda de validación.

### Celda 7 — Construir modelo ⚠️ REVISAR
- **Salida correcta:**
  ```
  [OK] SRNetLite construido: X parámetros entrenables (Y.YYM)
  ```
- El número de parámetros debe estar entre **500K y 2M** — si ves algo muy diferente, el modelo no se construyó correctamente.

### Celdas 8–10 — Loop de entrenamiento ⚠️ LA MÁS IMPORTANTE
- Itera por épocas, imprimiendo métricas por época
- **Salida correcta por época:**
  ```
  Época 01/50 | train_loss=0.693 | val_loss=0.680 | val_auc=0.541 | lr=1e-3
  Época 02/50 | train_loss=0.670 | val_loss=0.655 | val_auc=0.583 | lr=1e-3
  Época 05/50 | train_loss=0.630 | val_loss=0.610 | val_auc=0.640 | ...
  ...
  [EarlyStopping] Mejor val_auc: 0.XXX en época YY
  ```

**Señales de entrenamiento saludable:**
- `val_loss` decrece en las primeras 5–10 épocas
- `val_auc` supera 0.60 antes de la época 10
- `val_auc` final > 0.70 es un buen resultado para LSB con p005

**Señales de colapso (abortar y revisar):**
- `val_auc` se queda en `0.500` después de 10 épocas → el modelo predice siempre la misma clase
- `val_loss` = `0.693` constante → predice 50/50 siempre (sin aprender)
- `val_loss` sube en lugar de bajar → overfitting muy temprano

**Si hay colapso:**
1. Verifica que el dataset esté balanceado (50% cover, 50% stego)
2. Reduce learning rate: cambia `lr` de `1e-3` a `1e-4` en `CONFIG`
3. Verifica que las imágenes stego realmente tienen bits modificados (ejecuta un par manualmente con el generador)

### Celda 11 — Evaluación en test set
- **Salida correcta:**
  ```
  [Test] AUC: 0.XXX | F1: 0.XXX | Accuracy: XX.X%
  Threshold óptimo: 0.XXX
  Matriz de confusión guardada
  ```

### Celda 12 — Guardar checkpoint ⚠️ REVISAR
- Guarda `srnet_lite_best.pt` y `model_metadata.json` en Drive
- **Salida correcta:**
  ```
  [OK] Checkpoint guardado: .../checkpoints/srnet_lite_best.pt
  [OK] Metadata guardada:   .../checkpoints/model_metadata.json
  [OK] Threshold óptimo: 0.XXX (guardado en metadata)
  ```

---

## Paso 4 — Descargar desde Drive a tu máquina

Después de que el notebook 02 termine:

1. Ve a [drive.google.com](https://drive.google.com)
2. Navega a `Mi unidad / stegadetect_replit / checkpoints /`
3. Descarga **ambos archivos**:
   - `srnet_lite_best.pt`
   - `model_metadata.json`

---

## Errores comunes y soluciones

| Error | Causa probable | Solución |
|-------|---------------|----------|
| `Mounted at /content/drive` no aparece | Popup bloqueado por el navegador | Permite popups para colab.research.google.com |
| `RuntimeError: CUDA out of memory` | Batch muy grande para la GPU | Reduce `batch_size` de 32 a 16 en `CONFIG` |
| `FileNotFoundError: train_manifest.csv` | Notebook 01 no terminó | Ejecuta notebook 01 completo primero |
| La descarga de BOSSBase falla | Servidor CTU caído | Reintenta más tarde o usa un mirror alternativo |
| `AssertionError` en sanity check | Imágenes con forma incorrecta | Verifica que las rutas en los manifests son absolutas |
| `val_auc = 0.5` constante | Dataset desbalanceado o stego mal generado | Verifica balance con `train_manifest.csv` |
| Sesión de Colab se desconecta | Timeout por inactividad | Ejecuta con Pro o mantén la pestaña activa; el checkpoint se guarda por época |
| `torch.cuda.is_available()` → False | Sin GPU asignada | Runtime → Change runtime type → T4 GPU |

---

## Resumen de tiempos estimados (T4 gratuito)

| Fase | Tiempo |
|------|--------|
| Montar Drive | 1 min |
| Descargar BOSSBase | 15–30 min |
| Descomprimir | 10–20 min |
| Convertir PGM → PNG | 20–40 min |
| Generar stego ×3 | 45–90 min |
| Crear manifests | 5 min |
| Entrenamiento (50 épocas) | 2–4 horas |
| Evaluación test | 5–10 min |
| **Total primera vez** | **~5–8 horas** |
| **Total ejecuciones posteriores** (dataset ya en Drive) | **~2–4 horas** |
