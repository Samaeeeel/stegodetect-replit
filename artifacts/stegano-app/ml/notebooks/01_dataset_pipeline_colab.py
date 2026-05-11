# %% [markdown]
# # Notebook 01 — Pipeline de Dataset para Estegoanálisis
#
# **Proyecto:** Sistema inteligente de detección de mensajes ocultos en imágenes
# mediante esteganografía y Machine Learning (Tesis de grado)
#
# **Objetivo de este notebook:**
# Generar el dataset completo de entrenamiento a partir de BOSSBase v1.01,
# aplicando esteganografía LSB con múltiples payloads y creando los manifests
# de train/val/test sin leakage entre splits.
#
# **Ejecutar en Google Colab** con GPU habilitada (Runtime → Change runtime type → GPU T4)
#
# **Tiempo estimado:** 60–90 minutos (descarga + conversión + generación de stego)
#
# ---
# **Flujo:**
# 1. Montar Google Drive
# 2. Crear estructura de carpetas en Drive
# 3. Descargar y descomprimir BOSSBase v1.01
# 4. Convertir PGM → PNG (cover)
# 5. Generar imágenes stego con LSB (p=0.05, 0.10, 0.20)
# 6. Crear splits train/val/test sin leakage
# 7. Guardar manifests CSV en Drive
# 8. Validar el dataset

# %% [markdown]
# ## Celda 1 — Instalar dependencias

# %%
# Instalar librerías necesarias (solo las que no vienen en Colab)
# tqdm ya viene en Colab; agregamos scikit-learn si falta
import subprocess
subprocess.run(["pip", "install", "-q", "tqdm", "pillow", "scikit-learn"], check=True)
print("Dependencias listas.")

# %% [markdown]
# ## Celda 2 — Montar Google Drive

# %%
from google.colab import drive
drive.mount("/content/drive")
print("Google Drive montado correctamente.")

# %% [markdown]
# ## Celda 3 — Configuración de rutas

# %%
import sys
import os
from pathlib import Path

# ── Rutas del proyecto en Drive ───────────────────────────────────────────────
DRIVE_BASE       = Path("/content/drive/MyDrive/stego_project")
RAW_DIR          = DRIVE_BASE / "raw"
COVER_DIR        = DRIVE_BASE / "cover"
STEGO_BASE       = DRIVE_BASE / "stego"
STEGO_P005       = STEGO_BASE / "p005"
STEGO_P010       = STEGO_BASE / "p010"
STEGO_P020       = STEGO_BASE / "p020"
PROCESSED_DIR    = DRIVE_BASE / "processed"
REPORTS_DIR      = DRIVE_BASE / "reports"
CHECKPOINT_DIR   = DRIVE_BASE / "checkpoints"

# Manifests
TRAIN_MANIFEST   = PROCESSED_DIR / "train_manifest.csv"
VAL_MANIFEST     = PROCESSED_DIR / "val_manifest.csv"
TEST_MANIFEST    = PROCESSED_DIR / "test_manifest.csv"

# BOSSBase
BOSSBASE_ZIP     = RAW_DIR / "BOSSbase_1.01.zip"
LOCAL_CACHE      = Path("/content/cache")
BOSSBASE_EXTRACT = LOCAL_CACHE / "BOSSBase_1.01"

# ── Crear estructura de carpetas ──────────────────────────────────────────────
for d in [RAW_DIR, COVER_DIR, STEGO_P005, STEGO_P010, STEGO_P020,
          PROCESSED_DIR, REPORTS_DIR, CHECKPOINT_DIR, LOCAL_CACHE]:
    d.mkdir(parents=True, exist_ok=True)

# ── Configuración de splits ───────────────────────────────────────────────────
TRAIN_RATIO = 0.70   # 70% de imágenes base para entrenamiento
VAL_RATIO   = 0.15   # 15% para validación
TEST_RATIO  = 0.15   # 15% para test
SEED        = 42

# ── Payloads a generar ────────────────────────────────────────────────────────
PAYLOADS = {
    "p005": (0.05, STEGO_P005),
    "p010": (0.10, STEGO_P010),
    "p020": (0.20, STEGO_P020),
}

print("Rutas configuradas:")
print(f"  Base:       {DRIVE_BASE}")
print(f"  Cover:      {COVER_DIR}")
print(f"  Stego p005: {STEGO_P005}")
print(f"  Processed:  {PROCESSED_DIR}")
print(f"  Reports:    {REPORTS_DIR}")

# %% [markdown]
# ## Celda 4 — Descargar BOSSBase v1.01
#
# BOSSBase v1.01 es el conjunto de referencia estándar para estegoanálisis.
# Contiene 10.000 imágenes en escala de grises (512×512 píxeles) en formato PGM,
# capturadas con diferentes cámaras digitales.
#
# Fuente oficial: http://agents.fel.cvut.cz/stegodata/
#
# **IMPORTANTE:** El archivo ZIP pesa ~685 MB.
# Si ya lo descargaste antes y está en Drive, esta celda lo salta automáticamente.

# %%
import urllib.request

BOSSBASE_URL = "http://agents.fel.cvut.cz/stegodata/BossBase-1.01-cover.tar.gz"
# Alternativa si falla la URL principal:
# BOSSBASE_URL = "https://drive.google.com/uc?id=<ID_DEL_ARCHIVO>"

# Verificar si el ZIP ya está en Drive
if BOSSBASE_ZIP.exists() and BOSSBASE_ZIP.stat().st_size > 1_000_000:
    print(f"BOSSBase ya descargado en Drive: {BOSSBASE_ZIP}")
    print(f"Tamaño: {BOSSBASE_ZIP.stat().st_size / 1e6:.1f} MB")
else:
    print(f"Descargando BOSSBase desde: {BOSSBASE_URL}")
    print("Esto puede tardar varios minutos...")

    # Descarga con progreso
    def report_progress(count, block_size, total_size):
        pct = min(count * block_size / total_size * 100, 100)
        if count % 500 == 0:
            print(f"  {pct:.1f}% ({count * block_size / 1e6:.0f} MB / {total_size/1e6:.0f} MB)")

    urllib.request.urlretrieve(BOSSBASE_URL, str(BOSSBASE_ZIP), reporthook=report_progress)
    print(f"\nDescargado: {BOSSBASE_ZIP.stat().st_size / 1e6:.1f} MB")

# %% [markdown]
# ## Celda 5 — Descomprimir BOSSBase

# %%
import tarfile
import zipfile

# Verificar si ya están los PGM descomprimidos
pgm_files_in_cache = list(BOSSBASE_EXTRACT.glob("*.pgm")) if BOSSBASE_EXTRACT.exists() else []

if len(pgm_files_in_cache) >= 9000:
    print(f"BOSSBase ya descomprimido: {len(pgm_files_in_cache)} PGM en {BOSSBASE_EXTRACT}")
else:
    print(f"Descomprimiendo en {LOCAL_CACHE}...")
    BOSSBASE_EXTRACT.mkdir(parents=True, exist_ok=True)

    # Detectar formato del archivo
    if str(BOSSBASE_ZIP).endswith(".tar.gz") or str(BOSSBASE_ZIP).endswith(".tgz"):
        with tarfile.open(str(BOSSBASE_ZIP), "r:gz") as tar:
            tar.extractall(str(LOCAL_CACHE))
    elif str(BOSSBASE_ZIP).endswith(".zip"):
        with zipfile.ZipFile(str(BOSSBASE_ZIP), "r") as zf:
            zf.extractall(str(LOCAL_CACHE))

    pgm_files_in_cache = list(BOSSBASE_EXTRACT.rglob("*.pgm"))
    print(f"Descomprimido: {len(pgm_files_in_cache)} archivos PGM")

# Verificar cantidad esperada
if len(pgm_files_in_cache) < 9000:
    print(f"[WARNING] Se esperaban ~10.000 PGM pero solo hay {len(pgm_files_in_cache)}")
    print("Verifica que el archivo de descarga esté completo.")
else:
    print(f"[OK] {len(pgm_files_in_cache)} imágenes PGM disponibles")

# %% [markdown]
# ## Celda 6 — Convertir PGM → PNG (cover)
#
# Convertimos las imágenes PGM a PNG RGB porque:
# 1. PNG es sin pérdida → preserva todos los bits exactamente
# 2. Nuestro modelo SRNet-lite espera entrada RGB
# 3. La conversión grayscale→RGB replica el canal gris en los 3 canales RGB
#
# **NOTA IMPORTANTE:** NO aplicamos resize, blur, jitter ni ninguna transformación
# que modifique los valores de píxel. Solo conversión de formato.

# %%
from PIL import Image
from tqdm import tqdm

# Buscar todos los PGM en el directorio extraído
all_pgm = sorted(BOSSBASE_EXTRACT.rglob("*.pgm"))
if not all_pgm:
    # Si la estructura de directorios es diferente, buscar recursivamente
    all_pgm = sorted(LOCAL_CACHE.rglob("*.pgm"))

print(f"PGM encontrados: {len(all_pgm)}")

# Verificar cuántos PNG cover ya existen en Drive
existing_cover = set(p.stem for p in COVER_DIR.glob("*.png"))
to_convert = [p for p in all_pgm if p.stem not in existing_cover]

print(f"Ya convertidos: {len(existing_cover)}")
print(f"Por convertir:  {len(to_convert)}")

errors_convert = []
for pgm_path in tqdm(to_convert, desc="PGM → PNG", unit="img"):
    try:
        out_path = COVER_DIR / (pgm_path.stem + ".png")
        with Image.open(pgm_path) as img:
            # Convertir a RGB (PGM es grayscale, replicar canal)
            img_rgb = img.convert("RGB")
            # Guardar como PNG sin compresión destructiva
            img_rgb.save(str(out_path), format="PNG", optimize=False)
    except Exception as e:
        errors_convert.append((str(pgm_path), str(e)))

cover_files = list(COVER_DIR.glob("*.png"))
print(f"\n[OK] Imágenes cover en Drive: {len(cover_files)}")
if errors_convert:
    print(f"[WARNING] Errores de conversión: {len(errors_convert)}")
    for f, e in errors_convert[:5]:
        print(f"  {f}: {e}")

# %% [markdown]
# ## Celda 7 — Generar dataset stego con LSB
#
# Para cada payload (0.05, 0.10, 0.20), generamos una versión stego de cada
# imagen cover modificando aleatoriamente la fracción correspondiente de LSBs.
#
# El embebido es **determinista por imagen** (misma imagen → mismo stego),
# lo que garantiza reproducibilidad del dataset.
#
# **Por qué múltiples payloads:**
# - p=0.05: Señal muy débil — más difícil, más realista
# - p=0.10: Señal moderada — caso estándar en literatura
# - p=0.20: Señal clara — más fácil de detectar
# Entrenando con los tres, el modelo aprende a detectar señales de diferente intensidad.

# %%
import numpy as np
import hashlib

def embed_lsb_random(pixels: np.ndarray, payload_ratio: float, seed: int) -> np.ndarray:
    """Embebe datos aleatorios en LSBs de posiciones aleatorias."""
    rng      = np.random.default_rng(seed)
    flat     = pixels.reshape(-1)
    n_embed  = int(len(flat) * payload_ratio)
    if n_embed == 0:
        return pixels.copy()
    indices  = rng.choice(len(flat), size=n_embed, replace=False)
    bits     = rng.integers(0, 2, size=n_embed, dtype=np.uint8)
    modified = flat.copy()
    modified[indices] = (modified[indices] & np.uint8(0xFE)) | bits
    return modified.reshape(pixels.shape)

def get_image_seed(filename: str, base_seed: int = 42) -> int:
    """Seed determinista por imagen basada en su nombre."""
    return int(hashlib.md5(filename.encode()).hexdigest()[:8], 16) ^ base_seed

cover_pngs = sorted(COVER_DIR.glob("*.png"))
print(f"Imágenes cover disponibles: {len(cover_pngs)}")

for payload_key, (payload_ratio, stego_dir) in PAYLOADS.items():
    existing_stego = set(p.stem for p in stego_dir.glob("*.png"))
    to_process     = [p for p in cover_pngs if p.stem not in existing_stego]

    print(f"\n[{payload_key}] payload={payload_ratio} | Por procesar: {len(to_process)}")

    errors_stego = []
    for cover_path in tqdm(to_process, desc=f"LSB {payload_key}", unit="img"):
        try:
            with Image.open(cover_path) as img:
                pixels = np.array(img.convert("RGB"), dtype=np.uint8)

            img_seed     = get_image_seed(cover_path.name, SEED)
            stego_pixels = embed_lsb_random(pixels, payload_ratio, seed=img_seed)

            out_path = stego_dir / cover_path.name
            Image.fromarray(stego_pixels, mode="RGB").save(
                str(out_path), format="PNG", optimize=False
            )
        except Exception as e:
            errors_stego.append((str(cover_path), str(e)))

    total_stego = len(list(stego_dir.glob("*.png")))
    print(f"  [OK] {total_stego} imágenes stego en {stego_dir}")
    if errors_stego:
        print(f"  [WARNING] Errores: {len(errors_stego)}")

# %% [markdown]
# ## Celda 8 — Crear splits train/val/test sin leakage
#
# **Estrategia anti-leakage:**
# Primero dividimos las imágenes BASE (las 10.000 de BOSSBase) en tres grupos
# mutuamente excluyentes. Luego asignamos cada imagen cover y TODAS sus versiones
# stego al mismo grupo.
#
# Esto garantiza que si bossbase_0001.png va a train, sus stego p005/p010/p020
# también van a train y NUNCA aparecen en val o test.

# %%
import csv
import random

random.seed(SEED)

# Obtener todas las imágenes cover
cover_files = sorted(COVER_DIR.glob("*.png"))
cover_stems  = [f.stem for f in cover_files]

# Barajar con semilla fija
random.shuffle(cover_stems)

# Calcular tamaños de split
n_total = len(cover_stems)
n_train = int(n_total * TRAIN_RATIO)
n_val   = int(n_total * VAL_RATIO)

train_stems = set(cover_stems[:n_train])
val_stems   = set(cover_stems[n_train:n_train + n_val])
test_stems  = set(cover_stems[n_train + n_val:])

print(f"Total imágenes base: {n_total}")
print(f"Train: {len(train_stems)} ({len(train_stems)/n_total:.1%})")
print(f"Val:   {len(val_stems)} ({len(val_stems)/n_total:.1%})")
print(f"Test:  {len(test_stems)} ({len(test_stems)/n_total:.1%})")
print(f"Verificación anti-leakage: {len(train_stems & val_stems & test_stems) == 0}")

def get_split(stem: str) -> str:
    if stem in train_stems: return "train"
    if stem in val_stems:   return "val"
    return "test"

# %% [markdown]
# ## Celda 9 — Construir manifests CSV

# %%
from collections import defaultdict

all_rows = {"train": [], "val": [], "test": []}

# ── Cover (label=0) ───────────────────────────────────────────────────────────
for cover_path in sorted(COVER_DIR.glob("*.png")):
    split = get_split(cover_path.stem)
    all_rows[split].append({
        "image_path":   str(cover_path),
        "label":        0,
        "payload":      0.00,
        "source_image": cover_path.stem,
        "split":        split,
    })

# ── Stego (label=1) ───────────────────────────────────────────────────────────
for payload_key, (payload_ratio, stego_dir) in PAYLOADS.items():
    for stego_path in sorted(stego_dir.glob("*.png")):
        split = get_split(stego_path.stem)
        all_rows[split].append({
            "image_path":   str(stego_path),
            "label":        1,
            "payload":      payload_ratio,
            "source_image": stego_path.stem,
            "split":        split,
        })

# Guardar CSVs
FIELDNAMES = ["image_path", "label", "payload", "source_image", "split"]

manifest_paths = {
    "train": TRAIN_MANIFEST,
    "val":   VAL_MANIFEST,
    "test":  TEST_MANIFEST,
}

for split_name, manifest_path in manifest_paths.items():
    rows = all_rows[split_name]
    random.shuffle(rows)   # Mezclar dentro del split
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    cover_count = sum(1 for r in rows if r["label"] == 0)
    stego_count = sum(1 for r in rows if r["label"] == 1)
    print(f"[{split_name}] {len(rows)} filas | cover={cover_count} | stego={stego_count} "
          f"| ratio={stego_count/len(rows):.1%}")

print(f"\nManifests guardados en: {PROCESSED_DIR}")

# %% [markdown]
# ## Celda 10 — Validar el dataset
#
# Verifica integridad, balance y ausencia de leakage.

# %%
import json

def validate_manifest_quick(manifest_path: Path, split_name: str) -> dict:
    """Validación rápida de un manifest."""
    rows = []
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    cover  = sum(1 for r in rows if int(r["label"]) == 0)
    stego  = sum(1 for r in rows if int(r["label"]) == 1)
    sources = set(r["source_image"] for r in rows)

    # Verificar algunas imágenes (muestra de 20)
    import random as rnd
    sample = rnd.sample(rows, min(20, len(rows)))
    corrupt = [r for r in sample if not Path(r["image_path"]).exists()]

    return {
        "split":       split_name,
        "total":       len(rows),
        "cover":       cover,
        "stego":       stego,
        "stego_ratio": round(stego / len(rows), 4) if rows else 0,
        "sources":     len(sources),
        "missing_sample": len(corrupt),
    }

validation_report = {}
for split_name, manifest_path in manifest_paths.items():
    stats = validate_manifest_quick(manifest_path, split_name)
    validation_report[split_name] = stats
    print(f"[{split_name}] total={stats['total']} | cover={stats['cover']} | "
          f"stego={stats['stego']} | stego_ratio={stats['stego_ratio']:.1%} | "
          f"sources={stats['sources']}")

# Verificar leakage
train_src = set()
val_src   = set()
test_src  = set()
for split_name, manifest_path in manifest_paths.items():
    with open(manifest_path) as f:
        srcs = set(r["source_image"] for r in csv.DictReader(f))
    if split_name == "train": train_src = srcs
    elif split_name == "val":  val_src = srcs
    else:                      test_src = srcs

leakage = {
    "train_val":  len(train_src & val_src),
    "train_test": len(train_src & test_src),
    "val_test":   len(val_src & test_src),
}
validation_report["leakage"] = leakage

if any(v > 0 for v in leakage.values()):
    print(f"\n[ERROR] Leakage detectado: {leakage}")
else:
    print(f"\n[OK] Sin leakage. Splits completamente disjuntos.")

# Guardar reporte
report_path = REPORTS_DIR / "dataset_validation.json"
with open(report_path, "w") as f:
    json.dump(validation_report, f, indent=2)
print(f"\nReporte guardado en: {report_path}")

# %% [markdown]
# ## Celda 11 — Resumen final

# %%
print("\n" + "="*60)
print("  DATASET LISTO PARA ENTRENAMIENTO")
print("="*60)
print(f"  Train: {TRAIN_MANIFEST}")
print(f"  Val:   {VAL_MANIFEST}")
print(f"  Test:  {TEST_MANIFEST}")
print(f"  Checkpoints (destino): {CHECKPOINT_DIR}")
print("="*60)
print("\nSiguiente paso: ejecutar 02_model_training_colab.py")
