"""
generate_dataset_v2.py — Generador de dataset de esteganografía LSB.

¿Qué es el payload ratio?
==========================
El payload ratio (p) indica qué fracción de los bits disponibles de la imagen
se usan para ocultar información. Por ejemplo, p=0.1 significa que el 10% de
los pixels tienen su LSB modificado para contener datos ocultos.

¿Por qué payloads bajos son más difíciles de detectar?
=======================================================
Con p=0.05 solo se modifica 1 de cada 20 píxeles. La señal es tan débil que
es casi indistinguible del ruido cuántico natural de la imagen. El detector
necesita aprender patrones estadísticos muy sutiles.
Con p=0.20 la señal es 4× más fuerte y el detector tiene más información.

¿Por qué LSB random es más realista que LSB secuencial?
========================================================
LSB secuencial siempre modifica los primeros N píxeles en orden de lectura,
lo que crea un patrón predecible (los primeros píxeles tienen distribución
diferente a los últimos). Un atacante puede detectar esto trivialmente.
LSB random selecciona píxeles aleatoriamente usando una clave, lo que hace
la distribución estadística más uniforme y difícil de detectar.

¿Por qué PNG y no JPG?
========================
JPEG usa compresión con pérdida que modifica los valores de píxel al
guardar y cargar. Esto destruye la señal LSB exacta que insertamos.
PNG usa compresión sin pérdida: los bits que insertamos se preservan intactos.
Para estegoanálisis LSB, PNG es el único formato válido.

Uso:
    python ml/src/dataset/generate_dataset_v2.py \\
      --input-dir  /content/drive/MyDrive/stegadetect_replit/cover \\
      --output-dir /content/drive/MyDrive/stegadetect_replit/stego/p005 \\
      --payload-ratio 0.05 \\
      --mode random \\
      --seed 42 \\
      --min-size 128
"""

import os
import csv
import argparse
import hashlib
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[ERROR] Pillow no disponible: pip install Pillow")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = lambda x, **kw: x


def embed_lsb_random(
    pixels: np.ndarray,
    payload_ratio: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Embebe datos aleatorios en los bits menos significativos (LSB) de forma aleatoria.

    El mensaje "oculto" es ruido aleatorio (no necesitamos un mensaje real
    para entrenar el detector; solo necesitamos la perturbación estadística).

    Args:
        pixels:        Array numpy de la imagen, shape (H, W, C), dtype uint8
        payload_ratio: Fracción de píxeles a modificar [0.0, 1.0]
        seed:          Semilla para reproducibilidad

    Returns:
        Array modificado con los LSB alterados en las posiciones seleccionadas
    """
    rng = np.random.default_rng(seed)

    flat = pixels.reshape(-1)           # Aplanar a 1D
    n_pixels = len(flat)
    n_embed  = int(n_pixels * payload_ratio)

    if n_embed == 0:
        return pixels.copy()

    # Seleccionar posiciones aleatorias sin repetición
    indices = rng.choice(n_pixels, size=n_embed, replace=False)

    # Generar bits aleatorios (el "mensaje" es ruido)
    secret_bits = rng.integers(0, 2, size=n_embed, dtype=np.uint8)

    # Modificar LSB: (pixel & ~1) | bit
    # (pixel & ~1) limpia el LSB; | bit lo pone en 0 o 1
    modified = flat.copy()
    modified[indices] = (modified[indices] & np.uint8(0xFE)) | secret_bits

    return modified.reshape(pixels.shape)


def embed_lsb_sequential(
    pixels: np.ndarray,
    payload_ratio: float,
) -> np.ndarray:
    """
    Embebe datos en los primeros N píxeles en orden secuencial.
    Menos realista pero más simple. Útil para comparación.
    """
    flat = pixels.reshape(-1).copy()
    n_embed = int(len(flat) * payload_ratio)
    secret_bits = np.random.randint(0, 2, size=n_embed, dtype=np.uint8)
    flat[:n_embed] = (flat[:n_embed] & np.uint8(0xFE)) | secret_bits
    return flat.reshape(pixels.shape)


def process_image(
    input_path: Path,
    output_path: Path,
    payload_ratio: float,
    mode: str = "random",
    seed: int = 42,
    min_size: int = 128,
) -> Optional[str]:
    """
    Carga una imagen cover, embebe LSB y guarda como stego PNG.

    Args:
        input_path:    Imagen fuente (cover)
        output_path:   Destino (stego)
        payload_ratio: Fracción de píxeles a modificar
        mode:          "random" | "sequential"
        seed:          Semilla determinista por imagen
        min_size:      Tamaño mínimo en píxeles (width y height)

    Returns:
        Mensaje de error o None si tuvo éxito
    """
    if not PIL_AVAILABLE:
        return "Pillow no disponible"

    try:
        with Image.open(input_path) as img:
            # Validar dimensiones mínimas
            w, h = img.size
            if w < min_size or h < min_size:
                return f"Imagen demasiado pequeña: {w}x{h}"

            # Convertir a RGB (algunas imágenes PGM son grayscale o tienen alpha)
            img_rgb = img.convert("RGB")
            pixels  = np.array(img_rgb, dtype=np.uint8)

        # Usar seed derivada del nombre del archivo para reproducibilidad
        # Esto garantiza que la misma imagen siempre produce el mismo stego
        img_seed = int(hashlib.md5(input_path.name.encode()).hexdigest()[:8], 16) ^ seed

        if mode == "random":
            stego_pixels = embed_lsb_random(pixels, payload_ratio, seed=img_seed)
        else:
            stego_pixels = embed_lsb_sequential(pixels, payload_ratio)

        # Guardar como PNG (sin pérdida — obligatorio para preservar LSB)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stego_img = Image.fromarray(stego_pixels, mode="RGB")
        stego_img.save(str(output_path), format="PNG", optimize=False)

        return None  # Sin error

    except Exception as e:
        return str(e)


def generate_dataset(
    input_dir: Path,
    output_dir: Path,
    payload_ratio: float,
    mode: str = "random",
    seed: int = 42,
    min_size: int = 128,
    extensions: Tuple[str, ...] = (".png", ".pgm", ".jpg", ".jpeg"),
) -> List[dict]:
    """
    Procesa todos los archivos de input_dir y genera el dataset stego.

    Returns:
        Lista de dicts con info de cada imagen procesada (para el manifest)
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Encontrar todas las imágenes fuente
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in extensions
    ])

    if not image_files:
        raise ValueError(f"No se encontraron imágenes en: {input_dir}")

    print(f"\n[Generator] Payload ratio: {payload_ratio} ({payload_ratio*100:.0f}%)")
    print(f"[Generator] Modo: {mode}")
    print(f"[Generator] Imágenes fuente: {len(image_files)}")
    print(f"[Generator] Destino: {output_dir}\n")

    manifest = []
    errors   = []

    iterator = tqdm(image_files, desc=f"LSB p={payload_ratio}", unit="img") \
               if TQDM_AVAILABLE else image_files

    for img_file in iterator:
        # Forzar extensión .png en el output (preservar nombre base)
        out_name   = img_file.stem + ".png"
        out_path   = output_dir / out_name

        error = process_image(
            input_path=img_file,
            output_path=out_path,
            payload_ratio=payload_ratio,
            mode=mode,
            seed=seed,
            min_size=min_size,
        )

        if error:
            errors.append({"file": str(img_file), "error": error})
        else:
            manifest.append({
                "image_path":   str(out_path),
                "label":        1,              # 1 = stego
                "payload":      payload_ratio,
                "source_image": img_file.stem,
                "split":        "",             # Se asigna en el notebook
            })

    print(f"\n[Generator] Procesadas: {len(manifest)} | Errores: {len(errors)}")
    if errors:
        print(f"[Generator] Primeros errores:")
        for e in errors[:5]:
            print(f"  {e['file']}: {e['error']}")

    return manifest


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generador de dataset de esteganografía LSB"
    )
    parser.add_argument("--input-dir",      type=Path, required=True,
                        help="Directorio con imágenes cover (PNG/PGM)")
    parser.add_argument("--output-dir",     type=Path, required=True,
                        help="Directorio destino para imágenes stego")
    parser.add_argument("--payload-ratio",  type=float, default=0.10,
                        help="Fracción de bits a modificar (ej: 0.10 = 10%%)")
    parser.add_argument("--mode",           choices=["random", "sequential"],
                        default="random", help="Modo de embebido LSB")
    parser.add_argument("--seed",           type=int, default=42,
                        help="Semilla aleatoria para reproducibilidad")
    parser.add_argument("--min-size",       type=int, default=128,
                        help="Tamaño mínimo de imagen en píxeles")
    parser.add_argument("--manifest-out",   type=Path, default=None,
                        help="Ruta para guardar manifest CSV del payload")
    args = parser.parse_args()

    manifest = generate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        payload_ratio=args.payload_ratio,
        mode=args.mode,
        seed=args.seed,
        min_size=args.min_size,
    )

    if args.manifest_out:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
            writer.writeheader()
            writer.writerows(manifest)
        print(f"[Generator] Manifest guardado en: {args.manifest_out}")


if __name__ == "__main__":
    main()
