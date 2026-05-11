"""
dataset_validator.py — Validador completo del dataset de estegoanálisis.

Verifica:
  1. Integridad de imágenes (legibilidad con Pillow)
  2. Balance por clase (cover vs stego)
  3. Conteo por payload
  4. Dimensiones mínimas
  5. Leakage entre splits (misma imagen base en train y test)
  6. Rutas inexistentes
  7. Muestras corruptas
  8. Estadísticas generales

Uso como CLI:
    python ml/src/dataset/dataset_validator.py \\
      --train /content/drive/MyDrive/stego_project/processed/train_manifest.csv \\
      --val   /content/drive/MyDrive/stego_project/processed/val_manifest.csv \\
      --test  /content/drive/MyDrive/stego_project/processed/test_manifest.csv \\
      --output /content/drive/MyDrive/stego_project/reports/dataset_validation.json
"""

import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict, Counter

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def load_manifest(path: Path) -> List[Dict]:
    """Carga un manifest CSV y devuelve lista de filas como dicts."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest no encontrado: {path}")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def validate_manifest(
    rows: List[Dict],
    split_name: str,
    min_width: int = 128,
    min_height: int = 128,
    check_images: bool = True,
    max_corrupt_check: int = 200,
) -> Dict:
    """
    Valida un conjunto de imágenes de un split.

    Args:
        rows:              Filas del manifest
        split_name:        Nombre del split ("train", "val", "test")
        min_width/height:  Dimensiones mínimas aceptables
        check_images:      Si True, abre cada imagen con Pillow (lento pero completo)
        max_corrupt_check: Máximo de imágenes a verificar (para datasets grandes)

    Returns:
        Diccionario con estadísticas y lista de errores
    """
    result = {
        "split":         split_name,
        "total":         len(rows),
        "cover":         0,
        "stego":         0,
        "by_payload":    {},
        "missing_paths": [],
        "corrupt":       [],
        "small_images":  [],
        "errors":        [],
        "source_images": set(),
    }

    for i, row in enumerate(rows):
        label   = int(row.get("label", -1))
        payload = float(row.get("payload", 0.0))
        path    = Path(row.get("image_path", ""))
        source  = row.get("source_image", "")

        if source:
            result["source_images"].add(source)

        # Conteo por clase
        if label == 0:
            result["cover"] += 1
        elif label == 1:
            result["stego"] += 1
            key = f"p{int(payload*100):03d}"
            result["by_payload"][key] = result["by_payload"].get(key, 0) + 1
        else:
            result["errors"].append(f"Label inválido en fila {i}: {label}")

        # Verificar que el archivo existe
        if not path.exists():
            result["missing_paths"].append(str(path))
            continue

        # Verificar imagen (limitado a max_corrupt_check por performance)
        if check_images and i < max_corrupt_check and PIL_AVAILABLE:
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    if w < min_width or h < min_height:
                        result["small_images"].append({
                            "path": str(path),
                            "size": f"{w}x{h}"
                        })
            except Exception as e:
                result["corrupt"].append({"path": str(path), "error": str(e)})

    # Calcular balance
    total = result["total"]
    if total > 0:
        result["cover_ratio"] = round(result["cover"] / total, 4)
        result["stego_ratio"] = round(result["stego"] / total, 4)
    else:
        result["cover_ratio"] = 0.0
        result["stego_ratio"] = 0.0

    result["source_images"] = list(result["source_images"])
    return result


def check_leakage(
    train_sources: Set[str],
    val_sources: Set[str],
    test_sources: Set[str],
) -> Dict:
    """
    Verifica que no haya leakage entre splits.

    Leakage ocurre cuando la misma imagen base aparece en múltiples splits.
    Esto infla artificialmente las métricas: el modelo "memoriza" la imagen
    en train y la reconoce en test aunque sea stego/cover.

    Por ejemplo: si bossbase_0001.png está en train (como cover) y
    en test (como stego p005), el modelo podría aprender propiedades
    específicas de esa imagen, no del proceso de embebido LSB.
    """
    train_val = train_sources & val_sources
    train_test = train_sources & test_sources
    val_test = val_sources & test_sources

    has_leakage = bool(train_val or train_test or val_test)

    return {
        "has_leakage":       has_leakage,
        "train_val_overlap": sorted(list(train_val))[:20],   # Máx 20 ejemplos
        "train_test_overlap":sorted(list(train_test))[:20],
        "val_test_overlap":  sorted(list(val_test))[:20],
        "n_train_val":       len(train_val),
        "n_train_test":      len(train_test),
        "n_val_test":        len(val_test),
    }


def run_validation(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    output_path: Path,
    check_images: bool = True,
) -> Dict:
    """
    Ejecuta la validación completa del dataset y guarda el reporte.

    Returns:
        Diccionario completo con el reporte de validación
    """
    print(f"\n{'='*60}")
    print(f"  Validando dataset de estegoanálisis")
    print(f"{'='*60}")

    report = {"status": "ok", "warnings": [], "splits": {}}

    # Validar cada split
    for name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        print(f"\n[{name.upper()}] Cargando: {path}")
        try:
            rows = load_manifest(path)
            stats = validate_manifest(rows, name, check_images=check_images)
            report["splits"][name] = stats

            # Imprimir resumen
            print(f"  Total: {stats['total']}")
            print(f"  Cover: {stats['cover']} ({stats['cover_ratio']:.1%})")
            print(f"  Stego: {stats['stego']} ({stats['stego_ratio']:.1%})")
            print(f"  Por payload: {stats['by_payload']}")
            if stats["missing_paths"]:
                print(f"  [WARNING] Rutas inexistentes: {len(stats['missing_paths'])}")
                report["warnings"].append(f"{name}: {len(stats['missing_paths'])} rutas faltantes")
            if stats["corrupt"]:
                print(f"  [WARNING] Imágenes corruptas: {len(stats['corrupt'])}")
                report["warnings"].append(f"{name}: {len(stats['corrupt'])} imágenes corruptas")
            if stats["small_images"]:
                print(f"  [WARNING] Imágenes pequeñas: {len(stats['small_images'])}")

        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            report["splits"][name] = {"error": str(e)}
            report["status"] = "error"

    # Verificar leakage
    print(f"\n[LEAKAGE CHECK]")
    try:
        train_sources = set(report["splits"].get("train", {}).get("source_images", []))
        val_sources   = set(report["splits"].get("val",   {}).get("source_images", []))
        test_sources  = set(report["splits"].get("test",  {}).get("source_images", []))

        leakage = check_leakage(train_sources, val_sources, test_sources)
        report["leakage"] = leakage

        if leakage["has_leakage"]:
            print(f"  [ERROR] Leakage detectado!")
            print(f"    train↔val:  {leakage['n_train_val']} imágenes")
            print(f"    train↔test: {leakage['n_train_test']} imágenes")
            print(f"    val↔test:   {leakage['n_val_test']} imágenes")
            report["status"] = "error"
            report["warnings"].append("Leakage detectado entre splits!")
        else:
            print(f"  [OK] Sin leakage. Los splits son disjuntos.")
    except Exception as e:
        report["leakage"] = {"error": str(e)}

    # Verificar balance global
    print(f"\n[BALANCE]")
    total_cover = sum(
        report["splits"].get(s, {}).get("cover", 0)
        for s in ["train", "val", "test"]
    )
    total_stego = sum(
        report["splits"].get(s, {}).get("stego", 0)
        for s in ["train", "val", "test"]
    )
    total = total_cover + total_stego
    if total > 0:
        ratio = total_stego / total
        print(f"  Global: {total_cover} cover, {total_stego} stego ({ratio:.1%} stego)")
        if ratio < 0.3 or ratio > 0.7:
            msg = f"Dataset desbalanceado: {ratio:.1%} stego — usar pos_weight en loss"
            print(f"  [WARNING] {msg}")
            report["warnings"].append(msg)
        else:
            print(f"  [OK] Balance aceptable.")

    # Guardar reporte
    report_to_save = {k: v for k, v in report.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_to_save, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[OK] Reporte guardado en: {output_path}")

    print(f"\n{'='*60}")
    print(f"  Estado final: {report['status'].upper()}")
    if report["warnings"]:
        print(f"  Advertencias: {len(report['warnings'])}")
        for w in report["warnings"]:
            print(f"    - {w}")
    print(f"{'='*60}\n")

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validador de dataset de estegoanálisis"
    )
    parser.add_argument("--train",  type=Path, required=True,
                        help="Ruta al train_manifest.csv")
    parser.add_argument("--val",    type=Path, required=True,
                        help="Ruta al val_manifest.csv")
    parser.add_argument("--test",   type=Path, required=True,
                        help="Ruta al test_manifest.csv")
    parser.add_argument("--output", type=Path, required=True,
                        help="Ruta donde guardar el JSON de validación")
    parser.add_argument("--no-image-check", action="store_true",
                        help="Saltar la verificación de apertura de imágenes (más rápido)")
    args = parser.parse_args()

    run_validation(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        output_path=args.output,
        check_images=not args.no_image_check,
    )


if __name__ == "__main__":
    main()
