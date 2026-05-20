"""
Servicio de esteganografía LSB.

Implementa inserción, extracción y análisis técnico de payloads
ocultos en imágenes usando técnica LSB (Least Significant Bit).

Formato propio "STEGODETECTv1":
  [MAGIC 13 bytes][header_len 4 bytes big-endian][header JSON UTF-8][payload bytes]

El magic header permite detectar si una imagen fue generada por este sistema.
Si no se encuentra el magic, la extracción reporta "no payload compatible".

Flujo de inserción:
  1. Calcular capacidad de la imagen
  2. Construir MAGIC + header_len + header_json + payload_bytes
  3. Convertir a bits (MSB first por byte)
  4. Reemplazar el LSB de cada canal seleccionado en los píxeles necesarios
  5. Guardar como PNG (nunca JPG — la compresión destruye el LSB)

Flujo de extracción:
  1. Leer bits LSB de los canales
  2. Verificar MAGIC (si falla → no es imagen del sistema)
  3. Leer header_len → leer header_json
  4. Leer payload_bytes según payload_size del header
  5. Validar SHA-256 → devolver payload o reportar error de integridad
"""

import csv
import hashlib
import json
import math
import struct
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from backend.core.config import STEGO_ARTIFACTS_DIR

# ── Constantes del protocolo ──────────────────────────────────────────────────
MAGIC          = b"STEGODETECTv1"   # 13 bytes — firma del sistema
MAGIC_LEN      = len(MAGIC)         # 13
HEADER_LEN_BYTES = 4                # 4 bytes big-endian para el tamaño del header JSON
MAX_PAYLOAD_BYTES = 2 * 1024 * 1024 # 2 MB límite de payload (evitar bloqueos en Replit)
CHANNEL_MAP    = {"R": 0, "G": 1, "B": 2}


# ── Excepciones del servicio ──────────────────────────────────────────────────

class StegoCapacityError(Exception):
    """El payload excede la capacidad de la imagen."""

class StegoFormatError(Exception):
    """No se encontró el magic header del sistema."""

class StegoIntegrityError(Exception):
    """El checksum SHA256 del payload no coincide."""


# ── Servicio principal ────────────────────────────────────────────────────────

class LSBSteganographyService:
    """
    Servicio de esteganografía LSB controlada por StegoDetect.

    Soporta:
    - Inserción de texto y archivos en imágenes PNG
    - Extracción de payloads generados por este sistema
    - Análisis técnico de distribución LSB
    - Generación de mapas de píxeles usados y CSV de posiciones
    """

    # ── calculate_capacity ────────────────────────────────────────────────────

    def calculate_capacity(
        self,
        image_path: Path,
        bits_per_channel: int = 1,
        channels: Tuple[str, ...] = ("R", "G", "B"),
    ) -> Dict[str, Any]:
        """
        Calcula la capacidad máxima de payload para una imagen.
        Descuenta el espacio que ocupa la cabecera del sistema.
        """
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            w, h = img_rgb.size

        n_channels     = len(channels)
        total_pixels   = w * h
        bits_available = total_pixels * n_channels * bits_per_channel
        bytes_available = bits_available // 8

        # Overhead estimado: MAGIC + 4 + header_json (~200 bytes)
        header_overhead = MAGIC_LEN + HEADER_LEN_BYTES + 200
        usable_bytes    = max(0, bytes_available - header_overhead)

        return {
            "width":                 w,
            "height":                h,
            "total_pixels":          total_pixels,
            "channels_used":         list(channels),
            "bits_per_channel":      bits_per_channel,
            "bits_available":        bits_available,
            "bytes_available":       bytes_available,
            "header_overhead":       header_overhead,
            "usable_bytes":          usable_bytes,
            "usable_kb":             round(usable_bytes / 1024, 2),
            "max_recommended_bytes": usable_bytes,
        }

    # ── embed_payload ─────────────────────────────────────────────────────────

    def embed_payload(
        self,
        cover_image_path: Path,
        payload_bytes: bytes,
        payload_type: str,
        original_filename: Optional[str] = None,
        mime_type: Optional[str] = None,
        bits_per_channel: int = 1,
        channels: Tuple[str, ...] = ("R", "G", "B"),
        mode: str = "sequential",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Oculta payload_bytes dentro de la imagen cover usando LSB.

        Retorna metadatos técnicos del proceso.
        Lanza StegoCapacityError si el payload no cabe.
        """
        if len(payload_bytes) > MAX_PAYLOAD_BYTES:
            raise StegoCapacityError(
                f"Payload ({len(payload_bytes):,} B) excede el límite del sistema "
                f"({MAX_PAYLOAD_BYTES:,} B = 2 MB)."
            )

        # Verificar capacidad
        cap = self.calculate_capacity(cover_image_path, bits_per_channel, channels)
        if len(payload_bytes) > cap["usable_bytes"]:
            raise StegoCapacityError(
                f"Payload ({len(payload_bytes):,} B) excede la capacidad útil "
                f"de la imagen ({cap['usable_bytes']:,} B = {cap['usable_kb']} KB). "
                f"Usa una imagen más grande o un payload más pequeño."
            )

        # Construir cabecera del protocolo
        sha256 = hashlib.sha256(payload_bytes).hexdigest()
        header = {
            "version":          "1.0",
            "payload_type":     payload_type,
            "filename":         original_filename or "",
            "mime_type":        mime_type or "application/octet-stream",
            "payload_size":     len(payload_bytes),
            "sha256":           sha256,
            "bits_per_channel": bits_per_channel,
            "channels":         list(channels),
            "mode":             mode,
            "seed":             seed,
            "created_by":       "StegoDetect",
            "algorithm":        f"LSB-{bits_per_channel}bit-{''.join(channels)}",
        }
        header_json     = json.dumps(header, ensure_ascii=False).encode("utf-8")
        header_len_pack = struct.pack(">I", len(header_json))  # 4 bytes big-endian

        # Stream total: MAGIC + header_len + header_json + payload
        data_to_embed = MAGIC + header_len_pack + header_json + payload_bytes
        bits          = _bytes_to_bits(data_to_embed)
        total_bits    = len(bits)

        # Abrir imagen y convertir a lista de píxeles
        with Image.open(cover_image_path) as img:
            img_rgb    = img.convert("RGB")
            w, h       = img_rgb.size
            orig_pixels = list(img_rgb.getdata())

        if total_bits > w * h * len(channels) * bits_per_channel:
            raise StegoCapacityError("Los datos totales superan la capacidad de la imagen.")

        # Insertar bits en los LSB de los canales seleccionados
        channel_indices = [CHANNEL_MAP[c] for c in channels]
        new_pixels      = list(orig_pixels)
        all_positions   = []    # todas las posiciones usadas (para CSV)
        bit_idx         = 0

        for pixel_idx in range(len(orig_pixels)):
            if bit_idx >= total_bits:
                break

            x = pixel_idx % w
            y = pixel_idx // w
            pixel_list = list(orig_pixels[pixel_idx])

            for ch_i, ch_idx in enumerate(channel_indices):
                for b in range(bits_per_channel):
                    if bit_idx >= total_bits:
                        break
                    # Reemplazar el b-ésimo LSB del canal
                    bit_mask   = 1 << b
                    clear_mask = (~bit_mask) & 0xFF
                    pixel_list[ch_idx] = (pixel_list[ch_idx] & clear_mask) | (bits[bit_idx] << b)

                    all_positions.append({
                        "x":          x,
                        "y":          y,
                        "channel":    channels[ch_i],
                        "bit_index":  bit_idx,
                        "payload_bit": bits[bit_idx],
                        "byte_index": bit_idx // 8,
                    })
                    bit_idx += 1

            new_pixels[pixel_idx] = tuple(pixel_list)

        # Reconstruir imagen stego
        stego_img = Image.new("RGB", (w, h))
        stego_img.putdata(new_pixels)

        # Guardar artefactos
        artifact_id    = str(uuid.uuid4())
        stego_filename = f"stego_{artifact_id}.png"
        csv_filename   = f"positions_{artifact_id}.csv"
        map_filename   = f"lsb_map_{artifact_id}.png"

        stego_path = STEGO_ARTIFACTS_DIR / stego_filename
        csv_path   = STEGO_ARTIFACTS_DIR / csv_filename
        map_path   = STEGO_ARTIFACTS_DIR / map_filename

        stego_img.save(str(stego_path), format="PNG", optimize=False)
        _write_positions_csv(csv_path, all_positions)
        _generate_lsb_map(orig_pixels, new_pixels, w, h, map_path)

        # Resumen de posiciones
        total_pixels_used = bit_idx // (len(channels) * bits_per_channel) + 1
        positions_summary = {
            "first_pixel":        {"x": 0, "y": 0},
            "last_pixel":         {
                "x": all_positions[-1]["x"] if all_positions else 0,
                "y": all_positions[-1]["y"] if all_positions else 0,
            },
            "total_pixels_used":  total_pixels_used,
            "total_bits_used":    total_bits,
            "channels_used":      list(channels),
            "capacity_used_pct":  round(
                100 * total_bits / (w * h * len(channels) * bits_per_channel), 4
            ),
        }

        return {
            "artifact_id":       artifact_id,
            "stego_filename":    stego_filename,
            "stego_path":        str(stego_path),
            "csv_filename":      csv_filename,
            "map_filename":      map_filename,
            "capacity":          cap,
            "payload": {
                "type":          payload_type,
                "filename":      original_filename or "",
                "size":          len(payload_bytes),
                "sha256":        sha256,
                "mime_type":     mime_type or "",
            },
            "positions_summary": positions_summary,
            "first_positions":   all_positions[:100],
            "technical": {
                "bits_per_channel":      bits_per_channel,
                "channels":              list(channels),
                "mode":                  mode,
                "algorithm":             header["algorithm"],
                "total_bits_embedded":   total_bits,
                "header_size_bytes":     len(MAGIC) + HEADER_LEN_BYTES + len(header_json),
                "payload_size_bytes":    len(payload_bytes),
            },
        }

    # ── auto_extract_payload ──────────────────────────────────────────────────

    def auto_extract_payload(
        self,
        stego_image_path: Path,
        bits_to_try: Tuple[int, ...] = (1, 2, 3, 4),
        channels: Tuple[str, ...] = ("R", "G", "B"),
    ) -> Dict[str, Any]:
        """
        Intenta extraer un payload StegoDetect probando varias configuraciones
        de bits_per_channel. Se detiene en el primer intento con cabecera
        STEGODETECTv1 válida.

        Necesario porque /stego/full-analysis no conoce los parámetros de
        embebido y antes solo probaba bits_per_channel=1, lo que producía
        falsos negativos para imágenes embebidas con 2+ bits.

        Devuelve el dict de extract_payload enriquecido con
        bits_per_channel_detected y channels_detected.
        """
        last_result: Dict[str, Any] = {
            "payload_found": False,
            "message": "No se encontró payload con ninguna configuración probada.",
        }
        attempts: List[Dict[str, Any]] = []

        for bpc in bits_to_try:
            try:
                result = self.extract_payload(
                    stego_image_path,
                    bits_per_channel=bpc,
                    channels=channels,
                )
            except Exception as exc:
                attempts.append({"bits_per_channel": bpc, "error": str(exc)})
                continue

            attempts.append({
                "bits_per_channel": bpc,
                "payload_found":    bool(result.get("payload_found")),
                "sha256_valid":     bool(result.get("sha256_valid")),
            })

            if result.get("payload_found"):
                result["bits_per_channel_detected"] = bpc
                result["channels_detected"]         = list(channels)
                result["auto_extraction_attempts"]  = attempts
                return result

            last_result = result

        last_result["auto_extraction_attempts"] = attempts
        return last_result

    # ── extract_payload ───────────────────────────────────────────────────────

    def extract_payload(
        self,
        stego_image_path: Path,
        bits_per_channel: int = 1,
        channels: Tuple[str, ...] = ("R", "G", "B"),
        mode: str = "sequential",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Intenta extraer un payload del sistema desde una imagen.

        Si no encuentra el MAGIC → payload_found=False.
        Si el SHA-256 no coincide → sha256_valid=False (payload devuelto igualmente).
        """
        with Image.open(stego_image_path) as img:
            img_rgb = img.convert("RGB")
            w, h    = img_rgb.size
            pixels  = list(img_rgb.getdata())

        channel_indices = [CHANNEL_MAP[c] for c in channels]

        def _extract_n_bits(n: int) -> List[int]:
            bits = []
            for pixel in pixels:
                if len(bits) >= n:
                    break
                for ch_idx in channel_indices:
                    for b in range(bits_per_channel):
                        if len(bits) >= n:
                            break
                        bits.append((pixel[ch_idx] >> b) & 1)
            return bits

        # Leer prefijo: MAGIC + header_len
        prefix_bits  = _extract_n_bits((MAGIC_LEN + HEADER_LEN_BYTES) * 8)
        prefix_bytes = _bits_to_bytes(prefix_bits)

        # Verificar MAGIC
        if prefix_bytes[:MAGIC_LEN] != MAGIC:
            lsb_analysis = self.analyze_lsb_structure(stego_image_path)
            return {
                "payload_found": False,
                "message":       "No se encontró payload compatible con el formato LSB del sistema.",
                "lsb_analysis":  lsb_analysis,
            }

        # Leer longitud del header JSON
        header_len_raw  = prefix_bytes[MAGIC_LEN: MAGIC_LEN + HEADER_LEN_BYTES]
        header_json_len = struct.unpack(">I", header_len_raw)[0]

        if header_json_len > 65536:
            return {
                "payload_found": False,
                "message":       "Cabecera inválida — tamaño fuera de rango.",
                "lsb_analysis":  self.analyze_lsb_structure(stego_image_path),
            }

        # Leer header JSON completo
        total_prefix    = MAGIC_LEN + HEADER_LEN_BYTES + header_json_len
        prefix_full     = _bits_to_bytes(_extract_n_bits(total_prefix * 8))
        header_json_raw = prefix_full[MAGIC_LEN + HEADER_LEN_BYTES: total_prefix]

        try:
            header = json.loads(header_json_raw.decode("utf-8"))
        except Exception:
            return {
                "payload_found": False,
                "message":       "Cabecera JSON corrupta o ilegible.",
                "lsb_analysis":  self.analyze_lsb_structure(stego_image_path),
            }

        payload_size = header.get("payload_size", 0)
        if payload_size < 0 or payload_size > MAX_PAYLOAD_BYTES:
            return {
                "payload_found": False,
                "message":       f"Tamaño de payload declarado inválido: {payload_size}.",
                "lsb_analysis":  self.analyze_lsb_structure(stego_image_path),
            }

        # Leer payload completo
        total_bytes_needed = total_prefix + payload_size
        all_bits  = _extract_n_bits(total_bytes_needed * 8)
        all_bytes = _bits_to_bytes(all_bits)
        payload_bytes = all_bytes[total_prefix: total_prefix + payload_size]

        # Validar SHA-256
        actual_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        sha256_valid  = (actual_sha256 == header.get("sha256", ""))

        # Análisis LSB del archivo stego
        lsb_analysis = self.analyze_lsb_structure(stego_image_path)

        # Primeras 100 posiciones de bits usadas
        total_bits_used   = total_bytes_needed * 8
        positions_used    = []
        bit_idx           = 0
        for pixel_idx, pixel in enumerate(pixels):
            if len(positions_used) >= 100:
                break
            x = pixel_idx % w
            y = pixel_idx // w
            for ch_i, ch_idx in enumerate(channel_indices):
                for b in range(bits_per_channel):
                    if bit_idx >= total_bits_used or len(positions_used) >= 100:
                        break
                    positions_used.append({
                        "x": x, "y": y,
                        "channel": channels[ch_i],
                        "bit_index": bit_idx,
                    })
                    bit_idx += 1

        total_pixels_used = (total_bits_used + len(channels) * bits_per_channel - 1) // \
                            (len(channels) * bits_per_channel)

        result: Dict[str, Any] = {
            "payload_found":      True,
            "payload_type":       header.get("payload_type", "binary"),
            "filename":           header.get("filename", ""),
            "mime_type":          header.get("mime_type", ""),
            "payload_size":       payload_size,
            "sha256_header":      header.get("sha256", ""),
            "sha256_actual":      actual_sha256,
            "sha256_valid":       sha256_valid,
            "algorithm":          header.get("algorithm", ""),
            "bits_per_channel":   header.get("bits_per_channel", 1),
            "channels":           header.get("channels", []),
            "created_by":         header.get("created_by", ""),
            "lsb_analysis":       lsb_analysis,
            "positions_summary":  {
                "first_pixel":        {"x": 0, "y": 0},
                "last_pixel":         {
                    "x": positions_used[-1]["x"] if positions_used else 0,
                    "y": positions_used[-1]["y"] if positions_used else 0,
                },
                "total_pixels_used":  total_pixels_used,
                "total_bits_used":    total_bits_used,
                "channels_used":      list(channels),
            },
            "first_positions":    positions_used,
        }

        # Devolver payload según tipo
        payload_type = header.get("payload_type", "binary")
        if payload_type == "text":
            try:
                result["message_text"] = payload_bytes.decode("utf-8")
            except Exception:
                result["message_text"] = payload_bytes.decode("latin-1", errors="replace")
        else:
            # Guardar como archivo descargable
            artifact_id  = str(uuid.uuid4())
            ext          = _ext_from_mime(header.get("mime_type", ""), header.get("filename", ""))
            out_filename = f"extracted_{artifact_id}{ext}"
            out_path     = STEGO_ARTIFACTS_DIR / out_filename
            out_path.write_bytes(payload_bytes)
            result["extracted_filename"] = out_filename
            result["artifact_id"]        = artifact_id

        return result

    # ── analyze_lsb_structure ─────────────────────────────────────────────────

    def analyze_lsb_structure(self, image_path: Path) -> Dict[str, Any]:
        """
        Análisis técnico de la distribución de bits LSB por canal RGB.

        Calcula:
        - Histograma 0/1 por canal
        - Entropía binaria estimada
        - Si tiene la cabecera del sistema
        - Capacidad estimada
        - Nota de aleatoriedad
        """
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            w, h    = img_rgb.size
            pixels  = list(img_rgb.getdata())

        channel_names = ["R", "G", "B"]
        stats: Dict[str, Any] = {}

        for ch_name, ch_idx in zip(channel_names, [0, 1, 2]):
            bits     = [(p[ch_idx] & 1) for p in pixels]
            n1       = sum(bits)
            n0       = len(bits) - n1
            ratio_1  = n1 / len(bits) if bits else 0.0
            stats[ch_name] = {
                "zeros":       n0,
                "ones":        n1,
                "total":       len(bits),
                "ratio_ones":  round(ratio_1, 4),
                "ratio_zeros": round(1 - ratio_1, 4),
                "entropy":     round(_binary_entropy(ratio_1), 4),
            }

        # Detectar MAGIC en los primeros bits (R bit LSB, columna por columna)
        min_bits    = MAGIC_LEN * 8
        prefix_bits = []
        for pixel in pixels:
            if len(prefix_bits) >= min_bits:
                break
            prefix_bits.append(pixel[0] & 1)  # R
            prefix_bits.append(pixel[1] & 1)  # G
            prefix_bits.append(pixel[2] & 1)  # B

        prefix_bytes      = _bits_to_bytes(prefix_bits[:min_bits])
        has_system_header = (prefix_bytes[:MAGIC_LEN] == MAGIC)

        # Nota de aleatoriedad
        ratios = [stats[c]["ratio_ones"] for c in channel_names]
        near_50 = all(0.45 <= r <= 0.55 for r in ratios)

        return {
            "width":             w,
            "height":            h,
            "total_pixels":      w * h,
            "channel_stats":     stats,
            "has_system_header": has_system_header,
            "capacity_estimate": {
                "total_pixels":    w * h,
                "bytes_available": (w * h * 3) // 8,
                "kb_available":    round((w * h * 3) / (8 * 1024), 2),
            },
            "randomness_note": (
                "Distribución LSB cercana a 50/50 — posible contenido esteganografiado o ruido natural."
                if near_50 else
                "Distribución LSB sesgada — imagen probablemente natural (no esteganografiada)."
            ),
        }


# ── Helpers internos ──────────────────────────────────────────────────────────

def _bytes_to_bits(data: bytes) -> List[int]:
    """Convierte bytes a lista de bits (MSB first dentro de cada byte)."""
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _bits_to_bytes(bits: List[int]) -> bytes:
    """Convierte lista de bits a bytes. Rellena con 0 al final si es necesario."""
    result = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i: i + 8]
        while len(chunk) < 8:
            chunk.append(0)
        byte = 0
        for b in chunk:
            byte = (byte << 1) | b
        result.append(byte)
    return bytes(result)


def _binary_entropy(p: float) -> float:
    """Entropía binaria H(p) = -p·log₂(p) − (1−p)·log₂(1−p)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    q = 1.0 - p
    return -(p * math.log2(p) + q * math.log2(q))


def _ext_from_mime(mime: str, filename: str) -> str:
    """Determina la extensión de archivo desde mime type o nombre original."""
    if filename:
        suffix = Path(filename).suffix
        if suffix:
            return suffix
    return {
        "text/plain":           ".txt",
        "application/pdf":      ".pdf",
        "image/png":            ".png",
        "image/jpeg":           ".jpg",
        "image/jpg":            ".jpg",
    }.get(mime, ".bin")


def _write_positions_csv(csv_path: Path, positions: List[Dict]) -> None:
    """Escribe todas las posiciones usadas en un CSV."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["x", "y", "channel", "bit_index", "payload_bit", "byte_index"]
        )
        writer.writeheader()
        writer.writerows(positions)


def _generate_lsb_map(
    orig_pixels: List,
    new_pixels: List,
    w: int,
    h: int,
    map_path: Path,
) -> None:
    """
    Genera imagen PNG donde los píxeles modificados se marcan en rojo
    y los no modificados se convierten a escala de grises.
    """
    map_data = []
    for op, np_ in zip(orig_pixels, new_pixels):
        if op != np_:
            map_data.append((220, 38, 38))   # Rojo — píxel con bits LSB modificados
        else:
            gray = int(0.299 * op[0] + 0.587 * op[1] + 0.114 * op[2])
            map_data.append((gray, gray, gray))

    map_img = Image.new("RGB", (w, h))
    map_img.putdata(map_data)
    map_img.save(str(map_path), format="PNG")
