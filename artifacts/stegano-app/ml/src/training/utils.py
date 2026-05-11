"""
utils.py — Utilidades de entrenamiento y gestión de checkpoints.

Incluye:
  - EarlyStopping basado en val_auc
  - Guardado y carga de checkpoints
  - Sanity check de batches
  - Cálculo de pos_weight para datasets desbalanceados
  - Logging de métricas por época
"""

import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Para el entrenamiento cuando la métrica de validación deja de mejorar.

    Se basa en val_auc porque AUC es más estable que la pérdida y menos
    sensible al threshold. Guardar el mejor modelo automáticamente evita
    recuperar pesos de una época posterior con overfitting.

    Args:
        patience:    Épocas a esperar sin mejora antes de parar
        min_delta:   Mejora mínima para contar como progreso
        mode:        "max" para métricas como AUC (mayor es mejor)
        verbose:     Imprimir estado en cada verificación
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
        verbose: bool = True,
    ):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.verbose   = verbose
        self.counter   = 0
        self.best      = None
        self.stop      = False

    def __call__(self, metric: float) -> bool:
        """
        Returns True si se debe parar el entrenamiento.
        """
        if self.best is None:
            self.best = metric
            return False

        improved = (metric - self.best > self.min_delta) if self.mode == "max" \
                   else (self.best - metric > self.min_delta)

        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] Sin mejora por {self.counter}/{self.patience} épocas "
                      f"(mejor: {self.best:.4f})")
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"  [EarlyStopping] Deteniendo entrenamiento.")
                self.stop = True

        return self.stop


# ── Checkpoint ────────────────────────────────────────────────────────────────

@dataclass
class CheckpointInfo:
    """Metadatos del mejor checkpoint guardado."""
    epoch: int = 0
    val_auc: float = 0.0
    val_loss: float = float("inf")
    train_auc: float = 0.0
    threshold: float = 0.5
    model_version: str = "srnet-lite-v1.0"
    timestamp: str = ""
    architecture: Dict = field(default_factory=dict)


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    metrics: Dict,
    checkpoint_path: Path,
    state_dict_path: Path,
    metadata_path: Path,
    threshold: float = 0.5,
) -> None:
    """
    Guarda el checkpoint completo (modelo + optimizador + metadatos).

    Guarda dos archivos:
      - checkpoint completo (.pt): para reanudar entrenamiento
      - state_dict (.pt):          para inferencia en Replit (más liviano)

    Args:
        model:            Modelo SRNetLite
        optimizer:        Optimizador (para resume training)
        epoch:            Época actual
        metrics:          Diccionario con val_auc, val_loss, etc.
        checkpoint_path:  Ruta para checkpoint completo
        state_dict_path:  Ruta para solo los pesos del modelo
        metadata_path:    Ruta para JSON con metadatos
        threshold:        Threshold óptimo calculado sobre validación
    """
    import torch, datetime

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint completo (para resume)
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "metrics":     metrics,
        "threshold":   threshold,
    }, str(checkpoint_path))

    # Solo state_dict (para Replit inference)
    torch.save(model.state_dict(), str(state_dict_path))

    # Metadatos en JSON
    metadata = {
        "epoch":          epoch,
        "val_auc":        round(float(metrics.get("val_auc", 0)), 4),
        "val_loss":       round(float(metrics.get("val_loss", 0)), 4),
        "val_accuracy":   round(float(metrics.get("val_accuracy", 0)), 4),
        "val_f1":         round(float(metrics.get("val_f1", 0)), 4),
        "threshold":      round(float(threshold), 4),
        "model_version":  "srnet-lite-v1.0",
        "input_size":     128,
        "input_channels": 3,
        "saved_at":       datetime.datetime.utcnow().isoformat(),
        "architecture": {
            "srm_filters":    9,
            "use_attention":  True,
            "dropout_rate":   0.5,
        }
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  [Checkpoint] Guardado en época {epoch+1}")
    print(f"    val_auc  = {metadata['val_auc']:.4f}")
    print(f"    val_loss = {metadata['val_loss']:.4f}")
    print(f"    threshold= {metadata['threshold']:.4f}")
    print(f"    Ruta:      {checkpoint_path}")


def load_checkpoint(
    model,
    checkpoint_path: Path,
    optimizer=None,
    device: str = "cpu",
) -> Tuple[int, Dict]:
    """
    Carga un checkpoint para reanudar el entrenamiento o para inferencia.

    Returns:
        (epoch_inicial, metrics_del_checkpoint)
    """
    import torch

    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state["model_state"])

    if optimizer and "optim_state" in state:
        optimizer.load_state_dict(state["optim_state"])

    epoch = state.get("epoch", 0)
    metrics = state.get("metrics", {})
    print(f"  [Checkpoint] Reanudando desde época {epoch+1}, "
          f"val_auc={metrics.get('val_auc', 0):.4f}")
    return epoch + 1, metrics


# ── Sanity check de batches ───────────────────────────────────────────────────

def sanity_check_batch(images, labels, stage: str = "train") -> None:
    """
    Verifica que un batch tiene la forma y valores esperados.
    Ejecutar antes de la primera época para detectar problemas tempranos.

    Detecta:
      - Dimensiones incorrectas
      - Imágenes con NaN o Inf
      - Labels fuera de rango [0, 1]
      - Distribución de labels (detecta imbalance severo)
    """
    print(f"\n[SanityCheck] Batch de {stage}:")
    print(f"  images: shape={images.shape}, dtype={images.dtype}")
    print(f"  labels: shape={labels.shape}, dtype={labels.dtype}")
    print(f"  images: min={images.min():.3f}, max={images.max():.3f}, "
          f"mean={images.mean():.3f}")
    print(f"  labels: {labels.float().mean():.1%} stego, "
          f"{(1-labels.float()).mean():.1%} cover")

    import torch
    assert not torch.isnan(images).any(), "NaN detectado en imágenes!"
    assert not torch.isinf(images).any(), "Inf detectado en imágenes!"
    assert images.shape[1:] == torch.Size([3, 128, 128]), \
        f"Shape inesperado: {images.shape}, esperado [B,3,128,128]"
    assert labels.min() >= 0 and labels.max() <= 1, \
        f"Labels fuera de rango: [{labels.min()}, {labels.max()}]"

    label_balance = labels.float().mean().item()
    if label_balance < 0.2 or label_balance > 0.8:
        print(f"  [WARNING] Desbalance severo en batch: {label_balance:.1%} stego")

    print("  [OK] Batch verificado correctamente.")


# ── Utilidades varias ─────────────────────────────────────────────────────────

def compute_pos_weight(n_cover: int, n_stego: int) -> float:
    """
    Calcula pos_weight para BCEWithLogitsLoss en datasets desbalanceados.

    Fórmula: pos_weight = n_cover / n_stego
    Si hay más cover que stego, pos_weight > 1 → penaliza más FN de clase stego.
    """
    if n_stego == 0:
        return 1.0
    pw = n_cover / n_stego
    print(f"  [pos_weight] cover={n_cover}, stego={n_stego}, pos_weight={pw:.3f}")
    return pw


def set_seed(seed: int = 42) -> None:
    """Fija la semilla aleatoria para reproducibilidad del entrenamiento."""
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"  [Seed] Semilla fijada: {seed}")


def get_device() -> str:
    """Detecta GPU disponible, con fallback a CPU."""
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        print(f"  [Device] GPU detectada: {gpu}")
        return "cuda"
    print("  [Device] CPU — el entrenamiento será lento. Usa Colab con GPU T4.")
    return "cpu"


class AverageMeter:
    """Calcula y almacena la media acumulada de una métrica durante una época."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def format_time(seconds: float) -> str:
    """Formatea segundos en formato legible HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
