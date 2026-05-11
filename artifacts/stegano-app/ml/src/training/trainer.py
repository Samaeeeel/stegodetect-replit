"""
trainer.py — Loop de entrenamiento para SRNet-lite.

Implementa:
  - Entrenamiento con mixed precision (AMP) si hay GPU
  - Evaluación completa por época (loss, AUC, accuracy, F1)
  - Early stopping basado en val_auc
  - Guardado automático del mejor checkpoint
  - Detección de colapso de modelo
  - Logging completo del historial
"""

import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.src.training.utils import (
    EarlyStopping, AverageMeter, save_checkpoint, format_time,
    sanity_check_batch
)
from ml.src.training.losses import get_loss_fn
from ml.src.evaluation.metrics import (
    compute_all_metrics, find_optimal_threshold,
    check_model_collapse, print_metrics_summary
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    Orquesta el ciclo completo de entrenamiento de SRNet-lite.

    Args:
        model:           Instancia de SRNetLite
        train_loader:    DataLoader de entrenamiento
        val_loader:      DataLoader de validación
        device:          "cuda" o "cpu"
        config:          Diccionario con hiperparámetros (ver DEFAULT_CONFIG)
        output_dir:      Directorio base para guardar checkpoints y reportes
    """

    DEFAULT_CONFIG = {
        "lr":              1e-4,
        "weight_decay":    1e-4,
        "epochs":          50,
        "patience":        10,          # Para early stopping
        "loss_type":       "bce",       # "bce" | "weighted_bce" | "smooth_bce"
        "label_smoothing": 0.05,
        "pos_weight":      1.0,
        "grad_clip":       1.0,         # Clipping de gradientes
        "scheduler":       "cosine",    # "cosine" | "plateau" | "none"
        "mixed_precision": True,        # Usar AMP si hay GPU
        "threshold_method":"youden",    # Método para calcular threshold óptimo
        "seed":            42,
    }

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        config: Optional[Dict] = None,
        output_dir: Optional[Path] = None,
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.cfg          = {**self.DEFAULT_CONFIG, **(config or {})}
        self.output_dir   = Path(output_dir) if output_dir else Path("/tmp/stego_training")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizador
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )

        # Función de pérdida
        self.criterion = get_loss_fn(
            loss_type=self.cfg["loss_type"],
            pos_weight=self.cfg["pos_weight"],
            label_smoothing=self.cfg["label_smoothing"],
        ).to(device)

        # Scheduler de learning rate
        self.scheduler = self._build_scheduler()

        # Early stopping sobre val_auc
        self.early_stopping = EarlyStopping(
            patience=self.cfg["patience"],
            mode="max",
            verbose=True,
        )

        # Mixed precision (AMP) — solo en GPU
        self.use_amp = self.cfg["mixed_precision"] and device == "cuda"
        self.scaler  = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Historial de entrenamiento
        self.history: Dict[str, List] = {
            "train_loss": [], "val_loss": [],
            "train_auc":  [], "val_auc":  [],
            "train_acc":  [], "val_acc":  [],
            "lr":         [],
        }

        self.best_val_auc   = 0.0
        self.best_threshold = 0.5
        self.start_epoch    = 0

    # ── API pública ───────────────────────────────────────────────────────────

    def fit(
        self,
        checkpoint_path:   Path,
        state_dict_path:   Path,
        metadata_path:     Path,
        history_path:      Path,
        resume_from:       Optional[Path] = None,
    ) -> Dict:
        """
        Entrena el modelo desde start_epoch hasta cfg["epochs"] o early stopping.

        Args:
            checkpoint_path: Destino para el checkpoint completo
            state_dict_path: Destino para solo los pesos (para Replit)
            metadata_path:   Destino para metadatos JSON
            history_path:    Destino para historial de métricas JSON
            resume_from:     Si se especifica, carga este checkpoint y continúa

        Returns:
            Diccionario con métricas del mejor checkpoint
        """
        if resume_from and resume_from.exists():
            from ml.src.training.utils import load_checkpoint
            self.start_epoch, _ = load_checkpoint(
                self.model, resume_from, self.optimizer, self.device
            )

        print(f"\n{'='*60}")
        print(f"  Iniciando entrenamiento SRNet-lite")
        print(f"  Device:    {self.device}")
        print(f"  AMP:       {'Sí' if self.use_amp else 'No'}")
        print(f"  Épocas:    {self.cfg['epochs']}")
        print(f"  LR:        {self.cfg['lr']}")
        print(f"  Loss:      {self.cfg['loss_type']}")
        print(f"  Patience:  {self.cfg['patience']}")
        print(f"{'='*60}\n")

        # Sanity check del primer batch
        images, labels = next(iter(self.train_loader))
        sanity_check_batch(images, labels, stage="train")

        total_start = time.time()

        for epoch in range(self.start_epoch, self.cfg["epochs"]):
            epoch_start = time.time()

            # ── Entrenamiento ──
            train_metrics = self._train_epoch(epoch)

            # ── Validación ──
            val_metrics = self._val_epoch()

            # ── Threshold óptimo en val ──
            threshold, _ = find_optimal_threshold(
                val_metrics["y_true"],
                val_metrics["y_prob"],
                method=self.cfg["threshold_method"],
            )
            self.best_threshold = threshold

            # ── Métricas finales de la época ──
            val_eval = compute_all_metrics(
                val_metrics["y_true"],
                val_metrics["y_prob"],
                threshold=threshold,
            )

            train_eval = compute_all_metrics(
                train_metrics["y_true"],
                train_metrics["y_prob"],
                threshold=threshold,
            )

            # ── Actualizar scheduler ──
            self._step_scheduler(val_eval["auc_roc"])

            # ── Guardar historial ──
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_auc"].append(train_eval["auc_roc"])
            self.history["val_auc"].append(val_eval["auc_roc"])
            self.history["train_acc"].append(train_eval["accuracy"])
            self.history["val_acc"].append(val_eval["accuracy"])
            self.history["lr"].append(current_lr)

            # ── Log por época ──
            elapsed = format_time(time.time() - epoch_start)
            print(f"Época [{epoch+1:3d}/{self.cfg['epochs']}] "
                  f"| train_loss={train_metrics['loss']:.4f} "
                  f"| val_loss={val_metrics['loss']:.4f} "
                  f"| val_auc={val_eval['auc_roc']:.4f} "
                  f"| val_acc={val_eval['accuracy']:.4f} "
                  f"| lr={current_lr:.2e} "
                  f"| {elapsed}")

            # ── Advertencia de colapso ──
            check_model_collapse(val_metrics["y_prob"], threshold)

            # ── Guardar mejor modelo ──
            if val_eval["auc_roc"] > self.best_val_auc:
                self.best_val_auc = val_eval["auc_roc"]
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics={
                        "val_auc":      val_eval["auc_roc"],
                        "val_loss":     val_metrics["loss"],
                        "val_accuracy": val_eval["accuracy"],
                        "val_f1":       val_eval["f1"],
                    },
                    checkpoint_path=checkpoint_path,
                    state_dict_path=state_dict_path,
                    metadata_path=metadata_path,
                    threshold=threshold,
                )

            # ── Guardar historial actualizado ──
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(history_path, "w") as f:
                json.dump(self.history, f, indent=2)

            # ── Early stopping ──
            if self.early_stopping(val_eval["auc_roc"]):
                print(f"\n  Early stopping en época {epoch+1}. "
                      f"Mejor val_auc = {self.best_val_auc:.4f}")
                break

        total_time = format_time(time.time() - total_start)
        print(f"\n  Entrenamiento finalizado en {total_time}")
        print(f"  Mejor val_auc = {self.best_val_auc:.4f}")
        print(f"  Threshold óptimo = {self.best_threshold:.4f}")

        return {
            "best_val_auc":   self.best_val_auc,
            "best_threshold": self.best_threshold,
            "history":        self.history,
        }

    # ── Métodos internos ──────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict:
        """Loop de entrenamiento para una época."""
        self.model.train()
        loss_meter = AverageMeter()
        all_probs  = []
        all_labels = []

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float()

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(images).squeeze(1)
                    loss   = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["grad_clip"]
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images).squeeze(1)
                loss   = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["grad_clip"]
                )
                self.optimizer.step()

            loss_meter.update(loss.item(), images.size(0))

            with torch.no_grad():
                probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        return {
            "loss":    loss_meter.avg,
            "y_true":  np.array(all_labels),
            "y_prob":  np.array(all_probs),
        }

    @torch.no_grad()
    def _val_epoch(self) -> Dict:
        """Loop de validación para una época."""
        self.model.eval()
        loss_meter = AverageMeter()
        all_probs  = []
        all_labels = []

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(images).squeeze(1)
                    loss   = self.criterion(logits, labels)
            else:
                logits = self.model(images).squeeze(1)
                loss   = self.criterion(logits, labels)

            loss_meter.update(loss.item(), images.size(0))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        return {
            "loss":   loss_meter.avg,
            "y_true": np.array(all_labels),
            "y_prob": np.array(all_probs),
        }

    def _build_scheduler(self):
        sched = self.cfg.get("scheduler", "cosine")
        if sched == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg["epochs"],
                eta_min=1e-6,
            )
        elif sched == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=5, verbose=True
            )
        return None

    def _step_scheduler(self, val_auc: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler,
                       torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_auc)
        else:
            self.scheduler.step()
