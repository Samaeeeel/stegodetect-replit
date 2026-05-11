"""
plots.py — Visualizaciones para análisis del modelo de estegoanálisis.

Genera y guarda:
  - Curva ROC con AUC
  - Matriz de confusión
  - Historial de entrenamiento (loss, AUC por época)
  - Distribución de probabilidades predichas
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")   # Backend sin pantalla para Colab/servidor
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[plots] matplotlib no disponible — pip install matplotlib")


def _check_matplotlib() -> None:
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib requerido: pip install matplotlib")


# ── Curva ROC ─────────────────────────────────────────────────────────────────

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    threshold: float,
    output_path: Path,
) -> None:
    """
    Genera y guarda la curva ROC con AUC y threshold óptimo marcado.

    Args:
        fpr:          Array de False Positive Rate
        tpr:          Array de True Positive Rate
        auc:          Área bajo la curva
        threshold:    Threshold óptimo (se marca en la curva)
        output_path:  Ruta donde guardar el PNG
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr, color="#1a56db", lw=2.5,
            label=f"SRNet-lite (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Clasificador aleatorio")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#1a56db")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR / Recall)", fontsize=12)
    ax.set_title("Curva ROC — Detección de Esteganografía LSB", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plots] Curva ROC guardada en: {output_path}")


# ── Matriz de confusión ───────────────────────────────────────────────────────

def plot_confusion_matrix(
    tn: int, fp: int, fn: int, tp: int,
    output_path: Path,
    title: str = "Matriz de Confusión",
) -> None:
    """
    Genera y guarda la matriz de confusión normalizada y sin normalizar.

    Etiquetas: 0 = Cover (sin mensaje), 1 = Stego (con mensaje)
    """
    _check_matplotlib()

    cm = np.array([[tn, fp], [fn, tp]])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, fmt, t in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Valores absolutos", "Normalizada por fila"]
    ):
        im = ax.imshow(data, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax, fraction=0.046)

        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Cover (pred.)", "Stego (pred.)"],
            yticklabels=["Cover (real)", "Stego (real)"],
            title=f"{title} — {t}",
            ylabel="Etiqueta real",
            xlabel="Etiqueta predicha",
        )
        ax.tick_params(labelsize=10)

        thresh = data.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = f"{data[i,j]:{fmt}}"
                ax.text(j, i, val, ha="center", va="center", fontsize=13,
                        color="white" if data[i, j] > thresh else "black",
                        fontweight="bold")

    plt.suptitle("Análisis de Errores — SRNet-lite", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plots] Matriz de confusión guardada en: {output_path}")


# ── Historial de entrenamiento ────────────────────────────────────────────────

def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Path,
) -> None:
    """
    Genera y guarda las curvas de pérdida y AUC durante el entrenamiento.

    Args:
        history: Diccionario con claves:
                 "train_loss", "val_loss", "train_auc", "val_auc"
        output_path: Ruta donde guardar el PNG
    """
    _check_matplotlib()

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Loss ──
    ax = axes[0]
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], "r-o", markersize=4, label="Val Loss")
    ax.set_title("Pérdida por época", fontsize=12, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (BCE)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── AUC ──
    ax = axes[1]
    if "train_auc" in history:
        ax.plot(epochs, history["train_auc"], "b-o", markersize=4, label="Train AUC")
    if "val_auc" in history:
        ax.plot(epochs, history["val_auc"], "r-o", markersize=4, label="Val AUC")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Azar (0.5)")
    ax.set_title("AUC-ROC por época", fontsize=12, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim([0.4, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Historial de Entrenamiento — SRNet-lite", fontsize=13, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plots] Historial guardado en: {output_path}")


# ── Distribución de probabilidades ───────────────────────────────────────────

def plot_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """
    Histograma de probabilidades predichas separado por clase real.
    Permite visualizar si el modelo separa bien las dos clases.
    Un buen modelo muestra dos picos: cover cerca de 0, stego cerca de 1.
    """
    _check_matplotlib()

    cover_probs = y_prob[y_true == 0]
    stego_probs = y_prob[y_true == 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 40)

    ax.hist(cover_probs, bins=bins, alpha=0.65, color="#16a34a",
            label=f"Cover (n={len(cover_probs)})", density=True)
    ax.hist(stego_probs, bins=bins, alpha=0.65, color="#dc2626",
            label=f"Stego (n={len(stego_probs)})", density=True)
    ax.axvline(x=threshold, color="black", linestyle="--", lw=2,
               label=f"Threshold óptimo ({threshold:.3f})")

    ax.set_xlabel("Probabilidad predicha (P_stego)", fontsize=12)
    ax.set_ylabel("Densidad", fontsize=12)
    ax.set_title("Distribución de predicciones — SRNet-lite", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plots] Distribución de probabilidades guardada en: {output_path}")
