"""
metrics.py — Métricas de evaluación para el modelo de estegoanálisis.

¿Por qué AUC-ROC como métrica principal?
==========================================
En datasets balanceados, accuracy es suficiente.
En estegoanálisis, el dataset puede ser ligeramente desbalanceado y el costo
de un falso negativo (no detectar un mensaje oculto) puede diferir del costo
de un falso positivo. AUC-ROC mide la capacidad discriminativa del modelo
independientemente del threshold, lo que es más informativo que accuracy.

¿Por qué calcular threshold óptimo?
=====================================
El threshold por defecto de 0.5 asume que los costos de FP y FN son iguales.
Para estegoanálisis forense, puede ser preferible maximizar recall (detectar
más mensajes ocultos) a costa de más FP. El threshold óptimo se calcula sobre
el conjunto de validación usando el índice de Youden: J = TPR - FPR.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from sklearn.metrics import (
        roc_auc_score, roc_curve, confusion_matrix,
        precision_score, recall_score, f1_score,
        accuracy_score, classification_report,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[metrics] sklearn no disponible — instala con: pip install scikit-learn")


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """
    Calcula el conjunto completo de métricas de evaluación.

    Args:
        y_true:    Etiquetas verdaderas (0 = cover, 1 = stego), shape [N]
        y_prob:    Probabilidades predichas de clase stego (sigmoid), shape [N]
        threshold: Umbral de decisión (0.0 – 1.0)

    Returns:
        Diccionario con todas las métricas relevantes
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn requerido para métricas completas.")

    y_pred = (y_prob >= threshold).astype(int)

    # Métricas básicas
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    # AUC-ROC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    # Matriz de confusión: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Especificidad (TNR)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy":    round(float(acc), 4),
        "precision":   round(float(prec), 4),
        "recall":      round(float(rec), 4),
        "f1":          round(float(f1), 4),
        "auc_roc":     round(float(auc), 4),
        "specificity": round(float(specificity), 4),
        "threshold":   round(float(threshold), 4),
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
        },
        "n_samples":   int(len(y_true)),
        "n_cover":     int((y_true == 0).sum()),
        "n_stego":     int((y_true == 1).sum()),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "youden",
) -> Tuple[float, float]:
    """
    Encuentra el threshold óptimo sobre un conjunto de validación.

    Métodos disponibles:
      - "youden": maximiza J = TPR - FPR (equilibrio entre sensibilidad y especificidad)
      - "f1":     maximiza F1-score (útil cuando las clases tienen costos similares)

    Args:
        y_true:  Etiquetas verdaderas
        y_prob:  Probabilidades predichas
        method:  Criterio de optimización

    Returns:
        (threshold_optimo, valor_de_metrica)
    """
    if not SKLEARN_AVAILABLE:
        return 0.5, 0.0

    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx]), float(j_scores[best_idx])

    elif method == "f1":
        best_f1, best_thresh = 0.0, 0.5
        for t in np.arange(0.3, 0.8, 0.01):
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t
        return float(best_thresh), float(best_f1)

    return 0.5, 0.0


def get_roc_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (fpr, tpr, thresholds) para graficar la curva ROC."""
    if not SKLEARN_AVAILABLE:
        return np.array([0,1]), np.array([0,1]), np.array([1,0])
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds


def check_model_collapse(
    y_prob: np.ndarray,
    threshold: float = 0.5,
    collapse_ratio: float = 0.95,
) -> bool:
    """
    Detecta si el modelo colapsó a predecir siempre la misma clase.

    Esto es un problema común cuando:
      - El learning rate es demasiado alto
      - El dataset está muy desbalanceado
      - La inicialización de pesos es incorrecta

    Returns:
        True si el modelo parece haber colapsado (problema), False si no.
    """
    preds = (y_prob >= threshold).astype(int)
    ratio = preds.mean()
    collapsed = ratio < (1 - collapse_ratio) or ratio > collapse_ratio
    if collapsed:
        print(f"[WARNING] Posible colapso detectado: {ratio:.1%} de predicciones "
              f"son {'stego' if ratio > 0.5 else 'cover'}. "
              f"Verifica el balance del dataset y la tasa de aprendizaje.")
    return collapsed


def save_metrics(metrics: Dict, output_path: Path) -> None:
    """Guarda las métricas en formato JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[metrics] Métricas guardadas en: {output_path}")


def print_metrics_summary(metrics: Dict, title: str = "Evaluación") -> None:
    """Imprime un resumen legible de las métricas."""
    cm = metrics.get("confusion_matrix", {})
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  Accuracy:    {metrics.get('accuracy', 0):.4f}")
    print(f"  AUC-ROC:     {metrics.get('auc_roc', 0):.4f}  ← métrica principal")
    print(f"  Precision:   {metrics.get('precision', 0):.4f}")
    print(f"  Recall:      {metrics.get('recall', 0):.4f}")
    print(f"  F1-score:    {metrics.get('f1', 0):.4f}")
    print(f"  Specificity: {metrics.get('specificity', 0):.4f}")
    print(f"  Threshold:   {metrics.get('threshold', 0.5):.4f}")
    print(f"  Samples:     {metrics.get('n_samples', 0)} "
          f"(cover={cm.get('TN',0)+cm.get('FP',0)}, "
          f"stego={cm.get('FN',0)+cm.get('TP',0)})")
    print(f"  Conf. Matrix:")
    print(f"    TN={cm.get('TN',0):5d}  FP={cm.get('FP',0):5d}")
    print(f"    FN={cm.get('FN',0):5d}  TP={cm.get('TP',0):5d}")

    if metrics.get('auc_roc', 0) < 0.55:
        print(f"\n  [WARNING] AUC cercano a 0.5 — el modelo puede haber colapsado.")
    if metrics.get('recall', 0) == 0.0:
        print(f"  [WARNING] Recall=0 — el modelo nunca predice clase stego.")
    if metrics.get('precision', 0) == 0.0:
        print(f"  [WARNING] Precision=0 — verificar la pérdida y el balance.")
    print(f"{'='*50}\n")
