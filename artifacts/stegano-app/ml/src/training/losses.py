"""
losses.py — Funciones de pérdida para entrenamiento de estegoanálisis.

¿Por qué BCEWithLogitsLoss?
===========================
El modelo SRNet-lite devuelve logits (salida sin activación), lo que permite
usar BCEWithLogitsLoss que combina sigmoid + BCE en una sola operación numéricamente
estable. Esto evita problemas de gradientes con valores extremos.

¿Por qué no CrossEntropyLoss?
==============================
CrossEntropyLoss es para clasificación multiclase con argmax final.
Para clasificación binaria, BCEWithLogitsLoss es la opción correcta y
equivale a CE con dos clases pero es más eficiente.

Weighted loss:
==============
Si el dataset está desbalanceado (más cover que stego o viceversa),
se puede usar pos_weight para penalizar más los falsos negativos en la
clase minoritaria.
"""

import torch
import torch.nn as nn


class WeightedBCELoss(nn.Module):
    """
    BCE con logits y soporte para pesos de clase.

    pos_weight: ratio negativos/positivos para balancear clases desiguales.
    Si hay 6000 cover y 4000 stego: pos_weight = 6000/4000 = 1.5
    """

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: [B] — salida del modelo (sin sigmoid)
        labels: [B] — etiquetas binarias float (0.0 o 1.0)
        """
        device = logits.device
        self.criterion.pos_weight = self.criterion.pos_weight.to(device)
        return self.criterion(logits, labels.float())


class LabelSmoothingBCELoss(nn.Module):
    """
    BCE con suavizado de etiquetas para reducir sobreconfianza del modelo.

    En lugar de usar etiquetas duras (0, 1), usa (epsilon, 1-epsilon).
    Esto ayuda a que el modelo no colapse en predicciones extremas.

    Recomendado: epsilon=0.1 para estegoanálisis.
    """

    def __init__(self, epsilon: float = 0.1, pos_weight: float = 1.0):
        super().__init__()
        self.epsilon = epsilon
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Suavizar: 0 → epsilon, 1 → 1-epsilon
        smooth_labels = labels.float() * (1 - self.epsilon) + self.epsilon * 0.5
        return self.bce(logits, smooth_labels)


def get_loss_fn(
    loss_type: str = "bce",
    pos_weight: float = 1.0,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """
    Factoría de funciones de pérdida.

    Args:
        loss_type:        "bce" | "weighted_bce" | "smooth_bce"
        pos_weight:       peso para la clase positiva (stego)
        label_smoothing:  epsilon para suavizado de etiquetas (0 = sin suavizado)
    """
    if loss_type == "weighted_bce":
        return WeightedBCELoss(pos_weight=pos_weight)
    elif loss_type == "smooth_bce":
        return LabelSmoothingBCELoss(epsilon=label_smoothing, pos_weight=pos_weight)
    else:
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]) if pos_weight != 1.0 else None
        )
