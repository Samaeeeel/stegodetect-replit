"""
blocks.py — Bloques constitutivos de la arquitectura SRNet-lite.

¿Por qué High-Pass Filtering (HPF) en estegoanálisis?
======================================================
Las señales LSB se esconden en los bits menos significativos de los píxeles,
que se manifiestan como variaciones de amplitud muy pequeña (~1 en escala 0-255).
El contenido semántico de la imagen (formas, texturas, bordes) domina la
representación espacial y "enmascara" la señal esteganográfica.

Un filtro de paso alto (HPF) suprime el contenido de baja frecuencia (semántico)
y amplifica las variaciones de alta frecuencia (ruido, señal LSB).
Esto transforma el problema de "¿qué hay en la imagen?" a "¿hay patrones anómalos
en el ruido residual?", que es mucho más favorable para el detector.

¿Por qué NO usar modelos preentrenados en ImageNet?
====================================================
Los modelos como ResNet50 entrenados en ImageNet aprenden características semánticas
de alto nivel (formas, colores, texturas gruesas). Para estegoanálisis necesitamos
exactamente lo contrario: detectar perturbaciones de amplitud ±1 en la escala de grises.
Usar un modelo de ImageNet:
  1. No ha aprendido a detectar esas perturbaciones.
  2. Sus primeras capas ya destruyen la señal LSB con ReLU y stride.
  3. El fine-tuning raramente converge correctamente para esta tarea.

Referencias:
  - SRNet: Boroumand et al. (2018), "Deep Residual Network for Steganalysis..."
  - SRM:   Fridrich & Kodovsky (2012), "Rich Models for Steganalysis of Digital Images"
"""

import torch
import torch.nn as nn
import numpy as np


# ── Filtros SRM (Spatial Rich Model) ─────────────────────────────────────────
def _build_srm_kernels() -> torch.Tensor:
    """
    Construye 3 filtros SRM clásicos de 5x5 usados en estegoanálisis.

    Estos filtros están diseñados para extraer el ruido residual de la imagen,
    suprimiendo el contenido semántico. Son pesos FIJOS (no entrenables).

    Los kernels implementados son variantes del filtro de diferencia de orden 1, 2 y 3
    usados en el modelo SRM original (Fridrich & Kodovsky, 2012).
    """
    # Kernel 1: Filtro de diferencia de segundo orden (horizontal)
    # Predice el píxel central como promedio de vecinos y calcula el error
    k1 = np.array([
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [-1,  2, -2,  2, -1],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
    ], dtype=np.float32) / 4.0

    # Kernel 2: Filtro de diferencia de segundo orden (vertical)
    k2 = k1.T.copy()

    # Kernel 3: Filtro de diferencia cruzada (diagonal)
    k3 = np.array([
        [ 0,  0,  0,  0,  0],
        [ 0, -1,  2, -1,  0],
        [ 0,  2, -4,  2,  0],
        [ 0, -1,  2, -1,  0],
        [ 0,  0,  0,  0,  0],
    ], dtype=np.float32) / 4.0

    # Stack: shape (3, 1, 5, 5) — 3 filtros, 1 canal de entrada, 5x5
    kernels = np.stack([k1, k2, k3], axis=0)[:, np.newaxis, :, :]
    return torch.from_numpy(kernels)


class SRMLayer(nn.Module):
    """
    Capa de preprocesamiento SRM con filtros fijos no entrenables.

    Input:  [B, 3, H, W] — imagen RGB en [0,1] o normalizada
    Output: [B, 9, H, W] — 3 residuos × 3 canales RGB

    Al aplicar los filtros SRM a cada canal RGB de forma independiente,
    obtenemos 9 mapas de residuo que contienen principalmente la señal
    de alta frecuencia (ruido + posible señal LSB).
    """

    def __init__(self):
        super().__init__()
        # Conv 2D que aplica los 3 filtros SRM a los 3 canales RGB.
        # Resultado: 9 mapas de residuo (3 filtros × 3 canales).
        # El atributo se llama 'srm' — este nombre es parte del state_dict
        # y DEBE coincidir con las copias embebidas en notebook 02 y model_service.py.
        self.srm = nn.Conv2d(
            in_channels=3,
            out_channels=9,     # 3 filtros × 3 canales
            kernel_size=5,
            padding=2,
            bias=False,
            groups=1,
        )

        # Construir pesos (9, 3, 5, 5):
        # out_channel i*3+j = filtro SRM i aplicado al canal RGB j
        k = _build_srm_kernels()       # (3, 1, 5, 5)
        weight = torch.zeros(9, 3, 5, 5)
        for i in range(3):             # Para cada filtro SRM
            for j in range(3):         # Para cada canal RGB
                weight[i*3+j, j, :, :] = k[i, 0, :, :]

        with torch.no_grad():
            self.srm.weight.copy_(weight)

        # Congelar pesos — los filtros SRM NO se entrenan
        for param in self.srm.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.srm(x)


# ── Bloques residuales ────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Bloque residual básico: Conv → BN → ReLU → Conv → BN + skip connection.

    Los skip connections permiten que el gradiente fluya directamente al principio
    de la red, evitando el vanishing gradient en redes profundas.
    También preservan características de bajo nivel útiles para el detector.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class DownsampleBlock(nn.Module):
    """
    Bloque de reducción espacial con skip connection proyectada.

    Reduce la resolución espacial con stride=2 mientras aumenta los canales.
    La proyección en el skip connection garantiza que las dimensiones coincidan.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # Proyección para alinear dimensiones del skip
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.main(x) + self.skip(x))


class AttentionBlock(nn.Module):
    """
    Módulo de atención por canal (Squeeze-and-Excitation simplificado).

    Aprende a dar más peso a los canales que contienen señales relevantes
    para la detección de esteganografía, suprimiendo los que no aportan.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale
