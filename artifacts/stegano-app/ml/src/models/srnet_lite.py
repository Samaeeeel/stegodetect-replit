"""
srnet_lite.py — Arquitectura SRNet-lite para estegoanálisis.

Inspirada en:
  SRNet: Boroumand et al. (2018) "Deep Residual Network for Steganalysis
         of Digital Images", IEEE Transactions on Information Forensics and Security.

Diferencias con SRNet completo:
  - Menos bloques (más ligero para entrenamiento en Colab T4)
  - Atención por canal (SE) para seleccionar features relevantes
  - Compatible con entrada [B, 3, 128, 128] y salida [B] (logit)

Flujo del modelo:
  Input RGB [B,3,128,128]
      ↓
  SRM HPF (filtros fijos, sin entrenamiento)   → [B,9,128,128]
      ↓
  Conv inicial + BN + ReLU                     → [B,16,128,128]
      ↓
  Bloque Residual × 2                          → [B,16,128,128]
      ↓
  Downsample (stride=2)                        → [B,32,64,64]
      ↓
  Bloque Residual × 2 + Atención SE            → [B,32,64,64]
      ↓
  Downsample (stride=2)                        → [B,64,32,32]
      ↓
  Bloque Residual × 2 + Atención SE            → [B,64,32,32]
      ↓
  Downsample (stride=2)                        → [B,128,16,16]
      ↓
  Bloque Residual × 2 + Atención SE            → [B,128,16,16]
      ↓
  Global Average Pooling                       → [B,128]
      ↓
  Dropout(0.5) + Linear(128→64) + ReLU         → [B,64]
      ↓
  Linear(64→1) → logit                         → [B,1]
"""

import torch
import torch.nn as nn

from ml.src.models.blocks import SRMLayer, ResidualBlock, DownsampleBlock, AttentionBlock


class SRNetLite(nn.Module):
    """
    SRNet-lite: detector de esteganografía LSB basado en residuo de ruido.

    Entrada:  tensor RGB normalizado [B, 3, 128, 128]
    Salida:   logit binario [B, 1]  (sin sigmoid — usar BCEWithLogitsLoss)

    Para inferencia, aplicar sigmoid al logit para obtener probabilidad:
        prob = torch.sigmoid(model(x)).item()
    """

    def __init__(
        self,
        dropout_rate: float = 0.5,
        use_attention: bool = True,
    ):
        super().__init__()

        self.use_attention = use_attention

        # ── Capa 0: HPF/SRM — filtros fijos no entrenables ───────────────────
        # Transforma la imagen en mapas de residuo de alta frecuencia.
        # Salida: 9 canales (3 filtros SRM × 3 canales RGB)
        self.srm = SRMLayer()

        # ── Capa 1: Proyección inicial ────────────────────────────────────────
        # Transforma los 9 canales SRM a 16 features
        self.stem = nn.Sequential(
            nn.Conv2d(9, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # ── Etapa 1: 128×128, 16 canales ─────────────────────────────────────
        self.stage1 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
        )

        # ── Etapa 2: 64×64, 32 canales ───────────────────────────────────────
        self.down1  = DownsampleBlock(16, 32)
        self.stage2 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
        )
        self.attn2  = AttentionBlock(32) if use_attention else nn.Identity()

        # ── Etapa 3: 32×32, 64 canales ───────────────────────────────────────
        self.down2  = DownsampleBlock(32, 64)
        self.stage3 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
        )
        self.attn3  = AttentionBlock(64) if use_attention else nn.Identity()

        # ── Etapa 4: 16×16, 128 canales ──────────────────────────────────────
        self.down3  = DownsampleBlock(64, 128)
        self.stage4 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.attn4  = AttentionBlock(128) if use_attention else nn.Identity()

        # ── Clasificador ──────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # Global Average Pooling → [B, 128, 1, 1]
            nn.Flatten(),               # → [B, 128]
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 1),          # logit sin activación
        )

        # Inicialización de pesos
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Inicialización cuidadosa de pesos para evitar colapso inicial.
        Usa He/Kaiming para capas con ReLU.

        IMPORTANTE: salta los parámetros NO entrenables (SRM) para no
        sobreescribir los filtros fijos que se establecieron en SRMLayer.__init__.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Solo inicializar capas entrenables — los filtros SRM son fijos
                if m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor RGB normalizado [B, 3, 128, 128]
               Normalización esperada: (pixel/255 - 0.5) / 0.5  ∈ [-1, 1]

        Returns:
            Tensor de logits [B, 1] — aplicar sigmoid para obtener probabilidad
        """
        # HPF: extraer señal de residuo de alta frecuencia
        x = self.srm(x)      # [B, 9, 128, 128]

        # Proyección
        x = self.stem(x)     # [B, 16, 128, 128]

        # Etapas progresivas
        x = self.stage1(x)                   # [B, 16, 128, 128]
        x = self.attn2(self.stage2(self.down1(x)))  # [B, 32, 64, 64]
        x = self.attn3(self.stage3(self.down2(x)))  # [B, 64, 32, 32]
        x = self.attn4(self.stage4(self.down3(x)))  # [B, 128, 16, 16]

        # Clasificación
        return self.classifier(x)            # [B, 1]

    def count_parameters(self) -> int:
        """Devuelve el número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_srnet_lite(
    dropout_rate: float = 0.5,
    use_attention: bool = True,
) -> SRNetLite:
    """
    Constructor principal del modelo SRNet-lite.

    Args:
        dropout_rate:   Tasa de dropout en el clasificador (0.0 – 1.0)
        use_attention:  Activar módulos de atención SE entre etapas

    Returns:
        Instancia de SRNetLite lista para entrenamiento o inferencia
    """
    model = SRNetLite(dropout_rate=dropout_rate, use_attention=use_attention)
    params = model.count_parameters()
    print(f"[SRNetLite] Parámetros entrenables: {params:,} ({params/1e6:.2f}M)")
    return model


# ── Prueba rápida de sanidad ──────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    print("Verificación de SRNetLite...")

    model = build_srnet_lite()
    model.eval()

    # Batch de prueba: 2 imágenes RGB 128×128
    x = torch.randn(2, 3, 128, 128)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.sigmoid(logits)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Logits:       {logits.squeeze().tolist()}")
    print(f"  Probs:        {probs.squeeze().tolist()}")
    print("  OK — arquitectura verificada.")
