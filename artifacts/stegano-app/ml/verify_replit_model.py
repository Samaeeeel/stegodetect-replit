"""
verify_replit_model.py — Script de verificación del checkpoint en Replit.

Uso:
    cd artifacts/stegano-app
    python ml/verify_replit_model.py

Comprueba:
    1. Existencia de los archivos de checkpoint
    2. Lectura del model_metadata.json
    3. Disponibilidad de torch (necesario para el modo real)
    4. Carga del modelo desde el checkpoint
    5. Forward pass con imagen sintética [1, 3, 128, 128]

Salida:
    MODEL_OK   — el checkpoint puede cargarse en Replit
    MODEL_ERROR — el checkpoint no puede cargarse, con explicación
"""

import sys
import json
from pathlib import Path

# Ajustar PYTHONPATH para imports relativos al proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_DIR  = PROJECT_ROOT / "ml" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "srnet_lite_best.pt"
METADATA_PATH   = CHECKPOINT_DIR / "model_metadata.json"

SEP = "─" * 60


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "✓" if ok else "✗"
    print(f"  [{status}] {label}")
    if detail:
        print(f"       {detail}")
    return ok


def main() -> int:
    print(SEP)
    print("  SRNet-lite — Verificación de checkpoint en Replit")
    print(SEP)

    all_ok = True

    # ── 1. Existencia de archivos ─────────────────────────────────────────────
    print("\n[1/5] Archivos de checkpoint")

    pt_ok = CHECKPOINT_PATH.exists()
    check(
        "srnet_lite_best.pt existe",
        pt_ok,
        str(CHECKPOINT_PATH) if pt_ok else
        f"No encontrado en: {CHECKPOINT_PATH}\n"
        "       → Sube el archivo desde Drive a ml/checkpoints/",
    )
    all_ok = all_ok and pt_ok

    meta_ok = METADATA_PATH.exists()
    check(
        "model_metadata.json existe",
        meta_ok,
        str(METADATA_PATH) if meta_ok else
        f"No encontrado en: {METADATA_PATH}\n"
        "       → Sube el archivo desde Drive a ml/checkpoints/",
    )
    all_ok = all_ok and meta_ok

    # ── 2. Leer metadata ──────────────────────────────────────────────────────
    print("\n[2/5] Contenido de model_metadata.json")

    threshold = 0.5
    input_size = 128

    if meta_ok:
        try:
            with open(METADATA_PATH) as f:
                meta = json.load(f)

            threshold  = meta.get("threshold", 0.5)
            input_size = meta.get("input_size", 128)

            check("threshold legible",  True, f"threshold  = {threshold}")
            check("input_size legible", True, f"input_size = {input_size}")

            if input_size != 128:
                check(
                    "input_size == 128",
                    False,
                    f"Se esperaba 128 pero el metadata dice {input_size}.\n"
                    "       → La app usa MODEL_INPUT_SIZE=128 (ver config.py).",
                )
                all_ok = False
            else:
                check("input_size == 128", True)

        except (json.JSONDecodeError, KeyError) as e:
            check("metadata válido", False, f"Error al parsear JSON: {e}")
            all_ok = False
    else:
        print("  [↳] Saltado — archivo no existe")

    # ── 3. Disponibilidad de torch ────────────────────────────────────────────
    print("\n[3/5] Disponibilidad de PyTorch")

    try:
        import torch
        check("torch importable", True, f"versión: {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        check(
            "CUDA disponible",
            cuda_ok,
            "GPU detectada" if cuda_ok else
            "Sin GPU — el modelo cargará en CPU (más lento pero funcional)",
        )
    except ImportError:
        check(
            "torch importable",
            False,
            "PyTorch no está instalado en este entorno.\n"
            "       → En Replit, el modo real requiere PyTorch.\n"
            "       → Instala con: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
            "       → NOTA: PyTorch ocupa ~500 MB; verifica el espacio en disco antes.",
        )
        all_ok = False

        print("\n" + SEP)
        print("  MODEL_ERROR — PyTorch no disponible")
        print("  La aplicación funcionará en modo mock hasta que se instale torch.")
        print(SEP)
        return 1

    # ── 4. Cargar modelo ──────────────────────────────────────────────────────
    print("\n[4/5] Carga del modelo")

    if not pt_ok:
        print("  [↳] Saltado — checkpoint no existe")
        print("\n" + SEP)
        print("  MODEL_ERROR — Checkpoint no encontrado")
        print("  Sube srnet_lite_best.pt a ml/checkpoints/ y vuelve a ejecutar.")
        print(SEP)
        return 1

    try:
        # Intentar importar la arquitectura desde ml/src
        try:
            from ml.src.models.srnet_lite import SRNetLite
            check("SRNetLite importado desde ml/src/models/", True)
        except ImportError as ie:
            check(
                "SRNetLite importado desde ml/src/models/",
                False,
                f"Import falló: {ie}\n"
                "       → Usando definición embebida (fallback de emergencia).",
            )
            # Definición mínima embebida para el test
            SRNetLite = _build_minimal_srnet()

        model = SRNetLite()
        state = torch.load(str(CHECKPOINT_PATH), map_location="cpu")

        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]

        model.load_state_dict(state)
        check("load_state_dict() sin errores", True)
        model.eval()
        check("model.eval() OK", True)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        check("Parámetros entrenables", True, f"{n_params:,} ({n_params/1e6:.2f}M)")

    except RuntimeError as e:
        msg = str(e)
        detail = msg

        if "unexpected key" in msg or "missing key" in msg:
            detail = (
                f"Incompatibilidad en state_dict:\n       {msg[:200]}\n"
                "       → El checkpoint se entrenó con una arquitectura diferente.\n"
                "       → Asegúrate de usar el notebook 02 sin modificar."
            )
        elif "size mismatch" in msg:
            detail = (
                f"Tamaño de tensor no coincide:\n       {msg[:200]}\n"
                "       → La arquitectura tiene diferentes dimensiones que el checkpoint."
            )

        check("load_state_dict() sin errores", False, detail)
        all_ok = False

        print("\n" + SEP)
        print("  MODEL_ERROR — No se pudo cargar el checkpoint")
        print(SEP)
        return 1

    except Exception as e:
        check("Carga del checkpoint", False, f"{type(e).__name__}: {e}")
        all_ok = False

        print("\n" + SEP)
        print("  MODEL_ERROR — Error inesperado al cargar")
        print(SEP)
        return 1

    # ── 5. Forward pass ───────────────────────────────────────────────────────
    print("\n[5/5] Forward pass con imagen sintética [1, 3, 128, 128]")

    try:
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            logit = model(x)
            prob  = torch.sigmoid(logit).item()

        expected_shape = (1, 1)
        shape_ok = tuple(logit.shape) == expected_shape

        check("Output shape == (1, 1)", shape_ok, f"shape actual: {tuple(logit.shape)}")
        check("Sin NaN en output",      not torch.isnan(logit).any(), f"logit={logit.item():.4f}")
        check("Probabilidad ∈ (0, 1)",  0.0 < prob < 1.0,             f"prob={prob:.4f} → threshold={threshold}")

    except Exception as e:
        check("Forward pass", False, f"{type(e).__name__}: {e}")
        all_ok = False

        print("\n" + SEP)
        print("  MODEL_ERROR — Forward pass falló")
        print(SEP)
        return 1

    # ── Resultado final ───────────────────────────────────────────────────────
    print("\n" + SEP)
    if all_ok:
        print("  MODEL_OK — El checkpoint puede cargarse correctamente en Replit.")
        print(f"  Threshold: {threshold}  |  Input size: {input_size}×{input_size}")
        print("  Próximo paso: reiniciar el workflow 'StegaDetect' y verificar GET /health")
    else:
        print("  MODEL_ERROR — Revisa los errores marcados con [✗] arriba.")
    print(SEP)

    return 0 if all_ok else 1


def _build_minimal_srnet():
    """
    Definición mínima de SRNetLite para el forward pass de verificación.
    Solo se usa si ml/src/models/ no está en PYTHONPATH.
    Idéntica en nombres de capas a srnet_lite.py.
    """
    import torch
    import torch.nn as nn
    import numpy as np

    def _k():
        k1 = np.array([[0,0,0,0,0],[0,0,0,0,0],[-1,2,-2,2,-1],[0,0,0,0,0],[0,0,0,0,0]],
                      dtype=np.float32) / 4.0
        k2 = k1.T.copy()
        k3 = np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],
                      dtype=np.float32) / 4.0
        return np.stack([k1,k2,k3], axis=0)[:,np.newaxis]

    class SRMLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.srm = nn.Conv2d(3, 9, 5, padding=2, bias=False)
            k = torch.from_numpy(_k())
            w = torch.zeros(9, 3, 5, 5)
            for i in range(3):
                for j in range(3):
                    w[i*3+j, j] = k[i, 0]
            with torch.no_grad():
                self.srm.weight.copy_(w)
            for p in self.srm.parameters():
                p.requires_grad = False
        def forward(self, x): return self.srm(x)

    class Res(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(c,c,3,padding=1,bias=False), nn.BatchNorm2d(c), nn.ReLU(True),
                nn.Conv2d(c,c,3,padding=1,bias=False), nn.BatchNorm2d(c))
            self.relu = nn.ReLU(True)
        def forward(self, x): return self.relu(x + self.block(x))

    class Down(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.main = nn.Sequential(
                nn.Conv2d(ci,co,3,stride=2,padding=1,bias=False), nn.BatchNorm2d(co), nn.ReLU(True),
                nn.Conv2d(co,co,3,padding=1,bias=False), nn.BatchNorm2d(co))
            self.skip = nn.Sequential(nn.Conv2d(ci,co,1,stride=2,bias=False), nn.BatchNorm2d(co))
            self.relu = nn.ReLU(True)
        def forward(self, x): return self.relu(self.main(x) + self.skip(x))

    class Attn(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(c,c//4), nn.ReLU(True), nn.Linear(c//4,c), nn.Sigmoid())
        def forward(self, x): return x * self.se(x).unsqueeze(-1).unsqueeze(-1)

    class SRNetLite(nn.Module):
        def __init__(self):
            super().__init__()
            self.srm    = SRMLayer()
            self.stem   = nn.Sequential(nn.Conv2d(9,16,3,padding=1,bias=False), nn.BatchNorm2d(16), nn.ReLU(True))
            self.stage1 = nn.Sequential(Res(16), Res(16))
            self.down1  = Down(16, 32)
            self.stage2 = nn.Sequential(Res(32), Res(32))
            self.attn2  = Attn(32)
            self.down2  = Down(32, 64)
            self.stage3 = nn.Sequential(Res(64), Res(64))
            self.attn3  = Attn(64)
            self.down3  = Down(64, 128)
            self.stage4 = nn.Sequential(Res(128), Res(128))
            self.attn4  = Attn(128)
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Dropout(0.5), nn.Linear(128,64), nn.ReLU(True),
                nn.Dropout(0.25), nn.Linear(64,1))
        def forward(self, x):
            x = self.srm(x); x = self.stem(x); x = self.stage1(x)
            x = self.attn2(self.stage2(self.down1(x)))
            x = self.attn3(self.stage3(self.down2(x)))
            x = self.attn4(self.stage4(self.down3(x)))
            return self.classifier(x)

    return SRNetLite


if __name__ == "__main__":
    sys.exit(main())
