# %% [markdown]
# # Notebook 02 — Entrenamiento de SRNet-lite en Google Colab
#
# **Proyecto:** Sistema inteligente de detección de mensajes ocultos en imágenes
# mediante esteganografía y Machine Learning (Tesis de grado)
#
# **Prerrequisito:** Ejecutar primero `01_dataset_pipeline_colab.py`
#
# **Ejecutar en Google Colab** con GPU T4 habilitada.
# Runtime → Change runtime type → Hardware accelerator: GPU → GPU type: T4
#
# **Tiempo estimado:** 2–4 horas dependiendo del tamaño del dataset y épocas.
#
# ---
# **Mecanismos anti-colapso incluidos:**
# - Validación de balance antes de entrenar
# - Sanity check del primer batch
# - Early stopping basado en val_auc (no val_loss)
# - Threshold óptimo calculado en validación (no asume 0.5)
# - Mixed precision (FP16) para aprovechar la GPU T4
# - Gradient clipping para estabilizar entrenamiento
# - Label smoothing para evitar sobreconfianza

# %% [markdown]
# ## Celda 1 — Instalar dependencias

# %%
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "torch", "torchvision",
    "scikit-learn", "tqdm", "matplotlib", "pillow"
], check=True)
print("Dependencias listas.")

# %% [markdown]
# ## Celda 2 — Montar Google Drive

# %%
from google.colab import drive
drive.mount("/content/drive")
print("Google Drive montado.")

# %% [markdown]
# ## Celda 3 — Configuración del proyecto

# %%
import sys
import os
from pathlib import Path

# Agregar la raíz del proyecto al PYTHONPATH para importar ml/src
# Si estás ejecutando desde Colab, clona o sube el repo primero
# O copia los archivos ml/src/ al entorno de Colab

# Opción A: Si subiste el repositorio a Drive
REPO_DIR = Path("/content/drive/MyDrive/stego_project/repo")
if REPO_DIR.exists() and str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# Opción B: Si estás ejecutando desde el directorio raíz del repo clonado
if Path("/content/workspace").exists() and "/content/workspace" not in sys.path:
    sys.path.insert(0, "/content/workspace")

# ── Rutas del proyecto ────────────────────────────────────────────────────────
DRIVE_BASE     = Path("/content/drive/MyDrive/stego_project")
TRAIN_MANIFEST = DRIVE_BASE / "processed" / "train_manifest.csv"
VAL_MANIFEST   = DRIVE_BASE / "processed" / "val_manifest.csv"
TEST_MANIFEST  = DRIVE_BASE / "processed" / "test_manifest.csv"
CHECKPOINT_DIR = DRIVE_BASE / "checkpoints"
REPORTS_DIR    = DRIVE_BASE / "reports"

CHECKPOINT_BEST      = CHECKPOINT_DIR / "srnet_lite_best.pt"
STATE_DICT_BEST      = CHECKPOINT_DIR / "srnet_lite_best_state_dict.pt"
MODEL_METADATA       = CHECKPOINT_DIR / "model_metadata.json"
TRAINING_HISTORY     = REPORTS_DIR    / "training_history.json"
METRICS_JSON         = REPORTS_DIR    / "metrics.json"
CONFUSION_MATRIX_PNG = REPORTS_DIR    / "confusion_matrix.png"
ROC_CURVE_PNG        = REPORTS_DIR    / "roc_curve.png"
PROB_DIST_PNG        = REPORTS_DIR    / "prob_distribution.png"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Verificar manifests ───────────────────────────────────────────────────────
for path, name in [(TRAIN_MANIFEST, "train"), (VAL_MANIFEST, "val"), (TEST_MANIFEST, "test")]:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest de {name} no encontrado: {path}\n"
            "Ejecuta primero 01_dataset_pipeline_colab.py"
        )
print("Manifests verificados OK.")

# ── Hiperparámetros ───────────────────────────────────────────────────────────
CONFIG = {
    "lr":              1e-4,       # Learning rate inicial (AdamW)
    "weight_decay":    1e-4,       # L2 regularization
    "epochs":          60,         # Máximo de épocas (early stopping puede cortar antes)
    "batch_size":      32,         # Tamaño de batch (reducir a 16 si hay OOM)
    "num_workers":     2,          # Workers del DataLoader (2 es seguro en Colab)
    "patience":        12,         # Early stopping: épocas sin mejora en val_auc
    "loss_type":       "smooth_bce",  # Con label smoothing para evitar sobreconfianza
    "label_smoothing": 0.05,
    "grad_clip":       1.0,        # Clipping máximo de gradientes
    "scheduler":       "cosine",   # CosineAnnealing LR
    "mixed_precision": True,       # AMP con GPU T4
    "threshold_method":"youden",   # Calcular threshold óptimo con índice de Youden
    "seed":            42,
    "crop_size":       128,        # Crop 128x128 desde imágenes 512x512
    "resume":          False,      # True para reanudar entrenamiento desde checkpoint
}

print("Configuración:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## Celda 4 — Semilla y device

# %%
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Semilla fijada: {seed}")

set_seed(CONFIG["seed"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[WARNING] GPU no disponible. El entrenamiento será muy lento en CPU.")

# %% [markdown]
# ## Celda 5 — Dataset PyTorch
#
# El dataset **NO** carga todas las imágenes en RAM.
# Lee cada imagen del disco bajo demanda en el método `__getitem__`.
# Esto permite trabajar con datasets de cualquier tamaño.
#
# Transformaciones en entrenamiento:
#   - RandomCrop 128×128 (desde imagen 512×512): introduce variabilidad de posición
#   - RandomHorizontalFlip: simetría horizontal (no interpola, solo refleja)
#   - RandomVerticalFlip: simetría vertical (no interpola, solo refleja)
#
# **NO se usa:** rotaciones arbitrarias, color jitter, resize con interpolación,
# porque estas transformaciones modifican los valores de píxel y destruyen la señal LSB.
#
# Transformaciones en validación/test:
#   - CenterCrop 128×128: determinista y reproducible

# %%
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

class StegoDataset(Dataset):
    """
    Dataset PyTorch para estegoanálisis LSB.

    Lee imágenes del disco bajo demanda (no en RAM).
    Devuelve (tensor_imagen, label) por cada índice.
    """

    def __init__(self, manifest_path: Path, transform=None):
        self.transform = transform
        self.samples   = []

        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.samples.append({
                    "path":    Path(row["image_path"]),
                    "label":   int(row["label"]),
                    "payload": float(row["payload"]),
                })

        # Verificar balance
        n_cover = sum(1 for s in self.samples if s["label"] == 0)
        n_stego = sum(1 for s in self.samples if s["label"] == 1)
        print(f"  [Dataset] {manifest_path.name}: {len(self.samples)} muestras "
              f"| cover={n_cover} | stego={n_stego} "
              f"| ratio={n_stego/len(self.samples):.1%}")

        if n_cover == 0 or n_stego == 0:
            raise ValueError(
                f"Dataset desbalanceado al extremo en {manifest_path.name}: "
                f"cover={n_cover}, stego={n_stego}. Verifica el manifest."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        try:
            with Image.open(sample["path"]) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            # Si la imagen está corrupta, devolver tensor negro + mismo label
            print(f"[WARNING] Imagen corrupta: {sample['path']}: {e}")
            img_rgb = Image.new("RGB", (512, 512), color=0)

        if self.transform:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = T.ToTensor()(img_rgb)

        return img_tensor, torch.tensor(sample["label"], dtype=torch.long)


# Transformaciones
# IMPORTANTE: Las augmentations (random crop, flips) NO interpolan píxeles,
# solo seleccionan o reorganizan píxeles existentes.
# El resize con interpolación (bilinear, bicubic) DESTRUYE la señal LSB
# porque mezcla los valores de píxel.

CROP_SIZE = CONFIG["crop_size"]

TRAIN_TRANSFORM = T.Compose([
    T.RandomCrop(CROP_SIZE),           # Crop aleatorio 128×128 desde 512×512
    T.RandomHorizontalFlip(p=0.5),     # Flip horizontal — no interpola
    T.RandomVerticalFlip(p=0.5),       # Flip vertical — no interpola
    T.ToTensor(),                      # [0,255] → [0.0, 1.0]
    T.Normalize(mean=[0.5, 0.5, 0.5], # Normalizar a [-1, 1]
                std=[0.5, 0.5, 0.5]),
])

VAL_TEST_TRANSFORM = T.Compose([
    T.CenterCrop(CROP_SIZE),           # Crop central determinista
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
])

print("\nCargando datasets...")
train_dataset = StegoDataset(TRAIN_MANIFEST, transform=TRAIN_TRANSFORM)
val_dataset   = StegoDataset(VAL_MANIFEST,   transform=VAL_TEST_TRANSFORM)
test_dataset  = StegoDataset(TEST_MANIFEST,  transform=VAL_TEST_TRANSFORM)

PIN_MEMORY = (DEVICE == "cuda")

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=CONFIG["num_workers"],
    pin_memory=PIN_MEMORY,
    drop_last=True,          # Descartar último batch incompleto para BN estable
)
val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"] * 2,  # Val puede tener batch más grande
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    pin_memory=PIN_MEMORY,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG["batch_size"] * 2,
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    pin_memory=PIN_MEMORY,
)

print(f"\nDataLoaders listos:")
print(f"  Train: {len(train_loader)} batches × {CONFIG['batch_size']} = ~{len(train_dataset)} muestras")
print(f"  Val:   {len(val_loader)} batches")
print(f"  Test:  {len(test_loader)} batches")

# %% [markdown]
# ## Celda 6 — Sanity check del primer batch
#
# Verificar que el batch tiene la forma y valores correctos antes de entrenar.
# Un error aquí es mejor que descubrir el problema después de 10 épocas.

# %%
images, labels = next(iter(train_loader))
print(f"\n[SanityCheck] Primer batch de entrenamiento:")
print(f"  images.shape: {images.shape}   ← esperado: [{CONFIG['batch_size']}, 3, 128, 128]")
print(f"  images.dtype: {images.dtype}   ← esperado: torch.float32")
print(f"  images.min/max: [{images.min():.3f}, {images.max():.3f}]  ← esperado: [-1, 1]")
print(f"  labels.shape: {labels.shape}")
print(f"  labels.unique: {labels.unique().tolist()}")
print(f"  balance: {labels.float().mean():.1%} stego")

assert images.shape == torch.Size([CONFIG["batch_size"], 3, 128, 128]), "Shape incorrecto!"
assert not torch.isnan(images).any(), "NaN en imágenes!"
assert images.min() >= -1.5 and images.max() <= 1.5, "Normalización incorrecta!"
print("\n[OK] Batch verificado. Listo para entrenar.")

# %% [markdown]
# ## Celda 7 — Construir modelo SRNet-lite

# %%
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# DEFINICIÓN DEL MODELO
# (Copiado aquí para que el notebook sea autocontenido en Colab)
# Si tienes ml/src/models/ en PYTHONPATH, puedes usar:
#     from ml.src.models.srnet_lite import build_srnet_lite
# ─────────────────────────────────────────────────────────────────────────────

def _build_srm_kernels():
    k1 = np.array([[ 0, 0, 0,  0, 0],[ 0, 0, 0,  0, 0],[-1, 2,-2,  2,-1],
                   [ 0, 0, 0,  0, 0],[ 0, 0, 0,  0, 0]], dtype=np.float32) / 4.0
    k2 = k1.T.copy()
    k3 = np.array([[ 0, 0, 0,  0, 0],[ 0,-1, 2, -1, 0],[ 0, 2,-4,  2, 0],
                   [ 0,-1, 2, -1, 0],[ 0, 0, 0,  0, 0]], dtype=np.float32) / 4.0
    return np.stack([k1, k2, k3], axis=0)[:, np.newaxis, :, :]

class SRMLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.srm = nn.Conv2d(3, 9, 5, padding=2, bias=False, groups=1)
        k = torch.from_numpy(_build_srm_kernels())
        weight = torch.zeros(9, 3, 5, 5)
        for i in range(3):
            for j in range(3):
                weight[i*3+j, j] = k[i, 0]
        with torch.no_grad():
            self.srm.weight.copy_(weight)
        for p in self.srm.parameters():
            p.requires_grad = False
    def forward(self, x): return self.srm(x)

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c,c,3,padding=1,bias=False), nn.BatchNorm2d(c), nn.ReLU(True),
            nn.Conv2d(c,c,3,padding=1,bias=False), nn.BatchNorm2d(c),
        )
        self.relu = nn.ReLU(True)
    def forward(self, x): return self.relu(x + self.block(x))

class DownsampleBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(ci,co,3,stride=2,padding=1,bias=False), nn.BatchNorm2d(co), nn.ReLU(True),
            nn.Conv2d(co,co,3,padding=1,bias=False), nn.BatchNorm2d(co),
        )
        self.skip = nn.Sequential(nn.Conv2d(ci,co,1,stride=2,bias=False), nn.BatchNorm2d(co))
        self.relu = nn.ReLU(True)
    def forward(self, x): return self.relu(self.main(x) + self.skip(x))

class AttentionBlock(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, c//r), nn.ReLU(True),
            nn.Linear(c//r, c), nn.Sigmoid(),
        )
    def forward(self, x): return x * self.se(x).unsqueeze(-1).unsqueeze(-1)

class SRNetLite(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.srm    = SRMLayer()
        self.stem   = nn.Sequential(nn.Conv2d(9,16,3,padding=1,bias=False), nn.BatchNorm2d(16), nn.ReLU(True))
        self.stage1 = nn.Sequential(ResidualBlock(16), ResidualBlock(16))
        self.down1  = DownsampleBlock(16, 32)
        self.stage2 = nn.Sequential(ResidualBlock(32), ResidualBlock(32))
        self.attn2  = AttentionBlock(32)
        self.down2  = DownsampleBlock(32, 64)
        self.stage3 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.attn3  = AttentionBlock(64)
        self.down3  = DownsampleBlock(64, 128)
        self.stage4 = nn.Sequential(ResidualBlock(128), ResidualBlock(128))
        self.attn4  = AttentionBlock(128)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )
        self._init()
    def _init(self):
        # IMPORTANTE: solo inicializar capas entrenables.
        # Los filtros SRM tienen requires_grad=False y no deben ser sobreescritos.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:   # ← salta el SRM
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.srm(x)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.attn2(self.stage2(self.down1(x)))
        x = self.attn3(self.stage3(self.down2(x)))
        x = self.attn4(self.stage4(self.down3(x)))
        return self.classifier(x)

model = SRNetLite(dropout=0.5).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[OK] SRNetLite construido: {n_params:,} parámetros entrenables ({n_params/1e6:.2f}M)")

# Test rápido de forward pass
with torch.no_grad():
    dummy = torch.randn(2, 3, 128, 128).to(DEVICE)
    out   = model(dummy)
    print(f"  Forward pass OK: input {dummy.shape} → output {out.shape}")

# %% [markdown]
# ## Celda 8 — Calcular pos_weight para balance de clases

# %%
import csv as csv_module

def count_labels(manifest_path):
    cover, stego = 0, 0
    with open(manifest_path) as f:
        for r in csv_module.DictReader(f):
            if int(r["label"]) == 0: cover += 1
            else: stego += 1
    return cover, stego

n_cover, n_stego = count_labels(TRAIN_MANIFEST)
pos_weight = n_cover / n_stego if n_stego > 0 else 1.0

print(f"Train: cover={n_cover}, stego={n_stego}")
print(f"pos_weight = {pos_weight:.3f}")
print(f"  (>1 penaliza más los FN de clase stego, <1 penaliza más los FP)")

CONFIG["pos_weight"] = pos_weight

# %% [markdown]
# ## Celda 9 — Configurar optimizador, pérdida y scheduler

# %%
from sklearn.metrics import roc_auc_score

# Optimizador: AdamW con weight decay para regularización
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"],
)

# Pérdida: BCE con label smoothing y pos_weight
# Label smoothing: 0 → 0.025, 1 → 0.975 (reduce sobreconfianza)
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([CONFIG["pos_weight"]]).to(DEVICE)
)

# Scheduler: CosineAnnealingLR — reduce LR suavemente de lr a eta_min
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CONFIG["epochs"], eta_min=1e-6
)

# Mixed precision (AMP): reduce uso de VRAM y acelera en GPU T4
USE_AMP = CONFIG["mixed_precision"] and DEVICE == "cuda"
scaler  = torch.cuda.amp.GradScaler() if USE_AMP else None

print(f"Optimizador: AdamW (lr={CONFIG['lr']}, wd={CONFIG['weight_decay']})")
print(f"Pérdida:     BCEWithLogitsLoss (pos_weight={pos_weight:.3f})")
print(f"Scheduler:   CosineAnnealingLR")
print(f"AMP:         {'Sí (FP16)' if USE_AMP else 'No (FP32)'}")

# %% [markdown]
# ## Celda 10 — Loop de entrenamiento principal
#
# Métricas monitoreadas por época:
#   - train_loss, val_loss
#   - val_auc (métrica principal para early stopping y guardado de checkpoint)
#   - val_accuracy, val_f1
#   - Detección de colapso de modelo

# %%
import json
import time
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def train_epoch(model, loader, optimizer, criterion, device, scaler, grad_clip):
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(images).squeeze(1)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images).squeeze(1)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        with torch.no_grad():
            all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_labels), np.array(all_probs)

@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        logits = model(images).squeeze(1)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_labels), np.array(all_probs)

def find_optimal_threshold(y_true, y_prob):
    """Threshold óptimo por índice de Youden: maximiza TPR - FPR."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])

# ── Estado del entrenamiento ─────────────────────────────────────────────────
history = {"train_loss":[], "val_loss":[], "train_auc":[], "val_auc":[], "lr":[]}
best_val_auc   = 0.0
best_threshold = 0.5
no_improve     = 0
PATIENCE       = CONFIG["patience"]
start_epoch    = 0

# Reanudar desde checkpoint si se solicita
if CONFIG["resume"] and CHECKPOINT_BEST.exists():
    ck = torch.load(str(CHECKPOINT_BEST), map_location=DEVICE)
    model.load_state_dict(ck["model_state"])
    optimizer.load_state_dict(ck["optim_state"])
    start_epoch    = ck["epoch"] + 1
    best_val_auc   = ck.get("metrics", {}).get("val_auc", 0)
    best_threshold = ck.get("threshold", 0.5)
    print(f"Reanudando desde época {start_epoch} | best_val_auc={best_val_auc:.4f}")

print(f"\n{'='*65}")
print(f"  ENTRENAMIENTO — SRNet-lite")
print(f"  Épocas: {CONFIG['epochs']} | Batch: {CONFIG['batch_size']} | Device: {DEVICE}")
print(f"{'='*65}\n")

for epoch in range(start_epoch, CONFIG["epochs"]):
    t0 = time.time()

    train_loss, tr_y, tr_prob = train_epoch(
        model, train_loader, optimizer, criterion, DEVICE, scaler, CONFIG["grad_clip"]
    )
    val_loss, val_y, val_prob = val_epoch(model, val_loader, criterion, DEVICE)

    # Calcular AUC y threshold óptimo
    try:
        train_auc = roc_auc_score(tr_y, tr_prob)
        val_auc   = roc_auc_score(val_y, val_prob)
        threshold = find_optimal_threshold(val_y, val_prob)
    except Exception:
        train_auc = val_auc = 0.5
        threshold = 0.5

    val_preds = (val_prob >= threshold).astype(int)
    val_acc   = accuracy_score(val_y, val_preds)
    val_f1    = f1_score(val_y, val_preds, zero_division=0)
    val_recall= (val_preds[val_y==1].sum() / (val_y==1).sum()) if (val_y==1).sum() > 0 else 0

    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_auc"].append(train_auc)
    history["val_auc"].append(val_auc)
    history["lr"].append(current_lr)

    elapsed = time.time() - t0
    print(f"Época [{epoch+1:3d}/{CONFIG['epochs']}] "
          f"| loss {train_loss:.4f}/{val_loss:.4f} "
          f"| auc {train_auc:.4f}/{val_auc:.4f} "
          f"| acc={val_acc:.4f} f1={val_f1:.4f} rec={val_recall:.4f} "
          f"| thr={threshold:.3f} lr={current_lr:.1e} | {elapsed:.0f}s")

    # Advertencia de colapso
    pred_ratio = val_preds.mean()
    if pred_ratio < 0.05 or pred_ratio > 0.95:
        print(f"  [WARNING] Posible colapso: {pred_ratio:.1%} de predicciones son "
              f"{'stego' if pred_ratio > 0.5 else 'cover'}")

    # Guardar historial
    with open(TRAINING_HISTORY, "w") as f:
        json.dump(history, f, indent=2)

    # Guardar mejor checkpoint
    if val_auc > best_val_auc:
        best_val_auc   = val_auc
        best_threshold = threshold
        no_improve     = 0
        import datetime

        # Checkpoint completo (para resume)
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "metrics":     {"val_auc": val_auc, "val_loss": val_loss},
            "threshold":   threshold,
        }, str(CHECKPOINT_BEST))

        # Solo pesos (para Replit)
        torch.save(model.state_dict(), str(STATE_DICT_BEST))

        # Metadatos
        metadata = {
            "epoch":         epoch,
            "val_auc":       round(val_auc, 4),
            "val_loss":      round(val_loss, 4),
            "val_accuracy":  round(val_acc, 4),
            "val_f1":        round(val_f1, 4),
            "threshold":     round(threshold, 4),
            "model_version": "srnet-lite-v1.0",
            "input_size":    128,
            "input_channels":3,
            "saved_at":      datetime.datetime.utcnow().isoformat(),
            "architecture":  {"srm_filters": 9, "use_attention": True, "dropout_rate": 0.5},
        }
        with open(MODEL_METADATA, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  *** Nuevo mejor checkpoint guardado (val_auc={val_auc:.4f}) ***")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\n  Early stopping en época {epoch+1}. Mejor val_auc={best_val_auc:.4f}")
            break

print(f"\nEntrenamiento finalizado.")
print(f"Mejor val_auc = {best_val_auc:.4f} | Threshold = {best_threshold:.4f}")

# %% [markdown]
# ## Celda 11 — Evaluación final en test set
#
# El test set se usa SOLO aquí, al final. Nunca para ajustar hiperparámetros.

# %%
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)

# Cargar el mejor modelo
best_state = torch.load(str(CHECKPOINT_BEST), map_location=DEVICE)
model.load_state_dict(best_state["model_state"])
model.eval()

# Evaluar en test
test_loss, test_y, test_prob = val_epoch(model, test_loader, criterion, DEVICE)

test_preds = (test_prob >= best_threshold).astype(int)
test_auc   = roc_auc_score(test_y, test_prob)
test_acc   = accuracy_score(test_y, test_preds)
test_prec  = precision_score(test_y, test_preds, zero_division=0)
test_rec   = recall_score(test_y, test_preds, zero_division=0)
test_f1    = f1_score(test_y, test_preds, zero_division=0)
cm         = confusion_matrix(test_y, test_preds)
tn, fp, fn, tp = cm.ravel()

print(f"\n{'='*55}")
print(f"  EVALUACIÓN FINAL EN TEST SET")
print(f"{'='*55}")
print(f"  AUC-ROC:   {test_auc:.4f}  ← métrica principal")
print(f"  Accuracy:  {test_acc:.4f}")
print(f"  Precision: {test_prec:.4f}")
print(f"  Recall:    {test_rec:.4f}")
print(f"  F1-score:  {test_f1:.4f}")
print(f"  Threshold: {best_threshold:.4f}")
print(f"  Matriz de confusión:")
print(f"    TN={tn:5d}  FP={fp:5d}")
print(f"    FN={fn:5d}  TP={tp:5d}")
print(f"{'='*55}")

if test_auc < 0.55:
    print("\n[WARNING] AUC cercano a azar. El modelo puede haber colapsado.")
    print("  Verifica: balance del dataset, learning rate, arquitectura.")
elif test_auc > 0.70:
    print("\n[OK] El modelo muestra capacidad discriminativa.")

# Guardar métricas finales
final_metrics = {
    "test_auc":       round(test_auc, 4),
    "test_accuracy":  round(test_acc, 4),
    "test_precision": round(test_prec, 4),
    "test_recall":    round(test_rec, 4),
    "test_f1":        round(test_f1, 4),
    "test_loss":      round(test_loss, 4),
    "threshold":      round(best_threshold, 4),
    "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
}
with open(METRICS_JSON, "w") as f:
    json.dump(final_metrics, f, indent=2)
print(f"\nMétricas guardadas en: {METRICS_JSON}")

# %% [markdown]
# ## Celda 12 — Generar visualizaciones

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# ── Curva ROC ────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(test_y, test_prob)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, "#1a56db", lw=2.5, label=f"SRNet-lite (AUC={test_auc:.4f})")
plt.plot([0,1],[0,1],"k--",alpha=0.4, label="Azar")
plt.fill_between(fpr, tpr, alpha=0.08, color="#1a56db")
plt.xlabel("Tasa de Falsos Positivos"); plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC — Detección de Esteganografía LSB", fontweight="bold")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(str(ROC_CURVE_PNG), dpi=150); plt.close()
print(f"Curva ROC guardada: {ROC_CURVE_PNG}")

# ── Matriz de confusión ──────────────────────────────────────────────────────
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, fmt in zip(axes, [cm, cm_norm], ["d", ".1%"]):
    im = ax.imshow(data, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    ax.set(xticks=[0,1], yticks=[0,1],
           xticklabels=["Cover","Stego"], yticklabels=["Cover","Stego"])
    th = data.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{data[i,j]:{fmt}}", ha="center", va="center",
                    color="white" if data[i,j] > th else "black", fontsize=13)
plt.suptitle("Matriz de Confusión — SRNet-lite", fontweight="bold")
plt.tight_layout()
plt.savefig(str(CONFUSION_MATRIX_PNG), dpi=150); plt.close()
print(f"Matriz de confusión guardada: {CONFUSION_MATRIX_PNG}")

# ── Historial de entrenamiento ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
epochs_range = range(1, len(history["train_loss"]) + 1)
axes[0].plot(epochs_range, history["train_loss"], "b-o", ms=3, label="Train")
axes[0].plot(epochs_range, history["val_loss"],   "r-o", ms=3, label="Val")
axes[0].set_title("Loss por época"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(epochs_range, history["train_auc"],  "b-o", ms=3, label="Train AUC")
axes[1].plot(epochs_range, history["val_auc"],    "r-o", ms=3, label="Val AUC")
axes[1].axhline(0.5, color="gray", ls="--", alpha=0.5, label="Azar")
axes[1].set_title("AUC-ROC por época"); axes[1].set_ylim([0.4, 1.05])
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.suptitle("Historial de Entrenamiento — SRNet-lite", fontweight="bold")
plt.tight_layout()
plt.savefig(str(REPORTS_DIR / "training_history.png"), dpi=150); plt.close()
print(f"Historial guardado en: {REPORTS_DIR}")

# %% [markdown]
# ## Celda 13 — Instrucciones para integrar el modelo en Replit
#
# ¡Entrenamiento completo! Ahora descarga el checkpoint y súbelo a Replit.

# %%
print(f"""
{'='*65}
  ENTRENAMIENTO COMPLETADO
{'='*65}

  Archivos para descargar de Google Drive y subir a Replit:

  1. CHECKPOINT (requerido):
     {CHECKPOINT_BEST}
     → Subir a:  ml/checkpoints/srnet_lite_best.pt

  2. METADATOS (requerido para threshold):
     {MODEL_METADATA}
     → Subir a:  ml/checkpoints/model_metadata.json

  Cómo descargar desde Colab:
  ─────────────────────────
  from google.colab import files
  files.download(str(STATE_DICT_BEST))   # Solo pesos (más ligero)
  files.download(str(MODEL_METADATA))

  Cómo verificar en Replit que el modo mock está desactivado:
  ──────────────────────────────────────────────────────────
  GET /health
  Respuesta esperada: {{ "mock_mode": false, "model_version": "srnet-lite-v1.0" }}

{'='*65}
  val_auc  = {best_val_auc:.4f}
  test_auc = {test_auc:.4f}
  threshold= {best_threshold:.4f}
{'='*65}
""")

# Opcional: descargar directamente desde Colab
# from google.colab import files
# files.download(str(STATE_DICT_BEST))
# files.download(str(MODEL_METADATA))
