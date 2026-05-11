"""
drive_paths.py — Rutas centralizadas del proyecto en Google Drive.

Centralizar todas las rutas en un único archivo evita errores de tipeo
y facilita cambiar la ubicación del proyecto en Drive sin tocar cada notebook.

Uso en Colab:
    from ml.src.dataset.drive_paths import DrivePaths
    paths = DrivePaths()
    paths.ensure_all()
"""

from pathlib import Path


class DrivePaths:
    """
    Gestiona todas las rutas del proyecto stego en Google Drive.
    Crea los directorios si no existen al llamar a ensure_all().
    """

    def __init__(self, base: str = "/content/drive/MyDrive/stego_project"):
        self.base = Path(base)

        # ── Dataset fuente ────────────────────────────────────────────────────
        self.raw         = self.base / "raw"
        self.cover       = self.base / "cover"

        # ── Imágenes estego por payload ───────────────────────────────────────
        self.stego       = self.base / "stego"
        self.stego_p005  = self.stego / "p005"
        self.stego_p010  = self.stego / "p010"
        self.stego_p020  = self.stego / "p020"

        # ── Manifests y splits ────────────────────────────────────────────────
        self.processed        = self.base / "processed"
        self.train_manifest   = self.processed / "train_manifest.csv"
        self.val_manifest     = self.processed / "val_manifest.csv"
        self.test_manifest    = self.processed / "test_manifest.csv"

        # ── Reportes ──────────────────────────────────────────────────────────
        self.reports              = self.base / "reports"
        self.dataset_validation   = self.reports / "dataset_validation.json"
        self.training_history     = self.reports / "training_history.json"
        self.confusion_matrix_png = self.reports / "confusion_matrix.png"
        self.roc_curve_png        = self.reports / "roc_curve.png"
        self.metrics_json         = self.reports / "metrics.json"

        # ── Checkpoints ───────────────────────────────────────────────────────
        self.checkpoints           = self.base / "checkpoints"
        self.best_checkpoint       = self.checkpoints / "srnet_lite_best.pt"
        self.best_state_dict       = self.checkpoints / "srnet_lite_best_state_dict.pt"
        self.model_metadata        = self.checkpoints / "model_metadata.json"

        # ── Caché local (solo en Colab /content, no persistente) ──────────────
        self.local_cache     = Path("/content/cache")
        self.bossbase_zip    = self.raw / "BOSSbase_1.01.zip"
        self.bossbase_pgm    = Path("/content/cache/BOSSBase_1.01")

        # ── Payloads disponibles ──────────────────────────────────────────────
        self.payloads = {
            0.05: self.stego_p005,
            0.10: self.stego_p010,
            0.20: self.stego_p020,
        }

    def ensure_all(self) -> None:
        """Crea todos los directorios necesarios si no existen."""
        dirs = [
            self.raw, self.cover,
            self.stego_p005, self.stego_p010, self.stego_p020,
            self.processed, self.reports, self.checkpoints,
            self.local_cache,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        print(f"[DrivePaths] Estructura verificada en: {self.base}")

    def summary(self) -> None:
        """Imprime un resumen de las rutas del proyecto."""
        print(f"\n{'='*60}")
        print(f"  Proyecto stego en Drive: {self.base}")
        print(f"{'='*60}")
        print(f"  Cover:          {self.cover}")
        print(f"  Stego p=0.05:   {self.stego_p005}")
        print(f"  Stego p=0.10:   {self.stego_p010}")
        print(f"  Stego p=0.20:   {self.stego_p020}")
        print(f"  Train manifest: {self.train_manifest}")
        print(f"  Val manifest:   {self.val_manifest}")
        print(f"  Test manifest:  {self.test_manifest}")
        print(f"  Checkpoint:     {self.best_checkpoint}")
        print(f"  Metadata:       {self.model_metadata}")
        print(f"{'='*60}\n")
