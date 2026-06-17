---
name: Stego artifacts path
description: Ruta real del directorio de artefactos stego
---

## Regla
STEGO_ARTIFACTS_DIR = BASE_DIR / "backend" / "storage" / "stego_artifacts"
(NO en artifacts/stegano-app/stego_artifacts/)

**Why:** config.py línea 15. Confusión común al correr tests desde el workspace root.
