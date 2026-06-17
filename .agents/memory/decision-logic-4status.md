---
name: Decision logic 4-status
description: _build_final_decision nueva lógica sin dependencia de dominio
---

## Regla
El status final NO depende del domain_status. Solo depende del puntaje ML y de si hay payload LSB validado.

Casos en orden de prioridad:
- **A `payload_found`**: extraction.payload_found AND sha256_valid → evidencia directa, prioridad máxima
- **B `ml_suspicious`**: prob >= 0.46 (threshold del modelo)
- **C `inconclusive_low_ml`**: 0.30 <= prob < 0.46
- **D `no_stego_evidence`**: prob < 0.30

LOW_EVIDENCE_THRESHOLD = 0.30 (constante en stego_routes.py)

**Why:** El modelo fue fine-tuned con dataset externo (wallpapers, fotos, etc.) así que ya no hay razón para bloquear la detección ML fuera de dominio. La info de dominio sigue en model_applicability pero no controla el status.

**Campos nuevos en final_decision:** evidence_type, ml_evidence_band, explanation, primary_metric_label, primary_metric_value

**Backward compat:** PDF antiguo usa status `ml_suspicious_unverified` y `no_evidence` — report_service._build_conclusion tiene fallbacks para ambos.
