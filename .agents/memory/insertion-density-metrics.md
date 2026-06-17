---
name: Insertion density metrics
description: Métricas forenses de densidad de inserción LSB en embed_payload
---

## Regla
embed_payload computa insertion_density en un post-loop sobre los primeros `total_pixels_used` píxeles comparando orig_pixels vs new_pixels.

Campos en insertion_density:
- used_pixels, modified_pixels, modified_channel_values
- total_pixels_image (w*h)
- used_pixel_ratio, modified_pixel_ratio (porcentajes)
- embedded_bits

**Why:** Para tesis forense — permite medir el impacto real de la inserción (≠ píxeles usados vs ≠ píxeles modificados, ya que si el LSB original ya era el bit a insertar no hay cambio).

**Expuesto en:** embed/text y embed/file endpoints. El frontend (displayEmbedResult) lo muestra en #embed-density-wrap.
