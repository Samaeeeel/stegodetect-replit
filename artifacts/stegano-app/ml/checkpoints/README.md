# Directorio de Checkpoints del Modelo

Este directorio almacena los pesos entrenados del modelo de estegoanálisis.

## Archivos esperados

El sistema busca los siguientes archivos en este orden:

1. `srnet_lite_best.pt` — Checkpoint preferido (modelo SRNet-lite entrenado en Colab)
2. `model.pt`           — Nombre alternativo aceptado

## Cómo cargar el modelo entrenado

1. Entrena el modelo en Google Colab usando BOSSBase v1.01 con imágenes stego LSB.
2. Guarda el checkpoint con:

```python
torch.save(model.state_dict(), "srnet_lite_best.pt")
```

3. Descarga el archivo `.pt` desde Colab.
4. Súbelo a este directorio en Replit.
5. Reinicia el servidor — el sistema lo detectará automáticamente.

## Qué modificar en el código

En `backend/services/model_service.py`, busca la sección marcada:

```
# ── DEFINE TU ARQUITECTURA AQUÍ ──
```

Define tu clase `SRNetLite` (o la arquitectura que hayas usado) y reemplaza
`_DummyModel` en la función `_load_real_model()`.

## Modo mock

Mientras no exista ningún checkpoint aquí, la aplicación funciona en
**modo demostración (mock)**. Los resultados son simulados y no tienen
validez científica.
