# StegaDetect

Sistema inteligente para la detección de mensajes ocultos en imágenes mediante esteganografía y Machine Learning.

## Run & Operate

- `cd artifacts/stegano-app && uvicorn main:app --host 0.0.0.0 --port 8000 --reload` — run the FastAPI app
- Workflow: **StegaDetect** (auto-starts on port 8000)

## Stack

- Python 3.11, FastAPI, Uvicorn
- ML: PyTorch (optional — mock mode if no checkpoint)
- PDF: ReportLab
- Frontend: HTML5, Bootstrap 5.3, Vanilla JS

## Where things live

- `artifacts/stegano-app/main.py` — FastAPI entry point
- `artifacts/stegano-app/backend/core/config.py` — all constants and paths
- `artifacts/stegano-app/backend/domain/analysis_result.py` — AnalysisResult entity
- `artifacts/stegano-app/backend/services/model_service.py` — PyTorch inference + mock fallback
- `artifacts/stegano-app/backend/services/report_service.py` — PDF generation
- `artifacts/stegano-app/backend/api/routes.py` — API endpoints
- `artifacts/stegano-app/frontend/` — HTML/CSS/JS frontend
- `artifacts/stegano-app/ml/checkpoints/` — place .pt checkpoint here

## Architecture decisions

- Monolithic FastAPI app: simpler than microservices, appropriate for Replit prototype
- Mock mode is hash-based (deterministic): same image always returns same result for reproducible testing
- Frontend is served directly from FastAPI via StaticFiles + HTMLResponse — no separate server needed
- Results stored in `results.json` flat file — adequate for prototype, replace with DB for production

## Product

Upload an image (PNG/JPG/JPEG), analyze it for hidden messages via LSB steganography detection, view probability/confidence result, download PDF report. Works in mock mode without a trained model.

## User preferences

- Python/FastAPI backend, not Node.js
- No Docker, Redis, Celery, or PostgreSQL
- Monolithic modular architecture
- Mock/fallback mode mandatory (app must work without trained model)
- PDF report generation required
- Code should be commented and readable for academic thesis

## Gotchas

- PyTorch is NOT imported in mock mode — mock prediction uses hash-based logic only (no torch required)
- `verifyAndReplaceArtifactToml` requires an existing artifact.toml; bootstrap via bash first
- Disk quota on Replit: avoid running `pip install` inside the workflow command; install globally first
- Workflow **StegaDetect** (port 8000) is the active one — the auto-generated `artifacts/stegano-app: StegaDetect` can be ignored

## Pointers

- See `artifacts/stegano-app/ml/checkpoints/README.md` for model integration instructions
- See `artifacts/stegano-app/backend/services/model_service.py` section `# ── DEFINE TU ARQUITECTURA AQUÍ ──` to add SRNet-lite class
