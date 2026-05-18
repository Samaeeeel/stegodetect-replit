/**
 * app.js — Lógica del frontend de StegoDetect
 *
 * Maneja:
 *   1. Drag & drop y selección de imagen
 *   2. Vista previa de imagen
 *   3. Envío de formulario (POST /analyze)
 *   4. Visualización de estados (idle, loading, result, error)
 *   5. Descarga del reporte PDF (GET /report/{id})
 *   6. Verificación del estado del sistema (GET /health)
 */

/* ── Elementos del DOM ───────────────────────────────────────────────────── */
const dropZone       = document.getElementById('drop-zone');
const fileInput      = document.getElementById('file-input');
const previewImg     = document.getElementById('preview-img');
const dropPlaceholder= document.getElementById('drop-placeholder');
const fileInfo       = document.getElementById('file-info');
const fileName       = document.getElementById('file-name');
const fileSize       = document.getElementById('file-size');
const clearBtn       = document.getElementById('clear-btn');
const analyzeBtn     = document.getElementById('analyze-btn');
const modelBadge     = document.getElementById('model-status-badge');
const mockAlert      = document.getElementById('mock-alert');

// Paneles de estado
const stateIdle      = document.getElementById('state-idle');
const stateLoading   = document.getElementById('state-loading');
const stateError     = document.getElementById('state-error');
const stateResult    = document.getElementById('state-result');
const errorMessage   = document.getElementById('error-message');

// Elementos de resultado
const resultBadge    = document.getElementById('result-badge');
const resultIcon     = document.getElementById('result-icon');
const resultLabel    = document.getElementById('result-label');
const resultFilename = document.getElementById('result-filename');
const probBar        = document.getElementById('prob-bar');
const probText       = document.getElementById('prob-text');
const confidenceBadge= document.getElementById('confidence-badge');
const explanationText= document.getElementById('explanation-text');
const mockResultWarn = document.getElementById('mock-result-warning');
const pdfBtn         = document.getElementById('pdf-btn');

/* ── Estado de la aplicación ─────────────────────────────────────────────── */
let selectedFile = null;
let lastAnalysisId = null;

/* ── Inicialización ──────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  bindEvents();
});

/* ── Verificar estado del sistema ─────────────────────────────────────────── */
async function checkHealth() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error('Sin respuesta');
    const data = await res.json();

    if (data.mock_mode) {
      modelBadge.innerHTML = '<i class="bi bi-flask me-1"></i>Modo Demostración';
      modelBadge.className = 'badge bg-warning text-dark';
      modelBadge.title     = '';
      mockAlert.style.display = 'block';
    } else {
      const thr = data.threshold != null ? ` · thr=${data.threshold}` : '';
      const ckpt = data.checkpoint_loaded ? data.checkpoint_loaded.replace('srnet_lite_', '').replace('.pt','') : '';
      modelBadge.innerHTML = '<i class="bi bi-cpu-fill me-1"></i>Modelo Entrenado Activo';
      modelBadge.className = 'badge bg-success';
      modelBadge.title     = `Checkpoint: ${data.checkpoint_loaded || ''}${thr}`;
      mockAlert.style.display = 'none';
    }
  } catch {
    modelBadge.innerHTML = '<i class="bi bi-wifi-off me-1"></i>Sin conexión';
    modelBadge.className = 'badge bg-secondary';
  }
}

/* ── Vincular eventos ─────────────────────────────────────────────────────── */
function bindEvents() {
  // Clic en zona de drop → abrir selector de archivo
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') fileInput.click();
  });

  // Selección manual de archivo
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) handleFileSelected(fileInput.files[0]);
  });

  // Drag & drop
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer?.files[0];
    if (file) handleFileSelected(file);
  });

  // Botón limpiar
  clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
  });

  // Botón analizar
  analyzeBtn.addEventListener('click', runAnalysis);

  // Botón PDF
  pdfBtn.addEventListener('click', downloadPdf);
}

/* ── Manejo de archivo ────────────────────────────────────────────────────── */
function handleFileSelected(file) {
  const allowed = ['image/png', 'image/jpeg'];
  if (!allowed.includes(file.type)) {
    showError('Tipo de archivo no permitido. Usa PNG, JPG o JPEG.');
    return;
  }

  selectedFile = file;
  analyzeBtn.disabled = false;
  showState('idle');

  // Mostrar información del archivo
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  fileInfo.style.display = 'flex';

  // Vista previa
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.style.display = 'block';
    dropPlaceholder.style.display = 'none';
  };
  reader.readAsDataURL(file);
}

function clearFile() {
  selectedFile = null;
  lastAnalysisId = null;
  fileInput.value = '';
  previewImg.src = '';
  previewImg.style.display = 'none';
  dropPlaceholder.style.display = 'flex';
  fileInfo.style.display = 'none';
  analyzeBtn.disabled = true;
  showState('idle');
}

/* ── Análisis ─────────────────────────────────────────────────────────────── */
async function runAnalysis() {
  if (!selectedFile) return;

  showState('loading');
  analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res = await fetch('/analyze', {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.detail || 'Error desconocido en el servidor.');
      return;
    }

    lastAnalysisId = data.id;
    displayResult(data);
  } catch (err) {
    showError('No se pudo conectar con el servidor. Verifica que la aplicación esté corriendo.');
  } finally {
    analyzeBtn.disabled = false;
  }
}

/* ── Mostrar resultado ────────────────────────────────────────────────────── */
function displayResult(data) {
  const isStego = data.prediction === 'stego';
  const prob    = data.probability_percent;

  // Badge y colores
  resultBadge.className = `result-badge mx-auto mb-2 ${data.prediction}`;
  resultIcon.className  = `bi ${isStego ? 'bi-exclamation-triangle-fill' : 'bi-check-circle-fill'}`;
  resultLabel.textContent   = data.label;
  resultLabel.style.color   = isStego ? 'var(--danger)' : 'var(--success)';
  resultFilename.textContent = data.filename;

  // Barra de probabilidad
  probBar.style.width    = `${prob}%`;
  probBar.className      = `progress-bar ${data.prediction}`;
  probText.textContent   = `${prob}%`;

  // Confianza
  const confidenceColors = {
    'Alta':  ['bg-success', 'Alta'],
    'Media': ['bg-warning text-dark', 'Media'],
    'Baja':  ['bg-danger', 'Baja'],
  };
  const [cls, label] = confidenceColors[data.confidence] || ['bg-secondary', data.confidence];
  confidenceBadge.className   = `badge ${cls}`;
  confidenceBadge.textContent = label;

  // Explicación
  explanationText.textContent = data.explanation;

  // Mock warning
  mockResultWarn.style.display = data.mock_mode ? 'block' : 'none';

  showState('result');
}

/* ── Descargar PDF ────────────────────────────────────────────────────────── */
async function downloadPdf() {
  if (!lastAnalysisId) return;

  pdfBtn.disabled = true;
  pdfBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Generando PDF...';

  try {
    const res = await fetch(`/report/${lastAnalysisId}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showError(err.detail || 'Error generando el reporte.');
      return;
    }

    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `reporte_esteganografia_${lastAnalysisId.slice(0, 8)}.pdf`;
    a.click();
    URL.revokeObjectURL(url);
  } catch {
    showError('No se pudo descargar el reporte PDF.');
  } finally {
    pdfBtn.disabled = false;
    pdfBtn.innerHTML = '<i class="bi bi-file-pdf me-2"></i>Descargar reporte PDF';
  }
}

/* ── Utilidades de UI ─────────────────────────────────────────────────────── */
function showState(state) {
  stateIdle.style.display    = state === 'idle'    ? 'block' : 'none';
  stateLoading.style.display = state === 'loading' ? 'block' : 'none';
  stateError.style.display   = state === 'error'   ? 'block' : 'none';
  stateResult.style.display  = state === 'result'  ? 'block' : 'none';
}

function showError(msg) {
  errorMessage.textContent = msg;
  showState('error');
}

function formatBytes(bytes) {
  if (bytes < 1024)        return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
