/**
 * app.js — Lógica del frontend de StegoDetect
 *
 * Tabs:
 *   1. Analizar imagen  → POST /analyze + POST /stego/full-analysis
 *   2. Ocultar          → POST /stego/embed/text  |  POST /stego/embed/file
 *   3. Extraer          → POST /stego/extract
 *   4. Guía técnica     → estática
 */

/* ── Inicialización ──────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  initAnalyzeTab();
  initEmbedTab();
  initExtractTab();
});

/* ═══════════════════════════════════════════════════════════════════════════
   HEALTH CHECK
═══════════════════════════════════════════════════════════════════════════ */
async function checkHealth() {
  const badge     = document.getElementById('model-status-badge');
  const mockAlert = document.getElementById('mock-alert');
  try {
    const data = await fetchJSON('/health');
    if (data.mock_mode) {
      badge.innerHTML  = '<i class="bi bi-flask me-1"></i>Modo Demostración';
      badge.className  = 'badge bg-warning text-dark';
      badge.title      = '';
      mockAlert.style.display = 'block';
    } else {
      const thr  = data.threshold != null ? ` · thr=${data.threshold}` : '';
      badge.innerHTML = '<i class="bi bi-cpu-fill me-1"></i>Modelo Entrenado Activo';
      badge.className = 'badge bg-success';
      badge.title     = `${data.checkpoint_loaded || ''}${thr}`;
      mockAlert.style.display = 'none';
    }
  } catch {
    badge.innerHTML = '<i class="bi bi-wifi-off me-1"></i>Sin conexión';
    badge.className = 'badge bg-secondary';
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   TAB 1 — ANALIZAR IMAGEN
═══════════════════════════════════════════════════════════════════════════ */
function initAnalyzeTab() {
  const dropZone    = document.getElementById('drop-zone');
  const fileInput   = document.getElementById('file-input');
  const previewImg  = document.getElementById('preview-img');
  const placeholder = document.getElementById('drop-placeholder');
  const fileInfo    = document.getElementById('file-info');
  const clearBtn    = document.getElementById('clear-btn');
  const analyzeBtn  = document.getElementById('analyze-btn');
  const pdfBtn      = document.getElementById('pdf-btn');
  const fullAnaBtn  = document.getElementById('full-analysis-btn');

  let selectedFile    = null;
  let lastAnalysisId  = null;
  let lastFullData    = null;

  // ── Drop zone ────────────────────────────────────────────────────────────
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') fileInput.click();
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault(); dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault(); dropZone.classList.remove('drag-over');
    if (e.dataTransfer?.files[0]) handleFile(e.dataTransfer.files[0]);
  });
  clearBtn.addEventListener('click', (e) => { e.stopPropagation(); clearFile(); });

  function handleFile(file) {
    if (!isImageFile(file)) { showAnalyzeError('Tipo no permitido. Usa PNG, JPG o JPEG.'); return; }
    selectedFile = file;
    analyzeBtn.disabled = false;
    showAnalyzeState('idle');
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = formatBytes(file.size);
    fileInfo.style.display = 'flex';
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewImg.style.display = 'block';
      placeholder.style.display = 'none';
    };
    reader.readAsDataURL(file);
  }

  function clearFile() {
    selectedFile = lastAnalysisId = null;
    fileInput.value = '';
    previewImg.src = '';
    previewImg.style.display = 'none';
    placeholder.style.display = 'flex';
    fileInfo.style.display = 'none';
    analyzeBtn.disabled = true;
    showAnalyzeState('idle');
    document.getElementById('full-analysis-result').style.display = 'none';
  }

  // ── Análisis ML ───────────────────────────────────────────────────────────
  analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    showAnalyzeState('loading');
    analyzeBtn.disabled = true;
    document.getElementById('full-analysis-result').style.display = 'none';

    try {
      const fd = new FormData();
      fd.append('file', selectedFile);
      const data = await fetchJSON('/analyze', { method: 'POST', body: fd });
      lastAnalysisId = data.id;
      displayAnalyzeResult(data);
    } catch (err) {
      showAnalyzeError(err.message || 'Error desconocido.');
    } finally {
      analyzeBtn.disabled = false;
    }
  });

  function displayAnalyzeResult(data) {
    const isStego = data.prediction === 'stego';
    const prob    = data.probability_percent;

    document.getElementById('result-badge').className =
      `result-badge mx-auto mb-2 ${data.prediction}`;
    document.getElementById('result-icon').className =
      `bi ${isStego ? 'bi-exclamation-triangle-fill' : 'bi-check-circle-fill'}`;
    document.getElementById('result-label').textContent  = data.label;
    document.getElementById('result-label').style.color  = isStego ? 'var(--danger)' : 'var(--success)';
    document.getElementById('result-filename').textContent = data.filename;

    const probBar  = document.getElementById('prob-bar');
    probBar.style.width = `${prob}%`;
    probBar.className   = `progress-bar ${data.prediction}`;
    document.getElementById('prob-text').textContent = `${prob}%`;

    const confMap = {
      'Alta':  ['bg-success', 'Alta'],
      'Media': ['bg-warning text-dark', 'Media'],
      'Baja':  ['bg-danger', 'Baja'],
    };
    const [cls, lbl] = confMap[data.confidence] || ['bg-secondary', data.confidence];
    const cb = document.getElementById('confidence-badge');
    cb.className   = `badge ${cls}`;
    cb.textContent = lbl;

    document.getElementById('explanation-text').textContent = data.explanation;
    document.getElementById('mock-result-warning').style.display =
      data.mock_mode ? 'block' : 'none';

    showAnalyzeState('result');
  }

  // ── PDF ───────────────────────────────────────────────────────────────────
  pdfBtn.addEventListener('click', async () => {
    if (!lastAnalysisId) return;
    pdfBtn.disabled  = true;
    pdfBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Generando...';
    try {
      const res = await fetch(`/report/${lastAnalysisId}`);
      if (!res.ok) throw new Error('Error generando el reporte.');
      triggerDownload(await res.blob(), `reporte_${lastAnalysisId.slice(0,8)}.pdf`);
    } catch (e) { showAnalyzeError(e.message); }
    finally {
      pdfBtn.disabled  = false;
      pdfBtn.innerHTML = '<i class="bi bi-file-pdf me-2"></i>Reporte PDF';
    }
  });

  // ── Análisis LSB completo ─────────────────────────────────────────────────
  fullAnaBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    fullAnaBtn.disabled  = true;
    fullAnaBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analizando LSB...';

    try {
      const fd = new FormData();
      fd.append('image', selectedFile);
      const data = await fetchJSON('/stego/full-analysis', { method: 'POST', body: fd });
      lastFullData = data;
      displayFullAnalysis(data);
    } catch (e) { showAnalyzeError(e.message); }
    finally {
      fullAnaBtn.disabled  = false;
      fullAnaBtn.innerHTML = '<i class="bi bi-layers me-2"></i>Análisis LSB completo';
    }
  });

  function displayFullAnalysis(data) {
    const wrap = document.getElementById('full-analysis-result');
    wrap.style.display = 'block';

    // Stats por canal
    const grid = document.getElementById('lsb-stats-grid');
    grid.innerHTML = '';
    const lsb = data.lsb_analysis || {};
    const chStats = lsb.channel_stats || {};
    ['R','G','B'].forEach(ch => {
      const s = chStats[ch];
      if (!s) return;
      const color = ch === 'R' ? '#ef4444' : ch === 'G' ? '#22c55e' : '#3b82f6';
      grid.innerHTML += `
        <div class="col-4">
          <div class="stat-chip text-center p-2 rounded">
            <div class="fw-bold" style="color:${color}">Canal ${ch}</div>
            <div class="small text-muted">0s: ${s.zeros.toLocaleString()}</div>
            <div class="small text-muted">1s: ${s.ones.toLocaleString()}</div>
            <div class="small">ratio 1s: <strong>${(s.ratio_ones*100).toFixed(1)}%</strong></div>
            <div class="small text-muted">H: ${s.entropy}</div>
          </div>
        </div>`;
    });

    // Badge cabecera
    const headerBadge = document.getElementById('lsb-header-badge');
    if (lsb.has_system_header) {
      headerBadge.innerHTML = `<span class="badge bg-success fs-6">
        <i class="bi bi-check-circle-fill me-1"></i>Cabecera StegoDetect detectada</span>`;
    } else {
      headerBadge.innerHTML = `<span class="badge bg-secondary">
        <i class="bi bi-x-circle me-1"></i>Sin cabecera del sistema</span>`;
    }

    // Nota de aleatoriedad
    document.getElementById('lsb-randomness').textContent =
      lsb.randomness_note || '';

    // Extracción de payload
    const ext        = data.payload_extraction || {};
    const foundBlock = document.getElementById('payload-found-block');
    const notBlock   = document.getElementById('payload-not-found-block');

    if (ext.payload_found) {
      foundBlock.style.display = 'block';
      notBlock.style.display   = 'none';

      document.getElementById('payload-meta').innerHTML = buildPayloadMetaHTML(ext);

      // Texto extraído
      const textBlock = document.getElementById('extracted-text-block');
      if (ext.payload_type === 'text' && ext.message_text != null) {
        textBlock.style.display = 'block';
        document.getElementById('extracted-text').textContent = ext.message_text;
      } else {
        textBlock.style.display = 'none';
      }

      // Descarga archivo extraído
      const dlBtn = document.getElementById('download-extracted-btn');
      if (ext.download_payload_url) {
        dlBtn.style.display = 'inline-block';
        dlBtn.onclick = () => window.open(ext.download_payload_url, '_blank');
      } else {
        dlBtn.style.display = 'none';
      }

      // CSV posiciones
      const csvBtn = document.getElementById('download-csv-full-btn');
      if (ext.positions_summary) {
        csvBtn.style.display = 'inline-block';
        // El CSV es del artifact_id de la extracción si existe
      } else {
        csvBtn.style.display = 'none';
      }

    } else {
      foundBlock.style.display = 'none';
      notBlock.style.display   = 'block';
    }
  }

  // ── Estados ───────────────────────────────────────────────────────────────
  function showAnalyzeState(state) {
    ['idle','loading','error','result'].forEach(s => {
      document.getElementById(`state-${s}`).style.display = s === state ? 'block' : 'none';
    });
  }
  function showAnalyzeError(msg) {
    document.getElementById('error-message').textContent = msg;
    showAnalyzeState('error');
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   TAB 2 — OCULTAR INFORMACIÓN
═══════════════════════════════════════════════════════════════════════════ */
function initEmbedTab() {
  const coverInput    = document.getElementById('embed-cover-input');
  const modeRadios    = document.querySelectorAll('input[name="embed-mode"]');
  const textBlock     = document.getElementById('embed-text-block');
  const fileBlock     = document.getElementById('embed-file-block');
  const msgArea       = document.getElementById('embed-message');
  const payloadInput  = document.getElementById('embed-payload-input');
  const embedBtn      = document.getElementById('embed-btn');
  const bitsSelect    = document.getElementById('embed-bits');
  const chSelect      = document.getElementById('embed-channels');
  const capInfo       = document.getElementById('embed-capacity-info');
  const textSizeInfo  = document.getElementById('embed-text-size');

  let currentMode = 'text';
  let lastEmbedResult = null;

  // ── Cambio de modo ────────────────────────────────────────────────────────
  modeRadios.forEach(r => r.addEventListener('change', () => {
    currentMode = r.value;
    textBlock.style.display = currentMode === 'text' ? 'block' : 'none';
    fileBlock.style.display = currentMode === 'file' ? 'block' : 'none';
    checkEmbedReady();
  }));

  // ── Capacidad al seleccionar imagen ──────────────────────────────────────
  coverInput.addEventListener('change', async () => {
    checkEmbedReady();
    const file = coverInput.files[0];
    if (!file) { capInfo.style.display = 'none'; return; }
    try {
      const fd = new FormData();
      fd.append('cover_image', file);
      fd.append('bits_per_channel', bitsSelect.value);
      fd.append('channels', chSelect.value);
      // Use embed/text with tiny payload just to get capacity - actually call calculate separately
      // We'll show dimensions from the file instead
      const img = new Image();
      img.onload = () => {
        const ch = chSelect.value.length;
        const bits = parseInt(bitsSelect.value);
        const usable = Math.floor((img.width * img.height * ch * bits) / 8) - 217;
        capInfo.innerHTML =
          `<i class="bi bi-info-circle me-1"></i>Imagen ${img.width}×${img.height}px · ` +
          `Capacidad útil: <strong>~${formatBytes(Math.max(0, usable))}</strong>`;
        capInfo.style.display = 'block';
      };
      img.src = URL.createObjectURL(file);
    } catch {}
  });

  // ── Tamaño del mensaje ────────────────────────────────────────────────────
  msgArea.addEventListener('input', () => {
    const size = new TextEncoder().encode(msgArea.value).length;
    textSizeInfo.textContent = `${size} bytes`;
    checkEmbedReady();
  });

  payloadInput.addEventListener('change', checkEmbedReady);
  bitsSelect.addEventListener('change', () => { coverInput.dispatchEvent(new Event('change')); });
  chSelect.addEventListener('change', () => { coverInput.dispatchEvent(new Event('change')); });

  function checkEmbedReady() {
    const hasCover   = coverInput.files.length > 0;
    const hasPayload = currentMode === 'text'
      ? msgArea.value.trim().length > 0
      : payloadInput.files.length > 0;
    embedBtn.disabled = !(hasCover && hasPayload);
  }

  // ── Generar imagen stego ──────────────────────────────────────────────────
  embedBtn.addEventListener('click', async () => {
    if (embedBtn.disabled) return;
    showEmbedState('loading');
    embedBtn.disabled = true;

    try {
      const fd = new FormData();
      fd.append('cover_image', coverInput.files[0]);
      fd.append('bits_per_channel', bitsSelect.value);
      fd.append('channels', chSelect.value);

      let url;
      if (currentMode === 'text') {
        fd.append('message', msgArea.value);
        url = '/stego/embed/text';
      } else {
        fd.append('payload_file', payloadInput.files[0]);
        url = '/stego/embed/file';
      }

      const data = await fetchJSON(url, { method: 'POST', body: fd });
      lastEmbedResult = data;
      displayEmbedResult(data);
    } catch (err) {
      showEmbedError(err.message || 'Error en la inserción.');
    } finally {
      embedBtn.disabled = false;
    }
  });

  function displayEmbedResult(data) {
    showEmbedState('result');

    // Métricas rápidas
    const cap = data.capacity || {};
    const pos = data.positions_summary || {};
    const pld = data.payload || {};
    document.getElementById('embed-metrics').innerHTML = `
      <div class="col-6"><div class="stat-chip p-2 rounded text-center">
        <div class="small text-muted">Imagen</div>
        <div class="fw-bold small">${cap.width||'?'}×${cap.height||'?'}px</div>
      </div></div>
      <div class="col-6"><div class="stat-chip p-2 rounded text-center">
        <div class="small text-muted">Payload</div>
        <div class="fw-bold small">${formatBytes(pld.size||0)}</div>
      </div></div>
      <div class="col-6"><div class="stat-chip p-2 rounded text-center">
        <div class="small text-muted">Píxeles usados</div>
        <div class="fw-bold small">${(pos.total_pixels_used||0).toLocaleString()}</div>
      </div></div>
      <div class="col-6"><div class="stat-chip p-2 rounded text-center">
        <div class="small text-muted">Capacidad usada</div>
        <div class="fw-bold small">${pos.capacity_used_pct||0}%</div>
      </div></div>`;

    // Botones descarga
    document.getElementById('dl-stego-btn').onclick = () =>
      window.open(data.download_url, '_blank');
    document.getElementById('dl-csv-btn').onclick = () =>
      window.open(data.csv_url, '_blank');
    document.getElementById('dl-map-btn').onclick = () =>
      window.open(data.map_url, '_blank');

    // Resumen técnico
    const tech = data.technical || {};
    document.getElementById('embed-tech-summary').innerHTML = `
      <div class="mb-1"><strong>Algoritmo:</strong> ${tech.algorithm || 'N/D'}</div>
      <div class="mb-1"><strong>Bits por canal:</strong> ${tech.bits_per_channel}</div>
      <div class="mb-1"><strong>Canales:</strong> ${(tech.channels||[]).join(', ')}</div>
      <div class="mb-1"><strong>Bits totales insertados:</strong> ${(tech.total_bits_embedded||0).toLocaleString()}</div>
      <div class="mb-1"><strong>Tamaño cabecera:</strong> ${tech.header_size_bytes} bytes</div>
      <div class="mb-0"><strong>SHA-256:</strong> <span class="font-monospace">${pld.sha256||''}</span></div>`;

    // Tabla de primeras posiciones
    renderPositionsTable('positions-table-wrap', data.first_positions || []);
  }

  function showEmbedState(state) {
    ['idle','loading','error','result'].forEach(s => {
      document.getElementById(`embed-state-${s}`).style.display = s === state ? 'block' : 'none';
    });
  }
  function showEmbedError(msg) {
    document.getElementById('embed-error-msg').textContent = msg;
    showEmbedState('error');
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   TAB 3 — EXTRAER INFORMACIÓN
═══════════════════════════════════════════════════════════════════════════ */
function initExtractTab() {
  const extractInput = document.getElementById('extract-input');
  const extractBtn   = document.getElementById('extract-btn');
  const bitsSelect   = document.getElementById('extract-bits');
  const chSelect     = document.getElementById('extract-channels');

  let lastExtractResult = null;

  extractInput.addEventListener('change', () => {
    extractBtn.disabled = extractInput.files.length === 0;
  });

  extractBtn.addEventListener('click', async () => {
    if (!extractInput.files[0]) return;
    showExtractState('loading');
    extractBtn.disabled = true;

    try {
      const fd = new FormData();
      fd.append('stego_image', extractInput.files[0]);
      fd.append('bits_per_channel', bitsSelect.value);
      fd.append('channels', chSelect.value);
      const data = await fetchJSON('/stego/extract', { method: 'POST', body: fd });
      lastExtractResult = data;
      displayExtractResult(data);
    } catch (err) {
      document.getElementById('extract-error-msg').textContent =
        err.message || 'Error en extracción.';
      showExtractState('error');
    } finally {
      extractBtn.disabled = false;
    }
  });

  function displayExtractResult(data) {
    showExtractState('result');

    const foundBlock = document.getElementById('extract-found');
    const notBlock   = document.getElementById('extract-not-found');

    if (!data.payload_found) {
      foundBlock.style.display = 'none';
      notBlock.style.display   = 'block';
      // Mostrar análisis LSB
      const lsb = data.lsb_analysis || {};
      document.getElementById('extract-lsb-info').innerHTML =
        buildLSBInfoHTML(lsb);
      return;
    }

    foundBlock.style.display = 'block';
    notBlock.style.display   = 'none';

    // Badge SHA-256
    const shaBadge = document.getElementById('extract-sha-badge');
    shaBadge.textContent = data.sha256_valid ? '✓ SHA-256 OK' : '⚠ SHA-256 FALLO';
    shaBadge.className   = `badge ${data.sha256_valid ? 'bg-success' : 'bg-danger'}`;

    // Metadatos
    document.getElementById('extract-meta').innerHTML = `
      <div class="col-6"><div class="stat-chip p-2 rounded">
        <div class="small text-muted">Tipo</div>
        <div class="fw-bold small">${data.payload_type || 'desconocido'}</div>
      </div></div>
      <div class="col-6"><div class="stat-chip p-2 rounded">
        <div class="small text-muted">Nombre</div>
        <div class="fw-bold small">${data.filename || '(sin nombre)'}</div>
      </div></div>
      <div class="col-6"><div class="stat-chip p-2 rounded">
        <div class="small text-muted">Tamaño</div>
        <div class="fw-bold small">${formatBytes(data.payload_size||0)}</div>
      </div></div>
      <div class="col-6"><div class="stat-chip p-2 rounded">
        <div class="small text-muted">Algoritmo</div>
        <div class="fw-bold small">${data.algorithm || 'N/D'}</div>
      </div></div>`;

    // Texto
    const textBlock = document.getElementById('extract-text-block');
    if (data.payload_type === 'text' && data.message_text != null) {
      textBlock.style.display = 'block';
      document.getElementById('extract-text-content').textContent = data.message_text;
    } else {
      textBlock.style.display = 'none';
    }

    // Descarga archivo
    const fileBlock = document.getElementById('extract-file-block');
    const dlBtn     = document.getElementById('extract-dl-payload-btn');
    if (data.download_payload_url) {
      fileBlock.style.display = 'block';
      dlBtn.onclick = () => window.open(data.download_payload_url, '_blank');
    } else {
      fileBlock.style.display = 'none';
    }

    // Posiciones
    const ps = data.positions_summary || {};
    document.getElementById('extract-positions-summary').innerHTML = `
      <strong>Posiciones usadas:</strong> ${(ps.total_pixels_used||0).toLocaleString()} píxeles ·
      ${(ps.total_bits_used||0).toLocaleString()} bits ·
      Canales: ${(ps.channels_used||[]).join(', ')}`;

    renderPositionsTable('extract-positions-table', data.first_positions || []);
  }

  function showExtractState(state) {
    ['idle','loading','error','result'].forEach(s => {
      document.getElementById(`extract-state-${s}`).style.display = s === state ? 'block' : 'none';
    });
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   HELPERS COMPARTIDOS
═══════════════════════════════════════════════════════════════════════════ */

async function fetchJSON(url, opts = {}) {
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a   = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

function isImageFile(file) {
  return ['image/png', 'image/jpeg'].includes(file.type);
}

function formatBytes(bytes) {
  if (bytes < 1024)        return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function buildPayloadMetaHTML(ext) {
  return `
    <div class="mb-1"><strong>Tipo:</strong> ${ext.payload_type}</div>
    <div class="mb-1"><strong>Nombre:</strong> ${ext.filename || '(sin nombre)'}</div>
    <div class="mb-1"><strong>Tamaño:</strong> ${formatBytes(ext.payload_size||0)}</div>
    <div class="mb-1"><strong>MIME:</strong> ${ext.mime_type || 'N/D'}</div>
    <div class="mb-1"><strong>Algoritmo:</strong> ${ext.algorithm || 'N/D'}</div>
    <div class="mb-0"><strong>SHA-256:</strong>
      <span class="badge ${ext.sha256_valid ? 'bg-success' : 'bg-danger'} ms-1">
        ${ext.sha256_valid ? 'Válido' : 'No coincide'}
      </span>
    </div>`;
}

function buildLSBInfoHTML(lsb) {
  const cap  = lsb.capacity_estimate || {};
  const dims = lsb.width ? `${lsb.width}×${lsb.height}px` : '?';
  return `<strong>Dimensiones:</strong> ${dims} · ` +
    `<strong>Capacidad estimada:</strong> ${cap.kb_available || '?'} KB · ` +
    `<span>${lsb.randomness_note || ''}</span>`;
}

function renderPositionsTable(containerId, positions) {
  const wrap = document.getElementById(containerId);
  if (!wrap || !positions.length) {
    if (wrap) wrap.innerHTML = '<p class="small text-muted mb-0">Sin posiciones.</p>';
    return;
  }
  const rows = positions.slice(0, 100).map(p =>
    `<tr><td>${p.bit_index}</td><td>${p.x},${p.y}</td><td>${p.channel}</td><td>${p.payload_bit ?? ''}</td></tr>`
  ).join('');
  wrap.innerHTML = `
    <table class="table table-sm table-bordered small positions-table mb-0">
      <thead class="table-light">
        <tr><th>Bit#</th><th>x,y</th><th>Canal</th><th>Bit</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}
