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

  // ── Analizar: ML + LSB integrado en un solo clic ────────────────────────────
  //
  // Flujo:
  //   1. POST /analyze          → guarda resultado y genera ID para PDF
  //   2. POST /stego/full-analysis → análisis integrado ML + LSB + extracción
  //   Ambas llamadas se hacen en paralelo para reducir latencia.
  //
  // Lógica de decisión (prioridad):
  //   Caso A payload_found + sha256_valid  → "Mensaje oculto encontrado"    (LSB gana)
  //   Caso B ML stego, sin cabecera        → "Posible mensaje oculto"       (ML avisa)
  //   Caso C sin evidencia                 → "Sin evidencia detectable"     (ninguno)
  // ─────────────────────────────────────────────────────────────────────────────

  analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    showAnalyzeState('loading');
    analyzeBtn.disabled = true;
    document.getElementById('full-analysis-result').style.display = 'none';

    try {
      const fd1 = new FormData(); fd1.append('file',  selectedFile);
      const fd2 = new FormData(); fd2.append('image', selectedFile);

      const [mlData, fullData] = await Promise.all([
        fetchJSON('/analyze',               { method: 'POST', body: fd1 }),
        fetchJSON('/stego/full-analysis',   { method: 'POST', body: fd2 }),
      ]);

      lastAnalysisId = mlData.id;
      lastFullData   = fullData;
      displayIntegratedResult(mlData, fullData);
    } catch (err) {
      showAnalyzeError(err.message || 'Error desconocido.');
    } finally {
      analyzeBtn.disabled = false;
    }
  });

  /**
   * displayIntegratedResult — Resultado combinado ML + extracción LSB.
   *
   * El veredicto visible depende de final_decision.status (calculado en el backend):
   *   "payload_found"  → badge lila, "Mensaje oculto encontrado"
   *   "ml_suspicious"  → badge rojo, "Posible mensaje oculto detectado"
   *   "no_evidence"    → badge verde, "Sin evidencia detectable"
   *
   * El ML nunca sobreescribe la evidencia LSB directa.
   */
  function displayIntegratedResult(mlData, fullData) {
    // ── Debug logs (thesis diagnostics) ───────────────────────────────────────
    console.log('FULL_ANALYSIS_RESPONSE', fullData);
    console.log('FINAL_DECISION', fullData.final_decision);
    console.log('PAYLOAD_EXTRACTION', fullData.payload_extraction);
    console.log('MODEL_APPLICABILITY', fullData.model_applicability);
    console.log('RELIABILITY', fullData.reliability);
    // ──────────────────────────────────────────────────────────────────────────

    const decision      = fullData.final_decision      || {};
    const extraction    = fullData.payload_extraction  || {};
    const applicability = fullData.model_applicability || {};
    const reliability   = fullData.reliability         || {};
    const status        = decision.status || 'no_evidence';

    // ── Configuración visual por status ──────────────────────────────────────
    // Cuatro status activos + dos alias para compatibilidad con PDFs viejos.
    const statusCfg = {
      // Caso A: evidencia LSB directa (cabecera + SHA-256 válido)
      payload_found: {
        badgeCls: 'stego-lsb',
        icon:     'bi-lock-fill',
        color:    '#7c3aed',
      },
      // Caso B: ML ≥ 0.46 — puntaje supera el threshold
      ml_suspicious: {
        badgeCls: 'stego',
        icon:     'bi-exclamation-triangle-fill',
        color:    'var(--danger)',
      },
      // Caso C: 0.30 ≤ ML < 0.46 — señal débil, no concluyente
      inconclusive_low_ml: {
        badgeCls: 'unverified',
        icon:     'bi-question-circle-fill',
        color:    '#d97706',
      },
      // Caso D: ML < 0.30 — sin evidencia esteganográfica detectable
      no_stego_evidence: {
        badgeCls: 'cover',
        icon:     'bi-shield-check-fill',
        color:    'var(--success)',
      },
      // Alias de compatibilidad (status legacy de PDFs anteriores)
      ml_suspicious_unverified: {
        badgeCls: 'unverified',
        icon:     'bi-question-circle-fill',
        color:    '#d97706',
      },
      no_evidence: {
        badgeCls: 'cover',
        icon:     'bi-shield-check-fill',
        color:    'var(--success)',
      },
    };
    const cfg = statusCfg[status] || statusCfg.no_stego_evidence;

    // ── Badge + título ────────────────────────────────────────────────────────
    document.getElementById('result-badge').className =
      `result-badge mx-auto mb-2 ${cfg.badgeCls}`;
    document.getElementById('result-icon').className  = `bi ${cfg.icon}`;

    const labelEl = document.getElementById('result-label');
    labelEl.textContent = decision.title || 'Análisis completado';
    labelEl.style.color = cfg.color;

    document.getElementById('result-filename').textContent = mlData.filename;

    // ── Métrica principal: cambia según el status ────────────────────────────
    // payload_found     → Certeza de extracción LSB = 100% (ML es secundario)
    // ml_suspicious     → Puntaje ML (supera threshold)
    // inconclusive_low_ml → Puntaje ML (señal débil)
    // no_stego_evidence → Puntaje ML (bajo)
    const prob       = mlData.probability_percent;
    const probBar    = document.getElementById('prob-bar');
    const probText   = document.getElementById('prob-text');
    const primaryLbl = document.getElementById('primary-metric-label');
    const relLbl     = document.getElementById('reliability-label');
    const cb         = document.getElementById('confidence-badge');
    const domainRow  = document.getElementById('domain-row');
    const mlSecBlock = document.getElementById('ml-secondary-block');
    const mlSecScore = document.getElementById('ml-secondary-score');
    const sha_ok     = !!extraction.sha256_valid;

    // El label de la métrica principal viene del backend cuando está disponible
    const backendLabel = decision.primary_metric_label || null;
    const backendValue = decision.primary_metric_value != null ? decision.primary_metric_value : null;

    if (status === 'payload_found' && sha_ok) {
      // Caso A: evidencia directa LSB → métrica = certeza 100%
      primaryLbl.textContent = backendLabel || 'Certeza de extracción LSB';
      probBar.style.width    = '100%';
      probBar.className      = 'progress-bar bg-success';
      probText.textContent   = '100%';

      relLbl.textContent = 'Confiabilidad';
      cb.className       = 'badge bg-success';
      cb.textContent     = 'Alta — payload validado';
      cb.title           = 'Cabecera STEGODETECTv1 detectada y SHA-256 verificado';

      domainRow.style.display = 'none';

      mlSecBlock.style.display = 'block';
      mlSecScore.textContent   = `${prob}%`;
    } else {
      // Casos B/C/D: métrica principal = puntaje ML
      primaryLbl.textContent = backendLabel || 'Puntaje ML de esteganografía';
      const displayValue = backendValue != null ? backendValue : prob;
      probBar.style.width    = `${displayValue}%`;
      probBar.className      = `progress-bar ${mlData.prediction}`;
      probText.textContent   = `${displayValue}%`;

      relLbl.textContent = 'Fiabilidad de interpretación';
      const relMap = {
        high:   'bg-success',
        medium: 'bg-warning text-dark',
        low:    'bg-danger',
      };
      cb.className   = `badge ${relMap[reliability.level] || 'bg-secondary'}`;
      cb.textContent = reliability.label || 'No determinada';
      if (reliability.tooltip) cb.title = reliability.tooltip;

      domainRow.style.display = '';
      const domainMap = {
        in_domain:               ['bg-success', 'Dentro del dominio'],
        possibly_out_of_domain:  ['bg-warning text-dark', 'Parcialmente compatible'],
        out_of_domain:           ['bg-danger', 'Fuera del dominio'],
      };
      const [domCls, domLbl] = domainMap[applicability.domain_status] || ['bg-secondary', '—'];
      const db = document.getElementById('domain-badge');
      db.className   = `badge ${domCls}`;
      db.textContent = domLbl;

      mlSecBlock.style.display = 'none';
    }

    // ── Tipo de evidencia (nuevo campo del backend) ────────────────────────
    const evidenceTypeEl = document.getElementById('evidence-type-text');
    if (evidenceTypeEl && decision.evidence_type) {
      evidenceTypeEl.textContent = decision.evidence_type;
      evidenceTypeEl.parentElement.style.display = '';
    } else if (evidenceTypeEl) {
      evidenceTypeEl.parentElement.style.display = 'none';
    }

    // ── Banda de evidencia ML ──────────────────────────────────────────────
    const mlBandEl = document.getElementById('ml-band-badge');
    if (mlBandEl) {
      const bandMap = {
        low:          ['bg-success text-white', 'Banda ML: baja (<30%)'],
        intermediate: ['bg-warning text-dark',  'Banda ML: intermedia (30–46%)'],
        suspicious:   ['bg-danger text-white',   'Banda ML: sospechosa (≥46%)'],
      };
      const band = decision.ml_evidence_band || 'low';
      const [bandCls, bandLbl] = bandMap[band] || ['bg-secondary', band];
      mlBandEl.className   = `badge ${bandCls}`;
      mlBandEl.textContent = bandLbl;
      mlBandEl.style.display = '';
    }

    // ── Explicación integrada (honesta — sin "0% probabilidad que no…") ─────
    const explanationEl = document.getElementById('explanation-text');
    // Primero intentamos el campo `explanation` (más detallado), luego `summary`
    const explanationText = decision.explanation || decision.summary || '';
    if (status === 'payload_found' && sha_ok) {
      explanationEl.textContent = explanationText ||
        'Se encontró y validó información oculta mediante extracción LSB del sistema. ' +
        'La presencia del payload fue confirmada mediante cabecera STEGODETECTv1 y ' +
        'verificación SHA-256. Esta evidencia directa tiene prioridad sobre el puntaje ML.';
    } else {
      explanationEl.textContent = explanationText;
    }

    document.getElementById('mock-result-warning').style.display =
      mlData.mock_mode ? 'block' : 'none';

    showAnalyzeState('result');

    // Mostrar detalles técnicos LSB automáticamente (ya tenemos los datos)
    displayFullAnalysis(fullData);

    // ── Razones del juicio de dominio (sección técnica) ──────────────────────
    const reasonsBlock = document.getElementById('domain-reasons-block');
    const reasonsList  = document.getElementById('domain-reasons-list');
    if (reasonsBlock && reasonsList) {
      const reasons = applicability.reasons || [];
      reasonsList.innerHTML = '';
      reasons.forEach(r => {
        const li = document.createElement('li');
        li.textContent = r;
        reasonsList.appendChild(li);
      });
      reasonsBlock.style.display = reasons.length ? 'block' : 'none';
    }
  }

  // ── PDF integrado ─────────────────────────────────────────────────────────
  // Usa /stego/report/{id} (decisión integrada) en lugar de /report/{id}
  // (ML-only). El ID viene del resultado de /stego/full-analysis.
  pdfBtn.addEventListener('click', async () => {
    const integratedId = lastFullData && lastFullData.id;
    if (!integratedId) {
      showAnalyzeError('No hay análisis integrado disponible. Analiza una imagen primero.');
      return;
    }
    pdfBtn.disabled  = true;
    pdfBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Generando...';
    try {
      const res = await fetch(`/stego/report/${integratedId}`);
      if (!res.ok) throw new Error('Error generando el reporte integrado.');
      triggerDownload(await res.blob(), `reporte_integrado_${integratedId.slice(0,8)}.pdf`);
    } catch (e) { showAnalyzeError(e.message); }
    finally {
      pdfBtn.disabled  = false;
      pdfBtn.innerHTML = '<i class="bi bi-file-pdf me-2"></i>Reporte integrado PDF';
    }
  });

  // ── "Ver detalles técnicos" — hace scroll a la sección ya poblada ──────────
  fullAnaBtn.addEventListener('click', () => {
    const wrap = document.getElementById('full-analysis-result');
    if (wrap && wrap.style.display !== 'none') {
      wrap.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });

  // ── displayFullAnalysis — rellena la sección de detalles técnicos LSB ──────
  function displayFullAnalysis(data) {
    const wrap = document.getElementById('full-analysis-result');
    wrap.style.display = 'block';

    // Estadísticas por canal
    const grid    = document.getElementById('lsb-stats-grid');
    grid.innerHTML = '';
    const lsb     = data.lsb_analysis || {};
    const chStats = lsb.channel_stats || {};
    ['R', 'G', 'B'].forEach(ch => {
      const s     = chStats[ch];
      if (!s) return;
      const color = ch === 'R' ? '#ef4444' : ch === 'G' ? '#22c55e' : '#3b82f6';
      grid.innerHTML += `
        <div class="col-4">
          <div class="stat-chip text-center p-2 rounded">
            <div class="fw-bold" style="color:${color}">Canal ${ch}</div>
            <div class="small text-muted">0s: ${s.zeros.toLocaleString()}</div>
            <div class="small text-muted">1s: ${s.ones.toLocaleString()}</div>
            <div class="small">ratio 1s: <strong>${(s.ratio_ones * 100).toFixed(1)}%</strong></div>
            <div class="small text-muted">H: ${s.entropy}</div>
          </div>
        </div>`;
    });

    // Badge: cabecera del sistema
    const headerBadge = document.getElementById('lsb-header-badge');
    headerBadge.innerHTML = lsb.has_system_header
      ? `<span class="badge bg-success fs-6">
           <i class="bi bi-check-circle-fill me-1"></i>Cabecera StegoDetect detectada
         </span>`
      : `<span class="badge bg-secondary">
           <i class="bi bi-x-circle me-1"></i>Sin cabecera del sistema
         </span>`;

    // Nota de aleatoriedad
    document.getElementById('lsb-randomness').textContent = lsb.randomness_note || '';

    // Bloque de extracción
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
  // Muestra capacidad teórica de la imagen Y límite del sistema (2 MB).
  // Recomendación de modo: 1 bit/canal es el más imperceptible (modo estándar),
  // 2–4 bits/canal aumenta capacidad pero reduce la invisibilidad.
  const SYSTEM_LIMIT_BYTES = 2 * 1024 * 1024; // MAX_PAYLOAD_BYTES del backend

  function updateCapacityInfo() {
    const file = coverInput.files[0];
    if (!file) { capInfo.style.display = 'none'; return; }
    const img = new Image();
    img.onload = () => {
      const ch        = chSelect.value.length;
      const bits      = parseInt(bitsSelect.value);
      const HEADER_OH = 217; // overhead cabecera STEGODETECTv1
      const theoretical = Math.floor((img.width * img.height * ch * bits) / 8);
      const usable      = Math.max(0, theoretical - HEADER_OH);
      const effective   = Math.min(usable, SYSTEM_LIMIT_BYTES);

      const modeHint = bits === 1
        ? '<span class="text-success fw-semibold">Modo estándar (1 bit/canal)</span> — recomendado para máxima imperceptibilidad'
        : `<span class="text-warning fw-semibold">Modo avanzado (${bits} bits/canal)</span> — mayor capacidad, menor imperceptibilidad`;

      capInfo.innerHTML =
        `<i class="bi bi-info-circle me-1"></i>` +
        `Imagen <strong>${img.width}×${img.height}px</strong> · ` +
        `Capacidad teórica: <strong>${formatBytes(theoretical)}</strong> · ` +
        `Límite sistema: <strong>${formatBytes(SYSTEM_LIMIT_BYTES)}</strong> · ` +
        `Disponible: <strong>${formatBytes(effective)}</strong><br>` +
        `<small class="text-muted">${modeHint}</small>`;
      capInfo.style.display = 'block';
      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(file);
  }

  coverInput.addEventListener('change', () => { checkEmbedReady(); updateCapacityInfo(); });

  // ── Tamaño del mensaje ────────────────────────────────────────────────────
  msgArea.addEventListener('input', () => {
    const size = new TextEncoder().encode(msgArea.value).length;
    textSizeInfo.textContent = `${size} bytes`;
    checkEmbedReady();
  });

  payloadInput.addEventListener('change', checkEmbedReady);
  bitsSelect.addEventListener('change', () => { checkEmbedReady(); updateCapacityInfo(); });
  chSelect.addEventListener('change',   () => { checkEmbedReady(); updateCapacityInfo(); });

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

    // Densidad de inserción (métricas de trazabilidad forense)
    const density = data.insertion_density || {};
    const densityWrap = document.getElementById('embed-density-wrap');
    if (densityWrap && density.total_pixels_image) {
      densityWrap.style.display = 'block';
      densityWrap.innerHTML = `
        <div class="small fw-semibold text-muted mb-1">Densidad de inserción LSB</div>
        <div class="row g-2">
          <div class="col-6"><div class="stat-chip p-2 rounded text-center">
            <div class="small text-muted">Píxeles usados</div>
            <div class="fw-bold small">${(density.used_pixels||0).toLocaleString()}</div>
            <div class="small text-muted">${density.used_pixel_ratio||0}% del total</div>
          </div></div>
          <div class="col-6"><div class="stat-chip p-2 rounded text-center">
            <div class="small text-muted">Píxeles modificados</div>
            <div class="fw-bold small">${(density.modified_pixels||0).toLocaleString()}</div>
            <div class="small text-muted">${density.modified_pixel_ratio||0}% del total</div>
          </div></div>
          <div class="col-6"><div class="stat-chip p-2 rounded text-center">
            <div class="small text-muted">Canales modificados</div>
            <div class="fw-bold small">${(density.modified_channel_values||0).toLocaleString()}</div>
          </div></div>
          <div class="col-6"><div class="stat-chip p-2 rounded text-center">
            <div class="small text-muted">Bits embebidos</div>
            <div class="fw-bold small">${(density.embedded_bits||0).toLocaleString()}</div>
          </div></div>
        </div>`;
    } else if (densityWrap) {
      densityWrap.style.display = 'none';
    }

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
    shaBadge.textContent = data.sha256_valid ? 'SHA-256 OK' : 'SHA-256 FALLO';
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
  const bpc     = ext.bits_per_channel_detected != null ? ext.bits_per_channel_detected : (ext.bits_per_channel || 'N/D');
  const chs     = Array.isArray(ext.channels_detected) && ext.channels_detected.length
                    ? ext.channels_detected.join(', ')
                    : (Array.isArray(ext.channels) && ext.channels.length ? ext.channels.join(', ') : 'N/D');
  const alg     = ext.algorithm_detected || ext.algorithm || 'N/D';
  return `
    <div class="mb-1"><strong>Tipo:</strong> ${ext.payload_type}</div>
    <div class="mb-1"><strong>Nombre:</strong> ${ext.filename || '(sin nombre)'}</div>
    <div class="mb-1"><strong>Tamaño:</strong> ${formatBytes(ext.payload_size||0)}</div>
    <div class="mb-1"><strong>MIME:</strong> ${ext.mime_type || 'N/D'}</div>
    <div class="mb-1"><strong>Algoritmo detectado:</strong> ${alg}</div>
    <div class="mb-1"><strong>Bits/canal detectados:</strong> ${bpc}</div>
    <div class="mb-1"><strong>Canales detectados:</strong> ${chs}</div>
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
