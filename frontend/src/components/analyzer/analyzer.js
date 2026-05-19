import { analyzeFile, analyzeUrl, explainAnalysis } from '../../utils/api.js'

export function initAnalyzer() {
  const dropzone  = document.getElementById('dropzone')
  const fileInput = document.getElementById('file-input')
  const urlInput  = document.getElementById('url-input')
  const urlSubmit = document.getElementById('url-submit')
  const fileTab   = document.getElementById('file-tab')
  const urlTab    = document.getElementById('url-tab')
  const loading   = document.getElementById('loading')
  const results   = document.getElementById('results')
  const errToast  = document.getElementById('error-toast')
  const ensemble  = document.getElementById('ensemble-card')
  const modGrid   = document.getElementById('models-grid')
  const explain   = document.getElementById('explain-card')

  if (!dropzone) return

  document.querySelectorAll('.analyze-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.analyze-tab').forEach(t => t.classList.remove('active'))
      tab.classList.add('active')
      const isFile = tab.dataset.tab === 'file'
      fileTab.classList.toggle('hidden', !isFile)
      urlTab.classList.toggle('hidden', isFile)
    })
  })

  dropzone.addEventListener('dragover', e => {
    e.preventDefault()
    dropzone.classList.add('drag-over')
  })
  dropzone.addEventListener('dragleave', e => {
    if (!dropzone.contains(e.relatedTarget)) dropzone.classList.remove('drag-over')
  })
  dropzone.addEventListener('drop', e => {
    e.preventDefault()
    dropzone.classList.remove('drag-over')
    const file = e.dataTransfer.files[0]
    if (file) run({ type: 'file', file })
  })

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) run({ type: 'file', file: fileInput.files[0] })
    fileInput.value = ''
  })

  urlSubmit.addEventListener('click', () => {
    const url = urlInput.value.trim()
    if (url) run({ type: 'url', url })
  })
  urlInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') urlSubmit.click()
  })

  async function run(source) {
    hideError()
    showLoading()
    try {
      const data = source.type === 'file'
        ? await analyzeFile(source.file)
        : await analyzeUrl(source.url)
      renderResults(data)
      requestExplanation(source, data)
    } catch (err) {
      showError(err.message)
    } finally {
      hideLoading()
    }
  }

  function showLoading() {
    loading.classList.remove('hidden')
    results.classList.add('hidden')
  }

  function hideLoading() { loading.classList.add('hidden') }

  function showError(msg) {
    errToast.textContent = msg
    errToast.classList.remove('hidden')
  }

  function hideError() { errToast.classList.add('hidden') }

  function renderResults(data) {
    const { ensemble: ens, models, elapsed_ms, vlm_watermark } = data
    const isAI = ens.verdict === 'ai'
    const label = isAI ? 'AI-GENERATED' : 'REAL'

    ensemble.innerHTML = `
      <div class="ens-verdict ${isAI ? 'ens-ai' : 'ens-real'}">${label}</div>
      <div class="ens-conf-row">
        <span class="ens-label">AI_PROB</span>
        <div class="bar-wrap"><div class="bar-fill fill-ai" style="width:${(ens.ai_prob * 100).toFixed(1)}%"></div></div>
        <span class="ens-val">${(ens.ai_prob * 100).toFixed(1)}%</span>
      </div>
      <div class="ens-conf-row">
        <span class="ens-label">REAL_PROB</span>
        <div class="bar-wrap"><div class="bar-fill fill-real" style="width:${(ens.real_prob * 100).toFixed(1)}%"></div></div>
        <span class="ens-val">${(ens.real_prob * 100).toFixed(1)}%</span>
      </div>
      ${ens.disagreement ? '<div class="ens-warn">⚠ MODELS DISAGREE — LOW CONFIDENCE</div>' : ''}
      ${watermarkNote(vlm_watermark)}
      <div class="ens-meta">CONFIDENCE · ${(ens.confidence * 100).toFixed(0)}% &nbsp;·&nbsp; ELAPSED · ${elapsed_ms} ms</div>
    `

    modGrid.innerHTML = models.map(m => {
      const name = m.name.split('/').pop()
      const aiPct   = (m.ai_prob   * 100).toFixed(1)
      const realPct = (m.real_prob * 100).toFixed(1)
      return `
        <div class="model-card">
          <div class="model-name" title="${escapeHtml(m.name)}">${escapeHtml(name)}</div>
          <div class="model-row">
            <span class="model-label">AI</span>
            <div class="bar-wrap"><div class="bar-fill fill-ai" style="width:${aiPct}%"></div></div>
            <span class="model-val">${aiPct}%</span>
          </div>
          <div class="model-row">
            <span class="model-label">REAL</span>
            <div class="bar-wrap"><div class="bar-fill fill-real" style="width:${realPct}%"></div></div>
            <span class="model-val">${realPct}%</span>
          </div>
        </div>
      `
    }).join('')

    results.classList.remove('hidden')
    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }

  function watermarkNote(wm) {
    if (!wm) return ''
    if (wm.watermark_found) {
      const type = escapeHtml(String(wm.watermark_type).toUpperCase())
      const conf = (wm.watermark_confidence * 100).toFixed(0)
      return `<div class="ens-watermark ens-watermark-found">◆ ВОДЯНОЙ ЗНАК · ${type} · ${conf}%</div>`
    }
    return '<div class="ens-watermark">○ ВОДЯНОЙ ЗНАК НЕ ОБНАРУЖЕН</div>'
  }

  async function requestExplanation(source, data) {
    showExplainLoading()
    try {
      const analysis = { ensemble: data.ensemble, vlm_watermark: data.vlm_watermark }
      renderExplanation(await explainAnalysis(source, analysis))
    } catch (err) {
      renderExplainError(err.message)
    }
  }

  const EXPLAIN_HEAD = '<span class="explain-eyebrow">— VLM разбор</span>'

  function showExplainLoading() {
    explain.classList.remove('hidden')
    explain.innerHTML = `
      <div class="explain-head">${EXPLAIN_HEAD}</div>
      <div class="explain-loading">
        <div class="loading-spinner loading-spinner-sm"></div>
        <span>VLM генерирует разбор…</span>
      </div>
    `
  }

  function renderExplanation(r) {
    const evidence = r.evidence && r.evidence.length
      ? `<div class="explain-sublabel">ПРИЗНАКИ</div>
         <ul class="explain-evidence">${r.evidence.map(e => `<li>${escapeHtml(e)}</li>`).join('')}</ul>`
      : ''
    const caveat = r.caveat
      ? `<div class="explain-caveat">
           <div class="explain-caveat-label">⚠ ОГОВОРКА</div>
           <div class="explain-caveat-text">${escapeHtml(r.caveat)}</div>
         </div>`
      : ''
    explain.innerHTML = `
      <div class="explain-head">${EXPLAIN_HEAD}<span class="explain-dot"></span></div>
      <p class="explain-text${r.available ? '' : ' explain-text-muted'}">${escapeHtml(r.explanation)}</p>
      ${evidence}
      ${caveat}
    `
  }

  function renderExplainError(msg) {
    explain.innerHTML = `
      <div class="explain-head">${EXPLAIN_HEAD}</div>
      <p class="explain-text explain-text-muted">Не удалось получить разбор: ${escapeHtml(msg)}</p>
    `
  }

  function escapeHtml(value) {
    const div = document.createElement('div')
    div.textContent = String(value)
    return div.innerHTML
  }
}
