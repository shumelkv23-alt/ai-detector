import { analyzeFile, analyzeUrl } from '../../utils/api.js'

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

  if (!dropzone) return

  // Tab switching
  document.querySelectorAll('.analyze-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.analyze-tab').forEach(t => t.classList.remove('active'))
      tab.classList.add('active')
      const isFile = tab.dataset.tab === 'file'
      fileTab.classList.toggle('hidden', !isFile)
      urlTab.classList.toggle('hidden', isFile)
    })
  })

  // Drag and drop
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
    if (file) run(() => analyzeFile(file))
  })

  // File input change (triggered by the <label for="file-input"> click)
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) run(() => analyzeFile(fileInput.files[0]))
    fileInput.value = ''
  })

  // URL submit
  urlSubmit.addEventListener('click', () => {
    const url = urlInput.value.trim()
    if (url) run(() => analyzeUrl(url))
  })
  urlInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') urlSubmit.click()
  })

  async function run(fn) {
    hideError()
    showLoading()
    try {
      const data = await fn()
      renderResults(data)
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
    const { ensemble: ens, models, elapsed_ms } = data
    const isAI = ens.verdict === 'ai'
    const label = isAI ? 'AI-GENERATED' : 'REAL'

    ensemble.innerHTML = `
      <div class="ens-verdict ${isAI ? 'ens-ai' : 'ens-real'}">${label}</div>
      <div class="ens-conf-row">
        <span class="ens-label">AI_PROB</span>
        <div class="bar-wrap"><div class="bar-fill ${isAI ? 'fill-ai' : 'fill-real'}" style="width:${(ens.ai_prob * 100).toFixed(1)}%"></div></div>
        <span class="ens-val">${(ens.ai_prob * 100).toFixed(1)}%</span>
      </div>
      <div class="ens-conf-row">
        <span class="ens-label">REAL_PROB</span>
        <div class="bar-wrap"><div class="bar-fill ${isAI ? 'fill-real' : 'fill-real'}" style="width:${(ens.real_prob * 100).toFixed(1)}%"></div></div>
        <span class="ens-val">${(ens.real_prob * 100).toFixed(1)}%</span>
      </div>
      ${ens.disagreement ? '<div class="ens-warn">⚠ MODELS DISAGREE — LOW CONFIDENCE</div>' : ''}
      <div class="ens-meta">CONFIDENCE · ${(ens.confidence * 100).toFixed(0)}% &nbsp;·&nbsp; ELAPSED · ${elapsed_ms} ms</div>
    `

    modGrid.innerHTML = models.map(m => {
      const name = m.name.split('/').pop()
      const aiPct   = (m.ai_prob   * 100).toFixed(1)
      const realPct = (m.real_prob * 100).toFixed(1)
      return `
        <div class="model-card">
          <div class="model-name" title="${m.name}">${name}</div>
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
}
