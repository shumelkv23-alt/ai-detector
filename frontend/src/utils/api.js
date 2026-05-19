export async function analyzeFile(file) {
  const body = new FormData()
  body.append('file', file)
  const res = await fetch('/api/analyze', { method: 'POST', body })
  if (!res.ok) throw await _extractError(res)
  return res.json()
}

export async function analyzeUrl(url) {
  const res = await fetch('/api/analyze-url', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  })
  if (!res.ok) throw await _extractError(res)
  return res.json()
}

export async function explainAnalysis(source, analysis) {
  const body = new FormData()
  body.append('analysis', JSON.stringify(analysis))
  if (source.type === 'file') body.append('file', source.file)
  else body.append('url', source.url)
  const res = await fetch('/api/explain', { method: 'POST', body })
  if (!res.ok) throw await _extractError(res)
  return res.json()
}

async function _extractError(res) {
  const data = await res.json().catch(() => ({}))
  return new Error(data.detail || `HTTP ${res.status}`)
}
