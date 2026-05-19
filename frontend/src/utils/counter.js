function animateCounter(el) {
  const to       = parseFloat(el.dataset.counter)
  const decimals = parseInt(el.dataset.decimals ?? '0', 10)
  const suffix   = el.dataset.suffix ?? ''
  const duration = 1200
  const start    = performance.now()

  function tick(now) {
    const elapsed = now - start
    const p       = Math.min(1, elapsed / duration)
    const eased   = 1 - Math.pow(1 - p, 3)
    el.textContent = (to * eased).toFixed(decimals) + suffix
    if (p < 1) requestAnimationFrame(tick)
  }

  requestAnimationFrame(tick)
}

export function initCounters() {
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        animateCounter(e.target)
        obs.unobserve(e.target)
      }
    })
  }, { threshold: 0.4 })

  document.querySelectorAll('[data-counter]').forEach(el => {
    const decimals = parseInt(el.dataset.decimals ?? '0', 10)
    const suffix   = el.dataset.suffix ?? ''
    el.textContent = (0).toFixed(decimals) + suffix
    obs.observe(el)
  })
}
