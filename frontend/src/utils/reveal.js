export function initReveal() {
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) e.target.classList.add('in')
    })
  }, { threshold: 0.15 })

  document.querySelectorAll('.reveal').forEach(el => obs.observe(el))
}
