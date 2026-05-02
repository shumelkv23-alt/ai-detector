import './styles/tokens.css'
import './styles/reset.css'
import './styles/global.css'

import './components/header/header.css'
import './components/hero/hero.css'
import './components/scan-demo/scan-demo.css'
import './components/dropzone/dropzone.css'
import './components/result-card/result-card.css'
import './components/logos/logos.css'
import './components/signals/signals.css'
import './components/how/how.css'
import './components/metrics/metrics.css'
import './components/cta/cta.css'
import './components/footer/footer.css'

import { initCounters } from './utils/counter.js'
import { initReveal }   from './utils/reveal.js'
import { initAnalyzer } from './components/analyzer/analyzer.js'

initCounters()
initReveal()
initAnalyzer()
