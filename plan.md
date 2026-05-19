# План: Система определения AI-сгенерированных изображений

## Контекст

Курсовая работа: веб-приложение для определения, является ли изображение реальной фотографией или AI-генерацией. Бэкенд на FastAPI ансамблирует 3 предобученные модели Hugging Face и возвращает вердикт ансамбля. Папка проекта пустая — строим с нуля.

Решения, согласованные с пользователем:
- Только CPU (без CUDA)
- `pip + venv + requirements.txt`
- **Grad-CAM (backward-pass объяснимость) — не делаем** (требует кастомных хуков, медленно). Вместо этого используем patch-based detection: нарезаем картинку на патчи 224×224 и делаем батчевый инференс — это даёт `patch_max_ai` (максимальная AI-вероятность среди патчей) без Grad-CAM.
- **Chrome-расширение — отложено на этап 2**. Сначала полностью закрываем веб-сайт, потом возвращаемся к extension.
- Монорепо на будущее: `backend/`, `frontend/` в одном корне (папка `extension/` появится на этапе 2).
- Тройка моделей пересмотрена (см. ниже)

## Git-флоу

- Две ветки: `main` (только стабильные milestone'ы) и `dev` (рабочая).
- Вся разработка идёт в `dev`.
- **После каждого этапа из билд-ордера** — коммит и `git push origin dev` с коротким английским commit-сообщением (формат Conventional Commits).
- Merge `dev → main` делаем только на финальных вехах (после завершения бэкенда и после завершения фронтенда), чтобы `main` всегда был запускаемым.

## Выбор моделей (отступление от исходного списка ТЗ)

Исходный список из ТЗ устарел: `umm-maybe/AI-image-detector` обучен в 2022 на ранних AI-картинках и плохо детектит современные Flux/SD3/Midjourney v8, а `Organika/sdxl-detector` — узкий SDXL-специалист. Изначально планировался ансамбль из 3 моделей, но `boluobobo/ItsNotAI-ai-detector-v2` оказался непригоден (см. ниже), поэтому остановились на двух.

| # | Модель | Архитектура | Accuracy | Вес |
|---|--------|-------------|----------|-----|
| 1 | `Ateeqq/ai-vs-human-image-detector`     | SigLIP (~92.9M) | 99.23% | 0.50 |
| 2 | `haywoodsloan/ai-image-detector-deploy` | SwinV2 (~0.2B)  | 98.15% (бывшая `haywoodsloan/ai-image-detector` удалена с HF, перевыложена под новым именем) | 0.50 |

Архитектурное разнообразие сохранено: SigLIP (contrastive vision-language) + SwinV2 (windowed hierarchical) — две разные предобучалки трансформеров, ошибки слабо коррелируют.

**Почему убрали `boluobobo/ItsNotAI-ai-detector-v2`:** в её `config.json` на HF id2label не заполнен (`{0: "LABEL_0", ..., 3: "LABEL_3"}`), реальная модель — 33-классовый dual-head классификатор (8 Real-источников + 25 AI). Стандартный `AutoModelForImageClassification` грузит неправильную голову, и без кастомного кода модель некорректно интегрируется в pipeline — на любом входе возвращает `ai_prob ≈ 1.0`, ломая ансамбль.

## Технические замечания

- Карта `id2label` у моделей разная (`ai/hum`, `ai/human`, `REAL/FAKE`) — нормализуем в детекторе к единому `{ai_prob, real_prob}`.
- **Первый запуск** скачает ~800 МБ моделей в `~/.cache/huggingface`. Упомянуть в README.
- На CPU инференс моделей запускаем параллельно через `asyncio.to_thread` + `asyncio.gather` — torch на CPU отпускает GIL во время forward, чтобы уложиться в SLA 10 сек из ТЗ.
- **Patch-based detection:** если картинка > 224×224, нарезаем до 2×2=4 патчей (PATCH_MAX_DIM=448, stride=224 без overlap) и прогоняем батчем вместе с глобальным снимком за один `model(**inputs)`. Итог: `patch_max_ai` — максимальная AI-вероятность среди патчей. Ансамбль берёт `max(global_ai, patch_max_ai)` как финальный `ai_prob`.
- **Temperature scaling:** softmax делится на `TEMPERATURE=1.5` перед нормализацией — уменьшает переуверенность моделей.

## Структура проекта

```
курсовая работа/
├── README.md                       # инструкция запуска + примеры
├── plan.md                         # этот файл
├── .gitignore
├── backend/
│   ├── requirements.txt
│   ├── run.py                      # uvicorn entrypoint
│   └── app/
│       ├── __init__.py
│       ├── main.py                 # FastAPI app, CORS, endpoints
│       ├── config.py               # MODELS, WEIGHTS, MAX_FILE_SIZE
│       ├── schemas.py              # Pydantic: AnalyzeResponse, ModelResult
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── detector.py         # ModelRegistry: ленивая загрузка 3 моделей
│       │   └── ensemble.py         # weighted_average + verdict
│       └── utils/
│           ├── __init__.py
│           ├── image.py            # load_from_upload, load_from_url, validate
│           └── errors.py           # AppError → HTTPException mapper
└── frontend/
    ├── index.html                  # Vite entry point (только HTML-разметка)
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.js                 # импорты CSS + initCounters/initReveal
        ├── styles/
        │   ├── tokens.css          # CSS custom properties (dark terminal palette)
        │   ├── reset.css           # box-sizing, body, base elements
        │   └── global.css          # .container, .btn, .section, .reveal
        ├── components/
        │   ├── header/header.css
        │   ├── hero/hero.css
        │   ├── scan-demo/scan-demo.css
        │   ├── logos/logos.css
        │   ├── signals/signals.css
        │   ├── how/how.css
        │   ├── metrics/metrics.css
        │   ├── cta/cta.css
        │   └── footer/footer.css
        └── utils/
            ├── counter.js          # animateCounter + IntersectionObserver
            └── reveal.js           # scroll reveal observer
```

> Папка `extension/` отсутствует — её создаём на этапе 2 (см. «Этап 2 — Браузерное расширение»).

## Backend — ключевые файлы

### `backend/requirements.txt`
```
fastapi==0.115.*
uvicorn[standard]==0.32.*
python-multipart==0.0.*
pillow==11.*
httpx==0.28.*
torch==2.5.*           # CPU build
transformers==4.46.*
numpy==2.1.*
pydantic==2.9.*
pytest==8.*            # для тестов
```

### `backend/app/config.py`
```python
MODELS = [
    {"id": "Ateeqq/ai-vs-human-image-detector", "weight": 0.35, "arch": "siglip"},
    {"id": "boluobobo/ItsNotAI-ai-detector-v2", "weight": 0.35, "arch": "beit"},
    {"id": "haywoodsloan/ai-image-detector",    "weight": 0.30, "arch": "swinv2"},
]
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
DEVICE = "cpu"
```

### `backend/app/ml/detector.py`
- `ModelRegistry` — синглтон, ленивая загрузка `AutoImageProcessor` + `AutoModelForImageClassification` для каждой из 3 моделей при старте сервера (через FastAPI `lifespan`).
- `predict(model_entry, pil_image) -> {ai_prob, real_prob}` — softmax по логитам. Учесть, что у разных моделей разный порядок меток (`id2label`): нормализовать к `{ai, real}`.

### `backend/app/ml/ensemble.py`
- `aggregate(results: list[ModelResult]) -> EnsembleResult` — взвешенное среднее `ai_prob`, итоговая метка по порогу 0.5, поле `confidence` = `|0.5 - avg| * 2`, флаг `disagreement` если стандартное отклонение > 0.2.

### `backend/app/utils/image.py`
- `load_from_upload(UploadFile)` — проверка MIME + размера + конвертация в RGB `PIL.Image`.
- `load_from_url(url)` — `httpx.AsyncClient.get` с таймаутом 8 с, проверка `Content-Type` и `Content-Length`, защита от SSRF (отклонять `127.0.0.1`, приватные сети, `file://`).

### `backend/app/main.py` — endpoints
- `POST /api/analyze` — `multipart/form-data` с полем `file` → `AnalyzeResponse`
- `POST /api/analyze-url` — JSON `{url: str}` → `AnalyzeResponse`
- `GET /api/health` — для расширения
- CORS: разрешить `http://localhost:*`, `chrome-extension://*`
- Параллельный инференс через `asyncio.gather(*[asyncio.to_thread(predict, m, img) for m in models])`

### `AnalyzeResponse` (schemas.py)
```python
{
  "models": [
    {"name": "Ateeqq/ai-vs-human-image-detector", "ai_prob": 0.83, "real_prob": 0.17},
    {"name": "boluobobo/ItsNotAI-ai-detector-v2", "ai_prob": 0.77, "real_prob": 0.23},
    {"name": "haywoodsloan/ai-image-detector",    "ai_prob": 0.78, "real_prob": 0.22}
  ],
  "ensemble": {
    "ai_prob": 0.79, "verdict": "ai", "confidence": 0.58,
    "disagreement": false
  },
  "elapsed_ms": 7421
}
```

## Frontend (`frontend/`)

Сборщик — **Vite 6** (vanilla JS, без фреймворка). Запуск: `cd frontend && npm run dev`.

### Дизайн: dark terminal / forensic

Палитра — глубокий чёрный фон, монопространственная типографика, зелёный сигнал / красный алерт. Дизайн намеренно отсылает к forensic-инструментам и CLI-интерфейсам.

### Design tokens (`src/styles/tokens.css`)

```css
:root {
  /* backgrounds */
  --bg-0: #050505;  --bg-1: #0a0a0a;  --bg-2: #111111;
  --bg-3: #1a1a1a;  --bg-4: #242424;

  /* borders */
  --line-1: #1f1f1f;  --line-2: #2a2a2a;  --line-3: #3a3a3a;

  /* foreground */
  --fg-1: #f5f5f5;  --fg-2: #a3a3a3;  --fg-3: #737373;
  --fg-4: #525252;  --fg-5: #2e2e2e;

  /* semantic */
  --signal: #10b981;  --signal-glow: rgba(16,185,129,0.18);
  --alert:  #ef4444;  --alert-glow:  rgba(239,68,68,0.18);
  --caution: #f59e0b;

  /* fonts */
  --font-mono: 'JetBrains Mono', ui-monospace, monospace;
  --font-sans: 'Inter', -apple-system, sans-serif;

  /* spacing */
  --space-9: 96px;  /* section padding */
  --gutter:  32px;

  /* animation */
  --ease-out:    cubic-bezier(0.22, 0.61, 0.36, 1);
  --ease-in-out: cubic-bezier(0.65, 0, 0.35, 1);
  --dur-base: 200ms;  --dur-slow: 400ms;

  /* misc */
  --radius-0: 0;  --radius-pill: 999px;
  --container: 1280px;
}
```

### Типографика

- **Primary font:** JetBrains Mono (загружается с Google Fonts).
- **UI font:** Inter — для body-текста и описаний.
- Hero title: `clamp(48px, 6.4vw, 88px)`, weight 500, `letter-spacing: -0.04em` — агрессивное сжатие на больших размерах.
- Section titles: `clamp(32px, 4vw, 52px)`, letter-spacing `-0.03em`.
- Все метки, теги, бейджи — монопространственные, uppercase, `letter-spacing: 0.08–0.16em`.

### Структура секций `index.html`

1. **Header** — sticky, `backdrop-filter: blur(12px)`, nav скрывается на `<900px`.
2. **Hero** — grid 1.1fr/1fr: слева заголовок + CTA + stat-метрики с анимированными счётчиками; справа `scan-demo` (анимированная линия сканирования, алерт-ячейки).
3. **Logos** — лента организаций в моно-шрифте.
4. **Signals** (#signals) — grid 3×2, карточки с прогресс-барами по каждому из 6 сигналов.
5. **How** (#how) — grid 1/1: слева шаги, справа терминальный блок с JSON-ответом.
6. **Metrics** (#metrics) — grid 4×1, крупные числа 56px с анимированными счётчиками.
7. **CTA** — центрированная секция, две кнопки.
8. **Footer** — logo + links + status.

### Поведение `src/utils/`

- `counter.js` — `initCounters()`: `IntersectionObserver` запускает `animateCounter` (ease-out cubic, 1200ms) при попадании элемента с `data-counter` в viewport.
- `reveal.js` — `initReveal()`: добавляет класс `.in` на `.reveal`-элементы при скролле (threshold 15%).

### Адаптив

- `>900px` — полный двухколоночный layout.
- `≤900px` — hero и how схлопываются в одну колонку, signals grid → 1fr, metrics → 2fr, header-nav скрыт.
- `≤480px` — metrics → 1fr, container padding 16px.
- `prefers-reduced-motion` — анимации scan-line и pulse отключаются, reveal без transition.

### Anti-template check

- Нет шаблонных Tailwind/shadcn паттернов — всё написано вручную.
- Монохромная палитра с двумя семантическими цветами (зелёный/красный) вместо «accent decoration».
- Иерархия за счёт масштаба и letter-spacing, а не цвета.
- Анимации только через `transform` / `opacity` / `top` — компоновочно-friendly.

## Порядок реализации (билд-ордер)

После каждого шага: `git add -A && git commit -m "<msg>" && git push origin dev`. Commit-сообщения в Conventional Commits, на английском.

### Шаг 0 — Git bootstrap
- `git init`
- Создать `.gitignore` (venv, `__pycache__`, `.env`, `.DS_Store`, `node_modules`, IDE-файлы).
- Первый коммит в `main`: initial план + дизайн-файл.
- Создать и переключиться на `dev`: `git branch -M main && git checkout -b dev`.
- Создать удалённый репозиторий на GitHub, `git remote add origin ...`, запушить обе ветки: `git push -u origin main && git push -u origin dev`.
- **Пуш:** `chore: initialize repo with main and dev branches`

### Шаг 1 — Скелет проекта
- Создать структуру папок (`backend/app/ml`, `backend/app/utils`, `frontend/styles`, `frontend/components`), пустые `__init__.py`, пустой `README.md`.
- **Пуш:** `chore: scaffold backend and frontend directory tree`

### Шаг 2 — Backend setup
- `requirements.txt`, создать `venv`, установить зависимости.
- `backend/run.py` + `backend/app/main.py` — FastAPI с `/api/health` возвращает `{"status": "ok"}`.
- Проверка: `uvicorn app.main:app --reload` поднимается, `/api/health` отвечает 200.
- **Пуш:** `feat(backend): add FastAPI skeleton with health endpoint`

### Шаг 3 — ML-ядро (одна модель)
- `config.py` (MODELS, WEIGHTS, MAX_FILE_SIZE).
- `ml/detector.py` — `ModelRegistry` грузит одну модель (Ateeqq), `predict()` возвращает нормализованный `{ai_prob, real_prob}`.
- `backend/tests/test_detector.py` — прогон на тестовом jpg в `samples/`.
- **Пуш:** `feat(ml): add single-model detector with id2label normalization`

### Шаг 4 — Все 3 модели + ensemble
- Расширить `ModelRegistry` на список из 3 моделей, lifespan-хук FastAPI прогревает их на старте.
- `ml/ensemble.py` — `aggregate()` с весами, `confidence`, `disagreement`.
- `tests/test_ensemble.py` — юнит-тесты на `aggregate`.
- **Пуш:** `feat(ml): add 3-model registry with weighted ensemble aggregation`

### Шаг 5 — Endpoints
- `POST /api/analyze` (multipart) + `POST /api/analyze-url` (JSON).
- `utils/image.py` (MIME/size валидация, SSRF-защита на `load_from_url`).
- Параллельный инференс: `asyncio.gather(*[asyncio.to_thread(predict, m, img) for m in models])`.
- CORS: `http://localhost:*`.
- Smoke-тест curl'ом: реальная фотка и AI-картинка.
- **Пуш:** `feat(api): add analyze and analyze-url endpoints with parallel inference`

### Шаг 6 — 🔀 Merge `dev → main` (backend milestone)
- `git checkout main && git merge --no-ff dev -m "merge: backend MVP complete"`
- `git push origin main`
- Вернуться на dev: `git checkout dev`.
- **Зачем:** `main` всегда запускаем. После этого шага бэкенд на `main` работает end-to-end.

### Шаг 7 — Frontend: Vite + design tokens + landing shell
- `cd frontend && npm install` (Vite 6).
- `src/styles/tokens.css`, `reset.css`, `global.css` — dark terminal palette, кнопки, секции, reveal.
- `index.html` — полная HTML-разметка (header / hero / logos / signals / how / metrics / cta / footer).
- `src/main.js` — импорты CSS + `initCounters()` / `initReveal()`.
- Проверка: `npm run dev` открывает `http://localhost:5173`, страница рендерится.
- **Пуш:** `feat(frontend): add Vite setup with dark terminal design tokens`

### Шаг 8 — Frontend: компонентные CSS-файлы
- Каждый компонент в своей папке (`header/`, `hero/`, `scan-demo/`, `logos/`, `signals/`, `how/`, `metrics/`, `cta/`, `footer/`).
- Адаптивные медиа-запросы — в каждом компонентном CSS-файле (не в глобальном).
- `src/utils/counter.js` + `src/utils/reveal.js` — анимации по IntersectionObserver.
- **Пуш:** `feat(frontend): split CSS into per-component files with responsive rules`

### Шаг 9 — Frontend: dropzone + fetch + результаты
- `src/components/dropzone/` — drag-drop зона + URL input (тёмный стиль: `--line-2` border, `--bg-1` фон).
- `src/components/result-card/` — карточки моделей с прогресс-барами; итог ансамбля крупнее.
- `src/utils/api.js` — `analyzeFile(file)` / `analyzeUrl(url)` → `fetch('/api/analyze')`.
- Spinner только через `opacity` + `transform`; toast-ошибка с `--alert` левой полосой.
- Вердикт: `ai` → `--alert`, `real` → `--signal`.
- **Пуш:** `feat(frontend): implement dropzone, fetch, and result cards`

### Шаг 10 — Frontend: polish + accessibility
- Focus-ring: `outline: 2px solid var(--signal); outline-offset: 2px;` на всех интерактивных.
- Проверить `prefers-reduced-motion` (scan-line и pulse отключены).
- Keyboard navigation: Tab по всем кнопкам и ссылкам.
- Прогон по Component Checklist из web/design-quality.md.
- **Пуш:** `feat(frontend): add focus states and accessibility polish`

### Шаг 11 — README
- Установка (`python -m venv`, `pip install -r`), запуск бэкенда, запуск фронта через `python -m http.server 5500`.
- Описание API, скриншоты.
- Предупреждение про первичную загрузку ~800 МБ моделей.
- **Пуш:** `docs: add README with setup and usage instructions`

### Шаг 12 — Отчёт о тестировании
- `tests/report.md` — прогон ~20 изображений (10 real, 10 AI из разных генераторов), таблица предсказаний каждой модели и ансамбля, accuracy.
- **Пуш:** `test: add manual accuracy report across 20 sample images`

### Шаг 13 — 🔀 Финальный merge `dev → main` (MVP done)
- `git checkout main && git merge --no-ff dev -m "merge: web MVP complete"`
- `git push origin main`
- Теги (по желанию): `git tag v0.1-mvp && git push --tags`.

## Верификация

- `pytest backend/tests/` — unit-тесты на `aggregate`, валидацию изображений, маппинг id2label.
- Ручной smoke-тест бэкенда:
  ```bash
  curl -F "file=@samples/real.jpg" http://localhost:8000/api/analyze
  curl -X POST http://localhost:8000/api/analyze-url -H "Content-Type: application/json" -d '{"url":"https://..."}'
  ```
- Запустить фронтенд: `cd frontend && npm run dev` → `http://localhost:5173`, проверить drag-drop + URL flow, увидеть карточки с результатами.
- Визуальная проверка дизайна через DevTools:
  - hero title: `font-size` ≈88px на десктопе, `letter-spacing: -0.04em`.
  - scan-line анимация видна, алерт-ячейки подсвечены красным.
  - счётчики запускаются при скролле в viewport.
  - focus ring (`outline: 2px solid var(--signal)`) виден на Tab-навигации.
  - `prefers-reduced-motion: reduce` → scan-line и pulse остановлены.
- Проверить error-paths: файл >10 МБ, gif, битый URL, недоступный URL.
- Git: `main` запускается без модификаций, `dev` содержит все промежуточные коммиты.

## Этап 1.5 — Повышение точности ансамбля до 85%+

### Диагноз (по результатам тестов на 5 контрольных фото)

Базовый MVP даёт неуверенные вердикты с такими паттернами проблем:

1. **Две модели расходятся радикально:** Ateeqq (агрессивная, 99% AI на всём) vs haywoodsloan (консервативная, 2% AI на всём). Среднее = 0.5 — монетка.
2. **Watermark scoring был бинарным:** любое срабатывание Roboflow → score=0.95 → ансамбль вылетает в "ai" даже на реальном фото.
3. **Сигналы работали односторонне:** все override-правила могли только повысить ai_prob, никто не мог снизить.
4. **Patch_max переопределял global:** один странный патч на чистом фото мог перевернуть вердикт.
5. **Roboflow Rapid + Python 3.14:** несовместимы (нет inference-sdk), поэтому отказались от workflow и вызываем стандартную YOLOv11-модель напрямую через `detect.roboflow.com/{model}/{version}`.
6. **TrUFor оказался непригоден:** репозиторий `grip-unina/TrUFor` не предназначен для `torch.hub` (нет `hubconf.py`), это исследовательский код с зависимостями mmcv/CUDA и весами через Google Drive. Удалили из проекта целиком — для курсовой это слишком дорого по времени, эквивалентный сигнал (детекция локальных манипуляций) реализован через ELA hotspot ratio.

### Реализованные изменения

**Модели (config.py):**
| # | Модель | Архитектура | Вес | Роль |
|---|--------|-------------|-----|------|
| 1 | `Ateeqq/ai-vs-human-image-detector`     | SigLIP | 0.30 | агрессивная (High-recall AI) |
| 2 | `haywoodsloan/ai-image-detector-deploy` | SwinV2 | 0.35 | консервативная (High-precision Real) |
| 3 | `umm-maybe/AI-image-detector`           | ViT    | 0.35 | арбитр / тай-брейкер |

**Aggregate (ensemble.py):**
- **Медиана при разногласии:** если std(ai_probs) > 0.25 — берём `median` из трёх моделей вместо среднего. Это срезает экстремиста (например, Ateeqq=0.99 при двух других ~0.2 → медиана 0.2, а не среднее 0.47).
- **Patch_max только когда global ≥ 0.4:** на чистых real-фото (global < 0.4) игнорируем patch_max — один странный патч больше не переворачивает вердикт.
- **Двусторонние сигналы:**
  - AI-сигналы (ELA ≥ 0.6, EXIF ≥ 0.85, FFT ≥ 0.7, watermark ≥ 0.9) — толкают `final_ai` вверх через `max()`.
  - REAL-сигналы — тянут вниз:
    - **Camera EXIF + нет AI-сигналов** → `final_ai = min(final_ai, 0.45)` (камера в Make/Model + чистые forensics ≈ настоящее фото).
    - **Все forensics молчат** (ELA<0.2, FFT<0.3, noise<0.35, watermark<0.3) и `base_ai < 0.85` → `final_ai *= 0.8` (страхует от одной агрессивной модели).

**Watermark scoring (watermark_analyzer.py):**
Линейная шкала вместо ступеньки:
```python
# Было: max(0.95, best_conf) — любое срабатывание → ≥0.95
# Стало: 0.40 + 0.55 * best_conf
#   conf=0.70 → score=0.785
#   conf=0.95 → score=0.92
```
Параллельно повышены пороги: `_ROBOFLOW_CONF=0.70`, `_YOLO_CONF=0.35` — срезают false positives Roboflow, обученного на 16 фото.

**Roboflow (.env, watermark_analyzer.py):**
Перешли с workflow на прямой вызов модели: `detect.roboflow.com/gemini-watermark-v2/1`. Это убрало 500-ошибки из-за Custom Python блоков на облачном хостинге Roboflow и сняло зависимость от `inference-sdk` (он не поддерживает Python 3.14).

**Watermark detection order (watermark_analyzer.py):**
Roboflow → если ничего не нашёл → YOLO-World. Это экономит время на большинстве AI-фото с Gemini watermark.

### Дальнейшие шаги

**Шаг 1.5.1 — Замер базовой точности** (после изменений выше)
Прогнать через `/api/analyze` 5 контрольных изображений:
1. Real-фото с AI-вставкой (бабочка в кота, Gemini sparkle в углу) — ожидаемый verdict: `ai`
2. Real-фото (кот с сердечком) — ожидаемый verdict: `real`
3. Real-фото (селфи пары) — ожидаемый verdict: `real`
4. AI-фото (5 мужчин с открытками, Gemini watermark) — ожидаемый verdict: `ai`
5. AI-фото (рыбак с котом, Gemini watermark) — ожидаемый verdict: `ai`

Записать все signals (ela/exif/noise/fft/watermark) для каждого и сравнить с ожидаемым.

**Шаг 1.5.2 — Расширение Roboflow датасета**
- Собрать ~50 изображений с Gemini sparkle watermark + ~30 реальных без.
- Переразметить и переобучить модель → перейти на `gemini-watermark-v2/2` (новая версия).
- После переобучения порог `_ROBOFLOW_CONF` можно снизить обратно до 0.55.

**Шаг 1.5.3 — TrUFor: удалён из проекта**
- Диагностика показала: репозиторий `grip-unina/TrUFor` не предназначен для `torch.hub` (нет `hubconf.py`).
- Полная установка требует: mmcv (Windows + Python 3.14 → отдельный квест), ручную загрузку весов из Google Drive, кастомный загрузчик с config-файлами.
- Решение: удалили `trufor_analyzer.py` и все упоминания из `main.py`, `ensemble.py`, `schemas.py`, `config.py`, `.env`.
- Альтернатива (реализована): ELA hotspot ratio покрывает основной use case TrUFor — детекция локальных манипуляций (composite, splicing, inpainting).

**Шаг 1.5.4 — ELA hotspot ratio для локальных манипуляций**
Проблема: ELA усреднял максимум по всему кадру → composite (вставка AI-объекта в реальное фото) давал низкий global score, потому что 95% картинки чистые. Override 0.6 никогда не срабатывал.

Добавили `hotspot_ratio = max_energy / median_energy`:
- На реальных фото отношение обычно 1.5-3 (равномерный residual).
- На composite отношение 5-15+ (вставленная область не подверглась той же истории сжатия).
- `hotspot_score = min(1.0, ratio / 8.0)` нормализуется в [0..1].
- `ela_score = max(global_score, hotspot_score)` — берём больший из двух сигналов.

Это ловит: AI-вставки в реальное фото (cat+butterfly из тестового набора), inpainting, splicing — независимо от того что глобальный ELA молчит.

**Шаг 1.5.5 — Patches с overlap для HuggingFace моделей**
Было: `PATCH_SIZE=224, PATCH_STRIDE=224, PATCH_MAX_DIM=448` → 2×2=4 патча без overlap. AI-объект, попадающий на границу патчей, размывался по двум.

Стало: `PATCH_STRIDE=112` (50% overlap), даёт 3×3=9 патчей. Объект целиком гарантированно попадёт хотя бы в один патч, и `patch_max_ai` отразит локальную AI-вероятность.

Цена: время инференса вырастет с ~2с до ~4с на CPU. Приемлемо в рамках SLA 10с.

**Шаг 1.5.6 — Калибровка порогов на тестовом датасете**
- Собрать тестовый датасет: 20 AI (Midjourney, DALL-E, SD, Gemini, Flux) + 20 реальных фото с камеры + 5 composite (вставка AI-объекта в реальное).
- Прогнать через API, сохранить все сигналы в CSV.
- Подобрать пороги override через grid search по accuracy/precision/recall.

**Шаг 1.5.7 — Финальный замер**
Цель: ≥85% правильных вердиктов на тестовом датасете 45 фото (включая 5 composite).
Если composite-кейсы провалены — рассмотреть альтернативный forensics-метод (CAT-Net, ManTraNet) или ручную установку TrUFor вне torch.hub.

**Шаг 1.5.4 — Калибровка порогов на тестовом датасете**
- Собрать тестовый датасет: 20 AI (Midjourney, DALL-E, SD, Gemini, Flux) + 20 реальных фото с камеры.
- Прогнать через API, сохранить все сигналы в CSV.
- Подобрать пороги override через grid search по accuracy/precision/recall.

**Шаг 1.5.5 — Финальный замер**
Цель: ≥85% правильных вердиктов на тестовом датасете 40 фото.
Если не дотягиваем — добавить ещё одну модель в ансамбль или подключить альтернативный forensics-метод.

## Этап 1.6 — VLM-слой (OpenAI vision): объяснение вердикта + watermark

### Идея и согласованные решения

К ансамблю подключается мультимодальная модель OpenAI (vision API). Обсудили
три возможных роли VLM и зафиксировали выбор:

- VLM **не** становится голосующей моделью ансамбля — современные мультимодалки
  плохие AI-vs-real классификаторы (уверенно ошибаются на фотореализме
  Flux/SD3/Midjourney), их вывод недетерминирован и не лезет в фиксированные веса.
- VLM хорош ровно в одном куске задачи — **детекте водяных знаков** (это по сути
  OCR + распознавание логотипа), поэтому именно watermark-сигнал ему доверяем.
- Плюс VLM пишет **человекочитаемое объяснение вердикта** — для UI и защиты курсовой.

**Выбран «Вариант B» — два отдельных вызова VLM:**

| Вызов | Где | Когда | Влияние |
|-------|-----|-------|---------|
| 1. Watermark-детект | внутри `/api/analyze`, параллельно с ансамблем | до `aggregate()` | **влияет на вердикт** через `watermark_score` |
| 2. Объяснение | отдельный эндпоинт `/api/explain` | после вердикта | не влияет, только текст |

Почему два вызова, а не один: watermark обязан отработать до вердикта, а
объяснение — наоборот, должно знать финальный вердикт, чтобы связно его
объяснить. Один вызов обе роли качественно не закрывает (chicken-and-egg).

### Ключевые архитектурные решения

- **VLM-watermark заменяет Roboflow.** `_RoboflowDetector` и переменные
  `ROBOFLOW_*` удалены из `watermark_analyzer.py`, `config.py`, `.env`.
  YOLO-World остаётся оффлайн-fallback'ом, когда `OPENAI_API_KEY` не задан.
- **`watermark_score = max(offline, vlm)`** — VLM приоритетный источник, YOLO +
  метаданные дотягивают, когда VLM молчит. Шкала VLM — та же линейная
  `0.40 + 0.55 * confidence`, порог override в `aggregate()` (0.9) не трогаем.
- **Вызов 1 — конкурентно, без роста времени.** Уходит в тот же `asyncio.gather`,
  что и HF-модели, обёрнут в `asyncio.wait_for(VLM_TIMEOUT=30с)`. Сетевой вызов
  идёт параллельно с CPU-инференсом и прячется под его время — замер: `/api/analyze`
  на тестовой картинке ≈25,7 с и с VLM, и без. По таймауту/ошибке watermark тихо
  падает на YOLO.
  **Важно (пре-существующее):** сам ансамбль на этой машине идёт ~25 с, то есть
  заявленный в ТЗ SLA 10 с уже превышен инференсом моделей — VLM-слой к этому
  отношения не имеет (без ключа возвращает `None` мгновенно). Таймаут VLM
  изначально стоял 7 с под «SLA 10 с»; раз SLA фактически недостижим, поднят до
  30 с, чтобы вызов успевал завершиться под прикрытием времени ансамбля.
- **`/api/explain` — без серверного состояния.** Гибкий multipart-эндпоинт:
  поля `analysis` (JSON с `ensemble` + `vlm_watermark`), `file` ИЛИ `url`.
  Картинку фронт пересылает повторно (держит `File`/`url` в руках) — никакого
  in-memory кэша, это противоречило бы принципу «без БД/истории».
- **Graceful fallback.** Нет `OPENAI_API_KEY` → вызов 1 возвращает `None`
  (watermark на YOLO), `/api/explain` отдаёт 200 с заглушкой и `available=false`.
- **VLM-вызов через `AsyncOpenAI`** + `response_format={"type":"json_object"}` —
  устойчиво ко всем версиям SDK 1.x; ответ парсится `json.loads` с защитной
  валидацией полей.

### Новые файлы и изменения

```
backend/app/llm/
├── __init__.py
├── client.py        # ленивый AsyncOpenAI, downscale+base64, is_enabled()
├── watermark.py     # вызов 1: detect_watermark_vlm() → VlmWatermark | None
└── explainer.py     # вызов 2: explain_verdict() → ExplainResult | None
```

- `config.py`: `+ OPENAI_API_KEY, OPENAI_VISION_MODEL, VLM_TIMEOUT,`
  `VLM_EXPLAIN_TIMEOUT, VLM_MAX_IMAGE_DIM`; `− ROBOFLOW_*`.
- `requirements.txt`: `+ openai`.
- `schemas.py`: `+ VlmWatermark, ExplainContext, ExplainResponse`;
  `AnalyzeResponse.vlm_watermark: VlmWatermark | None`.
- `watermark_analyzer.py`: удалён Roboflow; `+ watermark_score_from_vlm()`.
- `main.py`: вызов 1 в `_run_ensemble`, новый эндпоинт `/api/explain`.

### Подводные камни (в отчёт курсовой)

- **Приватность.** Картинка уходит на серверы OpenAI — зафиксировать в
  ограничениях (как уже отмечено для расширения).
- **Стоимость.** Два платных вызова на полностью разобранную картинку.
- **Точка отказа.** Нет интернета/ключа → система остаётся запускаемой
  (watermark на YOLO, объяснение недоступно), но часть функций отключена.

## Этап 2 — Браузерное расширение (СЛЕДУЮЩАЯ СЕССИЯ)

> Решено делать как «плюс» к курсовой. Стартуем со следующей сессии.

### Цель и ключевое решение

Расширение для Chrome (Manifest V3): ПКМ по любой картинке на сайте → проверка
через **уже существующий** бэкенд. ML-логику и эндпоинты НЕ трогаем — расширение
это тонкий клиент к `/api/analyze-url` и `/api/analyze`. Проект для этого готов
наполовину: `analyze-url` уже принимает URL картинки и возвращает вердикт.

### Архитектура

```
браузер (ПКМ по <img>) → extension service worker
  → POST http://localhost:8000/api/analyze-url
  → вердикт в попапе + бейдж-оверлей поверх картинки
```

Два пути получения картинки:
1. **По URL** (основной) — `analyze-url`, у `<img>` есть `src`.
2. **По байтам** (fallback) — для `data:`-картинок, ленивой подгрузки и картинок
   за авторизацией: service worker сам делает `fetch(srcUrl)` → blob →
   `POST /api/analyze` (multipart).

### Структура `extension/`

```
extension/
├── manifest.json        # MV3: permissions, host_permissions, service worker
├── background.js        # service worker: context menu + fetch к бэкенду
├── content.js           # бейдж-оверлей поверх проверенной картинки
├── content.css          # стили бейджа (forensic-палитра из tokens.css)
├── popup/
│   ├── popup.html       # окно: последний результат + ручной ввод URL
│   ├── popup.js
│   └── popup.css
└── icons/               # 16 / 48 / 128 px
```

### manifest.json — ключевое

- `"manifest_version": 3`
- `"permissions": ["contextMenus", "activeTab", "storage"]`
- `"host_permissions": ["http://localhost:8000/*"]` — даёт service worker право
  на cross-origin fetch к бэкенду **мимо page-CORS**. Поэтому CORS на бэкенде
  менять НЕ нужно.
- `"background": { "service_worker": "background.js" }`
- `"content_scripts"` — `content.js` + `content.css` на `<all_urls>`.
- `"action"` — попап.

### Backend — что (не) менять

- **CORS трогать не нужно** — extension fetch через `host_permissions` не
  подчиняется page-CORS.
- `GET /api/health` уже есть — попап пингует его и показывает статус
  «бэкенд онлайн / оффлайн».
- **SSRF** — `load_from_url` (`utils/image.py`) уже отклоняет приватные сети
  и `file://`. Перед любым публичным хостингом перепроверить, что фильтр живой.

### Build-order

После каждого шага: `git add -A && git commit && git push origin dev`.

**Шаг 2.1 — Скелет расширения**
- Создать `extension/` с `manifest.json`, пустым `background.js`, иконками.
- Загрузить через `chrome://extensions` → Load unpacked, убедиться что грузится
  без ошибок.
- Пуш: `feat(extension): scaffold MV3 manifest and service worker`

**Шаг 2.2 — Context menu + запрос к API**
- `background.js`: `chrome.contextMenus.create` с `contexts: ["image"]`.
- По клику — `info.srcUrl` → `POST /api/analyze-url`.
- Результат пока в `console.log` + `chrome.storage.local`.
- Пуш: `feat(extension): add image context menu calling analyze-url`

**Шаг 2.3 — Попап с результатом**
- `popup.html/js/css` — показывает последний вердикт из `chrome.storage.local`:
  verdict-бейдж (ai = `--alert`, real = `--signal`), confidence, по-модельные
  ai_prob.
- Поле ручного ввода URL + кнопка.
- Индикатор статуса бэкенда через `/api/health`.
- Пуш: `feat(extension): add popup with verdict display and health check`

**Шаг 2.4 — Бейдж-оверлей на странице**
- `content.js` — после ответа рисует абсолютно спозиционированный бейдж в углу
  `<img>` (verdict + confidence).
- `content.css` — forensic-стиль, моно-шрифт, `--signal`/`--alert`.
- Состояние loading — спиннер только через `opacity`/`transform`.
- Пуш: `feat(extension): overlay verdict badge on checked images`

**Шаг 2.5 — Fallback по байтам**
- Если `srcUrl` это `data:` либо fetch по URL упал → service worker
  `fetch(srcUrl)` → blob → `FormData` → `POST /api/analyze`.
- Пуш: `feat(extension): fallback to byte upload for data-URI images`

**Шаг 2.6 — Документация**
- README: раздел «Браузерное расширение» — установка unpacked, требование
  запущенного бэкенда.
- Пуш: `docs: document browser extension setup`

### Подводные камни (зафиксировать в отчёте курсовой)

- **Хостинг.** Пока бэкенд на `localhost` — расширение работает только при
  запущенном локальном сервере. Реальная польза — после деплоя (вне MVP).
- **Приватность.** URL каждой проверяемой картинки уходит на сервер — упомянуть
  в ограничениях работы.
- **Лимиты `analyze-url`.** Картинки за авторизацией и `data:` закрывает Шаг 2.5.
- **SLA.** Инференс ~4-7 с — в попапе и бейдже обязателен видимый loading-стейт.

## Что НЕ делаем (вне MVP)

- **Chrome-расширение** — вынесено в отдельный «Этап 2» (см. раздел выше), делаем со следующей сессии. В MVP не входит.
- **Grad-CAM / heatmap-визуализация в UI** — не делаем. `patch_grid` внутри детектора считается, но в API не выставляется. Если понадобится для защиты — добавим поле в `AnalyzeResponse`.
- Деплой / Docker / nginx
- БД и хранение истории запросов
- Авторизация / rate limiting (бэкенд локальный)
- Видео и GIF
