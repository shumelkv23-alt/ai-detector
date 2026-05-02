# План: Система определения AI-сгенерированных изображений

## Контекст

Курсовая работа: веб-приложение для определения, является ли изображение реальной фотографией или AI-генерацией. Бэкенд на FastAPI ансамблирует 3 предобученные модели Hugging Face и возвращает вердикт ансамбля. Папка проекта пустая — строим с нуля.

Решения, согласованные с пользователем:
- Только CPU (без CUDA)
- `pip + venv + requirements.txt`
- **Heatmap (Grad-CAM) — пока не делаем** (отложено / убрано из MVP). Возможно вернём позже.
- **Chrome-расширение — отложено на этап 2**. Сначала полностью закрываем веб-сайт, потом возвращаемся к extension.
- Монорепо на будущее: `backend/`, `frontend/` в одном корне (папка `extension/` появится на этапе 2).
- Тройка моделей пересмотрена (см. ниже)

## Git-флоу

- Две ветки: `main` (только стабильные milestone'ы) и `dev` (рабочая).
- Вся разработка идёт в `dev`.
- **После каждого этапа из билд-ордера** — коммит и `git push origin dev` с коротким английским commit-сообщением (формат Conventional Commits).
- Merge `dev → main` делаем только на финальных вехах (после завершения бэкенда и после завершения фронтенда), чтобы `main` всегда был запускаемым.

## Выбор моделей (отступление от исходного списка ТЗ)

Исходный список из ТЗ устарел: `umm-maybe/AI-image-detector` обучен в 2022 на ранних AI-картинках и плохо детектит современные Flux/SD3/Midjourney v8, а `Organika/sdxl-detector` — узкий SDXL-специалист. Согласовано с пользователем: концепт ансамбля из 3 моделей сохраняем, конкретные веса меняем на более свежие и точные.

| # | Модель | Архитектура | Accuracy | Вес |
|---|--------|-------------|----------|-----|
| 1 | `Ateeqq/ai-vs-human-image-detector`  | SigLIP (~92.9M) | 99.23% | 0.35 |
| 2 | `boluobobo/ItsNotAI-ai-detector-v2`  | BEiT-Large (~304M, файн-тюн `microsoft/beit-large-patch16-224`) | 95.07% binary / 93.47% multiclass (25 AI-источников) | 0.35 |
| 3 | `haywoodsloan/ai-image-detector`     | SwinV2 (~88M) | универсал из исходного ТЗ | 0.30 |

Архитектурное разнообразие (SigLIP + BEiT + Swin) — три разные предобучалки трансформеров: contrastive vision-language (SigLIP) + masked image modeling (BEiT) + windowed hierarchical (Swin). Ошибки слабо коррелируют, ансамбль осмыслен. `boluobobo/ItsNotAI-ai-detector-v2` выбран вместо ранее рассматривавшихся `Dafilab/ai-image-detector` (0 downloads, сырая) и `mmanikanta/ConvNeXT_AI_image_detector` (нет safetensors, только pickle). У `boluobobo` есть safetensors без security-warnings, плюс бонус — мульти-классификация на 25 источников (Midjourney, SD, Flux, DALL-E, StyleGAN2 и т.д.), это можно показать в UI как дополнительную карточку «Похоже на: X».

## Технические замечания

- Карта `id2label` у моделей разная (`ai/hum`, `ai/human`, `REAL/FAKE`) — нормализуем в детекторе к единому `{ai_prob, real_prob}`.
- **Первый запуск** скачает ~800 МБ моделей в `~/.cache/huggingface`. Упомянуть в README.
- На CPU инференс трёх моделей запускаем параллельно через `asyncio.to_thread` + `asyncio.gather` — torch на CPU отпускает GIL во время forward, чтобы уложиться в SLA 10 сек из ТЗ.
- Если позже захочется heatmap — добавим отдельным эндпоинтом `/api/analyze?explain=true`, не ломая текущий API.

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

> Папка `extension/` отсутствует — расширение перенесено в «Что НЕ делаем (этап 2)».

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

## Что НЕ делаем (вне MVP)

- **Chrome-расширение — этап 2**. Вернёмся после того, как веб-сайт полностью готов и протестирован. Бэкенд API уже будет совместим (CORS под `chrome-extension://*` добавим на этапе 2).
- **Grad-CAM / heatmap-визуализация** — отложено по решению пользователя. Если позже понадобится для защиты курсовой — вернём отдельным эндпоинтом.
- Деплой / Docker / nginx
- БД и хранение истории запросов
- Авторизация / rate limiting (бэкенд локальный)
- Видео и GIF
