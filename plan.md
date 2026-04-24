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
- **Дизайн фронтенда** отталкивается от [DESIGN-notion.md](DESIGN-notion.md) — это источник правды по палитре, типографике, отступам, границам и теням.

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
├── DESIGN-notion.md                # источник правды по дизайну фронтенда
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
    ├── index.html
    ├── styles/
    │   ├── tokens.css              # CSS custom properties из DESIGN-notion.md
    │   ├── typography.css          # NotionInter scale, weights, letter-spacing
    │   └── global.css              # reset + layout-утилиты
    ├── components/
    │   ├── hero.css                # hero-секция: 64px display headline
    │   ├── dropzone.css            # drag-drop зона с whisper-border
    │   ├── result-card.css         # карточки моделей + итог ансамбля
    │   └── badge.css               # pill-badge для вердикта (AI / Real)
    └── app.js                      # drag-drop, fetch, рендер результатов
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

Чистый ванильный HTML/CSS/JS — никаких сборщиков (для курсовой это плюс: открывается одним кликом).

**Дизайн-система — по [DESIGN-notion.md](DESIGN-notion.md)**. Ниже — как именно применяем каждую часть.

### Design tokens (`styles/tokens.css`)

Все значения берутся из разделов 2 и 5 DESIGN-notion.md. Никаких хардкоженых цветов в компонентах:

```css
:root {
  /* palette */
  --color-bg:          #ffffff;
  --color-bg-alt:      #f6f5f4;              /* warm white для альтернации секций */
  --color-text:        rgba(0,0,0,0.95);     /* near-black, не чистый #000 */
  --color-text-muted:  #615d59;              /* warm gray 500 */
  --color-text-faint:  #a39e98;              /* warm gray 300 */
  --color-accent:      #0075de;              /* Notion Blue — единственный saturated */
  --color-accent-hover:#005bab;
  --color-focus:       #097fe8;
  --color-badge-bg:    #f2f9ff;
  --color-badge-text:  #097fe8;
  --color-success:     #1aae39;              /* вердикт Real */
  --color-warning:     #dd5b00;              /* вердикт AI */

  /* borders & shadows */
  --border-whisper:    1px solid rgba(0,0,0,0.1);
  --shadow-card:       rgba(0,0,0,0.04) 0 4px 18px,
                       rgba(0,0,0,0.027) 0 2.025px 7.84688px,
                       rgba(0,0,0,0.02) 0 0.8px 2.925px,
                       rgba(0,0,0,0.01) 0 0.175px 1.04062px;

  /* radii */
  --radius-btn:   4px;
  --radius-card:  12px;
  --radius-hero:  16px;
  --radius-pill:  9999px;

  /* spacing (8px base, non-rigid) */
  --space-section: clamp(48px, 5vw + 32px, 120px);
  --space-lg:      32px;
  --space-md:      16px;
  --space-sm:      8px;
}
```

### Типографика (`styles/typography.css`)

Стек шрифтов точно по DESIGN-notion.md §3 (NotionInter недоступен публично → используем Inter как primary, остальные fallback'и из файла):

```css
body {
  font-family: Inter, -apple-system, system-ui, "Segoe UI", Helvetica, Arial, sans-serif;
  font-feature-settings: "lnum", "locl";
  color: var(--color-text);
  background: var(--color-bg);
}
```

Иерархия — ровно по таблице из §3 DESIGN-notion.md. Ключевые роли для нашего UI:

| Где используем | Роль | Размер | Weight | letter-spacing | line-height |
|---|---|---|---|---|---|
| Hero headline («AI Image Detector») | Display Hero | 64px | 700 | -2.125px | 1.00 |
| Подзаголовок hero | Body Large | 20px | 600 | -0.125px | 1.40 |
| Заголовок секции результатов | Section Heading | 48px | 700 | -1.5px | 1.00 |
| Название модели в карточке | Card Title | 22px | 700 | -0.25px | 1.27 |
| Проценты (число) | Sub-heading Large | 40px | 700 | normal | 1.50 |
| Описание под процентом | Body | 16px | 400 | normal | 1.50 |
| Бейдж вердикта (AI / Real) | Badge | 12px | 600 | 0.125px | 1.33 |
| Кнопка «Проанализировать» | Nav / Button | 15px | 600 | normal | 1.33 |

Compression на display-размерах — это важная фишка Notion, не терять (§3, принцип «Compression at scale»).

### Макет `index.html`

Одна страница, три секции с вертикальным ритмом `var(--space-section)` между ними:

1. **Hero** (`components/hero.css`) — белый фон, центрированная колонка, max-width 1200px. Display-заголовок 64px, подзаголовок 20px warm-gray, синяя pill-кнопка как сигнал действия.
2. **Dropzone + URL input** (`components/dropzone.css`) — на фоне warm white (`--color-bg-alt`), чтобы сработала notion-альтернация секций (§5, «Warm alternation»). Сама зона drop — карточка: `--border-whisper`, `--radius-hero` (16px, featured), внутри — иконка + подсказка «Drop image or paste URL», плюс явный `<input type="file">` и `<input type="url">`.
3. **Results** (`components/result-card.css`) — белый фон, сетка: слева превью изображения (12px radius + whisper border, §4 «Image Treatment»), справа 4 карточки в grid. Первая карточка — **итог ансамбля** (крупнее, `--radius-hero`, layered `--shadow-card`, бейдж AI/Real pill'ом цвета вердикта). Остальные три — карточки моделей (12px radius, whisper border, процент крупно 40px/700, прогресс-бар, название модели 22px/700).

### Поведение `app.js`

- Обработчики `dragenter/dragover/drop` на dropzone + fallback через `<input type="file">`.
- По submit: `fetch('/api/analyze' | '/api/analyze-url')`, во время ожидания — spinner (только анимация `opacity` + `transform`, компоновочно-friendly, §8 web/coding-style).
- Рендер результатов: 4 карточки. Цвет pill-бейджа вердикта:
  - `verdict === "real"` → фон `rgba(26,174,57,0.1)`, текст `--color-success`
  - `verdict === "ai"`   → фон `rgba(221,91,0,0.1)`, текст `--color-warning`
  - `disagreement === true` → добавить маленький бейдж «Models disagree» (pill, `--color-badge-bg` / `--color-badge-text`)
- Ошибки: тост-карточка снизу с whisper-border, красная тонкая полоса слева, текст из `detail` ответа FastAPI (без внутренних трейсбеков).

### Адаптив (breakpoints из §7 DESIGN-notion.md)

- `>1200px` — полный layout, hero 64px.
- `768-1200px` — всё то же, отступы уменьшаются до `--space-lg`.
- `<768px` — hero масштабируется до 40px → 26px, грид результатов схлопывается в одну колонку, dropzone на всю ширину.
- Touch-таргеты: кнопки минимум 8-16px padding, pill-бейджи 4px/8px (§7 Touch Targets).

### Состояния (§8)

- **Focus** на всех интерактивных элементах: `outline: 2px solid var(--color-focus); outline-offset: 2px;` — обязательно, не убирать `outline: none`.
- **Hover** на кнопке CTA: фон → `--color-accent-hover`, `transform: scale(1.05)`.
- **Active**: `transform: scale(0.9)` (фирменное Notion-нажатие).
- **Disabled** (во время загрузки): текст `--color-text-faint`, `opacity: 0.6`, `cursor: not-allowed`.

### Anti-template check

Перед сдачей прогоняем фронт по чеклисту из web/design-quality.md §Component Checklist:
- Не выглядит как дефолтный Tailwind/shadcn.
- hover/focus/active — осознанные.
- Есть иерархия (контраст 64px display vs 16px body, а не одинаковый emphasis).
- Notion-альтернация секций (белый ↔ warm white) создаёт ритм без жёстких разделителей.

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

### Шаг 7 — Frontend: tokens + базовый layout
- `index.html` с разметкой трёх секций (hero / dropzone / results-placeholder).
- `styles/tokens.css` — все CSS custom properties из DESIGN-notion.md (см. выше).
- `styles/typography.css` — шрифтовая шкала.
- `styles/global.css` — reset + базовый layout.
- **Пуш:** `feat(frontend): add Notion design tokens and typography scale`

### Шаг 8 — Frontend: hero + dropzone
- `components/hero.css` — hero-секция (64px display headline, 20px subtitle, blue CTA).
- `components/dropzone.css` — drop-зона с whisper-border на warm-white фоне.
- `app.js` — `dragenter/dragover/drop` handlers + URL input.
- **Пуш:** `feat(frontend): implement hero section and drag-drop zone`

### Шаг 9 — Frontend: результаты + fetch
- `components/result-card.css` + `components/badge.css` — карточки моделей, итог-ансамбля, AI/Real pill-бейджи.
- В `app.js`: `fetch` с обработкой loading/error, рендер 4 карточек, toast на ошибках.
- Focus/hover/active состояния на всех интерактивных элементах (§8 дизайна).
- **Пуш:** `feat(frontend): render ensemble results with model cards and verdict badge`

### Шаг 10 — Адаптив и anti-template чек
- Медиа-запросы под breakpoints (§7 DESIGN-notion.md): 1200 / 768 / 400.
- Прогон по Component Checklist из web/design-quality.md.
- Проверить reduced-motion, контраст, keyboard navigation.
- **Пуш:** `feat(frontend): add responsive breakpoints and accessibility polish`

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
- Открыть `frontend/index.html` через `python -m http.server 5500` (CORS не сработает с `file://`), проверить drag-drop + URL flow, увидеть 4 карточки с результатами.
- Визуальная проверка соответствия DESIGN-notion.md:
  - hero headline: 64px, weight 700, letter-spacing -2.125px — через DevTools computed styles.
  - карточки: `box-shadow` — 4-слойный stack с max opacity 0.04.
  - альтернация секций: белый → warm white (`#f6f5f4`) → белый.
  - focus ring видим на Tab-навигации (`:focus-visible`).
- Проверить error-paths: файл >10 МБ, gif, битый URL, недоступный URL.
- Git: `main` запускается без модификаций, `dev` содержит все промежуточные коммиты.

## Что НЕ делаем (вне MVP)

- **Chrome-расширение — этап 2**. Вернёмся после того, как веб-сайт полностью готов и протестирован. Бэкенд API уже будет совместим (CORS под `chrome-extension://*` добавим на этапе 2).
- **Grad-CAM / heatmap-визуализация** — отложено по решению пользователя. Если позже понадобится для защиты курсовой — вернём отдельным эндпоинтом.
- Деплой / Docker / nginx
- БД и хранение истории запросов
- Авторизация / rate limiting (бэкенд локальный)
- Видео и GIF
