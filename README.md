# Детектор AI-сгенерированных изображений

Курсовой проект по дисциплине «Специальный семинар».
Тема:Методы отличия изображений, созданных искусственным
интеллектом, от реальных изображений
 
## О студенте
Шумель Кирилл Викторович СДП-КБ-231

## Возможности

- **Ансамбль из трёх предобученных моделей** (SigLIP, SwinV2, ViT с HuggingFace) —
  взвешенное голосование + патч-инференс для локальных артефактов.
- **Forensic-анализ** без нейросети: ELA (error level analysis), EXIF-метаданные,
  консистентность шума, частотный анализ (FFT) и поиск водяных знаков AI-генераторов.
- **Детекция водяных знаков** через YOLO-World (zero-shot) и метаданные генераторов
  (Midjourney, DALL-E, Firefly, C2PA и др.).
- **Опциональный VLM-слой** (OpenAI gpt-4o): дополнительная проверка водяных знаков
  и человекочитаемое объяснение вердикта. Работает как fallback — без ключа всё остальное живёт.
- **Веб-интерфейс** на Vite: загрузка файла или ссылки, карточка результата с разбивкой по сигналам.

## Структура проекта

```
.
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI: /api/analyze, /analyze-url, /explain, /health
│   │   ├── config.py               # модели ансамбля, лимиты, настройки VLM
│   │   ├── schemas.py              # Pydantic-схемы запросов/ответов
│   │   ├── ml/
│   │   │   ├── detector.py         # загрузка HF-моделей, патч-инференс
│   │   │   ├── ensemble.py         # агрегация вероятностей + forensic override
│   │   │   ├── ela.py              # Error Level Analysis
│   │   │   ├── exif_analyzer.py    # анализ EXIF-метаданных
│   │   │   ├── noise_analyzer.py   # консистентность шума
│   │   │   ├── fft_analyzer.py     # частотный анализ (FFT)
│   │   │   └── watermark_analyzer.py  # YOLO-World + метаданные генераторов
│   │   ├── llm/
│   │   │   ├── client.py           # OpenAI-клиент, кодирование изображений
│   │   │   ├── watermark.py        # VLM-детекция водяных знаков
│   │   │   └── explainer.py        # VLM-объяснение вердикта
│   │   └── utils/                  # загрузка/валидация изображений, обработка ошибок
│   ├── tests/                      # pytest
│   ├── train_watermark.py          # обучение YOLO-детектора watermark'ов (опционально)
│   ├── download_dataset.py         # загрузка датасета с Roboflow (опционально)
│   ├── requirements.txt
│   └── run.py                      # запуск uvicorn (порт 8000)
└── frontend/
    ├── index.html
    ├── src/
    │   ├── main.js
    │   ├── components/             # hero, dropzone, result-card, signals, …
    │   ├── utils/                  # api.js, reveal.js, counter.js
    │   └── styles/                 # tokens.css, global.css, reset.css
    ├── vite.config.js              # dev-proxy /api → localhost:8000
    └── package.json
```

## Установка

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### Backend

```bash
cd backend

python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# CPU-версия torch (рекомендуется, экономит ~2 ГБ против CUDA-сборки):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> При первом запуске модели ансамбля (~несколько ГБ) автоматически скачиваются
> с HuggingFace и кэшируются локально. Нужен интернет.

### Frontend

```bash
cd frontend
npm install
```

## Запуск

**Backend** (FastAPI на `http://localhost:8000`):

```bash
cd backend
python run.py
```

**Frontend** (Vite dev-сервер, проксирует `/api` на бэкенд):

```bash
cd frontend
npm run dev
```

Открой адрес, который покажет Vite (обычно `http://localhost:5173`).

## Настройка окружения (опционально)

VLM-слой включается только если задан ключ OpenAI. Без него работают ансамбль и
весь forensic-анализ — просто без VLM-проверки водяных знаков и без текстовых объяснений.

Создай файл `backend/.env`:

```bash
cd backend
cp .env.example .env   # если шаблона нет — создай .env вручную
```

Переменные:

| Ключ                  | По умолчанию | Назначение                                                        |
|-----------------------|--------------|-------------------------------------------------------------------|
| `OPENAI_API_KEY`      | *(пусто)*    | Ключ OpenAI. Пусто → VLM-слой отключён, остальное работает.       |
| `OPENAI_VISION_MODEL` | `gpt-4o`     | Vision-модель для проверки водяных знаков и объяснений.           |
| `VLM_TIMEOUT`         | `30.0`       | Таймаут VLM-детекции водяного знака, сек.                          |
| `VLM_EXPLAIN_TIMEOUT` | `20.0`       | Таймаут VLM-объяснения вердикта, сек.                              |
| `VLM_MAX_IMAGE_DIM`   | `1024`       | Макс. сторона изображения, отправляемого в VLM (px).              |

## Обучение детектора водяных знаков (опционально)

По умолчанию для поиска watermark'ов используется YOLO-World в zero-shot режиме —
обучать ничего не нужно. Если хочешь собственную YOLO-модель под конкретные знаки:

1. Скачай датасет с [Roboflow Universe](https://universe.roboflow.com)
   (`python download_dataset.py --help` подскажет шаги).
2. Запусти обучение:

```bash
cd backend
python train_watermark.py                 # стандартный режим
python train_watermark.py --tiny          # для маленьких датасетов (10–50 фото/класс)
```

Лучшие веса автоматически копируются в `backend/models/watermark_yolo.pt`,
сервер подхватит их при следующем запуске.

## API

| Метод | Эндпоинт            | Назначение                                            |
|-------|---------------------|-------------------------------------------------------|
| GET   | `/api/health`       | Health-check.                                         |
| POST  | `/api/analyze`      | Анализ загруженного файла (multipart).                |
| POST  | `/api/analyze-url`  | Анализ изображения по URL (JSON `{"url": "..."}`).    |
| POST  | `/api/explain`      | VLM-объяснение вердикта по результату анализа.        |

## Тесты

```bash
cd backend
pytest
```

## Используемые технологии

- **Python 3.10+** + **FastAPI / Uvicorn** — асинхронный backend и REST API
- **PyTorch / Transformers** — ансамбль предобученных классификаторов (SigLIP, SwinV2, ViT)
- **Ultralytics (YOLO-World / YOLO11)** — zero-shot и обучаемая детекция водяных знаков
- **Pillow / NumPy** — обработка изображений, ELA, FFT, анализ шума и EXIF
- **OpenAI SDK** — опциональный VLM-слой (gpt-4o)
- **Pydantic** — валидация запросов и ответов
- **Vite (vanilla JS)** — frontend и dev-сервер с прокси на API
- **pytest** — тесты backend
```
