import os
from dotenv import load_dotenv

load_dotenv(override=True)

MODELS = [
    {"id": "Ateeqq/ai-vs-human-image-detector",     "weight": 0.30, "arch": "siglip"},
    {"id": "haywoodsloan/ai-image-detector-deploy", "weight": 0.35, "arch": "swinv2"},
    {"id": "umm-maybe/AI-image-detector",           "weight": 0.35, "arch": "vit"},
]
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
DEVICE = "cpu"

TEMPERATURE = 1.5    # смягчает переуверенность softmax
PATCH_SIZE = 224     # размер патча = входной размер моделей
PATCH_STRIDE = 112   # 50% overlap → ловим объекты на границе патчей
PATCH_MAX_DIM = 448  # 3×3=9 патчей с overlap, было 2×2=4 без overlap

# Roboflow — прямой вызов модели detect.roboflow.com
ROBOFLOW_API_KEY:       str = os.environ.get("ROBOFLOW_API_KEY",       "")
ROBOFLOW_MODEL_ID:      str = os.environ.get("ROBOFLOW_MODEL_ID",      "")
ROBOFLOW_MODEL_VERSION: str = os.environ.get("ROBOFLOW_MODEL_VERSION", "1")
