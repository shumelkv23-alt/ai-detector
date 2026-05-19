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

TEMPERATURE = 1.5
PATCH_SIZE = 224
PATCH_STRIDE = 112
PATCH_MAX_DIM = 448

OPENAI_API_KEY:      str   = os.environ.get("OPENAI_API_KEY",      "")
OPENAI_VISION_MODEL: str   = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o")
VLM_TIMEOUT:         float = float(os.environ.get("VLM_TIMEOUT",         "30.0"))
VLM_EXPLAIN_TIMEOUT: float = float(os.environ.get("VLM_EXPLAIN_TIMEOUT", "20.0"))
VLM_MAX_IMAGE_DIM:   int   = int(os.environ.get("VLM_MAX_IMAGE_DIM",     "1024"))
