MODELS = [
    {"id": "Ateeqq/ai-vs-human-image-detector", "weight": 0.35, "arch": "siglip"},
    {"id": "boluobobo/ItsNotAI-ai-detector-v2",  "weight": 0.35, "arch": "beit"},
    {"id": "haywoodsloan/ai-image-detector",      "weight": 0.30, "arch": "swinv2"},
]
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
DEVICE = "cpu"
