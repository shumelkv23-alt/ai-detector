from typing import TypedDict

from PIL import Image
from PIL.ExifTags import TAGS


# Программы, однозначно указывающие на AI-генерацию.
_AI_KEYWORDS = [
    "stable diffusion", "comfyui", "automatic1111", "novelai",
    "midjourney", "dall-e", "dall·e", "dreamstudio", "firefly",
    "imagen", "leonardo", "invoke", "fooocus", "koboldai",
]

# Редакторы, которые могут говорить о постобработке/монтаже.
_EDIT_KEYWORDS = [
    "photoshop", "gimp", "lightroom", "affinity photo",
    "capture one", "luminar", "pixelmator",
]

# Порог: если exif_score >= этого значения — переопределяем вердикт на "ai".
EXIF_OVERRIDE_THRESHOLD = 0.85


class ExifResult(TypedDict):
    exif_present: bool
    has_camera_info: bool
    ai_software_detected: bool
    edit_software_detected: bool
    exif_score: float


def analyze_exif(image: Image.Image) -> ExifResult:
    """Извлекает EXIF-сигналы и возвращает вероятность AI по метаданным.

    Логика score:
    - AI software в поле Software → 0.95 (сильный сигнал)
    - Нет EXIF вообще → 0.3 (слабый, соцсети стрипают)
    - Нет камеры + есть редактор → 0.4 (среднеслабый)
    - Есть данные камеры → 0.05 (скорее реальный снимок)
    """
    try:
        raw = image.getexif()
        exif_data: dict = dict(raw) if raw else {}
    except Exception:
        exif_data = {}

    if not exif_data:
        return {
            "exif_present": False,
            "has_camera_info": False,
            "ai_software_detected": False,
            "edit_software_detected": False,
            "exif_score": 0.3,
        }

    named: dict[str, str] = {
        TAGS.get(k, str(k)): str(v) for k, v in exif_data.items()
    }

    make = named.get("Make", "").lower().strip()
    model = named.get("Model", "").lower().strip()
    software = named.get("Software", "").lower().strip()

    has_camera_info = bool(make or model)
    ai_detected = any(kw in software for kw in _AI_KEYWORDS)
    edit_detected = any(kw in software for kw in _EDIT_KEYWORDS)

    if ai_detected:
        score = 0.95
    elif has_camera_info:
        score = 0.05
    elif edit_detected:
        score = 0.4
    else:
        score = 0.3

    return {
        "exif_present": True,
        "has_camera_info": has_camera_info,
        "ai_software_detected": ai_detected,
        "edit_software_detected": edit_detected,
        "exif_score": round(score, 4),
    }
