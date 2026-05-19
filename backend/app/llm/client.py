import base64
import io
import logging

from PIL import Image

import app.config as config

logger = logging.getLogger(__name__)

_client: object | None = None


def is_enabled() -> bool:
    return bool(config.OPENAI_API_KEY)


def get_client() -> object | None:
    global _client
    if not is_enabled():
        return None
    if _client is None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            logger.warning("Пакет openai не установлен: pip install openai")
            return None
        _client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def reset_client() -> None:
    global _client
    _client = None


def encode_image(image: Image.Image, max_dim: int | None = None) -> str:
    limit = max_dim if max_dim is not None else config.VLM_MAX_IMAGE_DIM

    rgb = image.convert("RGB")
    width, height = rgb.size
    longest = max(width, height)
    if longest > limit:
        scale = limit / longest
        rgb = rgb.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"
