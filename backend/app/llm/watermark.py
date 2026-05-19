import json
import logging

from PIL import Image

import app.config as config
from app.llm.client import encode_image, get_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Ты — судебный аналитик изображений. Твоя задача — определить, есть ли на "
    "изображении видимый водяной знак, подпись или плашка AI-генератора. "
    "Примеры: Gemini sparkle (четырёхлучевая звёздочка), логотип Midjourney, "
    "пять цветных квадратов DALL-E, бейдж Adobe Firefly, плашка SynthID, "
    "надпись 'Made with AI' или подобные. Не считай водяным знаком обычные "
    "сток-копирайты фотобанков и подписи фотографов. Отвечай строго в формате JSON."
)

_USER_PROMPT = (
    "Осмотри изображение на наличие водяного знака AI-генератора. "
    "Верни JSON-объект ровно с такими полями:\n"
    '{"watermark_found": true|false, '
    '"watermark_type": "тип знака строкой (например gemini, midjourney, '
    'dalle, firefly) или none", '
    '"watermark_confidence": число от 0.0 до 1.0}\n'
    "watermark_confidence — насколько ты уверен, что это именно знак "
    "AI-генератора. Если знака нет — watermark_found=false, type=none, "
    "confidence=0.0."
)

_MAX_TOKENS = 200


def _clamp01(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def parse_watermark_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("VLM watermark: ответ не распарсился как JSON")
        return None
    if not isinstance(data, dict):
        return None

    found = bool(data.get("watermark_found", False))
    raw_type = data.get("watermark_type", "none")
    watermark_type = str(raw_type).strip().lower() if raw_type else "none"
    confidence = _clamp01(data.get("watermark_confidence", 0.0))

    if not found:
        watermark_type, confidence = "none", 0.0

    return {
        "watermark_found": found,
        "watermark_type": watermark_type,
        "watermark_confidence": round(confidence, 4),
    }


async def detect_watermark_vlm(image: Image.Image) -> dict | None:
    client = get_client()
    if client is None:
        return None

    data_url = encode_image(image)
    try:
        response = await client.chat.completions.create(  # type: ignore[attr-defined]
            model=config.OPENAI_VISION_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=_MAX_TOKENS,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("VLM watermark: вызов OpenAI не удался: %s", exc)
        return None

    return parse_watermark_json(response.choices[0].message.content)
