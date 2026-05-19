import json
import logging

from PIL import Image

import app.config as config
from app.llm.client import encode_image, get_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Ты — судебный аналитик изображений. Ансамбль специализированных "
    "детекторов уже вынес вердикт: фотография реальная или сгенерирована ИИ. "
    "Твоя задача — посмотреть на изображение и понятным языком объяснить, "
    "какие видимые признаки согласуются с этим вердиктом. Пиши по-русски, "
    "без воды, для защиты курсовой работы. Если визуально ты НЕ согласен с "
    "вердиктом — честно отметь это в поле caveat, не подстраивайся под него. "
    "Отвечай строго в формате JSON."
)

_VERDICT_LABEL = {"ai": "сгенерировано ИИ", "real": "реальная фотография"}

_MAX_TOKENS = 600


def _format_context(ensemble: dict, vlm_watermark: dict | None) -> str:
    verdict = str(ensemble.get("verdict", "")).lower()
    verdict_human = _VERDICT_LABEL.get(verdict, verdict or "неизвестно")

    lines = [
        f"Вердикт ансамбля: {verdict_human}.",
        f"AI-вероятность: {ensemble.get('ai_prob')}, "
        f"уверенность: {ensemble.get('confidence')}.",
        f"Детекторы между собой {'расходятся' if ensemble.get('disagreement') else 'согласны'}.",
        "Сигналы forensics (0 — чисто, 1 — аномалия): "
        f"ELA={ensemble.get('ela_score')}, EXIF={ensemble.get('exif_score')}, "
        f"шум={ensemble.get('noise_score')}, FFT={ensemble.get('fft_score')}, "
        f"watermark={ensemble.get('watermark_score')}.",
    ]
    if vlm_watermark and vlm_watermark.get("watermark_found"):
        lines.append(
            "Обнаружен водяной знак: "
            f"{vlm_watermark.get('watermark_type')} "
            f"(уверенность {vlm_watermark.get('watermark_confidence')})."
        )
    return "\n".join(lines)


def _build_user_prompt(ensemble: dict, vlm_watermark: dict | None) -> str:
    return (
        f"{_format_context(ensemble, vlm_watermark)}\n\n"
        "Посмотри на изображение и верни JSON-объект ровно с такими полями:\n"
        '{"explanation": "2-4 предложения: почему вердикт именно такой, '
        'опираясь на видимые признаки изображения", '
        '"evidence": ["короткий признак 1", "короткий признак 2"], '
        '"caveat": "оговорка, если ты визуально не согласен с вердиктом, '
        'иначе null"}'
    )


def parse_explain_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("VLM explain: ответ не распарсился как JSON")
        return None
    if not isinstance(data, dict):
        return None

    explanation = str(data.get("explanation", "")).strip()
    if not explanation:
        return None

    raw_evidence = data.get("evidence", [])
    evidence: list[str] = []
    if isinstance(raw_evidence, list):
        evidence = [str(item).strip() for item in raw_evidence if str(item).strip()]

    raw_caveat = data.get("caveat")
    caveat = str(raw_caveat).strip() if raw_caveat else None
    if caveat and caveat.lower() in ("null", "none", "нет"):
        caveat = None

    return {"explanation": explanation, "evidence": evidence, "caveat": caveat}


async def explain_verdict(
    image: Image.Image,
    ensemble: dict,
    vlm_watermark: dict | None,
) -> dict | None:
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
                        {"type": "text", "text": _build_user_prompt(ensemble, vlm_watermark)},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=_MAX_TOKENS,
            temperature=0.3,
        )
    except Exception as exc:
        logger.warning("VLM explain: вызов OpenAI не удался: %s", exc)
        return None

    return parse_explain_json(response.choices[0].message.content)
