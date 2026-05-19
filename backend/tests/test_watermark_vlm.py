"""Тесты VLM-вызова 1 (watermark) и его интеграции в watermark-сигнал."""

import asyncio
import json

from PIL import Image

from app.llm import client
from app.llm.watermark import detect_watermark_vlm, parse_watermark_json
from app.ml.watermark_analyzer import watermark_score_from_vlm


# --- parse_watermark_json -------------------------------------------------

def test_parse_valid_found():
    raw = json.dumps(
        {"watermark_found": True, "watermark_type": "Gemini", "watermark_confidence": 0.9}
    )
    result = parse_watermark_json(raw)
    assert result == {
        "watermark_found": True,
        "watermark_type": "gemini",
        "watermark_confidence": 0.9,
    }


def test_parse_not_found_normalizes_fields():
    raw = json.dumps(
        {"watermark_found": False, "watermark_type": "midjourney", "watermark_confidence": 0.8}
    )
    result = parse_watermark_json(raw)
    # found=False → тип и уверенность принудительно обнуляются.
    assert result == {
        "watermark_found": False,
        "watermark_type": "none",
        "watermark_confidence": 0.0,
    }


def test_parse_confidence_clamped():
    raw = json.dumps(
        {"watermark_found": True, "watermark_type": "x", "watermark_confidence": 5.0}
    )
    result = parse_watermark_json(raw)
    assert result is not None
    assert result["watermark_confidence"] == 1.0


def test_parse_garbage_confidence_becomes_zero():
    raw = json.dumps(
        {"watermark_found": True, "watermark_type": "x", "watermark_confidence": "abc"}
    )
    result = parse_watermark_json(raw)
    assert result is not None
    assert result["watermark_confidence"] == 0.0


def test_parse_invalid_json_returns_none():
    assert parse_watermark_json("not json at all") is None


def test_parse_non_dict_json_returns_none():
    assert parse_watermark_json("[1, 2, 3]") is None


def test_parse_empty_returns_none():
    assert parse_watermark_json("") is None
    assert parse_watermark_json(None) is None


# --- watermark_score_from_vlm --------------------------------------------

def test_score_not_found_is_zero():
    assert watermark_score_from_vlm(False, 0.9) == 0.0


def test_score_linear_scale():
    assert watermark_score_from_vlm(True, 0.0) == 0.40
    assert watermark_score_from_vlm(True, 1.0) == 0.95
    assert watermark_score_from_vlm(True, 0.70) == 0.785


def test_score_clamps_out_of_range():
    assert watermark_score_from_vlm(True, 2.0) == 0.95
    assert watermark_score_from_vlm(True, -1.0) == 0.40


# --- detect_watermark_vlm: graceful fallback без ключа --------------------

def test_detect_returns_none_without_key(monkeypatch):
    monkeypatch.setattr("app.config.OPENAI_API_KEY", "")
    client.reset_client()
    image = Image.new("RGB", (32, 32), "white")
    assert asyncio.run(detect_watermark_vlm(image)) is None
