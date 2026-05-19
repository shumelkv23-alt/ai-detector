"""Тесты VLM-вызова 2 (объяснение вердикта)."""

import asyncio
import json

from PIL import Image

from app.llm import client
from app.llm.explainer import explain_verdict, parse_explain_json


_ENSEMBLE = {
    "ai_prob": 0.82, "real_prob": 0.18, "verdict": "ai", "confidence": 0.64,
    "disagreement": False, "ela_score": 0.2, "exif_score": 0.3,
    "noise_score": 0.1, "fft_score": 0.6, "watermark_score": 0.9,
}


# --- parse_explain_json ---------------------------------------------------

def test_parse_valid():
    raw = json.dumps(
        {
            "explanation": "Кривые пальцы и гладкая кожа.",
            "evidence": ["шесть пальцев", "пластиковая текстура"],
            "caveat": None,
        }
    )
    result = parse_explain_json(raw)
    assert result == {
        "explanation": "Кривые пальцы и гладкая кожа.",
        "evidence": ["шесть пальцев", "пластиковая текстура"],
        "caveat": None,
    }


def test_parse_empty_explanation_returns_none():
    raw = json.dumps({"explanation": "  ", "evidence": [], "caveat": None})
    assert parse_explain_json(raw) is None


def test_parse_evidence_not_list_becomes_empty():
    raw = json.dumps({"explanation": "текст", "evidence": "строка", "caveat": None})
    result = parse_explain_json(raw)
    assert result is not None
    assert result["evidence"] == []


def test_parse_string_null_caveat_becomes_none():
    raw = json.dumps({"explanation": "текст", "evidence": [], "caveat": "null"})
    result = parse_explain_json(raw)
    assert result is not None
    assert result["caveat"] is None


def test_parse_real_caveat_kept():
    raw = json.dumps(
        {"explanation": "текст", "evidence": [], "caveat": "Визуально похоже на фото."}
    )
    result = parse_explain_json(raw)
    assert result is not None
    assert result["caveat"] == "Визуально похоже на фото."


def test_parse_invalid_json_returns_none():
    assert parse_explain_json("брокен") is None


def test_parse_empty_returns_none():
    assert parse_explain_json("") is None
    assert parse_explain_json(None) is None


# --- explain_verdict: graceful fallback без ключа -------------------------

def test_explain_returns_none_without_key(monkeypatch):
    monkeypatch.setattr("app.config.OPENAI_API_KEY", "")
    client.reset_client()
    image = Image.new("RGB", (32, 32), "white")
    assert asyncio.run(explain_verdict(image, _ENSEMBLE, None)) is None
