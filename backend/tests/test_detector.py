from unittest.mock import MagicMock
import torch
from PIL import Image
from app.ml.detector import ModelRegistry, _normalize_label


def test_normalize_label_ai_variants():
    assert _normalize_label("ai") == "ai"
    assert _normalize_label("AI") == "ai"
    assert _normalize_label("artificial") == "ai"
    assert _normalize_label("fake") == "ai"
    assert _normalize_label("Midjourney") == "ai"
    assert _normalize_label("StableDiffusion") == "ai"


def test_normalize_label_real_variants():
    assert _normalize_label("real") == "real"
    assert _normalize_label("REAL") == "real"
    assert _normalize_label("human") == "real"
    assert _normalize_label("Human") == "real"


def test_predict_returns_valid_probabilities():
    """predict() should return ai_prob + real_prob == 1.0 (mocked model)."""
    reg = ModelRegistry()

    mock_processor = MagicMock()
    mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}

    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[2.0, 1.0]])  # ai > real
    mock_model = MagicMock()
    mock_model.return_value = mock_output
    mock_model.config.id2label = {0: "ai", 1: "human"}

    reg._models["test/mock-model"] = (mock_processor, mock_model)

    result = reg.predict({"id": "test/mock-model", "weight": 1.0}, Image.new("RGB", (224, 224)))

    assert "ai_prob" in result and "real_prob" in result
    assert abs(result["ai_prob"] + result["real_prob"] - 1.0) < 1e-4
    assert result["ai_prob"] > result["real_prob"]


def test_predict_multiclass_beit_style():
    """boluobobo-style: many AI source labels + one Real label."""
    reg = ModelRegistry()

    mock_processor = MagicMock()
    mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}

    # 4 labels: Real + 3 AI sources
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[3.0, 1.0, 1.0, 1.0]])
    mock_model = MagicMock()
    mock_model.return_value = mock_output
    mock_model.config.id2label = {0: "Real", 1: "Midjourney", 2: "StableDiffusion", 3: "DALL-E"}

    reg._models["test/beit-style"] = (mock_processor, mock_model)

    result = reg.predict({"id": "test/beit-style", "weight": 0.35}, Image.new("RGB", (224, 224)))

    assert result["real_prob"] > result["ai_prob"]
    assert abs(result["ai_prob"] + result["real_prob"] - 1.0) < 1e-4
