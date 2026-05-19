"""Тесты общего слоя VLM (app.llm.client)."""

from PIL import Image

from app.llm import client


def test_is_enabled_false_without_key(monkeypatch):
    monkeypatch.setattr("app.config.OPENAI_API_KEY", "")
    assert client.is_enabled() is False


def test_is_enabled_true_with_key(monkeypatch):
    monkeypatch.setattr("app.config.OPENAI_API_KEY", "sk-test")
    assert client.is_enabled() is True


def test_get_client_none_without_key(monkeypatch):
    monkeypatch.setattr("app.config.OPENAI_API_KEY", "")
    client.reset_client()
    assert client.get_client() is None


def test_encode_image_returns_data_url():
    image = Image.new("RGB", (64, 48), "red")
    encoded = client.encode_image(image)
    assert encoded.startswith("data:image/jpeg;base64,")
    assert len(encoded) > len("data:image/jpeg;base64,")


def test_encode_image_downscales_long_side():
    image = Image.new("RGB", (4000, 2000), "blue")
    # Кодирование не должно падать на крупной картинке; max_dim ограничивает.
    encoded = client.encode_image(image, max_dim=256)
    assert encoded.startswith("data:image/jpeg;base64,")
