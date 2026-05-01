import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from typing import TypedDict


class ModelPrediction(TypedDict):
    ai_prob: float
    real_prob: float


# Keywords that indicate a real/human photo
_REAL_KEYWORDS = ("real", "human", "authentic", "natural")


def _normalize_label(label: str) -> str:
    """Map any model label to 'ai' or 'real'.

    Works across different id2label schemes:
    - Ateeqq: {0: "ai", 1: "human"}
    - boluobobo: {0: "Real", 1: "Midjourney", 2: "StableDiffusion", ...}
    - haywoodsloan: {0: "artificial", 1: "real"}
    """
    label_lower = label.lower().strip()
    if any(k in label_lower for k in _REAL_KEYWORDS):
        return "real"
    return "ai"


class ModelRegistry:
    """Singleton holding loaded processor+model pairs."""

    _instance: "ModelRegistry | None" = None

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models: dict = {}
        return cls._instance

    def load(self, model_entry: dict) -> None:
        model_id = model_entry["id"]
        if model_id in self._models:
            return
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
        model.eval()
        self._models[model_id] = (processor, model)

    def predict(self, model_entry: dict, image: Image.Image) -> ModelPrediction:
        model_id = model_entry["id"]
        processor, model = self._models[model_id]

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        if isinstance(probs, float):
            probs = [probs]

        id2label: dict = model.config.id2label
        ai_prob = 0.0
        real_prob = 0.0

        for idx, prob in enumerate(probs):
            if _normalize_label(id2label[idx]) == "real":
                real_prob += prob
            else:
                ai_prob += prob

        total = ai_prob + real_prob
        if total > 0:
            ai_prob /= total
            real_prob /= total

        return {"ai_prob": round(ai_prob, 4), "real_prob": round(real_prob, 4)}


registry = ModelRegistry()
