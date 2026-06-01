import torch

from PIL import Image

from transformers import AutoImageProcessor, AutoModelForImageClassification

from typing import TypedDict

from app.config import TEMPERATURE, PATCH_SIZE, PATCH_STRIDE, PATCH_MAX_DIM


class ModelPrediction(TypedDict):
    ai_prob: float
    real_prob: float


class PatchedPrediction(TypedDict):
    ai_prob: float
    real_prob: float
    patch_max_ai: float
    patch_grid: list[list[float]]


_REAL_KEYWORDS = ("real", "hum", "authentic", "natural")


def _normalize_label(label: str) -> str:
    label_lower = label.lower().strip()
    if any(k in label_lower for k in _REAL_KEYWORDS):
        return "real"
    return "ai"


def _logits_to_prediction(logits_row: torch.Tensor, id2label: dict) -> ModelPrediction:
    probs = torch.softmax(logits_row / TEMPERATURE, dim=-1).tolist()
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


def _extract_patches(image: Image.Image) -> tuple[list[Image.Image], tuple[int, int]]:
    w, h = image.size
    if max(w, h) > PATCH_MAX_DIM:
        scale = PATCH_MAX_DIM / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        w, h = image.size

    if w <= PATCH_SIZE and h <= PATCH_SIZE:
        return [], (1, 1)

    cols = max(1, (w + PATCH_STRIDE - 1) // PATCH_STRIDE)
    rows = max(1, (h + PATCH_STRIDE - 1) // PATCH_STRIDE)

    patches: list[Image.Image] = []
    for row in range(rows):
        for col in range(cols):
            x = min(col * PATCH_STRIDE, max(0, w - PATCH_SIZE))
            y = min(row * PATCH_STRIDE, max(0, h - PATCH_SIZE))
            patches.append(image.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE)))

    return patches, (rows, cols)


class ModelRegistry:
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

    def _run_batch(self, model_id: str, images: list[Image.Image]) -> list[ModelPrediction]:
        processor, model = self._models[model_id]
        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        id2label: dict = model.config.id2label
        return [_logits_to_prediction(logits[i], id2label) for i in range(logits.shape[0])]

    def predict(self, model_entry: dict, image: Image.Image) -> PatchedPrediction:
        model_id = model_entry["id"]
        patches, (rows, cols) = _extract_patches(image)

        all_preds = self._run_batch(model_id, [image] + patches)
        global_pred = all_preds[0]
        patch_preds = all_preds[1:]

        if patch_preds:
            patch_probs = [p["ai_prob"] for p in patch_preds]
            patch_max_ai = round(max(patch_probs), 4)
            patch_grid = [
                [round(patch_probs[r * cols + c], 4) for c in range(cols)]
                for r in range(rows)
            ]
        else:
            patch_max_ai = global_pred["ai_prob"]
            patch_grid = [[global_pred["ai_prob"]]]

        return {
            **global_pred,
            "patch_max_ai": patch_max_ai,
            "patch_grid": patch_grid,
        }


registry = ModelRegistry()
