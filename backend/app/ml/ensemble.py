import math
from typing import TypedDict


class ModelResult(TypedDict):
    name: str
    ai_prob: float
    real_prob: float
    weight: float


class EnsembleResult(TypedDict):
    ai_prob: float
    real_prob: float
    verdict: str
    confidence: float
    disagreement: bool


def aggregate(results: list[ModelResult]) -> EnsembleResult:
    total_weight = sum(r["weight"] for r in results)
    weighted_ai = sum(r["ai_prob"] * r["weight"] for r in results) / total_weight

    verdict = "ai" if weighted_ai >= 0.5 else "real"
    confidence = round(abs(weighted_ai - 0.5) * 2, 4)

    mean = sum(r["ai_prob"] for r in results) / len(results)
    variance = sum((r["ai_prob"] - mean) ** 2 for r in results) / len(results)
    disagreement = math.sqrt(variance) > 0.2

    return {
        "ai_prob": round(weighted_ai, 4),
        "real_prob": round(1.0 - weighted_ai, 4),
        "verdict": verdict,
        "confidence": confidence,
        "disagreement": disagreement,
    }
