import math
from typing import TypedDict

from app.ml.exif_analyzer import EXIF_OVERRIDE_THRESHOLD
from app.ml.noise_analyzer import NOISE_OVERRIDE_THRESHOLD
from app.ml.fft_analyzer import FFT_OVERRIDE_THRESHOLD
from app.ml.watermark_analyzer import WATERMARK_OVERRIDE_THRESHOLD


class ModelResult(TypedDict):
    name: str
    ai_prob: float
    real_prob: float
    weight: float
    patch_max_ai: float


class EnsembleResult(TypedDict):
    ai_prob: float
    real_prob: float
    verdict: str
    confidence: float
    disagreement: bool
    ela_score: float
    exif_score: float
    noise_score: float
    fft_score: float
    watermark_score: float


ELA_OVERRIDE_THRESHOLD = 0.6
DISAGREEMENT_STD = 0.25


def aggregate(
    results: list[ModelResult],
    ela_score: float = 0.0,
    exif_score: float = 0.0,
    noise_score: float = 0.0,
    fft_score: float = 0.0,
    watermark_score: float = 0.0,
) -> EnsembleResult:
    ai_probs = [r["ai_prob"] for r in results]
    weights = [r["weight"] for r in results]
    total_weight = sum(weights)

    mean = sum(ai_probs) / len(ai_probs)
    variance = sum((p - mean) ** 2 for p in ai_probs) / len(ai_probs)
    std = math.sqrt(variance)
    disagreement = std > DISAGREEMENT_STD

    if disagreement and len(ai_probs) >= 3:
        base_ai = sorted(ai_probs)[len(ai_probs) // 2]
    else:
        base_ai = sum(p * w for p, w in zip(ai_probs, weights)) / total_weight

    weighted_patch_max = sum(r["patch_max_ai"] * r["weight"] for r in results) / total_weight
    if base_ai >= 0.4:
        final_ai = max(base_ai, weighted_patch_max)
    else:
        final_ai = base_ai

    if ela_score >= ELA_OVERRIDE_THRESHOLD:
        final_ai = max(final_ai, ela_score)
    if exif_score >= EXIF_OVERRIDE_THRESHOLD:
        final_ai = max(final_ai, exif_score)
    if noise_score >= NOISE_OVERRIDE_THRESHOLD:
        final_ai = max(final_ai, noise_score)
    if fft_score >= FFT_OVERRIDE_THRESHOLD:
        final_ai = max(final_ai, fft_score)
    if watermark_score >= WATERMARK_OVERRIDE_THRESHOLD:
        final_ai = max(final_ai, watermark_score)

    has_camera_exif = exif_score <= 0.1
    no_ai_signals = (
        watermark_score < 0.3
        and ela_score < 0.4
    )
    if has_camera_exif and no_ai_signals:
        final_ai = min(final_ai, 0.45)

    all_forensics_quiet = (
        ela_score < 0.2
        and fft_score < 0.3
        and noise_score < 0.35
        and watermark_score < 0.3
    )
    if all_forensics_quiet and base_ai < 0.85:
        final_ai *= 0.8

    final_ai = max(0.0, min(1.0, final_ai))

    verdict = "ai" if final_ai >= 0.5 else "real"
    confidence = round(abs(final_ai - 0.5) * 2, 4)

    return {
        "ai_prob": round(final_ai, 4),
        "real_prob": round(1.0 - final_ai, 4),
        "verdict": verdict,
        "confidence": confidence,
        "disagreement": disagreement,
        "ela_score": round(ela_score, 4),
        "exif_score": round(exif_score, 4),
        "noise_score": round(noise_score, 4),
        "fft_score": round(fft_score, 4),
        "watermark_score": round(watermark_score, 4),
    }
