"""Анализ консистентности шума по локальным патчам.

В реальных снимках sensor noise относительно равномерен по всему кадру.
AI-вставка имеет другой характер шума: диффузионные модели дают более
гладкие текстуры, отличную дисперсию. Высокий коэффициент вариации
локальных дисперсий шума → аномалия консистентности.
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import TypedDict


NOISE_BLUR_RADIUS = 2
NOISE_PATCH = 64
NOISE_STRIDE = 32
NOISE_MAX_DIM = 1200
# Потолок CV. Реальные фото с разнообразными текстурами легко дают CV 1.5-2.5,
# поэтому потолок завышен — иначе насыщение в 1.0 у любого нормального снимка.
NOISE_CV_CEILING = 3.5
# Override отключён по умолчанию: разделение real vs composite слишком слабое
# для надёжного hard-veto. Сигнал используется только как value в отчёте.
NOISE_OVERRIDE_THRESHOLD = 0.95


class NoiseResult(TypedDict):
    noise_score: float


def analyze_noise_consistency(image: Image.Image) -> NoiseResult:
    """Возвращает score аномальности шума: 0.0 = равномерный, 1.0 = сильная аномалия."""
    gray_img = image.convert("L")

    w, h = gray_img.size
    if max(w, h) > NOISE_MAX_DIM:
        scale = NOISE_MAX_DIM / max(w, h)
        gray_img = gray_img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

    # Шум = оригинал − low-frequency component
    denoised_img = gray_img.filter(ImageFilter.GaussianBlur(radius=NOISE_BLUR_RADIUS))
    noise = np.array(gray_img, dtype=np.float32) - np.array(denoised_img, dtype=np.float32)

    img_h, img_w = noise.shape
    if img_h < NOISE_PATCH or img_w < NOISE_PATCH:
        return {"noise_score": 0.0}

    variances: list[float] = []
    for y in range(0, img_h - NOISE_PATCH + 1, NOISE_STRIDE):
        for x in range(0, img_w - NOISE_PATCH + 1, NOISE_STRIDE):
            variances.append(float(np.var(noise[y : y + NOISE_PATCH, x : x + NOISE_PATCH])))

    if len(variances) < 4:
        return {"noise_score": 0.0}

    mean_var = float(np.mean(variances))
    if mean_var < 1e-6:
        return {"noise_score": 0.0}

    cv = float(np.std(variances)) / mean_var
    return {"noise_score": round(min(1.0, cv / NOISE_CV_CEILING), 4)}
