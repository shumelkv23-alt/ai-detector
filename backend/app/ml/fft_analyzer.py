"""Спектральный анализ FFT по локальным патчам.

GAN и диффузионные модели оставляют характерные следы в частотном спектре:
GAN — периодические артефакты на высоких частотах; диффузия — более гладкий
спектр в определённых диапазонах. Высокая inter-patch вариация
нормализованных радиальных профилей → регионы с разной частотной сигнатурой.
"""

import numpy as np
from PIL import Image
from typing import TypedDict


FFT_PATCH = 128
FFT_STRIDE = 64
FFT_MAX_DIM = 1024
# Потолок inter-patch spectral variance (эмпирический, требует тюнинга)
FFT_VARIANCE_CEILING = 0.015
FFT_OVERRIDE_THRESHOLD = 0.7


class FFTResult(TypedDict):
    fft_score: float


def _radial_profile(magnitude: np.ndarray) -> np.ndarray:
    """Радиально усреднённый спектр — агрегируем по кольцам вокруг центра."""
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.ogrid[:h, :w]
    r = np.round(np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)).astype(int)
    max_r = min(cx, cy)
    return np.array(
        [magnitude[r == i].mean() if np.any(r == i) else 0.0 for i in range(max_r)]
    )


def analyze_fft(image: Image.Image) -> FFTResult:
    """Возвращает score спектральной аномалии: 0.0 = равномерный, 1.0 = аномалия."""
    w, h = image.size
    if max(w, h) > FFT_MAX_DIM:
        scale = FFT_MAX_DIM / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

    gray = np.array(image.convert("L"), dtype=np.float32)
    img_h, img_w = gray.shape

    if img_h < FFT_PATCH or img_w < FFT_PATCH:
        return {"fft_score": 0.0}

    profiles: list[np.ndarray] = []
    for y in range(0, img_h - FFT_PATCH + 1, FFT_STRIDE):
        for x in range(0, img_w - FFT_PATCH + 1, FFT_STRIDE):
            patch = gray[y : y + FFT_PATCH, x : x + FFT_PATCH]
            fshift = np.fft.fftshift(np.fft.fft2(patch))
            magnitude = np.log1p(np.abs(fshift))
            profiles.append(_radial_profile(magnitude))

    if len(profiles) < 4:
        return {"fft_score": 0.0}

    arr = np.array(profiles)
    # Нормализуем каждый профиль в [0..1] перед сравнением
    mx = arr.max(axis=1, keepdims=True)
    mx = np.where(mx < 1e-6, 1.0, mx)
    arr_norm = arr / mx

    inter_variance = float(np.var(arr_norm, axis=0).mean())
    return {"fft_score": round(min(1.0, inter_variance / FFT_VARIANCE_CEILING), 4)}
