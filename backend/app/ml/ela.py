"""Error Level Analysis — детектит локальные манипуляции в изображениях.

Идея: повторное JPEG-сжатие c quality=90 даёт residual, который в нетронутых
областях минимален (картинка уже была сжата похожим образом), а в AI-вставках
или отредактированных регионах резко выше. Это классический forensics-метод,
описанный Krawetz (2007).
"""

import io
from typing import TypedDict

import numpy as np
from PIL import Image, ImageChops


ELA_QUALITY = 90
# Эмпирический потолок энергии residual для нормализации в [0..1].
# Чистый JPEG обычно даёт средний residual 2-5, манипуляции 15-30+.
ELA_ENERGY_CEILING = 25.0
# Размер скользящего окна в пикселях для поиска максимума энергии.
ELA_WINDOW = 128
ELA_STRIDE = 64
# Защита от ОЗУ: ограничиваем длинную сторону, но достаточно крупно,
# чтобы сохранить high-frequency артефакты (LANCZOS-ресайз убивал ELA при 448).
ELA_MAX_DIM = 1600
# Hotspot detection: если max энергия превышает медианную в N раз → локальная
# аномалия (вставленный AI-объект в чистое фото). На реальных фото отношение
# обычно <3, на composite — 5-15+.
ELA_HOTSPOT_RATIO_CEILING = 8.0


class ELAResult(TypedDict):
    ela_score: float
    ela_grid: list[list[float]]


def _normalize_energy(energy: float) -> float:
    return round(min(1.0, energy / ELA_ENERGY_CEILING), 4)


def compute_ela(image: Image.Image) -> ELAResult:
    """Возвращает максимум локальной ELA-энергии и grid энергий по патчам.

    ВАЖНО: работаем на близком к оригиналу разрешении (только мягкий
    BICUBIC-ресайз при длинной стороне > ELA_MAX_DIM), потому что
    LANCZOS-ресайз до маленького размера выкидывает high-frequency
    информацию, на которой основан ELA.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    if max(w, h) > ELA_MAX_DIM:
        scale = ELA_MAX_DIM / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        w, h = image.size

    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=ELA_QUALITY)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    residual = ImageChops.difference(image, recompressed)
    energy = np.array(residual, dtype=np.float32).mean(axis=2)  # [H, W]

    if w <= ELA_WINDOW and h <= ELA_WINDOW:
        score = _normalize_energy(float(energy.mean()))
        return {"ela_score": score, "ela_grid": [[score]]}

    cols = max(1, (w - ELA_WINDOW) // ELA_STRIDE + 1)
    rows = max(1, (h - ELA_WINDOW) // ELA_STRIDE + 1)

    grid: list[list[float]] = []
    window_energies: list[float] = []
    for row in range(rows):
        grid_row: list[float] = []
        for col in range(cols):
            x = min(col * ELA_STRIDE, max(0, w - ELA_WINDOW))
            y = min(row * ELA_STRIDE, max(0, h - ELA_WINDOW))
            window_energy = float(energy[y : y + ELA_WINDOW, x : x + ELA_WINDOW].mean())
            grid_row.append(_normalize_energy(window_energy))
            window_energies.append(window_energy)
        grid.append(grid_row)

    max_energy = max(window_energies)
    base_score = _normalize_energy(max_energy)

    # Hotspot detection: max / median. Реальные фото: ratio ~1.5-3.
    # Composite (AI-вставка в JPEG): ratio 5-15+, потому что вставленная область
    # ещё не подверглась той же истории сжатия, что и остальное изображение.
    median_energy = float(np.median(window_energies))
    hotspot_ratio = max_energy / max(median_energy, 1e-3)
    hotspot_score = round(min(1.0, hotspot_ratio / ELA_HOTSPOT_RATIO_CEILING), 4)

    # Финальный score — максимум из глобального и hotspot. Это ловит
    # как полностью AI-сжатые изображения (высокий global), так и локальные
    # вставки в чистые фото (низкий global, но высокий hotspot).
    final_score = max(base_score, hotspot_score)

    return {"ela_score": final_score, "ela_grid": grid}
