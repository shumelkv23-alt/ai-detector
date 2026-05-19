import io
from typing import TypedDict

import numpy as np
from PIL import Image, ImageChops


ELA_QUALITY = 90
ELA_ENERGY_CEILING = 25.0
ELA_WINDOW = 128
ELA_STRIDE = 64
ELA_MAX_DIM = 1600
ELA_HOTSPOT_RATIO_CEILING = 8.0


class ELAResult(TypedDict):
    ela_score: float
    ela_grid: list[list[float]]


def _normalize_energy(energy: float) -> float:
    return round(min(1.0, energy / ELA_ENERGY_CEILING), 4)


def compute_ela(image: Image.Image) -> ELAResult:
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
    energy = np.array(residual, dtype=np.float32).mean(axis=2)

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

    median_energy = float(np.median(window_energies))
    hotspot_ratio = max_energy / max(median_energy, 1e-3)
    hotspot_score = round(min(1.0, hotspot_ratio / ELA_HOTSPOT_RATIO_CEILING), 4)

    final_score = max(base_score, hotspot_score)

    return {"ela_score": final_score, "ela_grid": grid}
