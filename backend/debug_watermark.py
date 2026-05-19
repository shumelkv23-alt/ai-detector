"""Показывает что именно детектирует watermark_analyzer."""
import sys
from PIL import Image
from dotenv import load_dotenv
load_dotenv(override=True)

path = sys.argv[1] if len(sys.argv) > 1 else None
if not path:
    print("Использование: python debug_watermark.py путь/к/фото.jpg")
    sys.exit(1)

image = Image.open(path).convert("RGB")

# Метаданные
from app.ml.watermark_analyzer import _scan_metadata, _roboflow_detector, _yolo_detector, _ROBOFLOW_CONF, _YOLO_CONF
gen, c2pa = _scan_metadata(image)
print(f"Metadata → generator={gen!r}, has_c2pa={c2pa}")

# Roboflow напрямую с сырыми данными
from app.config import ROBOFLOW_API_KEY, ROBOFLOW_MODEL_ID, ROBOFLOW_MODEL_VERSION
import base64, io, httpx
buf = io.BytesIO()
image.save(buf, format="JPEG", quality=90)
b64 = base64.b64encode(buf.getvalue()).decode()
url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}/{ROBOFLOW_MODEL_VERSION}"
r = httpx.post(url, params={"api_key": ROBOFLOW_API_KEY}, content=b64,
               headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=20.0)
data = r.json()
preds = data.get("predictions", [])
print(f"\nRoboflow raw predictions ({len(preds)} шт.):")
for p in preds:
    print(f"  class={p.get('class')!r}  conf={p.get('confidence', 0):.3f}  (порог={_ROBOFLOW_CONF})")
if not preds:
    print("  (пусто)")

# YOLO-World
print(f"\nYOLO-World (порог={_YOLO_CONF}):")
hits = _yolo_detector.detect(image)
if hits:
    for cls, conf in hits:
        print(f"  class={cls!r}  conf={conf:.3f}")
else:
    print("  (пусто)")
