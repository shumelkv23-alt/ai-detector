"""Тест прямого вызова стандартной модели detect.roboflow.com."""
import os, base64, io
from dotenv import load_dotenv
load_dotenv(override=True)

API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
print(f"key={API_KEY[:8]}...\n")

from PIL import Image
import httpx

img = Image.new("RGB", (64, 64), color=(128, 128, 128))
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=90)
b64 = base64.b64encode(buf.getvalue()).decode()

url = "https://detect.roboflow.com/gemini-watermark-v2/1"
print(f"URL: {url}\n")

r = httpx.post(
    url,
    params={"api_key": API_KEY},
    content=b64,
    headers={"Content-Type": "application/x-www-form-urlencoded"},
    timeout=20.0,
)
print(f"Status: {r.status_code}")
print(f"Body:   {r.text[:600]}")
