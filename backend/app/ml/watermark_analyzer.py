import logging
from typing import TypedDict

from PIL import Image

logger = logging.getLogger(__name__)

_WATERMARK_PROMPTS: list[tuple[str, str]] = [
    ("midjourney",    "Midjourney AI art logo watermark text"),
    ("adobe_firefly", "Adobe Firefly generative AI badge logo"),
    ("dalle2_squares","DALL-E five small colored squares watermark bottom corner"),
    ("nightcafe",     "NightCafe Creator logo watermark"),
    ("wombo_dream",   "WOMBO Dream AI logo watermark"),
    ("canva_ai",      "Canva logo watermark AI generated"),
    ("craiyon",       "Craiyon AI image generator watermark text"),
    ("bing_creator",  "Microsoft Designer Bing Image Creator badge watermark"),
    ("hotpot_ai",     "Made with Hotpot AI watermark badge text"),
    ("leonardo_ai",   "Leonardo AI image generator logo watermark"),
]

_YOLO_CONF = 0.35

_AI_KEYWORDS: tuple[str, ...] = (
    "stable diffusion", "comfyui", "automatic1111", "novelai",
    "midjourney", "dall-e", "dall·e", "dalle", "dreamstudio",
    "firefly", "adobe firefly", "imagen", "leonardo", "leonardo.ai",
    "bing image creator", "designer", "ideogram", "playground ai",
    "fooocus", "invokeai", "kandinsky", "flux",
)
_C2PA_MARKERS: tuple[str, ...] = ("c2pa", "contentcredentials", "jumbf")

WATERMARK_OVERRIDE_THRESHOLD = 0.9

_WM_SCORE_BASE = 0.40
_WM_SCORE_SLOPE = 0.55


class WatermarkResult(TypedDict):
    detected_generator: str
    detected_classes: list[str]
    has_c2pa: bool
    watermark_score: float


def _conf_to_score(confidence: float) -> float:
    clamped = max(0.0, min(1.0, confidence))
    return round(_WM_SCORE_BASE + _WM_SCORE_SLOPE * clamped, 4)


def watermark_score_from_vlm(found: bool, confidence: float) -> float:
    if not found:
        return 0.0
    return _conf_to_score(confidence)


class _YoloWorldDetector:
    _instance: "_YoloWorldDetector | None" = None
    _model: object = None
    _loaded: bool = False

    def __new__(cls) -> "_YoloWorldDetector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> bool:
        if self._loaded:
            return self._model is not None
        self._loaded = True

        try:
            from ultralytics import YOLOWorld
        except ImportError:
            logger.warning("ultralytics не установлен: pip install ultralytics")
            return False

        try:
            model = YOLOWorld("yolov8s-worldv2.pt")
            model.set_classes([prompt for _, prompt in _WATERMARK_PROMPTS])
            self._model = model
            logger.info("YOLO-World загружен (%d классов)", len(_WATERMARK_PROMPTS))
            return True
        except Exception as exc:
            logger.warning("YOLO-World не загружен: %s", exc)
            return False

    def detect(self, image: Image.Image) -> list[tuple[str, float]]:
        if not self.load() or self._model is None:
            return []
        try:
            results = self._model.predict(image, verbose=False, conf=_YOLO_CONF)
            detections: list[tuple[str, float]] = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id < len(_WATERMARK_PROMPTS):
                        detections.append((_WATERMARK_PROMPTS[cls_id][0], conf))
            return detections
        except Exception as exc:
            logger.error("YOLO-World inference error: %s", exc)
            return []


def _collect_metadata_text(image: Image.Image) -> str:
    parts: list[str] = []
    try:
        exif = image.getexif()
        for value in exif.values():
            parts.append(str(value))
    except Exception:
        pass
    for key, value in (image.info or {}).items():
        parts.append(f"{key} {value}")
    return " ".join(parts).lower()


def _scan_metadata(image: Image.Image) -> tuple[str, bool]:
    haystack = _collect_metadata_text(image)
    detected = next((kw for kw in _AI_KEYWORDS if kw in haystack), "")
    has_c2pa = any(marker in haystack for marker in _C2PA_MARKERS)
    return detected, has_c2pa


_yolo_detector = _YoloWorldDetector()


def analyze_watermark(image: Image.Image) -> WatermarkResult:
    detected_generator, has_c2pa = _scan_metadata(image)

    all_hits = _yolo_detector.detect(image)

    seen: dict[str, float] = {}
    for cls, conf in all_hits:
        seen[cls] = max(seen.get(cls, 0.0), conf)

    detected_classes = list(seen.keys())
    best_conf = max(seen.values(), default=0.0)

    if detected_classes and not detected_generator:
        detected_generator = max(seen, key=lambda k: seen[k])

    if detected_classes:
        score = _conf_to_score(best_conf)
    elif detected_generator:
        score = 0.80
    elif has_c2pa:
        score = 0.55
    else:
        score = 0.0

    return {
        "detected_generator": detected_generator,
        "detected_classes": detected_classes,
        "has_c2pa": has_c2pa,
        "watermark_score": round(score, 4),
    }
