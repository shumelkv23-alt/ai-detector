import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import ValidationError

import app.config as config
from app.config import MODELS
from app.ml.detector import registry
from app.ml.ela import compute_ela
from app.ml.exif_analyzer import analyze_exif
from app.ml.noise_analyzer import analyze_noise_consistency
from app.ml.fft_analyzer import analyze_fft
from app.ml.watermark_analyzer import analyze_watermark, watermark_score_from_vlm
from app.ml.ensemble import aggregate
from app.llm.watermark import detect_watermark_vlm
from app.llm.explainer import explain_verdict
from app.schemas import (
    AnalyzeResponse,
    AnalyzeUrlRequest,
    ExplainContext,
    ExplainResponse,
    VlmWatermark,
)
from app.utils.errors import AppError, register_error_handlers
from app.utils.image import load_from_upload, load_from_url

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    for model_entry in MODELS:
        registry.load(model_entry)
    yield


app = FastAPI(title="AI Image Detector", lifespan=lifespan)
register_error_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile) -> AnalyzeResponse:
    image = await load_from_upload(file)
    return await _run_ensemble(image)


@app.post("/api/analyze-url", response_model=AnalyzeResponse)
async def analyze_url(payload: AnalyzeUrlRequest) -> AnalyzeResponse:
    image = await load_from_url(payload.url)
    return await _run_ensemble(image)


@app.post("/api/explain", response_model=ExplainResponse)
async def explain(
    analysis: str = Form(...),
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
) -> ExplainResponse:
    context = _parse_explain_context(analysis)
    image = await _load_explain_image(file, url)

    result = await _safe_explain(image, context)
    if result is None:
        return ExplainResponse(
            explanation="Объяснение недоступно: VLM-слой не настроен "
            "(нет OPENAI_API_KEY) или вызов не удался.",
            evidence=[],
            caveat=None,
            available=False,
        )
    return ExplainResponse(
        explanation=result["explanation"],
        evidence=result["evidence"],
        caveat=result["caveat"],
        available=True,
    )


async def _run_ensemble(image: Image.Image) -> AnalyzeResponse:
    started = time.perf_counter()

    model_tasks = [asyncio.to_thread(registry.predict, m, image) for m in MODELS]
    ela_task = asyncio.to_thread(compute_ela, image)
    exif_task = asyncio.to_thread(analyze_exif, image)
    noise_task = asyncio.to_thread(analyze_noise_consistency, image)
    fft_task = asyncio.to_thread(analyze_fft, image)
    watermark_task = asyncio.to_thread(analyze_watermark, image)
    vlm_task = _safe_vlm_watermark(image)
    *raw_results, ela, exif, noise, fft, watermark, vlm_wm = await asyncio.gather(
        *model_tasks, ela_task, exif_task, noise_task, fft_task, watermark_task, vlm_task
    )

    model_results = [
        {
            "name": m["id"],
            "ai_prob": r["ai_prob"],
            "real_prob": r["real_prob"],
            "weight": m["weight"],
            "patch_max_ai": r["patch_max_ai"],
        }
        for m, r in zip(MODELS, raw_results)
    ]

    if vlm_wm is not None:
        vlm_wm_score = watermark_score_from_vlm(
            vlm_wm["watermark_found"], vlm_wm["watermark_confidence"]
        )
    else:
        vlm_wm_score = 0.0
    watermark_score = max(watermark["watermark_score"], vlm_wm_score)

    ensemble = aggregate(
        model_results,
        ela_score=ela["ela_score"],
        exif_score=exif["exif_score"],
        noise_score=noise["noise_score"],
        fft_score=fft["fft_score"],
        watermark_score=watermark_score,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return AnalyzeResponse(
        models=[
            {
                "name": r["name"],
                "ai_prob": r["ai_prob"],
                "real_prob": r["real_prob"],
                "patch_max_ai": r["patch_max_ai"],
            }
            for r in model_results
        ],
        ensemble=ensemble,
        elapsed_ms=elapsed_ms,
        vlm_watermark=VlmWatermark(**vlm_wm) if vlm_wm is not None else None,
    )


async def _safe_vlm_watermark(image: Image.Image) -> dict | None:
    try:
        return await asyncio.wait_for(
            detect_watermark_vlm(image), timeout=config.VLM_TIMEOUT
        )
    except asyncio.TimeoutError:
        logger.warning("VLM watermark: таймаут %s с — fallback на YOLO", config.VLM_TIMEOUT)
        return None
    except Exception as exc:
        logger.warning("VLM watermark: ошибка %s — fallback на YOLO", exc)
        return None


async def _safe_explain(image: Image.Image, context: ExplainContext) -> dict | None:
    vlm_wm = context.vlm_watermark.model_dump() if context.vlm_watermark else None
    try:
        return await asyncio.wait_for(
            explain_verdict(image, context.ensemble.model_dump(), vlm_wm),
            timeout=config.VLM_EXPLAIN_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("VLM explain: таймаут %s с", config.VLM_EXPLAIN_TIMEOUT)
        return None
    except Exception as exc:
        logger.warning("VLM explain: ошибка %s", exc)
        return None


def _parse_explain_context(raw: str) -> ExplainContext:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AppError(422, f"Поле analysis — невалидный JSON: {exc}") from exc
    try:
        return ExplainContext.model_validate(data)
    except ValidationError as exc:
        raise AppError(422, f"Поле analysis не соответствует схеме: {exc}") from exc


async def _load_explain_image(
    file: UploadFile | None, url: str | None
) -> Image.Image:
    has_file = file is not None
    has_url = bool(url and url.strip())
    if has_file == has_url:
        raise AppError(400, "Нужно передать ровно одно из полей: file или url")
    if has_file:
        return await load_from_upload(file)
    return await load_from_url(url)  # type: ignore[arg-type]
