import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.config import MODELS
from app.ml.detector import registry
from app.ml.ela import compute_ela
from app.ml.exif_analyzer import analyze_exif
from app.ml.noise_analyzer import analyze_noise_consistency
from app.ml.fft_analyzer import analyze_fft
from app.ml.watermark_analyzer import analyze_watermark
from app.ml.ensemble import aggregate
from app.schemas import AnalyzeResponse, AnalyzeUrlRequest
from app.utils.errors import register_error_handlers
from app.utils.image import load_from_upload, load_from_url


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


async def _run_ensemble(image: Image.Image) -> AnalyzeResponse:
    started = time.perf_counter()

    model_tasks = [asyncio.to_thread(registry.predict, m, image) for m in MODELS]
    ela_task = asyncio.to_thread(compute_ela, image)
    exif_task = asyncio.to_thread(analyze_exif, image)
    noise_task = asyncio.to_thread(analyze_noise_consistency, image)
    fft_task = asyncio.to_thread(analyze_fft, image)
    watermark_task = asyncio.to_thread(analyze_watermark, image)
    *raw_results, ela, exif, noise, fft, watermark = await asyncio.gather(
        *model_tasks, ela_task, exif_task, noise_task, fft_task, watermark_task
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

    ensemble = aggregate(
        model_results,
        ela_score=ela["ela_score"],
        exif_score=exif["exif_score"],
        noise_score=noise["noise_score"],
        fft_score=fft["fft_score"],
        watermark_score=watermark["watermark_score"],
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
    )
