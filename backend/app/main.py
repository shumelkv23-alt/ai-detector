import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.config import MODELS
from app.ml.detector import registry
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

    tasks = [asyncio.to_thread(registry.predict, m, image) for m in MODELS]
    raw_results = await asyncio.gather(*tasks)

    model_results = [
        {
            "name": m["id"],
            "ai_prob": r["ai_prob"],
            "real_prob": r["real_prob"],
            "weight": m["weight"],
        }
        for m, r in zip(MODELS, raw_results)
    ]

    ensemble = aggregate(model_results)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return AnalyzeResponse(
        models=[
            {"name": r["name"], "ai_prob": r["ai_prob"], "real_prob": r["real_prob"]}
            for r in model_results
        ],
        ensemble=ensemble,
        elapsed_ms=elapsed_ms,
    )
