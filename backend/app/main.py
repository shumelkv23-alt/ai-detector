from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import MODELS
from app.ml.detector import registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    for model_entry in MODELS:
        registry.load(model_entry)
    yield


app = FastAPI(title="AI Image Detector", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+",
    allow_origins=["http://127.0.0.1:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
