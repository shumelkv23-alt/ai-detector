from pydantic import BaseModel, Field


class ModelResult(BaseModel):
    name: str
    ai_prob: float = Field(ge=0.0, le=1.0)
    real_prob: float = Field(ge=0.0, le=1.0)
    patch_max_ai: float = Field(ge=0.0, le=1.0)


class EnsembleResult(BaseModel):
    ai_prob: float = Field(ge=0.0, le=1.0)
    real_prob: float = Field(ge=0.0, le=1.0)
    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)
    disagreement: bool
    ela_score: float = Field(ge=0.0, le=1.0)
    exif_score: float = Field(ge=0.0, le=1.0)
    noise_score: float = Field(ge=0.0, le=1.0)
    fft_score: float = Field(ge=0.0, le=1.0)
    watermark_score: float = Field(ge=0.0, le=1.0)


class VlmWatermark(BaseModel):
    watermark_found: bool
    watermark_type: str
    watermark_confidence: float = Field(ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    models: list[ModelResult]
    ensemble: EnsembleResult
    elapsed_ms: int
    vlm_watermark: VlmWatermark | None = None


class AnalyzeUrlRequest(BaseModel):
    url: str


class ExplainContext(BaseModel):
    ensemble: EnsembleResult
    vlm_watermark: VlmWatermark | None = None


class ExplainResponse(BaseModel):
    explanation: str
    evidence: list[str]
    caveat: str | None = None
    available: bool
