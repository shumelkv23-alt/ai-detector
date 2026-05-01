from pydantic import BaseModel, Field


class ModelResult(BaseModel):
    name: str
    ai_prob: float = Field(ge=0.0, le=1.0)
    real_prob: float = Field(ge=0.0, le=1.0)


class EnsembleResult(BaseModel):
    ai_prob: float = Field(ge=0.0, le=1.0)
    real_prob: float = Field(ge=0.0, le=1.0)
    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)
    disagreement: bool


class AnalyzeResponse(BaseModel):
    models: list[ModelResult]
    ensemble: EnsembleResult
    elapsed_ms: int


class AnalyzeUrlRequest(BaseModel):
    url: str
