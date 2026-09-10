# api/schemas.py
from pydantic import BaseModel, Field


class TopKPrediction(BaseModel):
    class_name: str
    label: int
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_label: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_k: list[TopKPrediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    onnx_path: str