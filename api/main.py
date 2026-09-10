# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_predictor
from api.routers import predict
from api.schemas import HealthResponse
from src.config import cfg

app = FastAPI(
    title="AgroML API",
    description="Plant disease classification API using PlantVillage-trained model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, tags=["prediction"])


@app.get("/health", response_model=HealthResponse)
def health_check():
    onnx_path = cfg.ONNX_DIR / "efficientnet_b0.onnx"
    model_loaded = False

    try:
        get_predictor()
        model_loaded = True
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        onnx_path=str(onnx_path),
    )


@app.get("/")
def root():
    return {"message": "AgroML API is running. See /docs for API documentation."}