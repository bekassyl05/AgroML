# api/routers/predict.py
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.dependencies import get_predictor
from api.schemas import PredictionResponse
from src.inference.predictor import Predictor

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    predictor: Predictor = Depends(get_predictor),
):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed types: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predictor.predict(image_bytes, top_k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return PredictionResponse(**result)