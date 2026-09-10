# api/dependencies.py
from functools import lru_cache

from src.config import cfg
from src.inference.predictor import Predictor


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """
    Singleton loader for the Predictor. lru_cache ensures the ONNX model
    and onnxruntime session are loaded only once, on first request.
    """
    onnx_path = cfg.ONNX_DIR / "efficientnet_b0.onnx"
    return Predictor(onnx_path=onnx_path)