# tests/test_api.py
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from src.config import cfg


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def onnx_model_available():
    onnx_path = cfg.ONNX_DIR / "efficientnet_b0.onnx"
    if not onnx_path.exists():
        pytest.skip(f"ONNX model not found at {onnx_path}. Run scripts/export_onnx.py first.")
    return onnx_path


@pytest.fixture
def sample_image_bytes():
    """Generate a synthetic in-memory RGB image (avoids depending on dataset files)."""
    image = Image.new("RGB", (256, 256), color=(34, 139, 34))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint(client, onnx_model_available):
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "onnx_path" in data


def test_predict_endpoint_valid_image(client, onnx_model_available, sample_image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("test_image.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200

    data = response.json()
    assert "predicted_class" in data
    assert "predicted_label" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["top_k"], list)
    assert len(data["top_k"]) > 0


def test_predict_endpoint_invalid_content_type(client, onnx_model_available):
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_endpoint_empty_file(client, onnx_model_available):
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400


def test_predict_endpoint_missing_file(client, onnx_model_available):
    response = client.post("/predict")
    assert response.status_code == 422  # FastAPI validation error for missing required field