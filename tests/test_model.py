# tests/test_model.py
import pytest
import torch

from src.config import cfg
from src.models.factory import build_model


@pytest.mark.parametrize("model_name", ["efficientnet_b0", "resnet18"])
def test_model_forward_pass_shape(model_name):
    model = build_model(model_name, num_classes=cfg.NUM_CLASSES, pretrained=False)
    model.eval()

    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (batch_size, cfg.NUM_CLASSES)
    assert output.dtype == torch.float32


@pytest.mark.parametrize("model_name", ["efficientnet_b0", "resnet18"])
def test_model_forward_single_sample(model_name):
    model = build_model(model_name, num_classes=cfg.NUM_CLASSES, pretrained=False)
    model.eval()

    dummy_input = torch.randn(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, cfg.NUM_CLASSES)


def test_build_model_invalid_name_raises():
    with pytest.raises(ValueError):
        build_model("nonexistent_architecture", num_classes=cfg.NUM_CLASSES)


def test_model_output_is_finite():
    model = build_model("efficientnet_b0", num_classes=cfg.NUM_CLASSES, pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    with torch.no_grad():
        output = model(dummy_input)

    assert torch.isfinite(output).all()