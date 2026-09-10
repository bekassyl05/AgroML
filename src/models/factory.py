# src/models/factory.py
import torch.nn as nn

from src.models.architecture import EfficientNetB0Classifier, ResNet18Classifier

_MODEL_REGISTRY = {
    "efficientnet_b0": EfficientNetB0Classifier,
    "resnet18": ResNet18Classifier,
}


def build_model(name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    name = name.lower()
    if name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model name '{name}'. Available: {available}")

    model_cls = _MODEL_REGISTRY[name]
    return model_cls(num_classes=num_classes, pretrained=pretrained, dropout=dropout)


if __name__ == "__main__":
    import torch
    from src.config import cfg

    for model_name in _MODEL_REGISTRY:
        model = build_model(model_name, num_classes=cfg.NUM_CLASSES)
        dummy_input = torch.randn(2, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
        output = model(dummy_input)
        print(f"{model_name}: output shape = {output.shape}")
        assert output.shape == (2, cfg.NUM_CLASSES), f"Unexpected output shape for {model_name}"

    print("All models passed smoke test.")