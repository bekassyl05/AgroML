# src/inference/predictor.py
import csv
import io
from pathlib import Path
from typing import Union

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.config import cfg
from src.data.transforms import get_test_transforms


class Predictor:
    def __init__(self, onnx_path: Path = None, class_names_source: Path = None):
        self.onnx_path = Path(onnx_path) if onnx_path else cfg.ONNX_DIR / "efficientnet_b0.onnx"
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cfg.DEVICE == "cuda" \
            else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        self.transform = get_test_transforms()

        source = Path(class_names_source) if class_names_source else cfg.SPLITS_DIR / "test.csv"
        self.class_names = self._load_class_names(source)

    @staticmethod
    def _load_class_names(csv_path: Path) -> list[str]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Class names source not found: {csv_path}")

        label_to_name = {}
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_to_name[int(row["label"])] = row["class_name"]

        return [label_to_name[i] for i in sorted(label_to_name.keys())]

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image_np = np.array(image.convert("RGB"))
        augmented = self.transform(image=image_np)
        image_hwc = augmented["image"]  # numpy HWC, normalized
        image_chw = np.transpose(image_hwc, (2, 0, 1)).astype(np.float32)
        batch = np.expand_dims(image_chw, axis=0)  # (1, 3, H, W)
        return batch

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def predict(self, image: Union[str, Path, bytes, Image.Image], top_k: int = 5) -> dict:
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        elif isinstance(image, bytes):
            pil_image = Image.open(io.BytesIO(image))
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        input_tensor = self._preprocess(pil_image)
        logits = self.session.run(None, {self.input_name: input_tensor})[0]
        probs = self._softmax(logits)[0]

        top_k = min(top_k, len(self.class_names))
        top_indices = np.argsort(probs)[::-1][:top_k]

        predicted_idx = int(top_indices[0])
        result = {
            "predicted_class": self.class_names[predicted_idx],
            "predicted_label": predicted_idx,
            "confidence": float(probs[predicted_idx]),
            "top_k": [
                {
                    "class_name": self.class_names[idx],
                    "label": int(idx),
                    "confidence": float(probs[idx]),
                }
                for idx in top_indices
            ],
        }
        return result