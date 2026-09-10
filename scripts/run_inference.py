# scripts/run_inference.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

from src.config import cfg
from src.inference.predictor import Predictor


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--onnx_path", type=str, default=None,
                         help="Path to ONNX model. Defaults to models_store/onnx/efficientnet_b0.onnx")
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    onnx_path = Path(args.onnx_path) if args.onnx_path else cfg.ONNX_DIR / "efficientnet_b0.onnx"

    predictor = Predictor(onnx_path=onnx_path)
    result = predictor.predict(image_path, top_k=args.top_k)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()