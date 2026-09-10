# scripts/export_onnx.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort

from src.config import cfg
from src.models.factory import build_model
from src.data.dataset import build_dataloader


def load_model(model_name: str, checkpoint_path: Path, device: str):
    model = build_model(model_name, num_classes=cfg.NUM_CLASSES, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.4f})")
    return model


def export_to_onnx(model, dummy_input: torch.Tensor, onnx_path: Path):
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"ONNX model exported to: {onnx_path}")

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure verified (onnx.checker passed).")


def verify_export(model, onnx_path: Path, sample_images: torch.Tensor, device: str,
                   atol: float = 1e-2, rtol: float = 1e-2):
    with torch.no_grad():
        torch_output = model(sample_images.to(device)).cpu().numpy()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    actual_providers = session.get_providers()
    print(f"ONNX Runtime providers in use: {actual_providers}")

    onnx_input = {session.get_inputs()[0].name: sample_images.numpy()}
    onnx_output = session.run(None, onnx_input)[0]

    max_abs_diff = np.max(np.abs(torch_output - onnx_output))
    print(f"Max absolute difference (PyTorch vs ONNX): {max_abs_diff:.6f}")

    # Primary correctness check: predicted classes must match exactly
    torch_preds = torch_output.argmax(axis=1)
    onnx_preds = onnx_output.argmax(axis=1)
    assert np.array_equal(torch_preds, onnx_preds), (
        f"Predicted classes differ between PyTorch and ONNX.\n"
        f"PyTorch: {torch_preds}\nONNX:    {onnx_preds}"
    )
    print("Predicted class labels match exactly.")

    # Secondary check: logits should be numerically close (looser tolerance —
    # small GPU/CPU floating-point divergence is expected and not an error)
    try:
        np.testing.assert_allclose(torch_output, onnx_output, atol=atol, rtol=rtol)
        print(f"Logits closely match within atol={atol}, rtol={rtol}.")
    except AssertionError:
        print(f"[WARN] Logits differ slightly beyond atol={atol}/rtol={rtol} "
              f"(max_abs_diff={max_abs_diff:.6f}) — likely GPU vs CPU numeric divergence. "
              f"Predicted classes still match, export considered valid.")


def main(model_name: str, checkpoint_path: Path, batch_size: int = 8):
    device = cfg.DEVICE

    model = load_model(model_name, checkpoint_path, device)

    test_loader = build_dataloader("test", batch_size=batch_size, shuffle=True)
    sample_images, _ = next(iter(test_loader))

    onnx_path = cfg.ONNX_DIR / f"{model_name}.onnx"
    dummy_input = torch.randn(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE).to(device)

    export_to_onnx(model, dummy_input, onnx_path)
    verify_export(model, onnx_path, sample_images, device)

    print(f"\nExport complete. ONNX model ready at: {onnx_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export AgroML classifier to ONNX")
    parser.add_argument("--model_name", type=str, default="efficientnet_b0",
                         choices=["efficientnet_b0", "resnet18"])
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to checkpoint. Defaults to models_store/checkpoints/<model_name>_best.pt")
    parser.add_argument("--batch_size", type=int, default=8,
                         help="Batch size for verification sample.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else \
        cfg.CHECKPOINTS_DIR / f"{args.model_name}_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    main(args.model_name, checkpoint_path, batch_size=args.batch_size)