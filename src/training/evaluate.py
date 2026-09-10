# src/training/evaluate.py
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

from src.config import cfg
from src.data.dataset import build_dataloader, PlantVillageDataset
from src.models.factory import build_model


def load_class_names(split_csv: Path) -> list[str]:
    """Recover class_name -> label mapping from a split CSV (label is index-sorted)."""
    import csv
    label_to_name = {}
    with open(split_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_to_name[int(row["label"])] = row["class_name"]
    return [label_to_name[i] for i in sorted(label_to_name.keys())]


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.array(all_preds)


def evaluate(model_name: str, checkpoint_path: Path, batch_size: int = 32):
    device = cfg.DEVICE

    model = build_model(model_name, num_classes=cfg.NUM_CLASSES, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.4f})")

    test_loader = build_dataloader("test", batch_size=batch_size, shuffle=False)
    class_names = load_class_names(cfg.SPLITS_DIR / "test.csv")

    y_true, y_pred = run_inference(model, test_loader, device)

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(cfg.NUM_CLASSES)
    )

    cm = confusion_matrix(y_true, y_pred, labels=range(cfg.NUM_CLASSES))

    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f} | Macro Recall: {recall_macro:.4f} | Macro F1: {f1_macro:.4f}")

    results = {
        "model_name": model_name,
        "checkpoint_epoch": checkpoint["epoch"],
        "test_accuracy": accuracy,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "per_class": {
            class_names[i]: {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i]),
                "support": int(support_per_class[i]),
            }
            for i in range(cfg.NUM_CLASSES)
        },
        "classification_report": report,
    }

    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = cfg.LOGS_DIR / f"evaluation_{model_name}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    cm_path = cfg.LOGS_DIR / f"confusion_matrix_{model_name}.npy"
    np.save(cm_path, cm)
    print(f"Confusion matrix saved to: {cm_path}")

    cm_csv_path = cfg.LOGS_DIR / f"confusion_matrix_{model_name}.csv"
    with open(cm_csv_path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(class_names) + "\n")
        for i, row in enumerate(cm):
            f.write(class_names[i] + "," + ",".join(map(str, row)) + "\n")
    print(f"Confusion matrix CSV saved to: {cm_csv_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AgroML classifier on test split")
    parser.add_argument("--model_name", type=str, default="efficientnet_b0",
                         choices=["efficientnet_b0", "resnet18"])
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to checkpoint. Defaults to models_store/checkpoints/<model_name>_best.pt")
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else \
        cfg.CHECKPOINTS_DIR / f"{args.model_name}_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    evaluate(args.model_name, checkpoint_path, batch_size=args.batch_size)