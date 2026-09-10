# src/data/dataset.py
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import cfg
from src.data.transforms import get_train_transforms, get_val_transforms


class PlantVillageDataset(Dataset):
    def __init__(self, csv_path: Path, transform=None):
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["filepath"], int(row["label"])))

        if not self.samples:
            raise RuntimeError(f"No samples loaded from {self.csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        filepath, label = self.samples[idx]

        image = cv2.imread(filepath)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {filepath}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        return image_tensor, label


def build_dataloader(split: str, batch_size: int = None, shuffle: bool = None,
                      num_workers: int = None) -> DataLoader:
    split = split.lower()
    assert split in {"train", "val", "test"}, "split must be 'train', 'val', or 'test'"

    csv_path = cfg.SPLITS_DIR / f"{split}.csv"
    transform = get_train_transforms() if split == "train" else get_val_transforms()
    dataset = PlantVillageDataset(csv_path, transform=transform)

    if shuffle is None:
        shuffle = (split == "train")
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE
    if num_workers is None:
        num_workers = cfg.NUM_WORKERS

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(cfg.DEVICE == "cuda"),
    )


if __name__ == "__main__":
    loader = build_dataloader("train")
    images, labels = next(iter(loader))
    print(f"Batch images shape: {images.shape}")   # (B, 3, H, W)
    print(f"Batch labels shape: {labels.shape}")   # (B,)
    print(f"Image dtype: {images.dtype}, Label dtype: {labels.dtype}")
    print(f"Sample label values: {labels[:5].tolist()}")