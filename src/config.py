# src/config.py
from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass(frozen=True)
class Config:
    # Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw" / "PlantVillage" / "plantvillage dataset" / "color"
    SPLITS_DIR: Path = PROJECT_ROOT / "data" / "splits"
    PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    CHECKPOINTS_DIR: Path = PROJECT_ROOT / "models_store" / "checkpoints"
    ONNX_DIR: Path = PROJECT_ROOT / "models_store" / "onnx"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # Data
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 2
    NUM_CLASSES: int = 38  # PlantVillage color subset; verify against actual folder count

    # Split ratios
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # Training
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        for d in [self.SPLITS_DIR, self.PROCESSED_DIR, self.CHECKPOINTS_DIR,
                  self.ONNX_DIR, self.LOGS_DIR]:
            d.mkdir(parents=True, exist_ok=True)


cfg = Config()