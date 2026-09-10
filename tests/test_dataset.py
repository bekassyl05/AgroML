# tests/test_dataset.py
import pytest
import torch

from src.config import cfg
from src.data.dataset import PlantVillageDataset, build_dataloader
from src.data.transforms import get_train_transforms, get_val_transforms


@pytest.fixture(scope="module")
def train_csv_path():
    path = cfg.SPLITS_DIR / "train.csv"
    if not path.exists():
        pytest.skip(f"train.csv not found at {path}. Run scripts/prepare_data.py first.")
    return path


def test_dataset_loads_samples(train_csv_path):
    dataset = PlantVillageDataset(train_csv_path, transform=get_val_transforms())
    assert len(dataset) > 0


def test_dataset_getitem_shapes(train_csv_path):
    dataset = PlantVillageDataset(train_csv_path, transform=get_val_transforms())
    image_tensor, label = dataset[0]

    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.shape == (3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    assert image_tensor.dtype == torch.float32
    assert isinstance(label, int)
    assert 0 <= label < cfg.NUM_CLASSES


def test_dataset_with_train_transforms(train_csv_path):
    dataset = PlantVillageDataset(train_csv_path, transform=get_train_transforms())
    image_tensor, label = dataset[0]
    assert image_tensor.shape == (3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)


def test_dataloader_batch_shapes():
    csv_path = cfg.SPLITS_DIR / "val.csv"
    if not csv_path.exists():
        pytest.skip(f"val.csv not found at {csv_path}. Run scripts/prepare_data.py first.")

    loader = build_dataloader("val", batch_size=4, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))

    assert images.shape == (4, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    assert labels.shape == (4,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_dataset_missing_csv_raises():
    with pytest.raises(FileNotFoundError):
        PlantVillageDataset(cfg.SPLITS_DIR / "nonexistent_split.csv")