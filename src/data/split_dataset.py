# src/data/split_dataset.py
import random
from pathlib import Path
from collections import defaultdict
import csv

from src.config import cfg


def scan_dataset(raw_dir: Path) -> dict[str, list[Path]]:
    """Scan class subfolders and return {class_name: [filepaths]}."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    class_to_files = defaultdict(list)
    valid_ext = {".jpg", ".jpeg", ".png"}

    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in class_dir.iterdir():
            if f.suffix.lower() in valid_ext:
                class_to_files[class_dir.name].append(f)

    if not class_to_files:
        raise RuntimeError(f"No class folders/images found under {raw_dir}")

    return class_to_files


def stratified_split(
    class_to_files: dict[str, list[Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    rng = random.Random(seed)
    class_names = sorted(class_to_files.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_rows, val_rows, test_rows = [], [], []

    for class_name, files in class_to_files.items():
        files = files.copy()
        rng.shuffle(files)

        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # remainder goes to test to avoid dropping samples due to rounding
        n_test = n - n_train - n_val

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        label = class_to_idx[class_name]

        for f in train_files:
            train_rows.append({"filepath": str(f), "label": label, "class_name": class_name})
        for f in val_files:
            val_rows.append({"filepath": str(f), "label": label, "class_name": class_name})
        for f in test_files:
            test_rows.append({"filepath": str(f), "label": label, "class_name": class_name})

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    return train_rows, val_rows, test_rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "class_name"])
        writer.writeheader()
        writer.writerows(rows)


def run():
    class_to_files = scan_dataset(cfg.RAW_DATA_DIR)

    num_classes_found = len(class_to_files)
    if num_classes_found != cfg.NUM_CLASSES:
        print(f"[WARN] cfg.NUM_CLASSES={cfg.NUM_CLASSES} but found {num_classes_found} "
              f"class folders. Update src/config.py accordingly.")

    train_rows, val_rows, test_rows = stratified_split(
        class_to_files,
        cfg.TRAIN_RATIO,
        cfg.VAL_RATIO,
        cfg.TEST_RATIO,
        cfg.SEED,
    )

    write_csv(train_rows, cfg.SPLITS_DIR / "train.csv")
    write_csv(val_rows, cfg.SPLITS_DIR / "val.csv")
    write_csv(test_rows, cfg.SPLITS_DIR / "test.csv")

    total = len(train_rows) + len(val_rows) + len(test_rows)
    print(f"Classes found: {num_classes_found}")
    print(f"Total images: {total}")
    print(f"Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")
    print(f"Splits written to: {cfg.SPLITS_DIR}")


if __name__ == "__main__":
    run()