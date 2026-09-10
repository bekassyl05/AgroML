# src/training/train.py
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from src.config import cfg
from src.data.dataset import build_dataloader
from src.models.factory import build_model
from src.training.callbacks import EarlyStopping, CheckpointSaver
from tqdm import tqdm
from torch.amp import autocast, GradScaler

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device, enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
        pbar.set_postfix(loss=loss.item())

    return running_loss / total, running_correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, running_correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Val", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with autocast(device_type=device, enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
        pbar.set_postfix(loss=loss.item())

    return running_loss / total, running_correct / total


def main(epochs: int, batch_size: int, lr: float, model_name: str, patience: int = 7):
    device = cfg.DEVICE
    torch.manual_seed(cfg.SEED)

    train_loader = build_dataloader("train", batch_size=batch_size)
    val_loader = build_dataloader("val", batch_size=batch_size)

    model = build_model(model_name, num_classes=cfg.NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(device, enabled=(device == "cuda"))

    log_dir = cfg.LOGS_DIR / "tensorboard" / model_name
    writer = SummaryWriter(log_dir=str(log_dir))

    early_stopper = EarlyStopping(patience=patience, mode="min")
    checkpoint_saver = CheckpointSaver(cfg.CHECKPOINTS_DIR, model_name)

    print(f"Training '{model_name}' on {device} | epochs={epochs}, batch_size={batch_size}, lr={lr}")

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - start

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        is_best = early_stopper.step(val_loss)
        saved_path = checkpoint_saver.save(model, optimizer, epoch, val_loss, val_acc, is_best)

        print(f"Epoch {epoch}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{elapsed:.1f}s | {'[BEST]' if is_best else ''}")

        if early_stopper.should_stop:
            print(f"Early stopping triggered at epoch {epoch}. Best val_loss={early_stopper.best_score:.4f}")
            break

    writer.close()
    print(f"Training complete. Best checkpoint saved in {cfg.CHECKPOINTS_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train AgroML classifier")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model_name", type=str, default="efficientnet_b0",
                         choices=["efficientnet_b0", "resnet18"])
    parser.add_argument("--patience", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_name=args.model_name,
        patience=args.patience,
    )