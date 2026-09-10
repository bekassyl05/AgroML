# src/training/callbacks.py
from pathlib import Path
import torch


class EarlyStopping:
    """Stops training when validation metric stops improving."""

    def __init__(self, patience: int = 5, mode: str = "min", min_delta: float = 1e-4):
        assert mode in {"min", "max"}
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, current_score: float) -> bool:
        """Returns True if current_score is the new best."""
        if self.best_score is None:
            self.best_score = current_score
            return True

        improved = (
            current_score < self.best_score - self.min_delta
            if self.mode == "min"
            else current_score > self.best_score + self.min_delta
        )

        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


class CheckpointSaver:
    def __init__(self, save_dir: Path, model_name: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

    def save(self, model, optimizer, epoch: int, val_loss: float, val_acc: float, is_best: bool):
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "model_name": self.model_name,
        }

        last_path = self.save_dir / f"{self.model_name}_last.pt"
        torch.save(state, last_path)

        if is_best:
            best_path = self.save_dir / f"{self.model_name}_best.pt"
            torch.save(state, best_path)
            return best_path

        return last_path