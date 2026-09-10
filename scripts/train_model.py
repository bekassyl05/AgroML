# scripts/train_model.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train import main, parse_args

if __name__ == "__main__":
    args = parse_args()
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_name=args.model_name,
        patience=args.patience,
    )