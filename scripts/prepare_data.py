# scripts/prepare_data.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.split_dataset import run

if __name__ == "__main__":
    run()