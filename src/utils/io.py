import yaml
from pathlib import Path

def load_config(path: str):
    # Resolve relative to project root
    root = Path(__file__).resolve().parents[2]  # go up from src/utils/io.py → Project2/
    full_path = root / path
    with open(full_path, "r") as f:
        return yaml.safe_load(f)
