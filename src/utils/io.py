import yaml
from pathlib import Path


def load_config(path: str):
    """
    Load a YAML configuration file and return as a dictionary.
    Resolves relative paths from the project root.
    """
    # Go up from src/utils/io.py → Project root (two levels)
    root = Path(__file__).resolve().parents[2]
    full_path = root / path

    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_path}")

    with open(full_path, "r") as f:
        return yaml.safe_load(f)


def load_yaml_config(path: str):
    """
    Alias for load_config(), to maintain compatibility with training scripts.
    """
    return load_config(path)
