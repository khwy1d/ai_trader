"""Load YAML configs from the repository's configs/ folder."""
from pathlib import Path
import yaml

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"

def load_config(name: str) -> dict:
    path = REPO_CONFIGS / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
