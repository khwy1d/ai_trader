"""Single place that knows where things live."""
import os
from pathlib import Path

DEFAULT_DATA_ROOT = "/content/drive/MyDrive/ai_trader_data"

def get_data_root() -> Path:
    return Path(os.environ.get("AI_TRADER_DATA_ROOT", DEFAULT_DATA_ROOT))

def data_dir(stage: str) -> Path:
    allowed = {"raw", "processed", "features", "labels", "models", "runs"}
    if stage not in allowed:
        raise ValueError(f"Unknown stage {stage!r}; allowed: {sorted(allowed)}")
    return get_data_root() / stage
