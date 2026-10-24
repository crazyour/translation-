from __future__ import annotations

import os
import sys
from pathlib import Path


APP_NAME = "实时英日翻译"
RESOURCE_DIR = Path(
    getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)
)


def _user_data_dir() -> Path:
    if not getattr(sys, "frozen", False):
        return RESOURCE_DIR
    if sys.platform == "darwin":
        base_dir = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base_dir = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base_dir / APP_NAME


DATA_DIR = _user_data_dir()
MODELS_DIR = DATA_DIR / "models"
CONFIG_FILE = DATA_DIR / "config.json"