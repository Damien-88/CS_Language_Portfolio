"""Shared filesystem and text normalization helpers for the project."""

import json
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "outputs"


def normalize_text(text, lowercase = True):
    """Normalize Unicode representation and optionally lowercase the text."""
    normalized = unicodedata.normalize("NFKC", text).strip()
    return normalized.lower() if lowercase else normalized


def load_json_file(path):
    """Load a JSON object from disk, returning empty dict if missing."""
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json_file(path, payload):
    """Write JSON payload to disk using UTF-8 and pretty formatting."""
    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open("w", encoding = "utf-8") as handle:
        json.dump(payload, handle, ensure_ascii = False, indent = 2)


def ensure_output_dir():
    """Create and return the standard outputs directory."""
    OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
    return OUTPUT_DIR