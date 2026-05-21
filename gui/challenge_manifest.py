"""
Load PPT-exported challenge sets from challenge_inputs/manifest.json.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHALLENGE_INPUTS_DIR = os.path.join(ROOT, "challenge_inputs")
MANIFEST_FILENAME = "manifest.json"


def challenge_inputs_dir() -> str:
    env = os.environ.get("SPECKLE_CHALLENGE_INPUTS_DIR", "").strip()
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return DEFAULT_CHALLENGE_INPUTS_DIR


def manifest_path(base_dir: Optional[str] = None) -> str:
    base = base_dir or challenge_inputs_dir()
    return os.path.join(base, MANIFEST_FILENAME)


def load_challenge_manifest(base_dir: Optional[str] = None) -> Optional[dict]:
    """Return manifest dict or None if missing/invalid."""
    path = manifest_path(base_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def resolve_manifest_entries(manifest: dict, base_dir: Optional[str] = None) -> List[dict]:
    """
    Return list of {label, image, source_slide, ...} with absolute image paths.

    Skips entries whose image file is missing.
    """
    base = base_dir or challenge_inputs_dir()
    raw = manifest.get("challenges")
    if not isinstance(raw, list):
        return []

    entries: List[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        image = str(item.get("image", "")).strip()
        if not label or not image:
            continue
        if not os.path.isabs(image):
            image = os.path.join(base, os.path.basename(image))
            if not os.path.isfile(image):
                image = os.path.normpath(os.path.join(ROOT, item.get("image", "").lstrip("./")))
        if not os.path.isfile(image):
            continue
        entries.append({
            "label": label,
            "image": os.path.abspath(image),
            "source_slide": item.get("source_slide"),
            "source_file": item.get("source_file"),
        })
    return entries
