"""Downstream configuration helpers.

This module intentionally duplicates minimal config loading utilities so that
downstream remains self-contained and decoupled from mae_pretrain utils.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_yaml_config(path: str | Path) -> dict:
    """Load a YAML file into dictionary.

    Args:
        path: YAML file path.

    Returns:
        Parsed dictionary (empty dict for empty file).
    """
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def load_json_config(path: str | Path) -> dict:
    """Load a JSON file into dictionary.

    Args:
        path: JSON file path.

    Returns:
        Parsed dictionary.
    """
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)
