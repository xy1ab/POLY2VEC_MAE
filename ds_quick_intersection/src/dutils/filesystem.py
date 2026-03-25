"""Downstream filesystem helpers.

This module provides local directory creation helpers and avoids importing
mae_pretrain utility modules to preserve module-level decoupling.
"""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path.

    Args:
        path: Directory path.

    Returns:
        Created/existing Path object.
    """
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
