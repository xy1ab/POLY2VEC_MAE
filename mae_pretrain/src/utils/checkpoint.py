"""Checkpoint save/export helpers for MAE pretraining.

This module centralizes checkpoint casting and export bundle generation, so the
trainer can keep a concise training loop and avoid repeated I/O code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .config import dump_yaml_config
from .filesystem import copy_if_exists, ensure_dir
from .precision import normalize_precision, precision_to_torch_dtype


def cast_state_dict_floats(state_dict: Mapping[str, Any], precision: str) -> dict[str, Any]:
    """Cast floating tensors in a state dict to target precision.

    Args:
        state_dict: Input state dict to convert.
        precision: Target precision string.

    Returns:
        Converted state dict (CPU tensors).
    """
    precision = normalize_precision(precision)
    target_dtype = precision_to_torch_dtype(precision)

    converted: dict[str, Any] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if torch.is_floating_point(tensor):
                tensor = tensor.to(dtype=target_dtype)
            converted[key] = tensor
        else:
            converted[key] = value
    return converted


def save_checkpoint(path: str | Path, state_dict: Mapping[str, Any], precision: str = "fp32") -> Path:
    """Save a checkpoint file with optional float cast.

    Args:
        path: Output checkpoint path.
        state_dict: Model state dict.
        precision: Output float precision.

    Returns:
        Saved checkpoint path.
    """
    out_path = Path(path)
    ensure_dir(out_path.parent)
    torch.save(cast_state_dict_floats(state_dict, precision), out_path)
    return out_path


def export_model_bundle(
    export_root: str | Path,
    run_name: str,
    run_config: Mapping[str, Any],
    full_state_dict: Mapping[str, Any],
    encoder_state_dict: Mapping[str, Any],
    checkpoint_precision: str,
    train_log_path: str | Path | None = None,
) -> Path:
    """Export a portable model bundle for downstream usage.

    Args:
        export_root: Root export directory.
        run_name: Folder name for this export (e.g., `mae_20260325_0830`).
        run_config: Training configuration to persist.
        full_state_dict: Full MAE checkpoint state dict.
        encoder_state_dict: Encoder-only checkpoint state dict.
        checkpoint_precision: Precision for saved weight tensors.
        train_log_path: Optional path to `train_log.txt`.

    Returns:
        Path to created export directory.
    """
    export_dir = ensure_dir(Path(export_root) / run_name)

    dump_yaml_config(dict(run_config), export_dir / "config.yaml")
    save_checkpoint(export_dir / "encoder_decoder.pth", full_state_dict, precision=checkpoint_precision)
    save_checkpoint(export_dir / "encoder.pth", encoder_state_dict, precision=checkpoint_precision)

    if train_log_path is not None:
        copy_if_exists(train_log_path, export_dir / "train_log.txt")

    return export_dir
