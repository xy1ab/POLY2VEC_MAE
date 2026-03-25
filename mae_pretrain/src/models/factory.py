"""Model factory and checkpoint loading APIs.

This module provides stable construction/loading entrypoints for scripts,
trainer, and downstream pipeline. It is the only module that knows how to
rebuild model topology from configuration dictionaries/files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models.encoder import PolyEncoder
from models.mae import MaskedAutoencoderViTPoly
from utils.checkpoint import save_checkpoint
from utils.config import load_config_any
from utils.precision import normalize_precision, precision_to_torch_dtype, resolve_precision_for_device


def infer_img_size_from_config(config: dict[str, Any]) -> tuple[int, int]:
    """Infer frequency-grid image size from config when `img_size` is absent.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple `(H, W)` after patch-size alignment padding.
    """
    if "img_size" in config and config["img_size"] is not None:
        img_size = config["img_size"]
        if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            return int(img_size[0]), int(img_size[1])

    pos_freqs = int(config.get("pos_freqs", 31))
    patch_size = int(config.get("patch_size", 2))

    len_h = 2 * pos_freqs + 1
    len_w = pos_freqs + 1

    pad_h = (patch_size - (len_h % patch_size)) % patch_size
    pad_w = (patch_size - (len_w % patch_size)) % patch_size
    return len_h + pad_h, len_w + pad_w


def _move_model_to_runtime_precision(model: torch.nn.Module, device: torch.device, precision: str) -> tuple[torch.nn.Module, str]:
    """Move model to device and runtime precision.

    Args:
        model: Model instance.
        device: Target device.
        precision: Requested precision.

    Returns:
        Tuple `(model, resolved_precision)`.
    """
    resolved = resolve_precision_for_device(device, precision)
    if resolved == "fp32":
        return model.to(device), resolved
    return model.to(device=device, dtype=precision_to_torch_dtype(resolved)), resolved


def build_mae_model_from_config(config: dict[str, Any], device: str | torch.device = "cpu", precision: str = "fp32") -> MaskedAutoencoderViTPoly:
    """Construct MAE model from config dictionary.

    Args:
        config: Model configuration dictionary.
        device: Runtime device.
        precision: Requested runtime precision.

    Returns:
        Constructed MAE model.
    """
    img_size = infer_img_size_from_config(config)
    model = MaskedAutoencoderViTPoly(
        img_size=img_size,
        patch_size=int(config.get("patch_size", 2)),
        in_chans=int(config.get("in_chans", 3)),
        embed_dim=int(config.get("embed_dim", 256)),
        depth=int(config.get("depth", 12)),
        num_heads=int(config.get("num_heads", 8)),
        decoder_embed_dim=int(config.get("dec_embed_dim", 128)),
        decoder_depth=int(config.get("dec_depth", 4)),
        decoder_num_heads=int(config.get("dec_num_heads", 4)),
    )
    model, _ = _move_model_to_runtime_precision(model, torch.device(device), normalize_precision(precision))
    return model


def load_mae_model(
    weight_path: str | Path,
    config_path: str | Path,
    device: str | torch.device = "cpu",
    precision: str = "fp32",
) -> tuple[MaskedAutoencoderViTPoly, dict[str, Any]]:
    """Load a full MAE model from checkpoint and config.

    Args:
        weight_path: MAE checkpoint path.
        config_path: Config file path.
        device: Runtime device.
        precision: Requested runtime precision.

    Returns:
        Tuple `(model, runtime_config)`.
    """
    config = load_config_any(config_path)
    model = build_mae_model_from_config(config, device="cpu", precision="fp32")
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model, runtime_precision = _move_model_to_runtime_precision(model, torch.device(device), precision)
    model.eval()

    runtime_config = dict(config)
    runtime_config["img_size"] = infer_img_size_from_config(runtime_config)
    runtime_config["runtime_precision"] = runtime_precision
    return model, runtime_config


def _extract_encoder_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract encoder-only state dict from a full MAE checkpoint.

    Args:
        state_dict: Full loaded checkpoint dictionary.

    Returns:
        Encoder-only state dict with keys matching `PolyEncoder`.
    """
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    if any(str(k).startswith("encoder.") for k in state_dict.keys()):
        encoder_state = {k[len("encoder.") :]: v for k, v in state_dict.items() if str(k).startswith("encoder.")}
        if encoder_state:
            return encoder_state

    return state_dict


def load_pretrained_encoder(
    weight_path: str | Path,
    config_path: str | Path,
    device: str | torch.device = "cpu",
    precision: str = "fp32",
) -> PolyEncoder:
    """Load frozen encoder weights for downstream tasks.

    Args:
        weight_path: Encoder checkpoint path or full MAE checkpoint path.
        config_path: Config file path.
        device: Runtime device.
        precision: Requested runtime precision.

    Returns:
        Frozen encoder model in eval mode.
    """
    config = load_config_any(config_path)
    img_size = infer_img_size_from_config(config)

    encoder = PolyEncoder(
        img_size=img_size,
        patch_size=int(config.get("patch_size", 2)),
        in_chans=int(config.get("in_chans", 3)),
        embed_dim=int(config.get("embed_dim", 256)),
        depth=int(config.get("depth", 12)),
        num_heads=int(config.get("num_heads", 8)),
    )

    raw_state_dict = torch.load(weight_path, map_location="cpu")
    state_dict = _extract_encoder_state_dict(raw_state_dict)
    encoder.load_state_dict(state_dict, strict=True)

    for param in encoder.parameters():
        param.requires_grad = False

    encoder, _ = _move_model_to_runtime_precision(encoder, torch.device(device), precision)
    encoder.eval()
    return encoder


def export_encoder_from_mae_checkpoint(
    mae_ckpt_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    precision: str = "fp32",
) -> Path:
    """Export encoder-only checkpoint from a full MAE checkpoint.

    Args:
        mae_ckpt_path: Full MAE checkpoint path.
        config_path: Config file path used for model build.
        output_path: Destination encoder checkpoint path.
        precision: Output precision.

    Returns:
        Saved output path.
    """
    model, _ = load_mae_model(mae_ckpt_path, config_path, device="cpu", precision="fp32")
    save_checkpoint(output_path, model.encoder.state_dict(), precision=precision)
    return Path(output_path)
