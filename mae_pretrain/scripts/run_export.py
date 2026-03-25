"""Checkpoint export launcher script.

This script exports an encoder-only checkpoint from a full MAE checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from runtime_bootstrap import ensure_cuda_runtime_libs


def _inject_src_path() -> Path:
    """Inject local `src` directory into `sys.path`.

    Returns:
        Project root path.
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return project_root


def main() -> None:
    """CLI main function for encoder export."""
    ensure_cuda_runtime_libs()
    project_root = _inject_src_path()

    from models.factory import export_encoder_from_mae_checkpoint
    from utils.config import load_yaml_config

    parser = argparse.ArgumentParser(description="Export encoder from MAE checkpoint")
    parser.add_argument(
        "--config",
        default=str(project_root / "configs" / "export_default.yaml"),
        type=str,
    )
    parser.add_argument("--mae_ckpt_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--precision", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    mae_ckpt_path = args.mae_ckpt_path or cfg.get("mae_ckpt_path")
    config_path = args.config_path or cfg.get("config_path")
    output_path = args.output_path or cfg.get("output_path")
    precision = args.precision or cfg.get("precision", "bf16")

    if not mae_ckpt_path or not config_path or not output_path:
        raise ValueError("mae_ckpt_path, config_path, output_path must be provided via config or CLI")

    saved_path = export_encoder_from_mae_checkpoint(
        mae_ckpt_path=mae_ckpt_path,
        config_path=config_path,
        output_path=output_path,
        precision=precision,
    )
    print(f"[INFO] Exported encoder checkpoint to: {saved_path}")


if __name__ == "__main__":
    main()
