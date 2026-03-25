"""Evaluation launcher script.

This script loads evaluator defaults from YAML and forwards merged arguments to
`engine.evaluator`.
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


def _build_cli_args_from_config(config_dict: dict) -> list[str]:
    """Convert config dictionary into CLI argument list.

    Args:
        config_dict: Parsed config dictionary.

    Returns:
        Flat CLI argument list.
    """
    cli_args: list[str] = []
    for key, value in config_dict.items():
        arg_name = f"--{key}"
        if isinstance(value, bool):
            if value:
                cli_args.append(arg_name)
        else:
            cli_args.extend([arg_name, str(value)])
    return cli_args


def main() -> None:
    """CLI main function for evaluation launch."""
    ensure_cuda_runtime_libs()
    project_root = _inject_src_path()

    from engine.evaluator import run_cli
    from utils.config import load_yaml_config

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        default=str(project_root / "configs" / "eval_default.yaml"),
        type=str,
    )
    pre_args, remaining = pre_parser.parse_known_args()

    config = load_yaml_config(pre_args.config)
    config_cli_args = _build_cli_args_from_config(config)
    run_cli(config_cli_args + remaining)


if __name__ == "__main__":
    main()
