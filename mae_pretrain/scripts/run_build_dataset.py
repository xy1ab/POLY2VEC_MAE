"""Dataset build launcher script.

This script scans vector files, triangulates polygons, and saves processed
triangle tensors into `data/processed`.
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
    """CLI main function for dataset building."""
    ensure_cuda_runtime_libs()
    project_root = _inject_src_path()

    from datasets.build_dataset import process_and_save

    parser = argparse.ArgumentParser(description="Build triangulated polygon dataset")
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="One or more raw vector directories.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(project_root / "data" / "processed" / "polygon_triangles_normalized.pt"),
    )
    args = parser.parse_args()

    process_and_save(args.input_dirs, args.output_path)


if __name__ == "__main__":
    main()
