"""Batch encode triangle shards to RVQ indices with paired meta export."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    _CURRENT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _CURRENT_DIR.parent
    _REPO_ROOT = _PROJECT_ROOT.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    import importlib

    ensure_cuda_runtime_libs = importlib.import_module(
        "rvqae_pretrain.scripts.runtime_bootstrap"
    ).ensure_cuda_runtime_libs
else:
    from .runtime_bootstrap import ensure_cuda_runtime_libs


def _inject_repo_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root


def main() -> None:
    ensure_cuda_runtime_libs()
    project_root = _inject_repo_root()

    import torch

    if __package__ in {None, ""}:
        import importlib

        pipeline_module = importlib.import_module("rvqae_pretrain.src.engine.pipeline")
        helpers = importlib.import_module("rvqae_pretrain.scripts.batch_infer_common")
        PolyRvqAePipeline = pipeline_module.PolyRvqAePipeline
    else:
        from ..src.engine.pipeline import PolyRvqAePipeline
        from . import batch_infer_common as helpers

    parser = argparse.ArgumentParser(description="Encode triangle shards to RVQ indices (with meta).")
    parser.add_argument("--tri_dir", type=str, required=True, help="Directory containing triangle+meta shard files.")
    parser.add_argument("--model_dir", type=str, required=True, help="Training `best` directory or its parent.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for `tri2ind_part_*.pt`.")
    parser.add_argument("--shard_size", type=int, default=10000, help="Number of samples per output shard.")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference micro-batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Runtime device, e.g. `cuda`, `cuda:0`, `cpu`.")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(f"`batch_size` must be > 0, got {args.batch_size}")
    if args.shard_size <= 0:
        raise ValueError(f"`shard_size` must be > 0, got {args.shard_size}")

    runtime_device = helpers.resolve_runtime_device(args.device)
    checkpoint_path, config_path = helpers.resolve_model_paths(args.model_dir)

    pipeline = PolyRvqAePipeline(
        weight_path=str(checkpoint_path),
        config_path=str(config_path),
        device=runtime_device,
        precision="fp32",
    )

    tri_meta_pairs = helpers.resolve_tri_meta_pairs(args.tri_dir)
    total_samples = helpers.preflight_validate_tri_meta_pairs(tri_meta_pairs)
    print(f"[INFO] Tri/meta preflight passed: {len(tri_meta_pairs)} shard pairs, total_samples={total_samples}")

    writer = helpers.SampleShardWriter(
        output_dir=args.output_dir,
        shard_prefix="tri2ind",
        manifest_name="tri2ind.manifest.json",
        shard_size=args.shard_size,
        metadata={
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "tri_dir": str(Path(args.tri_dir).expanduser().resolve()),
            "model_dir": str(Path(args.model_dir).expanduser().resolve()),
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_path),
            "batch_size": int(args.batch_size),
            "device": runtime_device,
            "project_root": str(project_root),
        },
    )

    processed = 0
    for shard_index, pair in enumerate(tri_meta_pairs, start=1):
        tri_samples = helpers.load_torch_list(pair.tri_path)
        meta_samples = helpers.load_torch_list(pair.meta_path)
        if len(tri_samples) != len(meta_samples):
            raise ValueError(
                f"Triangle/meta sample count mismatch for {pair.tri_path.name} and {pair.meta_path.name}: "
                f"{len(tri_samples)} vs {len(meta_samples)}"
            )

        for start in range(0, len(tri_samples), args.batch_size):
            end = min(start + args.batch_size, len(tri_samples))
            tri_batch = [np.asarray(sample, dtype=np.float32) for sample in tri_samples[start:end]]
            meta_batch = meta_samples[start:end]

            indices_batch = pipeline.quantize_triangles(tri_batch)
            if indices_batch.ndim == 3:
                indices_batch = indices_batch.unsqueeze(1)
            indices_batch = helpers.to_uint16_indices(indices_batch, context="tri2ind")

            for sample_offset in range(len(tri_batch)):
                writer.add(
                    {
                        "indices": indices_batch[sample_offset].cpu(),
                        "meta": torch.as_tensor(meta_batch[sample_offset], dtype=torch.float32).cpu(),
                    }
                )
            processed += len(tri_batch)

        print(
            f"[INFO] Encoded shard {shard_index}/{len(tri_meta_pairs)}: "
            f"{pair.tri_path.name} | cumulative={processed}/{total_samples}"
        )

    manifest_path = writer.finalize()
    print(f"[INFO] Saved tri2ind shards to: {Path(args.output_dir).expanduser().resolve()}")
    print(f"[INFO] Manifest: {manifest_path}")
    print(f"[INFO] Exported samples: {processed}")


if __name__ == "__main__":
    main()
