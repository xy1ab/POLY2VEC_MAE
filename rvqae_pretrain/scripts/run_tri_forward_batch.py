"""Batch full-forward RVQAE export from triangles to indices/real/imag (+optional ICFT)."""

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

    parser = argparse.ArgumentParser(description="Full-forward RVQAE export for triangle shards.")
    parser.add_argument("--tri_dir", type=str, required=True, help="Directory containing triangle+meta shard files.")
    parser.add_argument("--model_dir", type=str, required=True, help="Training `best` directory or its parent.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for `tri_forward_part_*.pt`.")
    parser.add_argument("--nicft", type=int, default=0, help="ICFT output size. <=0 disables ICFT export.")
    parser.add_argument("--shard_size", type=int, default=10000, help="Number of samples per output shard.")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference micro-batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Runtime device, e.g. `cuda`, `cuda:0`, `cpu`.")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(f"`batch_size` must be > 0, got {args.batch_size}")
    if args.shard_size <= 0:
        raise ValueError(f"`shard_size` must be > 0, got {args.shard_size}")
    if args.nicft < 0:
        raise ValueError(f"`nicft` must be >= 0, got {args.nicft}")

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
        shard_prefix="tri_forward",
        manifest_name="tri_forward.manifest.json",
        shard_size=args.shard_size,
        metadata={
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "tri_dir": str(Path(args.tri_dir).expanduser().resolve()),
            "model_dir": str(Path(args.model_dir).expanduser().resolve()),
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_path),
            "batch_size": int(args.batch_size),
            "device": runtime_device,
            "nicft": int(args.nicft),
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

            imgs = pipeline.triangles_to_images(tri_batch)
            with torch.no_grad():
                outputs = pipeline.model(imgs, use_vq=True)

            indices_batch = outputs.indices.long()
            if indices_batch.ndim == 3:
                indices_batch = indices_batch.unsqueeze(1)
            indices_batch_u16 = helpers.to_uint16_indices(indices_batch, context="tri_forward")

            recon = outputs.recon_imgs.float()
            mag_valid = recon[:, 0, : pipeline.valid_h, : pipeline.valid_w]
            cos_valid = recon[:, 1, : pipeline.valid_h, : pipeline.valid_w]
            sin_valid = recon[:, 2, : pipeline.valid_h, : pipeline.valid_w]
            phase_valid = torch.atan2(sin_valid, cos_valid)
            raw_mag_valid = torch.expm1(mag_valid)
            real_batch = (raw_mag_valid * torch.cos(phase_valid)).float().cpu()
            imag_batch = (raw_mag_valid * torch.sin(phase_valid)).float().cpu()

            icft_batch = None
            if args.nicft > 0:
                icft_batch = pipeline.codec.icft_2d(
                    real_batch.to(pipeline.device),
                    f_uv_imag=imag_batch.to(pipeline.device),
                    spatial_size=args.nicft,
                ).float().cpu()

            for sample_offset in range(len(tri_batch)):
                record = {
                    "indices": indices_batch_u16[sample_offset].cpu(),
                    "meta": torch.as_tensor(meta_batch[sample_offset], dtype=torch.float32).cpu(),
                    "real": real_batch[sample_offset].float().cpu(),
                    "imag": imag_batch[sample_offset].float().cpu(),
                }
                if icft_batch is not None:
                    record["icft"] = icft_batch[sample_offset].float().cpu()
                writer.add(record)

            processed += len(tri_batch)

        print(
            f"[INFO] Forwarded shard {shard_index}/{len(tri_meta_pairs)}: "
            f"{pair.tri_path.name} | cumulative={processed}/{total_samples}"
        )

    manifest_path = writer.finalize()
    print(f"[INFO] Saved tri_forward shards to: {Path(args.output_dir).expanduser().resolve()}")
    print(f"[INFO] Manifest: {manifest_path}")
    print(f"[INFO] Exported samples: {processed}")


if __name__ == "__main__":
    main()
