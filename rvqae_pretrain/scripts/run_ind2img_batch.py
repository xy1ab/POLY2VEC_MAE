"""Batch decode RVQ index shards to real/imag frequency maps (and optional ICFT)."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

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
        PolyRvqDecodePipeline = pipeline_module.PolyRvqDecodePipeline
    else:
        from ..src.engine.pipeline import PolyRvqDecodePipeline
        from . import batch_infer_common as helpers

    parser = argparse.ArgumentParser(description="Decode index shards to real/imag maps.")
    parser.add_argument("--ind_dir", type=str, required=True, help="Directory containing tri2ind shard files.")
    parser.add_argument("--model_dir", type=str, required=True, help="Training `best` directory or its parent.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for `ind2img_part_*.pt`.")
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
    decoder_path, quantizer_path, config_path = helpers.resolve_decode_paths(args.model_dir)
    pipeline = PolyRvqDecodePipeline(
        decoder_path=str(decoder_path),
        quantizer_path=str(quantizer_path),
        config_path=str(config_path),
        device=runtime_device,
        precision="fp32",
    )

    ind_shards = helpers.resolve_ind_shards(args.ind_dir)
    print(f"[INFO] Resolved index shards: {len(ind_shards)}")

    writer = helpers.SampleShardWriter(
        output_dir=args.output_dir,
        shard_prefix="ind2img",
        manifest_name="ind2img.manifest.json",
        shard_size=args.shard_size,
        metadata={
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "ind_dir": str(Path(args.ind_dir).expanduser().resolve()),
            "model_dir": str(Path(args.model_dir).expanduser().resolve()),
            "decoder_path": str(decoder_path),
            "quantizer_path": str(quantizer_path),
            "config_path": str(config_path),
            "batch_size": int(args.batch_size),
            "device": runtime_device,
            "nicft": int(args.nicft),
            "project_root": str(project_root),
        },
    )

    processed = 0
    for shard_index, shard_path in enumerate(ind_shards, start=1):
        samples = helpers.load_torch_list(shard_path)
        for start in range(0, len(samples), args.batch_size):
            end = min(start + args.batch_size, len(samples))
            batch_samples = samples[start:end]

            batch_indices = []
            batch_meta = []
            for local_index, sample in enumerate(batch_samples):
                if not isinstance(sample, dict):
                    raise TypeError(f"Index shard sample must be dict: {shard_path}#{start + local_index}")
                if "indices" not in sample or "meta" not in sample:
                    raise KeyError(f"Index shard sample missing `indices`/`meta`: {shard_path}#{start + local_index}")
                batch_indices.append(helpers.normalize_indices_grid(sample["indices"]))
                batch_meta.append(sample["meta"])

            indices_batch = torch.stack(batch_indices, dim=0)
            indices_batch_u16 = helpers.to_uint16_indices(indices_batch, context="ind2img")
            real_batch, imag_batch = pipeline.decode_indices(indices_batch)

            icft_batch = None
            if args.nicft > 0:
                icft_batch = pipeline.codec.icft_2d(
                    real_batch.to(pipeline.device),
                    f_uv_imag=imag_batch.to(pipeline.device),
                    spatial_size=args.nicft,
                ).float().cpu()

            for sample_offset in range(len(batch_samples)):
                record = {
                    "indices": indices_batch_u16[sample_offset].cpu(),
                    "meta": torch.as_tensor(batch_meta[sample_offset], dtype=torch.float32).cpu(),
                    "real": real_batch[sample_offset].float().cpu(),
                    "imag": imag_batch[sample_offset].float().cpu(),
                }
                if icft_batch is not None:
                    record["icft"] = icft_batch[sample_offset].float().cpu()
                writer.add(record)

            processed += len(batch_samples)

        print(
            f"[INFO] Decoded shard {shard_index}/{len(ind_shards)}: "
            f"{shard_path.name} | cumulative={processed}"
        )

    manifest_path = writer.finalize()
    print(f"[INFO] Saved ind2img shards to: {Path(args.output_dir).expanduser().resolve()}")
    print(f"[INFO] Manifest: {manifest_path}")
    print(f"[INFO] Exported samples: {processed}")


if __name__ == "__main__":
    main()
