"""Downstream MAE reconstruction demo script.

This script demonstrates downstream usage that depends only on pretrain public
pipeline API (`engine/pipeline.py`) and exported/checkpoint artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union


def _inject_paths() -> tuple[Path, Path]:
    """Inject downstream/src and mae_pretrain/src into import paths.

    Returns:
        Tuple `(downstream_root, project_root)`.
    """
    script_dir = Path(__file__).resolve().parent
    downstream_root = script_dir.parent
    project_root = downstream_root.parent

    downstream_src = downstream_root / "src"
    pretrain_src = project_root / "mae_pretrain" / "src"

    if str(downstream_src) not in sys.path:
        sys.path.insert(0, str(downstream_src))
    if str(pretrain_src) not in sys.path:
        sys.path.insert(0, str(pretrain_src))

    return downstream_root, project_root


def _build_args(default_config_path: Path):
    """Parse CLI args with YAML defaults.

    Args:
        default_config_path: Default downstream config path.

    Returns:
        Parsed argparse namespace.
    """
    from dutils.config import load_yaml_config

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=str(default_config_path))
    pre_args, remaining = pre_parser.parse_known_args()

    defaults = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Downstream MAE reconstruction demo")
    parser.add_argument("--config", type=str, default=pre_args.config)
    parser.add_argument("--shape", type=str, default="circle", choices=["dataset", "circle", "pentagon"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--mask_ratio", type=float, default=0.0)
    parser.add_argument("--experiment_dir", type=str, default="")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--save_dir", type=str, default="")
    parser.set_defaults(**defaults)

    return parser.parse_args(remaining)


def _resolve_latest_run_dir(experiment_dir: Path) -> Path:
    """Resolve latest timestamped run directory.

    Args:
        experiment_dir: Path to a run directory or a parent ckpt directory.

    Returns:
        Resolved run directory containing model config.
    """
    if (experiment_dir / "poly_mae_config.json").is_file():
        return experiment_dir

    candidates = [path for path in experiment_dir.iterdir() if path.is_dir() and (path / "poly_mae_config.json").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No run directory with poly_mae_config.json found under: {experiment_dir}")

    return sorted(candidates)[-1]


def _resolve_mae_ckpt(run_dir: Path) -> Path:
    """Resolve latest MAE checkpoint file inside run directory.

    Args:
        run_dir: Run directory containing checkpoints.

    Returns:
        Selected checkpoint path.
    """
    ckpt_files = sorted(run_dir.glob("mae_ckpt_*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No mae_ckpt_*.pth found in {run_dir}")

    def _epoch(path: Path) -> int:
        stem = path.stem
        try:
            return int(stem.split("_")[-1])
        except Exception:
            return -1

    return sorted(ckpt_files, key=_epoch)[-1]


def _build_demo_polygon(shape: str) -> tuple[np.ndarray, str]:
    """Create synthetic polygon for demo mode.

    Args:
        shape: Shape name (`circle` or `pentagon`).

    Returns:
        Tuple `(polygon_coords, suffix)`.
    """
    if shape == "pentagon":
        angles = np.linspace(0, 2 * np.pi, 6)[:-1] + np.pi / 10
        poly_coords = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 0.8
        return poly_coords, "shape_pentagon"

    angles = np.linspace(0, 2 * np.pi, 65)[:-1]
    poly_coords = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 0.8
    return poly_coords, "shape_circle"


def main() -> None:
    """CLI main function for downstream reconstruction demo."""
    downstream_root, project_root = _inject_paths()

    from dutils.filesystem import ensure_dir
    from dutils.rasterize import rasterize_polygon
    from tasks.pipeline_mae import MaeReconstructionPipeline

    args = _build_args(downstream_root / "configs" / "recons.yaml")

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = (downstream_root / data_path).resolve()

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.is_absolute():
        experiment_dir = (downstream_root / experiment_dir).resolve()

    run_dir = _resolve_latest_run_dir(experiment_dir)
    ckpt_path = _resolve_mae_ckpt(run_dir)
    config_path = run_dir / "poly_mae_config.json"

    pipeline = MaeReconstructionPipeline(
        weight_path=str(ckpt_path),
        config_path=str(config_path),
        precision=args.precision,
    )

    if args.shape == "dataset":
        all_polys = torch.load(str(data_path), weights_only=False)
        if args.index >= len(all_polys):
            raise IndexError(f"Index {args.index} out of range. Total: {len(all_polys)}")

        tris = all_polys[args.index]
        shapely_tris = [ShapelyPolygon(t) for t in tris]
        merged_poly = unary_union(shapely_tris)
        if merged_poly.geom_type == "MultiPolygon":
            merged_poly = max(merged_poly.geoms, key=lambda item: item.area)

        poly_coords = np.array(merged_poly.exterior.coords)[:-1]
        save_suffix = f"idx_{args.index}"
    else:
        poly_coords, save_suffix = _build_demo_polygon(args.shape)

    raw_polygons = [poly_coords]

    captured_mask = None
    captured_imgs = None
    original_forward = pipeline.model.forward

    def hooked_forward(imgs, mask_ratio=args.mask_ratio):
        """Capture model input and generated mask for visualization.

        Args:
            imgs: MAE input image tensor.
            mask_ratio: MAE mask ratio.

        Returns:
            Original forward outputs.
        """
        nonlocal captured_mask, captured_imgs
        captured_imgs = imgs.clone().detach()
        out = original_forward(imgs, mask_ratio)
        _, _, mask, _, _ = out
        captured_mask = mask.clone().detach()
        return out

    pipeline.model.forward = hooked_forward
    real_part, imag_part = pipeline.reconstruct_real_imag(raw_polygons, mask_ratio=args.mask_ratio)
    pipeline.model.forward = original_forward

    patch_size = int(pipeline.config.get("patch_size", 2))
    h, w = captured_imgs.shape[2], captured_imgs.shape[3]
    h_p, w_p = h // patch_size, w // patch_size

    orig_mag = captured_imgs[0, 0].cpu()
    orig_cos = captured_imgs[0, 1].cpu()
    orig_sin = captured_imgs[0, 2].cpu()
    orig_phase = torch.atan2(orig_sin, orig_cos)

    if args.mask_ratio > 0:
        mask_map = captured_mask[0].cpu().reshape(h_p, w_p, 1, 1).expand(-1, -1, patch_size, patch_size)
        mask_map = mask_map.permute(0, 2, 1, 3).reshape(h, w)
    else:
        mask_map = torch.zeros((h, w))

    valid_h = h - pipeline.codec.converter.pad_h
    valid_w = w - pipeline.codec.converter.pad_w

    orig_mag_valid = orig_mag[:valid_h, :valid_w].numpy()
    orig_phase_valid = orig_phase[:valid_h, :valid_w].numpy()
    mask_map_valid = mask_map[:valid_h, :valid_w].numpy()

    masked_mag_vis = orig_mag_valid.copy()
    masked_mag_vis[mask_map_valid == 1] = np.nan

    masked_phase_vis = orig_phase_valid.copy()
    masked_phase_vis[mask_map_valid == 1] = np.nan

    complex_valid = real_part + 1j * imag_part
    pad_h = pipeline.codec.converter.pad_h
    pad_w = pipeline.codec.converter.pad_w

    if pad_h > 0 or pad_w > 0:
        complex_padded = torch.nn.functional.pad(complex_valid, (0, pad_w, 0, pad_h), value=0.0)
    else:
        complex_padded = complex_valid

    mag_channel = torch.log1p(torch.abs(complex_padded))
    phase_channel = torch.angle(complex_padded)
    icft_raster_channel = pipeline.codec.icft_2d(complex_padded, spatial_size=256)

    gt_raster = rasterize_polygon(poly_coords)

    fig, axes = plt.subplots(1, 6, figsize=(36, 6))

    im0 = axes[0].imshow(gt_raster, cmap="gray", extent=[-1, 1, -1, 1])
    axes[0].set_title("Ground Truth Polygon")

    im1 = axes[1].imshow(masked_mag_vis, cmap="viridis")
    axes[1].set_title(f"MAE Input: Masked Mag ({int(args.mask_ratio * 100)}%)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(masked_phase_vis, cmap="viridis", vmin=-np.pi, vmax=np.pi)
    axes[2].set_title(f"MAE Input: Masked Phase ({int(args.mask_ratio * 100)}%)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(mag_channel[0].cpu().numpy(), cmap="viridis")
    axes[3].set_title("Downstream Ch1: Recon Mag(log)")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    im4 = axes[4].imshow(phase_channel[0].cpu().numpy(), cmap="viridis", vmin=-np.pi, vmax=np.pi)
    axes[4].set_title("Downstream Ch2: Recon Phase")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    im5 = axes[5].imshow(icft_raster_channel[0].cpu().numpy(), cmap="gray", vmin=0, vmax=1, extent=[-1, 1, -1, 1])
    axes[5].set_title("Downstream Ch3: ICFT Raster")
    plt.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = (downstream_root / save_dir).resolve()
    ensure_dir(save_dir)

    save_path = save_dir / f"resnet_integration_proof_{save_suffix}_mask_{int(args.mask_ratio * 100)}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"[INFO] Demo visualization saved to: {save_path}")


if __name__ == "__main__":
    main()
