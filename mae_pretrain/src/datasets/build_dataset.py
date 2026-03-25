"""Dataset building utilities for polygon triangulation.

This module reads vector files, normalizes polygons, triangulates each polygon,
and saves all triangle tensors into a compact `.pt` dataset.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import torch
from tqdm import tqdm

from datasets.geometry_polygon import PolyFourierConverter


def _collect_vector_files(input_dirs: Iterable[str | Path]) -> list[Path]:
    """Collect shapefile and geojson files recursively.

    Args:
        input_dirs: Directory list to scan.

    Returns:
        Sorted unique file path list.
    """
    files: list[Path] = []
    for directory in input_dirs:
        directory = Path(directory)
        files.extend(Path(p) for p in glob.glob(str(directory / "**" / "*.shp"), recursive=True))
        files.extend(Path(p) for p in glob.glob(str(directory / "**" / "*.geojson"), recursive=True))

    unique_sorted = sorted(set(files))
    return unique_sorted


def process_and_save(input_dirs: Iterable[str | Path], output_path: str | Path) -> None:
    """Build triangulated polygon dataset and save to disk.

    Args:
        input_dirs: Iterable of source directories containing vector files.
        output_path: Output `.pt` path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converter = PolyFourierConverter(device="cpu")
    file_list = _collect_vector_files(input_dirs)

    all_triangles: list[np.ndarray] = []

    for file_path in file_list:
        try:
            gdf = gpd.read_file(file_path)
        except Exception as exc:
            print(f"[WARN] Failed to read {file_path}: {exc}")
            continue

        for geom in tqdm(gdf.geometry, desc=f"Triangulating {file_path.name}"):
            if geom is None or geom.is_empty:
                continue

            if geom.geom_type == "Polygon":
                polygons = [geom]
            elif geom.geom_type == "MultiPolygon":
                polygons = list(geom.geoms)
            else:
                continue

            for poly in polygons:
                coords = np.array(poly.exterior.coords)
                if len(coords) < 4:
                    continue

                minx, miny = coords[:, 0].min(), coords[:, 1].min()
                maxx, maxy = coords[:, 0].max(), coords[:, 1].max()
                cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
                max_range = max(maxx - minx, maxy - miny) / 2.0
                if max_range < 1e-6:
                    continue

                norm_coords = (coords - np.array([cx, cy])) / max_range
                tris = converter.triangulate_polygon(norm_coords)
                if tris.shape[0] > 0:
                    all_triangles.append(tris.astype(np.float32))

    if not all_triangles:
        print("[WARN] No valid polygons were triangulated. Nothing to save.")
        return

    torch.save(all_triangles, output_path)
    print(f"[INFO] Saved {len(all_triangles)} triangulated polygons to: {output_path}")
