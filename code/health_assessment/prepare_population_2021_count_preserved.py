#!/usr/bin/env python3
"""Reproject raw 2021 population counts to the UCM grid without losing totals.

Population is an extensive quantity. GDAL's sum resampling distributes each
source-cell count across overlapping destination cells. Pixels whose centres
fall outside the London borough polygons are then set to zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector_checksums(path: Path) -> dict[str, str]:
    """Hash every Shapefile sidecar, or the single non-Shapefile dataset."""
    related = sorted(path.parent.glob(f"{path.stem}.*")) if path.suffix.lower() == ".shp" else [path]
    return {item.name: _sha256(item) for item in related if item.is_file()}


def _git_value(repository_root: Path, args: list[str]) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=repository_root, check=True,
            capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def prepare(args: argparse.Namespace) -> None:
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import Resampling, reproject

    source_path = Path(args.source_population).resolve()
    reference_path = Path(args.reference_raster).resolve()
    borough_path = Path(args.boroughs).resolve()
    output_path = Path(args.output).resolve()
    for path in (source_path, reference_path, borough_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output_path}; use --force after review")

    boroughs = gpd.read_file(borough_path)
    if args.borough_field not in boroughs.columns or boroughs.crs is None:
        raise ValueError("Borough layer must contain the requested ID field and a CRS")
    if boroughs[args.borough_field].nunique() != 33:
        raise ValueError("Expected 33 unique London borough identifiers")
    if boroughs.geometry.isna().any() or boroughs.geometry.is_empty.any():
        raise ValueError("Borough layer contains missing or empty geometries")
    if (~boroughs.geometry.is_valid).any():
        raise ValueError("Borough layer contains invalid geometries")

    with rasterio.open(source_path) as source, rasterio.open(reference_path) as reference:
        if source.crs is None or reference.crs is None:
            raise ValueError("Source population and reference raster must have CRSs")
        if source.count != 1 or reference.count != 1:
            raise ValueError("Source population and reference must each have one band")

        source_values = source.read(1, masked=True).filled(0).astype(np.float32)
        if np.any(~np.isfinite(source_values)) or np.any(source_values < 0):
            raise ValueError("Source population contains invalid or negative values")
        destination = np.zeros((reference.height, reference.width), dtype=np.float32)
        reproject(
            source=source_values,
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            src_nodata=None,
            dst_nodata=0,
            resampling=Resampling.sum,
        )

        boroughs = boroughs.to_crs(reference.crs)
        london_mask = rasterize(
            ((geometry, 1) for geometry in boroughs.geometry),
            out_shape=destination.shape,
            transform=reference.transform,
            fill=0,
            all_touched=False,
            dtype="uint8",
        ).astype(bool)
        rectangular_grid_total = float(destination.sum(dtype=np.float64))
        destination[~london_mask] = 0
        london_total = float(destination.sum(dtype=np.float64))
        source_total = float(source_values.sum(dtype=np.float64))
        relative_benchmark_difference = (
            (london_total - args.expected_london_population)
            / args.expected_london_population
        )
        if abs(relative_benchmark_difference) > args.max_benchmark_relative_difference:
            raise ValueError(
                f"London population {london_total:,.0f} differs from benchmark "
                f"{args.expected_london_population:,.0f} by "
                f"{relative_benchmark_difference:.2%}"
            )

        profile = reference.profile.copy()
        profile.update(driver="GTiff", dtype="float32", count=1, compress="lzw", nodata=0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as target:
            target.write(destination, 1)
        source_tags = source.tags()

    repository_root = Path(__file__).resolve().parents[2]
    dirty_text = _git_value(repository_root, ["status", "--porcelain"])
    manifest = {
        "schema_version": 1,
        "method": "count-preserving reprojection with GDAL sum resampling",
        "boundary_rule": "retain pixels whose centres fall inside London borough polygons",
        "source_population_total": source_total,
        "population_on_rectangular_reference_grid_before_mask": rectangular_grid_total,
        "london_population_after_borough_mask": london_total,
        "expected_london_population": args.expected_london_population,
        "relative_benchmark_difference": relative_benchmark_difference,
        "maximum_benchmark_relative_difference": args.max_benchmark_relative_difference,
        "inputs": {
            "source_population": {"path": str(source_path), "sha256": _sha256(source_path), "tags": source_tags},
            "reference_raster": {"path": str(reference_path), "sha256": _sha256(reference_path)},
            "boroughs": {
                "path": str(borough_path),
                "checksums": _vector_checksums(borough_path),
            },
        },
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "geopandas": gpd.__version__,
            "rasterio": rasterio.__version__,
            "git_commit": _git_value(repository_root, ["rev-parse", "HEAD"]),
            "git_worktree_dirty": bool(dirty_text) if dirty_text is not None else None,
            "preparation_script_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created count-preserved population raster: {output_path}")
    print(f"London population: {london_total:,.2f}")
    print(f"Benchmark difference: {relative_benchmark_difference:+.3%}")
    print(f"Manifest: {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-population", required=True, help="Raw population-count raster")
    parser.add_argument("--reference-raster", required=True, help="Trusted UCM target grid")
    parser.add_argument("--boroughs", required=True, help="London borough polygons")
    parser.add_argument("--borough-field", default="GSS_CODE")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-london-population", type=float, default=8_800_000)
    parser.add_argument("--max-benchmark-relative-difference", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    prepare(build_parser().parse_args())
