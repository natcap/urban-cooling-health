"""Rasterize screened candidate tree points to the baseline 10 m grid.

The candidate points were already screened for road type, distance from
existing canopy and buildings, and minimum point spacing. This script converts
their assumed 5 m crown radius to a reproducible pixel-centre opportunity mask
without constructing millions of temporary buffer polygons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio


DEFAULT_ELIGIBLE_CODES = (4, 20, 21)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point_coordinate_views(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-copy x/y views for a standard 2D Point shapefile."""
    size = path.stat().st_size
    with path.open("rb") as source:
        header = source.read(100)
    if len(header) != 100:
        raise ValueError(f"Invalid shapefile header: {path}")
    declared_size = struct.unpack(">i", header[24:28])[0] * 2
    shape_type = struct.unpack("<i", header[32:36])[0]
    if declared_size != size or shape_type != 1:
        raise ValueError("Candidate input must be a valid 2D Point shapefile")

    # A point record is 8 bytes of record metadata, a 4-byte type, and x/y.
    record_size = 28
    record_count, remainder = divmod(size - 100, record_size)
    if remainder:
        raise ValueError("Variable-length records found in Point shapefile")
    raw = np.memmap(path, mode="r", dtype="u1")
    content_lengths = np.ndarray(
        (record_count,), dtype=">i4", buffer=raw, offset=104, strides=(record_size,)
    )
    record_types = np.ndarray(
        (record_count,), dtype="<i4", buffer=raw, offset=108, strides=(record_size,)
    )
    if not np.all(content_lengths == 10) or not np.all(record_types == 1):
        raise ValueError("Candidate shapefile contains a non-Point record")
    x = np.ndarray(
        (record_count,), dtype="<f8", buffer=raw, offset=112, strides=(record_size,)
    )
    y = np.ndarray(
        (record_count,), dtype="<f8", buffer=raw, offset=120, strides=(record_size,)
    )
    return x, y


def _candidate_mask(
    x: np.ndarray,
    y: np.ndarray,
    transform: rasterio.Affine,
    shape: tuple[int, int],
    radius_m: float,
) -> np.ndarray:
    """Mark pixel centres within the specified radius of any candidate point."""
    if transform.b != 0 or transform.d != 0:
        raise ValueError("A north-up baseline grid is required")
    pixel_width = transform.a
    pixel_height = -transform.e
    if pixel_width <= 0 or pixel_height <= 0:
        raise ValueError("Invalid baseline pixel dimensions")

    height, width = shape
    left, top = transform.c, transform.f
    mask = np.zeros(shape, dtype=np.uint8)
    radius_squared = radius_m**2
    # Three neighbouring rows and columns cover every possible pixel centre for
    # a 5 m radius on the project's 10 m grid.
    row_reach = int(np.ceil(radius_m / pixel_height))
    column_reach = int(np.ceil(radius_m / pixel_width))
    chunk_size = 250_000
    for start in range(0, len(x), chunk_size):
        xx = x[start : start + chunk_size]
        yy = y[start : start + chunk_size]
        base_columns = np.floor((xx - left) / pixel_width).astype(np.int64)
        base_rows = np.floor((top - yy) / pixel_height).astype(np.int64)
        for row_offset in range(-row_reach, row_reach + 1):
            rows = base_rows + row_offset
            centre_y = top - (rows + 0.5) * pixel_height
            for column_offset in range(-column_reach, column_reach + 1):
                columns = base_columns + column_offset
                centre_x = left + (columns + 0.5) * pixel_width
                inside = (
                    (rows >= 0)
                    & (rows < height)
                    & (columns >= 0)
                    & (columns < width)
                    & ((xx - centre_x) ** 2 + (yy - centre_y) ** 2 < radius_squared)
                )
                mask[rows[inside], columns[inside]] = 1
    return mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-points", type=Path, required=True)
    parser.add_argument("--output-mask", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--radius-m", type=float, default=5.0)
    parser.add_argument(
        "--eligible-codes",
        type=int,
        nargs="+",
        default=list(DEFAULT_ELIGIBLE_CODES),
    )
    args = parser.parse_args()

    baseline_path = args.baseline.expanduser().resolve()
    points_path = args.candidate_points.expanduser().resolve()
    for path in (baseline_path, points_path):
        if not path.is_file():
            parser.error(f"Input not found: {path}")
    if args.radius_m <= 0:
        parser.error("--radius-m must be positive")

    x, y = point_coordinate_views(points_path)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Candidate coordinates contain non-finite values")

    with rasterio.open(baseline_path) as baseline:
        if baseline.crs is None or not baseline.crs.is_projected:
            raise ValueError("The baseline must use a projected CRS")
        values = baseline.read(1)
        valid = baseline.read_masks(1) > 0
        candidate = _candidate_mask(
            x, y, baseline.transform, baseline.shape, args.radius_m
        )
        candidate[~valid] = 0
        profile = baseline.profile.copy()
        profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")

    output_mask = args.output_mask.expanduser().resolve()
    output_audit = args.output_audit.expanduser().resolve()
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    output_audit.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_mask, "w", **profile) as target:
        target.write(candidate, 1)

    counts = Counter(int(code) for code in values[candidate == 1])
    eligible_codes = set(args.eligible_codes)
    eligible_pixels = sum(counts[code] for code in eligible_codes)
    no_urban_pixels = sum(counts[code] for code in eligible_codes - {20})
    pixel_area_m2 = abs(
        profile["transform"].a * profile["transform"].e
        - profile["transform"].b * profile["transform"].d
    )
    audit = {
        "method": "pixel centres within radius of screened candidate points",
        "candidate_points": str(points_path),
        "candidate_point_geometry_sha256": _sha256(points_path),
        "candidate_point_count": len(x),
        "baseline": str(baseline_path),
        "baseline_sha256": _sha256(baseline_path),
        "radius_m": args.radius_m,
        "pixel_area_m2": pixel_area_m2,
        "candidate_pixels_by_baseline_code": {
            str(code): counts[code] for code in sorted(counts)
        },
        "candidate_pixels_all_valid_codes": int(candidate.sum()),
        "candidate_area_all_valid_codes_km2": candidate.sum()
        * pixel_area_m2
        / 1_000_000,
        "eligible_codes": sorted(eligible_codes),
        "eligible_candidate_pixels": eligible_pixels,
        "eligible_candidate_area_km2": eligible_pixels
        * pixel_area_m2
        / 1_000_000,
        "eligible_without_urban_pixels": no_urban_pixels,
        "eligible_without_urban_area_km2": no_urban_pixels
        * pixel_area_m2
        / 1_000_000,
        "output_mask": str(output_mask),
        "output_mask_sha256": _sha256(output_mask),
    }
    output_audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Saved opportunity mask: {output_mask}")
    print(f"Eligible candidate area: {audit['eligible_candidate_area_km2']:.4f} km2")
    print(
        "Eligible area without Urban: "
        f"{audit['eligible_without_urban_area_km2']:.4f} km2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
