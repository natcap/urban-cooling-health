#!/usr/bin/env python3
"""Harmonize Green30 NoData encoding with baseline without changing valid cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--green30", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    green30_path = args.green30.expanduser().resolve()
    baseline_path = args.baseline.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    for path in (green30_path, baseline_path):
        if not path.is_file():
            parser.error(f"Missing input raster: {path}")
    for path in (output_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    replaced_pixels = 0
    valid_pixels = 0
    with rasterio.open(green30_path) as green30, rasterio.open(baseline_path) as baseline:
        if (
            green30.crs != baseline.crs
            or green30.transform != baseline.transform
            or green30.width != baseline.width
            or green30.height != baseline.height
        ):
            raise ValueError("Green30 and baseline LULC rasters do not share one exact grid")
        if green30.count != 1 or baseline.count != 1:
            raise ValueError("Both LULC rasters must contain one band")
        if green30.nodata != 255 or baseline.nodata != 0:
            raise ValueError(
                f"Expected Green30 NoData 255 and baseline NoData 0; found "
                f"{green30.nodata!r} and {baseline.nodata!r}"
            )

        profile = green30.profile.copy()
        profile.update(nodata=0, compress="lzw")
        with rasterio.open(output_path, "w", **profile) as output:
            for _, window in green30.block_windows(1):
                values = green30.read(1, window=window)
                source_nodata = values == green30.nodata
                # Zero is not a valid Green30 class, so it can safely become
                # the common baseline/Target30 NoData sentinel.
                if np.any((values == 0) & ~source_nodata):
                    raise ValueError("Green30 contains a valid zero-valued cell")
                result = values.copy()
                result[source_nodata] = 0
                output.write(result, 1, window=window)
                replaced_pixels += int(source_nodata.sum())
                valid_pixels += int((~source_nodata).sum())

    # Read back the product and assert that masking, values and geometry match
    # the intended transformation exactly.
    with rasterio.open(green30_path) as source, rasterio.open(output_path) as output:
        if output.nodata != 0 or output.crs != source.crs or output.transform != source.transform:
            raise ValueError("Harmonized output metadata failed validation")
        for _, window in source.block_windows(1):
            before = source.read(1, window=window)
            after = output.read(1, window=window)
            expected = np.where(before == source.nodata, 0, before)
            if not np.array_equal(after, expected):
                raise ValueError("Harmonization changed a valid Green30 land-cover value")

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "Replace Green30 NoData sentinel 255 with 0",
        "scientific_content_change": False,
        "source_nodata": 255,
        "target_nodata": 0,
        "valid_pixel_count": valid_pixels,
        "nodata_pixel_count": replaced_pixels,
        "valid_values_unchanged": True,
        "inputs": {
            "green30": {"path": str(green30_path), "sha256": _sha256(green30_path)},
            "baseline_reference": {
                "path": str(baseline_path), "sha256": _sha256(baseline_path)
            },
        },
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Valid cells unchanged: {valid_pixels:,}; NoData cells recoded: {replaced_pixels:,}")


if __name__ == "__main__":
    main()
