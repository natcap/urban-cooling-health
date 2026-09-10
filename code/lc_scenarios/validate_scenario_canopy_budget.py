"""Validate that Green30 and Target30 use comparable canopy budgets.

The check operates on the final LULC rasters, not intended tree counts. It
therefore captures overlap with existing canopy and rasterization effects.
Class 100 is treated as tree canopy by default.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio


def _same_grid(reference: rasterio.DatasetReader, other: rasterio.DatasetReader) -> bool:
    """Return True only when two rasters have the same analytical grid."""
    return (
        reference.width == other.width
        and reference.height == other.height
        and reference.crs == other.crs
        and np.allclose(tuple(reference.transform), tuple(other.transform))
    )


def _summarize_scenario(
    baseline_path: Path,
    scenario_path: Path,
    tree_class: int,
) -> dict[str, float | int | str]:
    """Count realized canopy changes using memory-safe raster blocks."""
    with rasterio.open(baseline_path) as baseline, rasterio.open(scenario_path) as scenario:
        if not _same_grid(baseline, scenario):
            raise ValueError(f"Grid mismatch: {scenario_path}")
        if baseline.crs is None or not baseline.crs.is_projected:
            raise ValueError("A projected CRS is required to calculate canopy area.")

        # Determinant handles rotated affine transforms as well as north-up grids.
        pixel_area_m2 = abs(
            baseline.transform.a * baseline.transform.e
            - baseline.transform.b * baseline.transform.d
        )
        crs_text = str(baseline.crs)
        counts = {
            "valid_pixels": 0,
            "baseline_tree_pixels": 0,
            "scenario_tree_pixels": 0,
            "added_tree_pixels": 0,
            "removed_tree_pixels": 0,
            "any_changed_pixels": 0,
        }

        for _, window in baseline.block_windows(1):
            base = baseline.read(1, window=window, masked=True)
            scen = scenario.read(1, window=window, masked=True)
            if not np.array_equal(np.ma.getmaskarray(base), np.ma.getmaskarray(scen)):
                raise ValueError(f"NoData footprint mismatch: {scenario_path}")

            valid = ~np.ma.getmaskarray(base)
            base_values = np.asarray(base.data)
            scenario_values = np.asarray(scen.data)
            base_tree = valid & (base_values == tree_class)
            scenario_tree = valid & (scenario_values == tree_class)

            counts["valid_pixels"] += int(np.count_nonzero(valid))
            counts["baseline_tree_pixels"] += int(np.count_nonzero(base_tree))
            counts["scenario_tree_pixels"] += int(np.count_nonzero(scenario_tree))
            counts["added_tree_pixels"] += int(np.count_nonzero(~base_tree & scenario_tree & valid))
            counts["removed_tree_pixels"] += int(np.count_nonzero(base_tree & ~scenario_tree & valid))
            counts["any_changed_pixels"] += int(
                np.count_nonzero(valid & (base_values != scenario_values))
            )

    baseline_tree_pixels = counts["baseline_tree_pixels"]
    net_tree_pixels = counts["scenario_tree_pixels"] - baseline_tree_pixels
    relative_increase_pct = (
        100.0 * net_tree_pixels / baseline_tree_pixels
        if baseline_tree_pixels > 0
        else float("nan")
    )
    return {
        "raster": str(scenario_path),
        "crs": crs_text,
        "pixel_area_m2": pixel_area_m2,
        **counts,
        "net_tree_pixels": net_tree_pixels,
        "added_tree_area_km2": counts["added_tree_pixels"] * pixel_area_m2 / 1_000_000,
        "net_tree_area_km2": net_tree_pixels * pixel_area_m2 / 1_000_000,
        "relative_tree_increase_pct": relative_increase_pct,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare realized Green30 and Target30 canopy additions."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--green30", type=Path, required=True)
    parser.add_argument("--target30", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tree-class", type=int, default=100)
    parser.add_argument(
        "--tolerance-pct",
        type=float,
        default=0.5,
        help="Maximum absolute percentage difference in added canopy versus Green30.",
    )
    args = parser.parse_args()

    for path in (args.baseline, args.green30, args.target30):
        if not path.is_file():
            parser.error(f"Raster not found: {path}")

    rows = []
    for name, path in (("Green30", args.green30), ("Target30", args.target30)):
        row = _summarize_scenario(args.baseline, path, args.tree_class)
        row["scenario"] = name
        rows.append(row)

    green_added = int(rows[0]["added_tree_pixels"])
    target_added = int(rows[1]["added_tree_pixels"])
    if green_added <= 0:
        raise ValueError("Green30 contains no added tree-canopy pixels.")

    difference_pct = 100.0 * (target_added - green_added) / green_added
    passed = abs(difference_pct) <= args.tolerance_pct
    for row in rows:
        row["target_minus_green_added_pixels"] = target_added - green_added
        row["target_minus_green_added_pct"] = difference_pct
        row["budget_tolerance_pct"] = args.tolerance_pct
        row["budget_check_passed"] = passed

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as target:
        # Keep generated audit tables byte-stable across operating systems.
        writer = csv.DictWriter(
            target, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved canopy-budget audit: {args.output}")
    print(
        f"Green30 added {green_added:,} pixels; Target30 added {target_added:,} "
        f"pixels; difference {difference_pct:+.3f}%."
    )
    print("PASS" if passed else "FAIL")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
