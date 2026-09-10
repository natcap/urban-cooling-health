"""Generate auditable Green10/20/30 rasters with InVEST Scenario Generator.

``historical-audit`` reconstructs the parameters recorded in the 2025 InVEST
3.14.1 logs. ``revised`` uses the approved canopy and planting definitions and
builds the levels sequentially so Green10 is contained in Green20 and Green20
is contained in Green30. Run this in the project's InVEST 3.20.2 environment.
The script writes new files only; it never modifies source rasters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import natcap.invest
import numpy as np
from natcap.invest import scenario_gen_proximity
from osgeo import gdal


EXPECTED_INVEST_VERSION = "3.20.2"
TREE_CODES = {1, 2, 100}
ELIGIBLE_CODES = {4, 20, 21}
REPLACEMENT_CODE = 100

# Eligible additions observed in the preserved Green rasters after excluding
# woodland-to-tree-cover relabeling.
REVISED_CUMULATIVE_PIXELS = {
    "Green10": 307_768,
    "Green20": 595_084,
    "Green30": 894_249,
}

# Confirmed by the preserved 3.14.1 Scenario Generator logs from 2025-09-18.
HISTORICAL_AREAS_HA = {
    "Green10": 3_200.0,
    "Green20": 6_200.0,
    "Green30": 9_300.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_grid(reference: gdal.Dataset, other: gdal.Dataset) -> bool:
    """Return whether raster dimensions, projection and transform agree."""
    return (
        reference.RasterXSize == other.RasterXSize
        and reference.RasterYSize == other.RasterYSize
        and reference.GetProjection() == other.GetProjection()
        and np.allclose(reference.GetGeoTransform(), other.GetGeoTransform())
    )


def _harmonize_nodata(raw_path: Path, baseline_path: Path, output_path: Path) -> None:
    """Copy an InVEST result while restoring the baseline NoData convention."""
    baseline = gdal.OpenEx(str(baseline_path), gdal.OF_RASTER)
    raw = gdal.OpenEx(str(raw_path), gdal.OF_RASTER)
    if baseline is None or raw is None:
        raise ValueError("Could not open a raster for NoData harmonization")
    if not _same_grid(baseline, raw):
        raise ValueError(f"Grid mismatch: {raw_path}")

    baseline_band = baseline.GetRasterBand(1)
    raw_band = raw.GetRasterBand(1)
    baseline_nodata = baseline_band.GetNoDataValue()
    if baseline_nodata is None:
        raise ValueError("The baseline must define a NoData value")

    driver = gdal.GetDriverByName("GTiff")
    output = driver.Create(
        str(output_path),
        baseline.RasterXSize,
        baseline.RasterYSize,
        1,
        baseline_band.DataType,
        options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
    )
    output.SetGeoTransform(baseline.GetGeoTransform())
    output.SetProjection(baseline.GetProjection())
    output_band = output.GetRasterBand(1)
    output_band.SetNoDataValue(baseline_nodata)

    block_x, block_y = baseline_band.GetBlockSize()
    for yoff in range(0, baseline.RasterYSize, block_y):
        rows = min(block_y, baseline.RasterYSize - yoff)
        for xoff in range(0, baseline.RasterXSize, block_x):
            cols = min(block_x, baseline.RasterXSize - xoff)
            base_array = baseline_band.ReadAsArray(xoff, yoff, cols, rows)
            output_array = raw_band.ReadAsArray(xoff, yoff, cols, rows)
            output_array[base_array == baseline_nodata] = baseline_nodata
            output_band.WriteArray(output_array, xoff, yoff)

    output_band.FlushCache()
    output.FlushCache()
    output = raw = baseline = None


def _audit_transition(
    baseline_path: Path, scenario_path: Path
) -> tuple[dict[str, int], int]:
    """Return changed source-to-target counts and eligible new-canopy count."""
    baseline = gdal.OpenEx(str(baseline_path), gdal.OF_RASTER)
    scenario = gdal.OpenEx(str(scenario_path), gdal.OF_RASTER)
    if baseline is None or scenario is None or not _same_grid(baseline, scenario):
        raise ValueError(f"Grid mismatch: {scenario_path}")

    baseline_band = baseline.GetRasterBand(1)
    scenario_band = scenario.GetRasterBand(1)
    nodata = baseline_band.GetNoDataValue()
    transitions: Counter[tuple[int, int]] = Counter()
    block_x, block_y = baseline_band.GetBlockSize()
    for yoff in range(0, baseline.RasterYSize, block_y):
        rows = min(block_y, baseline.RasterYSize - yoff)
        for xoff in range(0, baseline.RasterXSize, block_x):
            cols = min(block_x, baseline.RasterXSize - xoff)
            base = baseline_band.ReadAsArray(xoff, yoff, cols, rows)
            scen = scenario_band.ReadAsArray(xoff, yoff, cols, rows)
            changed = (base != scen) & (base != nodata)
            pairs, counts = np.unique(
                np.column_stack((base[changed], scen[changed])),
                axis=0,
                return_counts=True,
            )
            for (source, target), count in zip(pairs, counts, strict=True):
                transitions[(int(source), int(target))] += int(count)

    eligible_added = sum(
        count
        for (source, target), count in transitions.items()
        if source in ELIGIBLE_CODES and target == REPLACEMENT_CODE
    )
    return (
        {
            f"{source}->{target}": count
            for (source, target), count in sorted(transitions.items())
        },
        eligible_added,
    )


def _scenario_args(
    workspace: Path,
    baseline: Path,
    area_ha: float,
    profile: str,
    suffix: str,
) -> dict[str, object]:
    if profile == "historical-audit":
        focal_codes = convertible_codes = "1 2 4 20 21"
        steps = 2
    else:
        focal_codes = "1 2 100"
        convertible_codes = "4 20 21"
        # One step makes the distance surface for each tier unambiguous.
        steps = 1
    return {
        "workspace_dir": str(workspace),
        "results_suffix": suffix,
        "base_lulc_path": str(baseline),
        "area_to_convert": area_ha,
        "focal_landcover_codes": focal_codes,
        "convertible_landcover_codes": convertible_codes,
        "replacement_lucode": REPLACEMENT_CODE,
        "convert_farthest_from_edge": False,
        "convert_nearest_to_edge": True,
        "n_fragmentation_steps": steps,
        "n_workers": -1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("revised", "historical-audit"),
        default="revised",
    )
    parser.add_argument(
        "--allow-other-invest-version",
        action="store_true",
        help="Permit a non-3.20.2 run for an explicit version audit.",
    )
    args = parser.parse_args()

    baseline = args.baseline.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not baseline.is_file():
        parser.error(f"Baseline not found: {baseline}")
    invest_version = natcap.invest.__version__
    if invest_version != EXPECTED_INVEST_VERSION and not args.allow_other_invest_version:
        parser.error(
            f"Expected InVEST {EXPECTED_INVEST_VERSION}, found {invest_version}. "
            "Use --allow-other-invest-version only for a version audit."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dataset = gdal.OpenEx(str(baseline), gdal.OF_RASTER)
    _, pixel_width, row_rotation, _, column_rotation, pixel_height = (
        baseline_dataset.GetGeoTransform()
    )
    # The determinant also handles a rotated projected grid correctly.
    pixel_area_m2 = abs(
        pixel_width * pixel_height - row_rotation * column_rotation
    )
    baseline_dataset = None
    if not np.isclose(pixel_area_m2, 100.0):
        raise ValueError(f"Expected 100 m2 pixels, found {pixel_area_m2}")

    manifest: dict[str, object] = {
        "profile": args.profile,
        "invest_version": invest_version,
        "baseline": str(baseline),
        "baseline_sha256": _sha256(baseline),
        "tree_codes": sorted(TREE_CODES),
        "eligible_planting_codes": sorted(ELIGIBLE_CODES),
        "replacement_code": REPLACEMENT_CODE,
        "scenarios": [],
    }
    previous = baseline
    previous_total = 0

    for label in ("Green10", "Green20", "Green30"):
        if args.profile == "historical-audit":
            run_base = baseline
            area_ha = HISTORICAL_AREAS_HA[label]
            expected_total = int(round(area_ha * 10_000 / pixel_area_m2))
        else:
            run_base = previous
            expected_total = REVISED_CUMULATIVE_PIXELS[label]
            incremental_pixels = expected_total - previous_total
            area_ha = incremental_pixels * pixel_area_m2 / 10_000

        workspace = output_dir / "invest_workspaces" / label.lower()
        suffix = label.lower()
        model_args = _scenario_args(workspace, run_base, area_ha, args.profile, suffix)
        scenario_gen_proximity.execute(model_args)
        raw_path = workspace / f"nearest_to_edge_{suffix}.tif"
        filename = (
            f"{label}_equal_area_eligible_v2.tif"
            if args.profile == "revised"
            else f"{label}_historical_parameters_invest_{invest_version}.tif"
        )
        final_path = output_dir / filename
        _harmonize_nodata(raw_path, baseline, final_path)

        transitions, eligible_added = _audit_transition(baseline, final_path)
        converted_pixels = sum(transitions.values())
        if args.profile == "revised":
            unexpected = set(transitions) - {"4->100", "20->100", "21->100"}
            if (
                unexpected
                or eligible_added != expected_total
                or converted_pixels != expected_total
            ):
                raise ValueError(
                    f"{label} failed QA: unexpected={sorted(unexpected)}, "
                    f"eligible={eligible_added:,}, converted={converted_pixels:,}, "
                    f"expected={expected_total:,}"
                )
        elif converted_pixels != expected_total:
            raise ValueError(
                f"{label} converted {converted_pixels:,} pixels; "
                f"expected {expected_total:,} from the requested area"
            )

        manifest["scenarios"].append(
            {
                "scenario": label,
                "run_base": str(run_base),
                "area_to_convert_ha": area_ha,
                "model_args": model_args,
                "output": str(final_path),
                "output_sha256": _sha256(final_path),
                "transitions": transitions,
                "converted_pixels": converted_pixels,
                "eligible_added_pixels": eligible_added,
                "eligible_added_area_km2": eligible_added * pixel_area_m2 / 1_000_000,
            }
        )
        previous = final_path
        previous_total = expected_total

    manifest_path = output_dir / f"green_scenario_manifest_{args.profile}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved scenarios and audit manifest to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
