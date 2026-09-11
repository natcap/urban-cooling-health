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
LOCKED_CODE = 254

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


def _prepare_opportunity_base(
    run_base_path: Path,
    opportunity_mask_path: Path,
    output_path: Path,
    eligible_codes: set[int],
) -> None:
    """Lock convertible cells outside an optional sensitivity-test mask."""
    run_base = gdal.OpenEx(str(run_base_path), gdal.OF_RASTER)
    opportunity = gdal.OpenEx(str(opportunity_mask_path), gdal.OF_RASTER)
    if run_base is None or opportunity is None or not _same_grid(run_base, opportunity):
        raise ValueError("Opportunity mask and run base must share one exact grid")
    run_band = run_base.GetRasterBand(1)
    opportunity_band = opportunity.GetRasterBand(1)
    nodata = run_band.GetNoDataValue()

    driver = gdal.GetDriverByName("GTiff")
    output = driver.Create(
        str(output_path),
        run_base.RasterXSize,
        run_base.RasterYSize,
        1,
        run_band.DataType,
        options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
    )
    output.SetGeoTransform(run_base.GetGeoTransform())
    output.SetProjection(run_base.GetProjection())
    output_band = output.GetRasterBand(1)
    output_band.SetNoDataValue(nodata)

    block_x, block_y = run_band.GetBlockSize()
    for yoff in range(0, run_base.RasterYSize, block_y):
        rows = min(block_y, run_base.RasterYSize - yoff)
        for xoff in range(0, run_base.RasterXSize, block_x):
            cols = min(block_x, run_base.RasterXSize - xoff)
            values = run_band.ReadAsArray(xoff, yoff, cols, rows)
            allowed = opportunity_band.ReadAsArray(xoff, yoff, cols, rows) == 1
            if np.any(values == LOCKED_CODE):
                raise ValueError(f"Reserved lock code {LOCKED_CODE} occurs in run base")
            lock = np.isin(values, list(eligible_codes)) & ~allowed
            values[lock] = LOCKED_CODE
            output_band.WriteArray(values, xoff, yoff)

    output_band.FlushCache()
    output.FlushCache()
    output = opportunity = run_base = None


def _compose_output(
    raw_path: Path,
    run_base_path: Path,
    baseline_path: Path,
    output_path: Path,
) -> None:
    """Apply only InVEST's new-canopy cells and restore baseline NoData."""
    baseline = gdal.OpenEx(str(baseline_path), gdal.OF_RASTER)
    run_base = gdal.OpenEx(str(run_base_path), gdal.OF_RASTER)
    raw = gdal.OpenEx(str(raw_path), gdal.OF_RASTER)
    if baseline is None or run_base is None or raw is None:
        raise ValueError("Could not open a scenario-composition raster")
    if not _same_grid(baseline, run_base) or not _same_grid(baseline, raw):
        raise ValueError(f"Grid mismatch: {raw_path}")

    baseline_band = baseline.GetRasterBand(1)
    run_base_band = run_base.GetRasterBand(1)
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
            run_array = run_base_band.ReadAsArray(xoff, yoff, cols, rows)
            raw_array = raw_band.ReadAsArray(xoff, yoff, cols, rows)
            output_array = run_array.copy()
            newly_converted = (raw_array == REPLACEMENT_CODE) & (
                run_array != REPLACEMENT_CODE
            )
            output_array[newly_converted] = REPLACEMENT_CODE
            output_array[base_array == baseline_nodata] = baseline_nodata
            output_band.WriteArray(output_array, xoff, yoff)

    output_band.FlushCache()
    output.FlushCache()
    output = raw = run_base = baseline = None


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
    eligible_codes: set[int],
) -> dict[str, object]:
    if profile == "historical-audit":
        focal_codes = convertible_codes = "1 2 4 20 21"
        steps = 2
    else:
        focal_codes = "1 2 100"
        convertible_codes = " ".join(str(code) for code in sorted(eligible_codes))
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
        "--opportunity-mask",
        type=Path,
        help=(
            "Optional 0/1 raster for an explicitly labelled sensitivity test. "
            "Only value-1 cells whose baseline code is eligible may convert. "
            "A Target-derived mask is not the primary Green definition."
        ),
    )
    parser.add_argument(
        "--eligible-codes",
        type=int,
        nargs="+",
        default=sorted(ELIGIBLE_CODES),
        help="Convertible source codes for the revised profile.",
    )
    parser.add_argument(
        "--cumulative-pixels",
        type=int,
        nargs=3,
        metavar=("GREEN10", "GREEN20", "GREEN30"),
        default=list(REVISED_CUMULATIVE_PIXELS.values()),
        help=(
            "Cumulative revised-scenario budgets. Override only for a clearly "
            "labelled capacity sensitivity."
        ),
    )
    parser.add_argument(
        "--allow-other-invest-version",
        action="store_true",
        help="Permit a non-3.20.2 run for an explicit version audit.",
    )
    args = parser.parse_args()

    baseline = args.baseline.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    opportunity_mask = (
        args.opportunity_mask.expanduser().resolve()
        if args.opportunity_mask is not None
        else None
    )
    eligible_codes = set(args.eligible_codes)
    cumulative_pixels = dict(
        zip(("Green10", "Green20", "Green30"), args.cumulative_pixels, strict=True)
    )
    if not baseline.is_file():
        parser.error(f"Baseline not found: {baseline}")
    if opportunity_mask is not None and not opportunity_mask.is_file():
        parser.error(f"Opportunity mask not found: {opportunity_mask}")
    if args.profile == "historical-audit" and opportunity_mask is not None:
        parser.error("The historical-audit profile cannot use an opportunity mask")
    if args.profile == "historical-audit" and eligible_codes != ELIGIBLE_CODES:
        parser.error("The historical-audit profile has fixed historical code lists")
    if (
        args.profile == "historical-audit"
        and cumulative_pixels != REVISED_CUMULATIVE_PIXELS
    ):
        parser.error("The historical-audit profile has fixed historical areas")
    if not eligible_codes:
        parser.error("At least one eligible code is required")
    if not (
        0 < cumulative_pixels["Green10"]
        < cumulative_pixels["Green20"]
        < cumulative_pixels["Green30"]
    ):
        parser.error("Revised cumulative pixel budgets must be positive and increasing")
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
        "eligible_planting_codes": sorted(eligible_codes),
        "replacement_code": REPLACEMENT_CODE,
        "cumulative_pixel_budgets": cumulative_pixels,
        "opportunity_mask": str(opportunity_mask) if opportunity_mask else None,
        "opportunity_mask_sha256": (
            _sha256(opportunity_mask) if opportunity_mask else None
        ),
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
            expected_total = cumulative_pixels[label]
            incremental_pixels = expected_total - previous_total
            area_ha = incremental_pixels * pixel_area_m2 / 10_000

        workspace = output_dir / "invest_workspaces" / label.lower()
        workspace.mkdir(parents=True, exist_ok=True)
        suffix = label.lower()
        model_base = run_base
        if opportunity_mask is not None:
            model_base = workspace / f"opportunity_masked_base_{suffix}.tif"
            _prepare_opportunity_base(
                run_base, opportunity_mask, model_base, eligible_codes
            )
        model_args = _scenario_args(
            workspace,
            model_base,
            area_ha,
            args.profile,
            suffix,
            eligible_codes,
        )
        scenario_gen_proximity.execute(model_args)
        raw_path = workspace / f"nearest_to_edge_{suffix}.tif"
        filename = (
            (
                f"{label}_masked_opportunity_sensitivity_v3.tif"
                if opportunity_mask is not None
                else f"{label}_equal_area_eligible_v2.tif"
            )
            if args.profile == "revised"
            else f"{label}_historical_parameters_invest_{invest_version}.tif"
        )
        final_path = output_dir / filename
        _compose_output(raw_path, run_base, baseline, final_path)

        transitions, eligible_added = _audit_transition(baseline, final_path)
        converted_pixels = sum(transitions.values())
        if args.profile == "revised":
            expected_transitions = {
                f"{code}->{REPLACEMENT_CODE}" for code in eligible_codes
            }
            unexpected = set(transitions) - expected_transitions
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
