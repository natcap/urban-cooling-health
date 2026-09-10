#!/usr/bin/env python3
"""Run the equal-budget Target30 v4 UCM with InVEST 3.20.2."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from osgeo import gdal
import natcap.invest
import natcap.invest.urban_cooling_model
import natcap.invest.utils
import pygeoprocessing


LOGGER = logging.getLogger(__name__)
REQUIRED_INVEST_VERSION = "3.20.2"


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for an input or output file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict:
    """Record a path, size and checksum for later provenance checks."""
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _raster_record(path: Path) -> dict:
    """Record output identity, grid geometry, NoData and statistics."""
    info = pygeoprocessing.get_raster_info(str(path))
    stats = info.get("statistics")
    return {
        **_file_record(path),
        "raster_size": list(info["raster_size"]),
        "pixel_size": list(info["pixel_size"]),
        "bounding_box": list(info["bounding_box"]),
        "projection_wkt": info["projection_wkt"],
        "nodata": list(info["nodata"]),
        "statistics": list(stats[0]) if stats else None,
    }


def _git_value(repository: Path, *arguments: str) -> str | None:
    """Return a Git value when the script is running inside a repository."""
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_manifest(
    path: Path,
    temperature: float,
    run_args: dict,
    required_inputs: list[Path],
    output_path: Path,
    repository: Path,
) -> None:
    """Write the complete production record only after a successful run."""
    git_status = _git_value(repository, "status", "--porcelain")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": "target30_equal_budget_v4",
        "purpose": "Air-temperature input for population-weighted health assessment",
        "temperature_setting_c": temperature,
        "model_arguments": run_args,
        "inputs": [_file_record(input_path) for input_path in required_inputs],
        "output": _raster_record(output_path),
        "software": {
            "invest": natcap.invest.__version__,
            "python": platform.python_version(),
            "gdal": gdal.VersionInfo("RELEASE_NAME"),
            "pygeoprocessing": pygeoprocessing.__version__,
            "git_commit": _git_value(repository, "rev-parse", "HEAD"),
            "git_worktree_dirty": bool(git_status),
            "runner_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[25, 28])
    parser.add_argument("--uhi-max", type=float, default=5)
    parser.add_argument("--humidity", type=float, default=45)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate both runs and inputs without creating model outputs.",
    )
    parser.add_argument(
        "--include-valuations",
        action="store_true",
        help="Also calculate productivity and building-energy outputs.",
    )
    parser.add_argument(
        "--lulc",
        type=Path,
        help="Override the approved Target30 v4 raster beneath the data root.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Override the separate scenario730v4_equal_budget workspace.",
    )
    args = parser.parse_args()

    if natcap.invest.__version__ != REQUIRED_INVEST_VERSION:
        parser.error(
            f"This production runner requires InVEST {REQUIRED_INVEST_VERSION}; "
            f"the active environment provides {natcap.invest.__version__}."
        )

    data_root = args.data_root.expanduser().resolve()
    input_root = data_root / "1_preprocess/UrbanCoolingModel/OfficialWorkingInputs"
    lulc_path = args.lulc.expanduser().resolve() if args.lulc else (
        input_root
        / "LULC/lc_tree_equity_scenarios_output/LULC_Scenario730v4_equal_budget.tif"
    )
    workspace = args.workspace.expanduser().resolve() if args.workspace else (
        data_root
        / "2_postprocess_intermediate/UCM_official_runs/"
        "scenario730v4_equal_budget_health_invest3202"
    )

    model_args = {
        "aoi_vector_path": str(input_root / "AOIs/London_Borough_aoi.shp"),
        "avg_rel_humidity": args.humidity,
        "biophysical_table_path": str(
            input_root / "LULC/Biophysical_table_ukech_2021_london_with_TCC.csv"
        ),
        "building_vector_path": str(
            input_root / "energy_buildings/bld_with_attr_compact_ucm2.gpkg"
        ),
        "cc_method": "factors",
        "cc_weight_albedo": "",
        "cc_weight_eti": "",
        "cc_weight_shade": "",
        "do_energy_valuation": args.include_valuations,
        "do_productivity_valuation": args.include_valuations,
        "energy_consumption_table_path": str(
            input_root / "energy_buildings/_UCM_Energy Consumption Table.csv"
        ),
        "green_area_cooling_distance": 450,
        "lulc_raster_path": str(lulc_path),
        "ref_eto_raster_path": str(
            input_root / "evapotranspiration/et0_V3_07_clipped_reprojected.tif"
        ),
        "t_air_average_radius": 500,
        "uhi_max": args.uhi_max,
        "workspace_dir": str(workspace),
    }

    # Valuation inputs are not needed for the temperature rasters used by the
    # health assessment. Check them only when valuations are explicitly enabled.
    required_inputs = [
        Path(model_args[key])
        for key in (
            "aoi_vector_path",
            "biophysical_table_path",
            "lulc_raster_path",
            "ref_eto_raster_path",
        )
    ]
    if args.include_valuations:
        required_inputs.extend(
            [
                Path(model_args["building_vector_path"]),
                Path(model_args["energy_consumption_table_path"]),
            ]
        )
    missing = [path for path in required_inputs if not path.exists()]
    if missing:
        parser.error("Missing required input(s): " + ", ".join(map(str, missing)))

    # InVEST validates that the workspace exists and is writable. Validation
    # mode may create this empty directory but never writes model outputs.
    workspace.mkdir(parents=True, exist_ok=True)
    for temperature in args.temperatures:
        temperature_label = f"{temperature:g}"
        suffix = (
            f"london_scenario730v4_equal_budget_{temperature_label}deg_"
            f"{args.uhi_max:g}uhi_{args.humidity:g}hum"
        )
        # Since InVEST 3.20.0, T_air is a primary output in the workspace root.
        expected_air_temperature = workspace / f"T_air_{suffix}.tif"
        manifest_path = workspace / f"run_manifest_{suffix}.json"
        if (
            expected_air_temperature.exists() or manifest_path.exists()
        ) and not args.validate_only:
            raise FileExistsError(
                "Refusing to overwrite completed output or manifest: "
                f"{expected_air_temperature}"
            )
        run_args = {
            **model_args,
            "t_ref": temperature,
            "results_suffix": suffix,
        }
        warnings = natcap.invest.urban_cooling_model.validate(run_args)
        if warnings:
            formatted = "\n".join(
                f"{', '.join(keys)}: {message}" for keys, message in warnings
            )
            raise ValueError(
                f"InVEST validation failed at {temperature_label} C:\n{formatted}"
            )
        if args.validate_only:
            LOGGER.info("Validated Target30 v4 at %s C", temperature_label)
            continue

        LOGGER.info("Running Target30 v4 at %s C", temperature_label)
        natcap.invest.urban_cooling_model.execute(run_args)
        if not expected_air_temperature.is_file():
            raise FileNotFoundError(
                f"InVEST completed without expected output: {expected_air_temperature}"
            )
        repository = Path(__file__).resolve().parents[2]
        _write_manifest(
            manifest_path,
            temperature,
            run_args,
            required_inputs,
            expected_air_temperature,
            repository,
        )
        LOGGER.info("Wrote manifest %s", manifest_path)


if __name__ == "__main__":
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(natcap.invest.utils.LOG_FMT))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    main()
