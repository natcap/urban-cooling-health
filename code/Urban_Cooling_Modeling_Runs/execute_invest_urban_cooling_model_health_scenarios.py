#!/usr/bin/env python3
"""Run the baseline and Green30 health-temperature scenarios with InVEST 3.20.2.

This runner intentionally calculates only the air-temperature outputs required
by the health assessment. Historical InVEST workspaces are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

from osgeo import gdal
import natcap.invest
import natcap.invest.urban_cooling_model
import natcap.invest.utils
import pygeoprocessing


LOGGER = logging.getLogger(__name__)
REQUIRED_INVEST_VERSION = "3.20.2"

# These LULC rasters reproduce the project definitions used by the historical
# baseline and Green30 runs. Only the InVEST software version is changed.
SCENARIOS = {
    "baseline": {
        "label": "2023 baseline land cover",
        "lulc": "LULC/LCM2023_London_10m_clip2aoi_tcc24.tif",
        "workspace": "baseline_health_invest3202",
    },
    "green30": {
        "label": "Green30 nearest-to-edge 30 percent canopy scenario",
        "lulc": (
            "LULC/LCM2023_London_10m_clip2aoi_tcc24_"
            "scenario4_nearest_to_edge_30prc_canopy_increase.tif"
        ),
        "workspace": "green30_health_invest3202",
    },
}


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a raster into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_record(path: Path) -> dict:
    """Record enough provenance to detect a changed input in a later rerun."""
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _raster_record(path: Path) -> dict:
    """Record output identity, grid geometry, nodata, and basic statistics."""
    info = pygeoprocessing.get_raster_info(str(path))
    stats = info.get("statistics")
    return {
        **_input_record(path),
        "raster_size": list(info["raster_size"]),
        "pixel_size": list(info["pixel_size"]),
        "bounding_box": list(info["bounding_box"]),
        "projection_wkt": info["projection_wkt"],
        "nodata": list(info["nodata"]),
        "statistics": list(stats[0]) if stats else None,
    }


def _write_manifest(
    manifest_path: Path,
    scenario_name: str,
    temperature: float,
    run_args: dict,
    required_inputs: list[Path],
    output_path: Path,
) -> None:
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario_name,
        "scenario_label": SCENARIOS[scenario_name]["label"],
        "purpose": "Air-temperature input for population-weighted health assessment",
        "temperature_setting_c": temperature,
        "model_arguments": run_args,
        "inputs": [_input_record(path) for path in required_inputs],
        "output": _raster_record(output_path),
        "software": {
            "invest": natcap.invest.__version__,
            "python": platform.python_version(),
            "gdal": gdal.VersionInfo("RELEASE_NAME"),
            "pygeoprocessing": pygeoprocessing.__version__,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SCENARIOS),
        default=list(SCENARIOS),
    )
    parser.add_argument("--temperatures", nargs="+", type=float, default=[25, 28])
    parser.add_argument("--uhi-max", type=float, default=5)
    parser.add_argument("--humidity", type=float, default=45)
    parser.add_argument(
        "--green30-lulc",
        type=Path,
        help="Use a reviewed Green30 LULC override, such as the NoData-harmonized copy.",
    )
    parser.add_argument(
        "--green30-workspace",
        type=Path,
        help="Write an overridden Green30 run to a separate reviewed workspace.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and model arguments without running InVEST.",
    )
    args = parser.parse_args()

    if natcap.invest.__version__ != REQUIRED_INVEST_VERSION:
        parser.error(
            f"This production runner requires InVEST {REQUIRED_INVEST_VERSION}; "
            f"the active environment provides {natcap.invest.__version__}."
        )

    data_root = args.data_root.expanduser().resolve()
    input_root = data_root / "1_preprocess/UrbanCoolingModel/OfficialWorkingInputs"
    common_inputs = {
        "aoi_vector_path": input_root / "AOIs/London_Borough_aoi.shp",
        "biophysical_table_path": (
            input_root / "LULC/Biophysical_table_ukech_2021_london_with_TCC.csv"
        ),
        "ref_eto_raster_path": (
            input_root / "evapotranspiration/et0_V3_07_clipped_reprojected.tif"
        ),
    }
    output_parent = data_root / "2_postprocess_intermediate/UCM_official_runs"

    for scenario_name in args.scenarios:
        definition = SCENARIOS[scenario_name]
        if scenario_name == "green30" and args.green30_lulc:
            lulc_path = args.green30_lulc.expanduser().resolve()
        else:
            lulc_path = input_root / definition["lulc"]
        if scenario_name == "green30" and args.green30_workspace:
            workspace = args.green30_workspace.expanduser().resolve()
        else:
            workspace = output_parent / definition["workspace"]
        required_inputs = [*common_inputs.values(), lulc_path]
        missing = [path for path in required_inputs if not path.exists()]
        if missing:
            parser.error("Missing required input(s): " + ", ".join(map(str, missing)))

        # InVEST validation requires a writable workspace. Validation may create
        # this empty directory, but it never creates model outputs.
        workspace.mkdir(parents=True, exist_ok=True)
        for temperature in args.temperatures:
            temperature_label = f"{temperature:g}"
            suffix = (
                f"london_{scenario_name}_{temperature_label}deg_"
                f"{args.uhi_max:g}uhi_{args.humidity:g}hum"
            )
            output_path = workspace / f"T_air_{suffix}.tif"
            manifest_path = workspace / f"run_manifest_{suffix}.json"
            if (output_path.exists() or manifest_path.exists()) and not args.validate_only:
                raise FileExistsError(
                    "Refusing to overwrite a completed or documented run: "
                    f"{output_path}"
                )

            run_args = {
                "aoi_vector_path": str(common_inputs["aoi_vector_path"]),
                "avg_rel_humidity": args.humidity,
                "biophysical_table_path": str(common_inputs["biophysical_table_path"]),
                "cc_method": "factors",
                "cc_weight_albedo": "",
                "cc_weight_eti": "",
                "cc_weight_shade": "",
                "do_energy_valuation": False,
                "do_productivity_valuation": False,
                "green_area_cooling_distance": 450,
                "lulc_raster_path": str(lulc_path),
                "ref_eto_raster_path": str(common_inputs["ref_eto_raster_path"]),
                "results_suffix": suffix,
                "t_air_average_radius": 500,
                "t_ref": temperature,
                "uhi_max": args.uhi_max,
                "workspace_dir": str(workspace),
            }
            warnings = natcap.invest.urban_cooling_model.validate(run_args)
            if warnings:
                formatted = "\n".join(
                    f"{', '.join(keys)}: {message}" for keys, message in warnings
                )
                raise ValueError(
                    f"InVEST validation failed for {scenario_name} at "
                    f"{temperature_label} C:\n{formatted}"
                )
            if args.validate_only:
                LOGGER.info("Validated %s at %s C", scenario_name, temperature_label)
                continue

            LOGGER.info("Running %s at %s C", scenario_name, temperature_label)
            natcap.invest.urban_cooling_model.execute(run_args)
            if not output_path.exists():
                raise FileNotFoundError(f"Expected InVEST output was not created: {output_path}")
            _write_manifest(
                manifest_path,
                scenario_name,
                temperature,
                run_args,
                required_inputs,
                output_path,
            )
            LOGGER.info("Documented run in %s", manifest_path)


if __name__ == "__main__":
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(natcap.invest.utils.LOG_FMT))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    main()
