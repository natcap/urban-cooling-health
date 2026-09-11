#!/usr/bin/env python3
"""Run revised paired Green/Target scenarios in InVEST 3.20.2.

The default health run writes temperature only.  ``--include-valuations``
also produces building-energy and WBGT outputs.  The project-specific Hothaps
work-intensity calculation must be run separately on the resulting WBGT
rasters; InVEST's built-in work-loss layers are retained only as intermediate
outputs.
"""

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
SCENARIOS = {
    "baseline": "LULC/LCM2023_London_10m_clip2aoi_tcc24.tif",
    "green10": (
        "LULC/lc_green_scenarios_output/revised_v2_invest_3.20.2_2026-09-10/"
        "Green10_equal_area_eligible_v2.tif"
    ),
    "green20": (
        "LULC/lc_green_scenarios_output/revised_v2_invest_3.20.2_2026-09-10/"
        "Green20_equal_area_eligible_v2.tif"
    ),
    "green30": (
        "LULC/lc_green_scenarios_output/revised_v2_invest_3.20.2_2026-09-10/"
        "Green30_equal_area_eligible_v2.tif"
    ),
    "target10": (
        "LULC/lc_tree_equity_scenarios_output/"
        "revised_v2_rank_fid_equal_area_2026-09-10/"
        "LULC_Target10_equal_area_eligible_v2.tif"
    ),
    "target20": (
        "LULC/lc_tree_equity_scenarios_output/"
        "revised_v2_rank_fid_equal_area_2026-09-10/"
        "LULC_Target20_equal_area_eligible_v2.tif"
    ),
    "target30": (
        "LULC/lc_tree_equity_scenarios_output/"
        "revised_v2_rank_fid_equal_area_2026-09-10/"
        "LULC_Target30_equal_area_eligible_v2.tif"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_record(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _raster_record(path: Path) -> dict[str, object]:
    info = pygeoprocessing.get_raster_info(str(path))
    statistics = info.get("statistics")
    return {
        **_input_record(path),
        "raster_size": list(info["raster_size"]),
        "pixel_size": list(info["pixel_size"]),
        "bounding_box": list(info["bounding_box"]),
        "projection_wkt": info["projection_wkt"],
        "nodata": list(info["nodata"]),
        "statistics": list(statistics[0]) if statistics else None,
    }


def _git_record(repo_root: Path | None) -> dict[str, object] | None:
    if repo_root is None:
        return None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout
        return {"root": str(repo_root), "commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"root": str(repo_root), "commit": None, "dirty": None}


def _parse_overrides(values: list[str]) -> dict[str, Path]:
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Scenario overrides must use NAME=/absolute/path.tif")
        name, raw_path = value.split("=", 1)
        if name not in SCENARIOS:
            raise ValueError(f"Unknown scenario override: {name}")
        overrides[name] = Path(raw_path).expanduser().resolve()
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--temperatures", nargs="+", type=float, default=[25, 28])
    parser.add_argument("--uhi-max", type=float, default=5)
    parser.add_argument("--humidity", type=float, default=45)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--scenario-lulc", action="append", default=[], metavar="NAME=PATH",
        help="Override a scenario LULC without editing this script.",
    )
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument(
        "--include-valuations",
        action="store_true",
        help=(
            "Also calculate building-energy savings and WBGT. The project's "
            "separate Hothaps script should be used for work productivity."
        ),
    )
    parser.add_argument(
        "--building-vector",
        type=Path,
        help="Override the default energy-buildings vector.",
    )
    parser.add_argument(
        "--energy-table",
        type=Path,
        help="Override the default energy-consumption table.",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if natcap.invest.__version__ != REQUIRED_INVEST_VERSION:
        parser.error(
            f"This runner requires InVEST {REQUIRED_INVEST_VERSION}; "
            f"found {natcap.invest.__version__}."
        )
    data_root = args.data_root.expanduser().resolve()
    input_root = data_root / "1_preprocess/UrbanCoolingModel/OfficialWorkingInputs"
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else data_root
        / "2_postprocess_intermediate/UCM_official_runs/"
        "revised_equal_area_invest3202_2026-09-10"
    )
    repo_root = args.repo_root.expanduser().resolve() if args.repo_root else None
    try:
        overrides = _parse_overrides(args.scenario_lulc)
    except ValueError as error:
        parser.error(str(error))
    common_inputs = {
        "aoi_vector_path": input_root / "AOIs/London_Borough_aoi.shp",
        "biophysical_table_path": (
            input_root / "LULC/Biophysical_table_ukech_2021_london_with_TCC.csv"
        ),
        "ref_eto_raster_path": (
            input_root / "evapotranspiration/et0_V3_07_clipped_reprojected.tif"
        ),
    }
    if args.include_valuations:
        common_inputs.update({
            "building_vector_path": (
                args.building_vector.expanduser().resolve()
                if args.building_vector
                else input_root / "energy_buildings/bld_with_attr_compact_ucm2.gpkg"
            ),
            "energy_consumption_table_path": (
                args.energy_table.expanduser().resolve()
                if args.energy_table
                else input_root / "energy_buildings/_UCM_Energy Consumption Table.csv"
            ),
        })

    for scenario_name in args.scenarios:
        lulc_path = overrides.get(scenario_name, input_root / SCENARIOS[scenario_name])
        workspace = output_root / scenario_name
        required_inputs = [*common_inputs.values(), lulc_path]
        missing = [path for path in required_inputs if not path.is_file()]
        if missing:
            parser.error("Missing required input(s): " + ", ".join(map(str, missing)))
        workspace.mkdir(parents=True, exist_ok=True)
        for temperature in args.temperatures:
            temperature_label = f"{temperature:g}"
            suffix = (
                f"london_{scenario_name}_{temperature_label}deg_"
                f"{args.uhi_max:g}uhi_{args.humidity:g}hum_revised_equal_area"
            )
            output_path = workspace / f"T_air_{suffix}.tif"
            manifest_path = workspace / f"run_manifest_{suffix}.json"
            if not args.validate_only and (output_path.exists() or manifest_path.exists()):
                raise FileExistsError(f"Refusing to overwrite a documented run: {output_path}")
            run_args = {
                "aoi_vector_path": str(common_inputs["aoi_vector_path"]),
                "avg_rel_humidity": args.humidity,
                "biophysical_table_path": str(common_inputs["biophysical_table_path"]),
                "cc_method": "factors",
                "cc_weight_albedo": "",
                "cc_weight_eti": "",
                "cc_weight_shade": "",
                "do_energy_valuation": args.include_valuations,
                "do_productivity_valuation": args.include_valuations,
                "green_area_cooling_distance": 450,
                "lulc_raster_path": str(lulc_path),
                "ref_eto_raster_path": str(common_inputs["ref_eto_raster_path"]),
                "results_suffix": suffix,
                "t_air_average_radius": 500,
                "t_ref": temperature,
                "uhi_max": args.uhi_max,
                "workspace_dir": str(workspace),
            }
            if args.include_valuations:
                run_args.update({
                    "building_vector_path": str(common_inputs["building_vector_path"]),
                    "energy_consumption_table_path": str(
                        common_inputs["energy_consumption_table_path"]
                    ),
                })
            warnings = natcap.invest.urban_cooling_model.validate(run_args)
            if warnings:
                formatted = "\n".join(
                    f"{', '.join(keys)}: {message}" for keys, message in warnings
                )
                raise ValueError(
                    f"Validation failed for {scenario_name} at {temperature_label} C:\n"
                    f"{formatted}"
                )
            if args.validate_only:
                LOGGER.info("Validated %s at %s C", scenario_name, temperature_label)
                continue

            LOGGER.info("Running %s at %s C", scenario_name, temperature_label)
            natcap.invest.urban_cooling_model.execute(run_args)
            if not output_path.is_file():
                raise FileNotFoundError(f"Expected output was not created: {output_path}")
            additional_outputs = []
            if args.include_valuations:
                additional_outputs = [
                    workspace / f"buildings_with_stats_{suffix}.shp",
                    workspace / "intermediate" / f"wbgt_{suffix}.tif",
                ]
                missing_outputs = [path for path in additional_outputs if not path.is_file()]
                if missing_outputs:
                    raise FileNotFoundError(
                        "Expected valuation output(s) were not created: "
                        + ", ".join(map(str, missing_outputs))
                    )
            manifest = {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "scenario": scenario_name,
                "purpose": (
                    "Air temperature, building energy and WBGT for revised "
                    "scenario comparison"
                    if args.include_valuations
                    else "Air-temperature input for population-weighted health assessment"
                ),
                "temperature_setting_c": temperature,
                "model_arguments": run_args,
                "inputs": [_input_record(path) for path in required_inputs],
                "output": _raster_record(output_path),
                "additional_outputs": [
                    _raster_record(path) if path.suffix.lower() == ".tif" else _input_record(path)
                    for path in additional_outputs
                ],
                "software": {
                    "invest": natcap.invest.__version__,
                    "python": platform.python_version(),
                    "gdal": gdal.VersionInfo("RELEASE_NAME"),
                    "pygeoprocessing": pygeoprocessing.__version__,
                },
                "git": _git_record(repo_root),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            LOGGER.info("Documented run in %s", manifest_path)
    return 0


if __name__ == "__main__":
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(natcap.invest.utils.LOG_FMT))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    raise SystemExit(main())
