#!/usr/bin/env python3
"""Aggregate revised health rasters to LSOA11 with a verified code crosswalk."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize


CAUSES = (
    "all_cause",
    "mental_disorder",
    "cardiovascular",
    "respiratory",
    "self_harm",
)
HISTORICAL_SCENARIOS = {
    "green30_25c": ("Green30", 25),
    "target30_25c": ("Target30", 25),
    "green30_28c": ("Green30", 28),
    "target30_28c": ("Target30", 28),
}
REVISED_SCENARIOS = {
    f"{family}{level}_{temperature}c": (
        f"{family.capitalize()}{level}", temperature
    )
    for temperature in (25, 28)
    for level in (10, 20, 30)
    for family in ("green", "target")
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, relative_to: Path | None = None) -> dict:
    """Record identity without embedding a user's machine-specific root."""
    display_path = path
    if relative_to is not None:
        try:
            display_path = path.relative_to(relative_to)
        except ValueError:
            pass
    return {
        "path": str(display_path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _shapefile_records(path: Path, relative_to: Path | None = None) -> list[dict]:
    return [
        _file_record(item, relative_to)
        for item in sorted(path.parent.glob(f"{path.stem}.*"))
    ]


def _make_crosswalk(svi: gpd.GeoDataFrame, official: gpd.GeoDataFrame) -> pd.DataFrame:
    """Match near-identical polygon versions by centroid and enforce one-to-one IDs."""
    required = {"id", "Neighborhood_name"}
    if not required.issubset(svi.columns):
        raise ValueError(f"SVI layer is missing columns: {sorted(required - set(svi.columns))}")
    if not {"LSOA11CD", "LSOA11NM"}.issubset(official.columns):
        raise ValueError("Official boundary layer must contain LSOA11CD and LSOA11NM")
    if svi.crs is None or official.crs is None:
        raise ValueError("Both LSOA layers must have a CRS")

    official = official.to_crs(svi.crs)
    svi_points = svi[["id", "Neighborhood_name", "geometry"]].copy()
    official_points = official[["LSOA11CD", "LSOA11NM", "geometry"]].copy()
    svi_points.geometry = svi_points.geometry.centroid
    official_points.geometry = official_points.geometry.centroid
    joined = gpd.sjoin_nearest(
        svi_points,
        official_points,
        how="left",
        distance_col="centroid_distance_m",
    ).drop(columns=["geometry", "index_right"])
    if len(joined) != len(svi) or joined["id"].nunique() != len(svi):
        raise ValueError("SVI-to-LSOA11 centroid join is not one-to-one on SVI id")
    if joined["LSOA11CD"].nunique() != len(official):
        raise ValueError("SVI-to-LSOA11 centroid join is not one-to-one on LSOA11CD")
    if joined["centroid_distance_m"].max() > 1.0:
        raise ValueError("An SVI polygon centroid is more than 1 metre from its LSOA11 match")
    return joined.sort_values("id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--crosswalk-csv", type=Path, required=True)
    parser.add_argument("--vulnerability-gpkg", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--scenario-set",
        choices=("historical30", "revised-equal-area"),
        default="historical30",
    )
    parser.add_argument(
        "--health-root",
        type=Path,
        help="Override the selected scenario set's health-results directory.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="Optional subset of keys from the selected scenario set.",
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    svi_path = (
        data_root / "1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/"
        "Social_Vulnerability_Index_london.gpkg"
    )
    official_path = (
        data_root / "0_source_data/UK boundary shapefile/"
        "statistical-gis-boundaries-london/statistical-gis-boundaries-london/ESRI/"
        "LSOA_2011_London_gen_MHW.shp"
    )
    scenarios = (
        REVISED_SCENARIOS
        if args.scenario_set == "revised-equal-area"
        else HISTORICAL_SCENARIOS
    )
    if args.scenarios:
        unknown = set(args.scenarios) - set(scenarios)
        if unknown:
            parser.error(f"Unknown scenario(s): {sorted(unknown)}")
        scenarios = {name: scenarios[name] for name in args.scenarios}
    default_health_folder = (
        "health_v3_revised_equal_area_population_weighted_2021_2026-09-10"
        if args.scenario_set == "revised-equal-area"
        else "health_v2_invest3202_population_weighted_2021_nodata_harmonized"
    )
    health_root = (
        args.health_root.expanduser().resolve()
        if args.health_root
        else data_root / "2_postprocess_intermediate/UCM_official_runs" / default_health_folder
    )
    for path in (svi_path, official_path):
        if not path.exists():
            parser.error(f"Missing required input: {path}")

    svi = gpd.read_file(svi_path)
    official = gpd.read_file(official_path)
    if len(svi) != 4835 or len(official) != 4835:
        raise ValueError(f"Expected 4,835 London LSOA11 polygons; found {len(svi)} and {len(official)}")
    crosswalk = _make_crosswalk(svi, official)

    raster_paths = {
        (scenario, cause): health_root / scenario / f"Excess_{cause}.tif"
        for scenario in scenarios
        for cause in CAUSES
    }
    missing = [path for path in raster_paths.values() if not path.exists()]
    if missing:
        parser.error("Missing health raster(s): " + ", ".join(map(str, missing)))

    reference_path = next(iter(raster_paths.values()))
    with rasterio.open(reference_path) as reference:
        if svi.crs != reference.crs:
            svi = svi.to_crs(reference.crs)
        reference_grid = (reference.crs, reference.transform, reference.width, reference.height)
        # Pixel-centre membership matches the historical rasterstats default,
        # while a single zone raster makes all 20 aggregations deterministic.
        zone_ids = rasterize(
            ((geometry, int(identifier)) for geometry, identifier in zip(svi.geometry, svi["id"])),
            out_shape=(reference.height, reference.width),
            transform=reference.transform,
            fill=0,
            dtype="int32",
            all_touched=False,
        )
        reference_values = reference.read(1, masked=True).filled(np.nan)
        unassigned = np.isfinite(reference_values) & (zone_ids == 0)
        unassigned_rows, unassigned_cols = np.nonzero(unassigned)
        edge_assignment_max_distance_m = 0.0
        if unassigned_rows.size:
            # The borough mask and generalized LSOA outline differ slightly at
            # their shared outer edge. Assign only valid, otherwise-unassigned
            # cell centres to the nearest LSOA so zonal and city totals close.
            x_coords, y_coords = rasterio.transform.xy(
                reference.transform, unassigned_rows, unassigned_cols, offset="center"
            )
            edge_points = gpd.GeoDataFrame(
                {"point_index": np.arange(unassigned_rows.size)},
                geometry=gpd.points_from_xy(x_coords, y_coords),
                crs=reference.crs,
            )
            edge_join = gpd.sjoin_nearest(
                edge_points,
                svi[["id", "geometry"]],
                how="left",
                distance_col="nearest_lsoa_distance_m",
            )
            # Equidistant boundary ties are resolved by the smallest stable id.
            edge_join = (
                edge_join.sort_values(["point_index", "nearest_lsoa_distance_m", "id"])
                .drop_duplicates("point_index")
                .sort_values("point_index")
            )
            if edge_join["id"].isna().any():
                raise ValueError("At least one valid edge cell could not be assigned to an LSOA")
            edge_assignment_max_distance_m = float(
                edge_join["nearest_lsoa_distance_m"].max()
            )
            zone_ids[unassigned_rows, unassigned_cols] = edge_join["id"].astype(int)
    zone_pixel_counts = np.bincount(zone_ids.ravel(), minlength=int(svi["id"].max()) + 1)
    if np.any(zone_pixel_counts[svi["id"].astype(int).to_numpy()] == 0):
        raise ValueError("At least one LSOA received no raster-cell centres")

    lookup = crosswalk.set_index("id")
    rows = []
    qa_rows = []
    for (scenario, cause), raster_path in raster_paths.items():
        with rasterio.open(raster_path) as source:
            grid = (source.crs, source.transform, source.width, source.height)
            if grid != reference_grid:
                raise ValueError(f"Raster grid differs from reference: {raster_path}")
            values = source.read(1, masked=True).filled(np.nan).astype(np.float64)
        valid = np.isfinite(values) & (zone_ids > 0)
        remaining_unassigned = np.isfinite(values) & (zone_ids == 0)
        if remaining_unassigned.any():
            raise ValueError(
                f"{int(remaining_unassigned.sum())} valid cells remain outside LSOAs: "
                f"{raster_path}"
            )
        sums = np.bincount(
            zone_ids[valid], weights=values[valid], minlength=len(zone_pixel_counts)
        )
        scenario_label, temperature = scenarios[scenario]
        for identifier in svi["id"].astype(int):
            rows.append({
                "id": identifier,
                "LSOA11CD": lookup.at[identifier, "LSOA11CD"],
                "LSOA11NM": lookup.at[identifier, "LSOA11NM"],
                "Neighborhood_name": lookup.at[identifier, "Neighborhood_name"],
                "scenario": scenario,
                "lc_scenario": scenario_label,
                "temperature_setting_c": temperature,
                "cause": cause,
                "indicator": "All_cause" if cause == "all_cause" else cause,
                "excess_deaths": sums[identifier],
                "deaths_averted": -sums[identifier],
            })
        qa_rows.append({
            "scenario": scenario,
            "cause": cause,
            "lsoa_sum_excess_deaths": float(sums.sum()),
            "valid_raster_sum_excess_deaths": float(np.nansum(values[valid])),
            "unassigned_valid_pixel_count": int(remaining_unassigned.sum()),
        })

    output_csv = args.output_csv.resolve()
    crosswalk_csv = args.crosswalk_csv.resolve()
    vulnerability_gpkg = args.vulnerability_gpkg.resolve()
    manifest_path = args.manifest.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    for output in (output_csv, crosswalk_csv, vulnerability_gpkg, manifest_path):
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    crosswalk.to_csv(crosswalk_csv, index=False)
    # This compact, scenario-independent artifact lets Figure 7 run from a
    # clone without exposing or depending on the external shared-drive path.
    vulnerability = svi.merge(
        crosswalk[["id", "LSOA11CD", "LSOA11NM"]], on="id", validate="one_to_one"
    )
    vulnerability.to_file(vulnerability_gpkg, layer="fig7_vulnerability_lsoa11", driver="GPKG")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "pixel-centre rasterization followed by zonal sum",
        "scenario_set": args.scenario_set,
        "health_root": str(health_root.relative_to(data_root)),
        "lsoa_count": len(svi),
        "crosswalk_method": "one-to-one nearest polygon-centroid match",
        "crosswalk_max_centroid_distance_m": float(crosswalk["centroid_distance_m"].max()),
        "outer_edge_cell_assignment": (
            "Valid raster-cell centres outside the generalized LSOA outline were assigned "
            "to the nearest LSOA; equidistant ties use the smallest stable id"
        ),
        "outer_edge_cell_count": int(unassigned_rows.size),
        "outer_edge_max_assignment_distance_m": edge_assignment_max_distance_m,
        "software": {
            "python": platform.python_version(),
            "geopandas": gpd.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rasterio": rasterio.__version__,
        },
        "inputs": {
            "svi": _file_record(svi_path, data_root),
            "official_lsoa11_shapefile": _shapefile_records(official_path, data_root),
            "health_rasters": [
                _file_record(path, data_root) for path in raster_paths.values()
            ],
        },
        "outputs": {
            "zonal_csv": _file_record(output_csv, repository_root),
            "crosswalk_csv": _file_record(crosswalk_csv, repository_root),
            "vulnerability_gpkg": _file_record(vulnerability_gpkg, repository_root),
        },
        "quality_control": qa_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows):,} zonal rows to {output_csv}")
    print(f"Maximum crosswalk centroid distance: {crosswalk['centroid_distance_m'].max():.4f} m")


if __name__ == "__main__":
    main()
