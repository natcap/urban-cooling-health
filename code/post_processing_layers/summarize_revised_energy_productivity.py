#!/usr/bin/env python3
"""Summarize revised UCM energy and Hothaps productivity outputs.

Citywide energy totals are summed once across the unique building layer. The
InVEST borough field is retained for spatial summaries and reconciled against
that unique total because a building intersecting two boroughs can otherwise
be counted twice. Productivity is the area-weighted mean Hothaps workability
fraction over valid 10 m cells; scenario changes are percentage points versus
the matching-temperature baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy
import pygeoprocessing
from osgeo import gdal


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sum_field(vector_path: Path, field_name: str) -> tuple[float, int, int]:
    dataset = gdal.OpenEx(str(vector_path), gdal.OF_VECTOR)
    if dataset is None:
        raise ValueError(f"Could not open vector: {vector_path}")
    layer = dataset.GetLayer()
    # Skip geometry and unused attributes; large building shapefiles then scan
    # the DBF only, which is substantially faster and uses less memory.
    definition = layer.GetLayerDefn()
    ignored = [
        definition.GetFieldDefn(index).GetName()
        for index in range(definition.GetFieldCount())
        if definition.GetFieldDefn(index).GetName() != field_name
    ]
    layer.SetIgnoredFields(["OGR_GEOMETRY", *ignored])
    total = 0.0
    valid = 0
    missing = 0
    for feature in layer:
        value = feature.GetField(field_name)
        if value is None:
            missing += 1
        else:
            total += float(value)
            valid += 1
    return total, valid, missing


def _raster_mean(raster_path: Path) -> tuple[float, int]:
    info = pygeoprocessing.get_raster_info(str(raster_path))
    nodata = info["nodata"][0]
    total = 0.0
    count = 0
    for _, block in pygeoprocessing.iterblocks((str(raster_path), 1)):
        valid = numpy.isfinite(block)
        if nodata is not None:
            valid &= ~numpy.isclose(block, nodata)
        total += float(block[valid].sum(dtype=numpy.float64))
        count += int(valid.sum())
    if count == 0:
        raise ValueError(f"No valid productivity pixels in {raster_path}")
    return total / count, count


def _borough_productivity(
    borough_path: Path, raster_path: Path
) -> list[dict[str, object]]:
    dataset = gdal.OpenEx(str(borough_path), gdal.OF_VECTOR)
    layer = dataset.GetLayer()
    names = {
        feature.GetFID(): {
            "borough": feature.GetField("NAME"),
            "borough_code": feature.GetField("GSS_CODE"),
        }
        for feature in layer
    }
    stats = pygeoprocessing.zonal_statistics(
        (str(raster_path), 1), str(borough_path), polygons_might_overlap=False
    )
    rows = []
    for fid, identity in names.items():
        stat = stats[fid]
        count = int(stat["count"])
        rows.append({
            **identity,
            "valid_pixels": count,
            "workability_fraction": float(stat["sum"]) / count if count else None,
        })
    return rows


def _borough_energy(uhi_path: Path) -> list[dict[str, object]]:
    dataset = gdal.OpenEx(str(uhi_path), gdal.OF_VECTOR)
    layer = dataset.GetLayer()
    rows = []
    for feature in layer:
        rows.append({
            "borough": feature.GetField("NAME"),
            "borough_code": feature.GetField("GSS_CODE"),
            "energy_savings": feature.GetField("avd_eng_cn"),
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ucm_output_root", type=Path)
    parser.add_argument("borough_vector", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    ucm_root = args.ucm_output_root.expanduser().resolve()
    borough_path = args.borough_vector.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else ucm_root / "summary"
    )
    hothaps_root = ucm_root / "hothaps_work_intensity"
    city_rows = []
    borough_rows = []

    for workability_path in sorted(hothaps_root.glob("*/high_work_productivity_*.tif")):
        scenario = workability_path.parent.name
        suffix = workability_path.stem.removeprefix("high_work_productivity_")
        workspace = ucm_root / scenario
        buildings_path = workspace / f"buildings_with_stats_{suffix}.shp"
        uhi_path = workspace / f"uhi_results_{suffix}.shp"
        for required in (buildings_path, uhi_path, borough_path):
            if not required.is_file():
                parser.error(f"Missing required output: {required}")

        temperature = float(suffix.split("deg_", 1)[0].rsplit("_", 1)[-1])
        energy_unique, valid_buildings, missing_energy = _sum_field(
            buildings_path, "energy_sav"
        )
        energy_borough_sum, _, _ = _sum_field(uhi_path, "avd_eng_cn")
        workability_mean, valid_pixels = _raster_mean(workability_path)
        city_rows.append({
            "scenario": scenario,
            "temperature_c": temperature,
            "energy_savings_unique_buildings": energy_unique,
            "energy_savings_borough_sum": energy_borough_sum,
            "borough_intersection_overcount": energy_borough_sum - energy_unique,
            "borough_intersection_overcount_pct": (
                (energy_borough_sum / energy_unique - 1) * 100
                if energy_unique else None
            ),
            "buildings_with_energy": valid_buildings,
            "buildings_missing_energy": missing_energy,
            "workability_fraction": workability_mean,
            "valid_productivity_pixels": valid_pixels,
        })

        energy_by_borough = {
            row["borough_code"]: row for row in _borough_energy(uhi_path)
        }
        for row in _borough_productivity(borough_path, workability_path):
            energy = energy_by_borough.get(row["borough_code"], {})
            borough_rows.append({
                "scenario": scenario,
                "temperature_c": temperature,
                **row,
                "energy_savings": energy.get("energy_savings"),
            })

    if not city_rows:
        parser.error(f"No Hothaps outputs found under {hothaps_root}")

    baseline = {
        row["temperature_c"]: row for row in city_rows if row["scenario"] == "baseline"
    }
    if set(baseline) != {row["temperature_c"] for row in city_rows}:
        parser.error("A matching-temperature baseline is required for every scenario")
    for row in city_rows:
        reference = baseline[row["temperature_c"]]
        row["energy_change_vs_baseline"] = (
            row["energy_savings_unique_buildings"]
            - reference["energy_savings_unique_buildings"]
        )
        row["productivity_change_vs_baseline_pp"] = (
            row["workability_fraction"] - reference["workability_fraction"]
        ) * 100

    city_path = output_dir / "citywide_energy_productivity_summary.csv"
    borough_output_path = output_dir / "borough_energy_productivity_summary.csv"
    _write_csv(city_path, city_rows)
    _write_csv(borough_output_path, borough_rows)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ucm_output_root": str(ucm_root),
        "borough_vector": str(borough_path),
        "methods": {
            "energy_citywide": "sum energy_sav once across unique buildings",
            "energy_borough": (
                "InVEST avd_eng_cn intersection aggregate; compare its city sum "
                "with the unique-building total before reporting"
            ),
            "productivity": "area-weighted mean Hothaps workability over valid 10 m cells",
        },
        "outputs": {
            "citywide": {"path": str(city_path), "sha256": _sha256(city_path)},
            "borough": {
                "path": str(borough_output_path),
                "sha256": _sha256(borough_output_path),
            },
        },
    }
    manifest_path = output_dir / "energy_productivity_summary_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created {city_path}")
    print(f"Created {borough_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
