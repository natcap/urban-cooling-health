#!/usr/bin/env python3
"""Audit InVEST 3.20.2 health inputs and compare them with legacy outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio


def _same_grid(left: rasterio.DatasetReader, right: rasterio.DatasetReader) -> bool:
    return (
        left.width == right.width
        and left.height == right.height
        and left.transform == right.transform
        and left.crs == right.crs
    )


def _valid(array: np.ndarray, nodata: float | None) -> np.ndarray:
    valid = np.isfinite(array)
    if nodata is not None:
        valid &= array != nodata
    return valid


def _pair_metrics(left_path: Path, right_path: Path, population_path: Path) -> dict:
    """Summarize right minus left and its population-weighted mean."""
    count = 0
    sum_diff = 0.0
    sum_abs = 0.0
    max_abs = 0.0
    weighted_sum = 0.0
    population_sum = 0.0
    with (
        rasterio.open(left_path) as left,
        rasterio.open(right_path) as right,
        rasterio.open(population_path) as population,
    ):
        if not _same_grid(left, right) or not _same_grid(left, population):
            raise ValueError(f"Raster grid mismatch: {left_path}, {right_path}, {population_path}")
        for _, window in left.block_windows(1):
            left_array = left.read(1, window=window)
            right_array = right.read(1, window=window)
            pop_array = population.read(1, window=window)
            valid = (
                _valid(left_array, left.nodata)
                & _valid(right_array, right.nodata)
                & _valid(pop_array, population.nodata)
            )
            if not valid.any():
                continue
            difference = right_array[valid].astype(np.float64) - left_array[valid]
            weights = np.maximum(pop_array[valid].astype(np.float64), 0.0)
            count += difference.size
            sum_diff += difference.sum()
            sum_abs += np.abs(difference).sum()
            max_abs = max(max_abs, float(np.abs(difference).max()))
            weighted_sum += float(np.dot(difference, weights))
            population_sum += weights.sum()
    return {
        "valid_pixels": count,
        "mean_difference_c": sum_diff / count,
        "mean_absolute_difference_c": sum_abs / count,
        "max_absolute_difference_c": max_abs,
        "population_weighted_mean_difference_c": weighted_sum / population_sum,
        "population_sum": population_sum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    runs = data_root / "2_postprocess_intermediate/UCM_official_runs"
    population = (
        data_root / "1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/pop_raster/"
        "resampled_10m/gbr_pop_2021_10m_count_preserved_bng.tif"
    )
    rasters = {
        "population_2021": population,
        "baseline_new_25": runs / "baseline_health_invest3202/T_air_london_baseline_25deg_5uhi_45hum.tif",
        "baseline_new_28": runs / "baseline_health_invest3202/T_air_london_baseline_28deg_5uhi_45hum.tif",
        "green30_new_25": runs / "green30_health_invest3202/T_air_london_green30_25deg_5uhi_45hum.tif",
        "green30_new_28": runs / "green30_health_invest3202/T_air_london_green30_28deg_5uhi_45hum.tif",
        "green30_harmonized_25": runs / "green30_health_invest3202_nodata_harmonized/T_air_london_green30_25deg_5uhi_45hum.tif",
        "green30_harmonized_28": runs / "green30_health_invest3202_nodata_harmonized/T_air_london_green30_28deg_5uhi_45hum.tif",
        "target30_new_25": runs / "scenario730v4_equal_budget_health_invest3202/T_air_london_scenario730v4_equal_budget_25deg_5uhi_45hum.tif",
        "target30_new_28": runs / "scenario730v4_equal_budget_health_invest3202/T_air_london_scenario730v4_equal_budget_28deg_5uhi_45hum.tif",
        "baseline_old_25": runs / "scenario0/work_and_energy_runs/intermediate/T_air_london_scenario_25.0deg_5.0uhi_45.0hum_energy_productivity.tif",
        "baseline_old_28": runs / "scenario0/work_and_energy_runs/intermediate/T_air_london_scenario_28deg_5uhi_45hum_energy_productivity.tif",
        "green30_old_25": runs / "scenario43/work_and_energy_runs/tcc_30prc/intermediate/T_air_london_scenario4_30prc_25.0deg_5.0uhi_45.0hum_energy_productivity.tif",
        "green30_old_28": runs / "scenario43/work_and_energy_runs/tcc_30prc/intermediate/T_air_london_scenario4_30prc_28deg_5uhi_45hum_energy_productivity.tif",
    }
    missing = [path for path in rasters.values() if not path.exists()]
    if missing:
        parser.error("Missing required raster(s): " + ", ".join(map(str, missing)))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_rows = []
    with rasterio.open(population) as reference:
        for name, path in rasters.items():
            with rasterio.open(path) as dataset:
                grid_rows.append({
                    "raster": name,
                    "path": str(path),
                    "same_grid_as_population": _same_grid(reference, dataset),
                    "width": dataset.width,
                    "height": dataset.height,
                    "pixel_width": dataset.transform.a,
                    "pixel_height": dataset.transform.e,
                    "left": dataset.bounds.left,
                    "top": dataset.bounds.top,
                    "crs": str(dataset.crs),
                })
    if not all(row["same_grid_as_population"] for row in grid_rows):
        raise ValueError("At least one model raster does not match the population grid")

    comparisons = [
        ("baseline_version_change_25", "baseline_old_25", "baseline_new_25"),
        ("baseline_version_change_28", "baseline_old_28", "baseline_new_28"),
        ("green30_version_change_25", "green30_old_25", "green30_new_25"),
        ("green30_version_change_28", "green30_old_28", "green30_new_28"),
        ("green30_nodata_harmonization_25", "green30_new_25", "green30_harmonized_25"),
        ("green30_nodata_harmonization_28", "green30_new_28", "green30_harmonized_28"),
        # For these rows, right minus left is the cooling delivered by the scenario.
        ("green30_cooling_new_25", "green30_new_25", "baseline_new_25"),
        ("green30_cooling_new_28", "green30_new_28", "baseline_new_28"),
        ("green30_cooling_harmonized_25", "green30_harmonized_25", "baseline_new_25"),
        ("green30_cooling_harmonized_28", "green30_harmonized_28", "baseline_new_28"),
        ("target30_cooling_new_25", "target30_new_25", "baseline_new_25"),
        ("target30_cooling_new_28", "target30_new_28", "baseline_new_28"),
        ("green30_cooling_old_25", "green30_old_25", "baseline_old_25"),
        ("green30_cooling_old_28", "green30_old_28", "baseline_old_28"),
    ]
    result_rows = []
    for comparison, left_name, right_name in comparisons:
        result_rows.append({
            "comparison": comparison,
            "left": left_name,
            "right": right_name,
            "difference_definition": "right minus left",
            **_pair_metrics(rasters[left_name], rasters[right_name], population),
        })

    for filename, rows in (
        ("invest_ucm_health_grid_audit.csv", grid_rows),
        ("invest_ucm_health_rerun_comparison.csv", result_rows),
    ):
        output_path = output_dir / filename
        with output_path.open("w", newline="", encoding="utf-8") as file_obj:
            # Use one stable line ending so regenerated audit tables have
            # identical text formatting on Windows, macOS and Linux.
            writer = csv.DictWriter(
                file_obj, fieldnames=rows[0].keys(), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
