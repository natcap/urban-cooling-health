"""Plot the daily temperature context for the Figure 4b capacity estimands.

This diagnostic preserves the exact spatial support used by the JJA capacity
candidate.  It contrasts (1) spatially averaged, reconstructed daily maximum
temperatures and (2) a single central-London grid cell with the corresponding
fixed 25/5 C UCM temperatures.  The central cell is illustrative only: daily
climate offsets are spatially uniform in this reconstruction, so it is not an
additional citywide estimand.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, osr

try:
    from .valuation_accounting import london_grid, sha256
except ImportError:
    from valuation_accounting import london_grid, sha256


CH_CROSS_LON_LAT = (-0.1278, 51.5074)


def _local_xy(dataset):
    """Return the raster coordinate of Charing Cross in the raster CRS."""
    src = osr.SpatialReference()
    src.ImportFromEPSG(4326)
    dst = osr.SpatialReference()
    dst.ImportFromWkt(dataset.GetProjection())
    # GDAL 3 otherwise follows EPSG's latitude-longitude axis ordering for
    # EPSG:4326. The supplied coordinate is intentionally longitude, latitude.
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(src, dst)
    return transform.TransformPoint(*CH_CROSS_LON_LAT)[:2]


def _nearest_valid_cell(valid, dataset):
    """Find nearest valid 10-m cell to Charing Cross, the urban-centre proxy."""
    x, y = _local_xy(dataset)
    gt = dataset.GetGeoTransform()
    rows, cols = np.indices(valid.shape)
    xs = gt[0] + (cols + 0.5) * gt[1] + (rows + 0.5) * gt[2]
    ys = gt[3] + (cols + 0.5) * gt[4] + (rows + 0.5) * gt[5]
    distance2 = (xs - x) ** 2 + (ys - y) ** 2
    distance2[~valid] = np.inf
    row, col = np.unravel_index(np.argmin(distance2), valid.shape)
    return int(row), int(col), float(xs[row, col]), float(ys[row, col])


def _write_csv(path, rows):
    with path.open("x", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    gdal.UseExceptions()
    gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")
    ucm_root = args.data_root / "2_postprocess_intermediate/UCM_official_runs"
    corrected_manifest = (
        ucm_root / "diagnostics/remaining_corrected_temperatures_2026-11-12_v1/manifest.json"
    )
    corrected = json.loads(corrected_manifest.read_text())
    scenario_paths = {k: Path(v) for k, v in corrected["scenario_paths"].items()}
    for name, path in scenario_paths.items():
        if sha256(path) != corrected["outputs"][name]:
            raise ValueError(f"Changed corrected raster: {path}")

    offsets_path = (
        ucm_root
        / "diagnostics/nine_scenario_health_readiness_2026-11-12_v2/shared_daily_offsets.csv"
    )
    with offsets_path.open() as source:
        offsets = [
            row for row in csv.DictReader(source)
            if row["reconstruction"] == "worldclim_aligned" and row["ssp"] == "ssp370"
        ]
    if len(offsets) != 15 * 92:
        raise ValueError("Expected 15 climate-weather members and 92 JJA days")

    boundary = args.data_root / "1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/London_Borough_aoi.shp"
    _, zones, _ = london_grid(boundary, list(scenario_paths.values()))
    base_dataset = gdal.Open(str(scenario_paths["baseline"]))
    base = base_dataset.ReadAsArray().astype(float)
    valid = zones > 0
    city_baseline = float(base[valid].mean())
    row, col, x, y = _nearest_valid_cell(valid, base_dataset)
    centre_baseline = float(base[row, col])

    # The earlier fixed benchmark is an unmodified 25 C reference-temperature,
    # 5 C UHI UCM run.  Use its paired baseline temperature for both the
    # citywide mean and the same central grid cell.
    fixed_root = ucm_root / "production/equal_area_2026-09/03_energy_work_capacity/baseline"
    fixed_paths = sorted(fixed_root.glob("T_air_*.tif"))
    if len(fixed_paths) != 1:
        raise ValueError(f"Expected one fixed baseline raster, found {fixed_paths}")
    fixed_dataset = gdal.Open(str(fixed_paths[0]))
    if (fixed_dataset.RasterXSize, fixed_dataset.RasterYSize) != base.shape[::-1]:
        raise ValueError("Fixed and reconstructed raster grids do not match")
    fixed = fixed_dataset.ReadAsArray().astype(float)
    fixed_city = float(fixed[valid].mean())
    fixed_centre = float(fixed[row, col])

    series = []
    for record in offsets:
        series.append({
            "date": record["date"], "gcm": record["gcm"], "weather_year": record["weather_year"],
            "offset_c": float(record["offset_c"]),
            "city_tmax_c": city_baseline + float(record["offset_c"]),
            "central_london_tmax_c": centre_baseline + float(record["offset_c"]),
        })
    args.output_dir.mkdir(parents=True)
    _write_csv(args.output_dir / "jja_daily_tmax_context_20261113_v01_data.csv", series)

    members = {}
    for value in series:
        members.setdefault((value["gcm"], value["weather_year"]), []).append(value)
    for values in members.values():
        values.sort(key=lambda value: value["date"])
        if len(values) != 92:
            raise ValueError("A climate-weather member does not contain 92 daily records")
    city_mean = float(np.mean([value["city_tmax_c"] for value in series]))
    centre_mean = float(np.mean([value["central_london_tmax_c"] for value in series]))

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.2), constrained_layout=True)
    specs = [
        (axes[0], "city_tmax_c", "London area-weighted mean", fixed_city, city_mean,
         "a", "Reconstructed daily maximum temperature across London"),
        (axes[1], "central_london_tmax_c", "Central-London grid cell", fixed_centre, centre_mean,
         "b", "Illustrative central-London grid cell"),
    ]
    for ax, field, label, fixed_value, mean_value, letter, title in specs:
        for values in members.values():
            ax.plot(range(1, 93), [value[field] for value in values], color="#5E81AC", alpha=0.20, lw=0.7)
        daily_mean = [np.mean([values[i][field] for values in members.values()]) for i in range(92)]
        ax.plot(range(1, 93), daily_mean, color="#1F4E79", lw=1.8,
                label="Mean of 15 reconstructed series")
        ax.axhline(fixed_value, color="#A84321", lw=1.6, ls="--",
                   label=f"Fixed 25/5°C UCM condition ({fixed_value:.2f}°C)")
        ax.axhline(mean_value, color="#1F4E79", lw=0.9, ls=":",
                   label=f"JJA mean ({mean_value:.2f}°C)")
        for xline in (31, 62):
            ax.axvline(xline + 0.5, color="#B8B8B8", lw=0.6, zorder=0)
        ax.set_xlim(1, 92)
        ax.set_xticks((1, 31, 62, 92), ("1 Jun", "1 Jul", "1 Aug", "31 Aug"))
        ax.set_xlabel("Day of reconstructed June–August period")
        ax.set_ylabel("Daily maximum temperature (°C)")
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")
        ax.grid(axis="y", color="#DDDDDD", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.8, loc="upper left")
    axes[1].text(
        0.02, -0.34,
        "The local series is illustrative. The reconstruction adds the same daily climate offset\n"
        "to every grid cell; spatial differences are retained through the UCM baseline field.",
        transform=axes[1].transAxes, fontsize=7.5, color="#444444", va="top",
    )
    figure.savefig(args.output_dir / "jja_daily_tmax_vs_fixed_25_5_20261113_v01.png", dpi=300,
                   bbox_inches="tight", facecolor="white")
    plt.close(figure)

    metadata = {
        "status": "context diagnostic only; no production outputs changed",
        "city_series": "Area-weighted mean across the exact corrected-capacity support",
        "central_series": "Nearest valid 10-m cell to Charing Cross (-0.1278, 51.5074); illustrative only",
        "cell_row_col": [row, col], "cell_x_y": [x, y],
        "reconstruction": "WorldClim-aligned, SSP370, 15 GCM-weather members, JJA, current-day offset_c",
        "city_baseline_tmax_c": city_baseline, "central_baseline_tmax_c": centre_baseline,
        "fixed_city_tmax_c": fixed_city, "fixed_central_tmax_c": fixed_centre,
        "city_jja_mean_tmax_c": city_mean, "central_jja_mean_tmax_c": centre_mean,
        "valid_cells": int(valid.sum()),
        "input_sha256": {str(path): sha256(path) for path in [corrected_manifest, offsets_path, fixed_paths[0]]},
    }
    (args.output_dir / "jja_daily_tmax_vs_fixed_25_5_20261113_v01_manifest.json").write_text(
        json.dumps(metadata, indent=2)
    )


if __name__ == "__main__":
    main()
