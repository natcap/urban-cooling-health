"""Prepare a bounded pilot of observation-anchored UCM climate parameters.

This script does not run InVEST or overwrite production data. It uses the four
locally available London MIDAS stations for July 2021, an explicitly provisional
Kenley rural reference, and matched WorldClim historical/future July Tmax grids.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, ogr, osr


STATIONS = {
    "00697": "urban_candidate",
    "00723": "urban_candidate",
    "00726": "rural_candidate",
    "62084": "urban_candidate",
}
MODELS = ["ACCESS-CM2", "IPSL-CM6A-LR", "MRI-ESM2-0"]
SSPS = ["ssp370", "ssp585"]
PERIODS = ["2041-2060", "2061-2080"]
LAPSE_C_PER_M = 0.0065
TARGET_ELEVATION_M = 5.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def table_after_data(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", errors="replace") as file:
        rows = list(csv.reader(file))
    marker = next(i for i, row in enumerate(rows) if row == ["data"])
    header = rows[marker + 1]
    return [dict(zip(header, row)) for row in rows[marker + 2:] if row]


def numeric(value: str | None) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if np.isfinite(result) else None


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def transform_point(lon: float, lat: float, projection: str) -> tuple[float, float]:
    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target = osr.SpatialReference(wkt=projection)
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(source, target).TransformPoint(lon, lat)[:2]


def landcover_profile(path: Path, lon: float, lat: float, radius: float) -> dict:
    dataset = gdal.Open(str(path))
    x, y = transform_point(lon, lat, dataset.GetProjection())
    transform = dataset.GetGeoTransform()
    pixel = abs(transform[1])
    col = int((x - transform[0]) / transform[1])
    row = int((y - transform[3]) / transform[5])
    pad = int(np.ceil(radius / pixel)) + 1
    xoff, yoff = max(0, col - pad), max(0, row - pad)
    xend = min(dataset.RasterXSize, col + pad + 1)
    yend = min(dataset.RasterYSize, row + pad + 1)
    values = dataset.GetRasterBand(1).ReadAsArray(xoff, yoff, xend-xoff, yend-yoff)
    cols = np.arange(xoff, xend)
    rows = np.arange(yoff, yend)
    xx = transform[0] + (cols + 0.5) * transform[1]
    yy = transform[3] + (rows + 0.5) * transform[5]
    inside = (xx[None, :] - x) ** 2 + (yy[:, None] - y) ** 2 <= radius ** 2
    nodata = dataset.GetRasterBand(1).GetNoDataValue()
    valid = inside & np.isfinite(values)
    if nodata is not None:
        valid &= values != nodata
    selected = values[valid].astype(int)
    if not selected.size:
        raise ValueError(f"No land-cover cells around {lon}, {lat}")
    return {
        f"urban_fraction_{int(radius)}m": float(np.isin(selected, [20, 21]).mean()),
        f"canopy_fraction_{int(radius)}m": float(np.isin(selected, [1, 2, 100]).mean()),
        f"grass_fraction_{int(radius)}m": float(np.isin(selected, [4, 5, 6, 7, 10]).mean()),
        f"valid_lulc_cells_{int(radius)}m": int(selected.size),
    }


def sample_raster(path: Path, lon: float, lat: float, band_number: int = 7) -> float:
    dataset = gdal.Open(str(path))
    if not dataset or band_number > dataset.RasterCount:
        raise ValueError(f"Missing raster/band: {path}, band {band_number}")
    x, y = transform_point(lon, lat, dataset.GetProjection())
    transform = dataset.GetGeoTransform()
    col = int((x - transform[0]) / transform[1])
    row = int((y - transform[3]) / transform[5])
    value = float(dataset.GetRasterBand(band_number).ReadAsArray(col, row, 1, 1)[0, 0])
    nodata = dataset.GetRasterBand(band_number).GetNoDataValue()
    if not np.isfinite(value) or (nodata is not None and value == nodata):
        raise ValueError(f"Invalid raster value at station: {path}")
    return value


def plot_geometry(axis, geometry, **kwargs) -> None:
    """Draw polygon or multipolygon OGR geometry without an extra GIS dependency."""
    if geometry is None:
        return
    geometry_name = geometry.GetGeometryName().upper()
    if geometry_name == "POLYGON":
        for ring_index in range(geometry.GetGeometryCount()):
            ring = geometry.GetGeometryRef(ring_index)
            points = np.asarray(ring.GetPoints())
            if len(points):
                axis.plot(points[:, 0], points[:, 1], **kwargs)
    elif geometry_name in {"MULTIPOLYGON", "GEOMETRYCOLLECTION"}:
        for index in range(geometry.GetGeometryCount()):
            plot_geometry(axis, geometry.GetGeometryRef(index), **kwargs)


def station_map(path: Path, inventory: list[dict], output_path: Path) -> None:
    """Map the four pilot stations against the actual London boundary."""
    dataset = ogr.Open(str(path))
    if dataset is None:
        raise FileNotFoundError(path)
    layer = dataset.GetLayer()
    projection = layer.GetSpatialRef().ExportToWkt()
    fig, axis = plt.subplots(figsize=(6.8, 5.3), constrained_layout=True)
    layer.ResetReading()
    for feature in layer:
        plot_geometry(axis, feature.GetGeometryRef(), color="#B8B8B8", linewidth=.45)

    label_offsets = {
        "00697": (8, 5), "00723": (8, -13), "00726": (8, 5), "62084": (8, 5),
    }
    for row in inventory:
        x, y = transform_point(float(row["longitude"]), float(row["latitude"]), projection)
        rural = row["role"] == "rural_candidate"
        axis.scatter(x, y, s=95 if rural else 55, marker="*" if rural else "o",
                     color="#F28E2B" if rural else "#4E79A7", edgecolor="white",
                     linewidth=.8, zorder=3)
        axis.annotate(row["station_name"].replace("LONDON, ", "").title(), (x, y),
                      xytext=label_offsets[row["station_id"]], textcoords="offset points",
                      fontsize=8, color="#333333")
    axis.scatter([], [], s=55, marker="o", color="#4E79A7", edgecolor="white",
                 label="Urban candidate")
    axis.scatter([], [], s=95, marker="*", color="#F28E2B", edgecolor="white",
                 label="Provisional rural reference")
    axis.legend(frameon=False, loc="lower left", fontsize=8)
    axis.set_title("Pilot temperature stations", loc="left", fontweight="bold")
    axis.set_aspect("equal")
    axis.set_axis_off()
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    weather_root = args.data_root/"0_source_data/uk-hourly-weather-obs"
    observation_root = weather_root/"dataset-version-202407"
    metadata_path = weather_root/"midas-open_uk-hourly-weather-obs_dv-202407_station-metadata.csv"
    lulc = args.data_root/("1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/"
                           "LULC/LCM2023_London_10m_clip2aoi_tcc24.tif")
    boundary = args.data_root/("1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/"
                               "AOIs/London_Borough_aoi.shp")
    historical = args.data_root/("1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/"
                                  "climate/worldclim_v21_historical/wc2.1_30s_tmax_07.tif")
    future_root = args.repo_root/"data/02_preprocess/GCMs/clipped"

    metadata = {row["src_id"]: row for row in table_after_data(metadata_path)
                if row.get("src_id") in STATIONS}
    if set(metadata) != set(STATIONS):
        raise ValueError(f"Missing station metadata: {set(STATIONS)-set(metadata)}")

    station_inventory, daily_records, station_summary = [], [], []
    daily_by_station: dict[str, dict[str, float]] = {}
    inputs = [metadata_path, lulc, historical, boundary]
    for station_id, role in STATIONS.items():
        matches = list(observation_root.glob(f"*_{station_id}_*_qcv-1_2021.csv"))
        if len(matches) != 1:
            raise ValueError(f"Expected one 2021 file for {station_id}; found {matches}")
        path = matches[0]
        inputs.append(path)
        meta = metadata[station_id]
        lon, lat = float(meta["station_longitude"]), float(meta["station_latitude"])
        elevation = float(meta["station_elevation"])
        profile = {}
        for radius in [500, 1000]:
            profile.update(landcover_profile(lulc, lon, lat, radius))
        inventory = dict(station_id=station_id, station_name=meta["station_name"], role=role,
                         latitude=lat, longitude=lon, elevation_m=elevation,
                         first_year=meta["first_year"], last_year=meta["last_year"], **profile)
        station_inventory.append(inventory)

        hours: dict[str, dict[datetime, float]] = defaultdict(dict)
        qc_codes = Counter()
        july_rows = 0
        duplicate_timestamps = 0
        for row in table_after_data(path):
            if not row.get("ob_time", "").startswith("2021-07"):
                continue
            july_rows += 1
            value = numeric(row.get("air_temperature"))
            qc_codes[row.get("air_temperature_q", "NA")] += 1
            if value is None:
                continue
            timestamp = datetime.fromisoformat(row["ob_time"])
            date = timestamp.date().isoformat()
            if timestamp in hours[date]:
                duplicate_timestamps += 1
            hours[date][timestamp] = value
        daily = {}
        for date, readings in sorted(hours.items()):
            values = list(readings.values())
            # Pilot completeness rule; record excluded days rather than filling gaps.
            complete = len(values) >= 18
            record = dict(station_id=station_id, station_name=meta["station_name"], date=date,
                          hourly_observations=len(values), complete_day=complete,
                          daily_max_c=max(values) if complete else "")
            daily_records.append(record)
            if complete:
                daily[date] = max(values)
        daily_by_station[station_id] = daily
        values = np.array(list(daily.values()))
        station_summary.append(dict(station_id=station_id, station_name=meta["station_name"],
            role=role, valid_july_hours=sum(len(v) for v in hours.values()), expected_july_hours=744,
            hourly_completeness=sum(len(v) for v in hours.values())/744,
            complete_days=len(values), mean_daily_max_c=float(values.mean()),
            median_daily_max_c=float(np.median(values)), p90_daily_max_c=float(np.quantile(values, .9)),
            duplicate_timestamps=duplicate_timestamps,
            air_temperature_q_counts=json.dumps(dict(qc_codes), sort_keys=True)))

    rural_id = "00726"
    rural_meta = metadata[rural_id]
    rural_elevation = float(rural_meta["station_elevation"])
    rural_adjustment = LAPSE_C_PER_M * (rural_elevation - TARGET_ELEVATION_M)
    common_dates = sorted(set.intersection(*(set(values) for values in daily_by_station.values())))
    uhi_daily = []
    urban_ids = [station for station, role in STATIONS.items() if role == "urban_candidate"]
    for date in common_dates:
        urban_values = {station: daily_by_station[station][date] for station in urban_ids}
        urban_id = max(urban_values, key=urban_values.get)
        urban_max = urban_values[urban_id]
        rural_raw = daily_by_station[rural_id][date]
        rural_adjusted = rural_raw + rural_adjustment
        uhi_daily.append(dict(date=date, warmest_urban_station_id=urban_id,
            warmest_urban_daily_max_c=urban_max, rural_station_id=rural_id,
            rural_daily_max_raw_c=rural_raw, rural_daily_max_elevation_adjusted_c=rural_adjusted,
            uhi_raw_c=urban_max-rural_raw, uhi_elevation_adjusted_c=urban_max-rural_adjusted))
    raw_uhi = np.array([row["uhi_raw_c"] for row in uhi_daily])
    adjusted_uhi = np.array([row["uhi_elevation_adjusted_c"] for row in uhi_daily])
    uhi_summary = []
    for method, values in [("raw", raw_uhi), ("elevation_adjusted", adjusted_uhi)]:
        uhi_summary.append(dict(method=method, paired_days=len(values), mean_c=float(values.mean()),
            median_c=float(np.median(values)), p75_c=float(np.quantile(values, .75)),
            p90_c=float(np.quantile(values, .9)), p95_c=float(np.quantile(values, .95)),
            minimum_c=float(values.min()), maximum_c=float(values.max())))

    kenley = next(row for row in station_summary if row["station_id"] == rural_id)
    observed_t_ref = kenley["mean_daily_max_c"] + rural_adjustment
    lon, lat = float(rural_meta["station_longitude"]), float(rural_meta["station_latitude"])
    # The historical source is the extracted July band; future rasters retain 12 bands.
    historical_at_rural = sample_raster(historical, lon, lat, 1)
    anomaly_records, parameter_records = [], []
    pilot_uhi = float(np.quantile(adjusted_uhi, .9))
    for period in PERIODS:
        for ssp in SSPS:
            scenario_trefs = []
            for model in MODELS:
                path = future_root/f"wc2.1_30s_tmax_{model}_{ssp}_{period}_London.tif"
                inputs.append(path)
                future = sample_raster(path, lon, lat, 7)
                anomaly = future-historical_at_rural
                t_ref = observed_t_ref+anomaly
                scenario_trefs.append(t_ref)
                anomaly_records.append(dict(station_id=rural_id, model=model, ssp=ssp, period=period,
                    historical_worldclim_july_tmax_c=historical_at_rural,
                    future_worldclim_july_tmax_c=future, anomaly_c=anomaly,
                    observed_rural_reference_c=observed_t_ref, future_t_ref_c=t_ref))
                parameter_records.append(dict(role="gcm_member", period=period, ssp=ssp, gcm=model,
                    t_ref_c=t_ref, uhi_max_c=pilot_uhi, relative_humidity_pct="held",
                    temperature_statistic="July mean daily maximum air temperature",
                    observation_period="July 2021 pilot", decision_status="diagnostic only"))
            parameter_records.append(dict(role="ensemble_mean_candidate", period=period, ssp=ssp,
                gcm="three-model mean", t_ref_c=float(np.mean(scenario_trefs)), uhi_max_c=pilot_uhi,
                relative_humidity_pct="held",
                temperature_statistic="July mean daily maximum air temperature",
                observation_period="July 2021 pilot", decision_status="diagnostic only"))
    parameter_records.extend([
        dict(role="existing_warm_scenario", period="nominal 2050", ssp="not coupled", gcm="none",
             t_ref_c=25, uhi_max_c=5, relative_humidity_pct=45,
             temperature_statistic="prescribed model setting", observation_period="none",
             decision_status="retained existing comparison"),
        dict(role="observed_pilot_baseline", period="2021", ssp="observed", gcm="none",
             t_ref_c=observed_t_ref, uhi_max_c=pilot_uhi, relative_humidity_pct="held",
             temperature_statistic="July mean daily maximum air temperature",
             observation_period="July 2021 pilot", decision_status="diagnostic only")])

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir/"station_inventory.csv", station_inventory)
    write_csv(args.output_dir/"station_daily_max.csv", daily_records)
    write_csv(args.output_dir/"station_summary.csv", station_summary)
    write_csv(args.output_dir/"uhi_daily.csv", uhi_daily)
    write_csv(args.output_dir/"uhi_summary.csv", uhi_summary)
    write_csv(args.output_dir/"rural_future_anomalies.csv", anomaly_records)
    write_csv(args.output_dir/"ucm_climate_parameters_pilot.csv", parameter_records)
    station_map(boundary, station_inventory, args.output_dir/"ucm_climate_station_selection.png")

    names = {row["station_id"]: row["station_name"].title() for row in station_inventory}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    x = np.arange(len(station_summary))
    means = [row["mean_daily_max_c"] for row in station_summary]
    axes[0].bar(x, means, color=["#4E79A7", "#59A14F", "#76B7B2", "#E15759"], alpha=.7,
                edgecolor="#333333", linewidth=.6)
    axes[0].set_xticks(x, [names[row["station_id"]].replace("London, ", "").replace(" ", "\n", 1)
                           for row in station_summary], fontsize=8)
    axes[0].set_ylabel("July 2021 mean daily maximum (°C)")
    axes[0].set_title("a  Available MIDAS stations", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=.5)
    candidates = [row for row in parameter_records if row["role"] == "ensemble_mean_candidate"]
    by_key = {(row["ssp"], row["period"]): row["t_ref_c"] for row in candidates}
    years = np.array([2021, 2050, 2070])
    for ssp, color, label in [("ssp370", "#F28E2B", "SSP370"),
                              ("ssp585", "#E15759", "SSP585")]:
        values = [observed_t_ref, by_key[(ssp, "2041-2060")], by_key[(ssp, "2061-2080")]]
        axes[1].plot(years, values, marker="o", markersize=4.5, color=color,
                     linewidth=1.5, label=label)
    axes[1].axhline(25, color="#777777", linestyle="--", linewidth=1,
                    label="Existing 25°C reference")
    axes[1].set_xticks(years, ["Observed\npilot", "2050", "2070"], fontsize=8)
    axes[1].set_ylabel("Reference temperature input (°C)")
    axes[1].set_title("b  Provisional UCM inputs", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=.5)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.output_dir/"ucm_climate_parameters_pilot.png", dpi=300, facecolor="white")
    plt.close(fig)

    manifest = dict(schema_version=1, generator=str(Path(__file__).resolve()),
        generator_sha256=sha256(Path(__file__)),
        created_utc=datetime.utcnow().isoformat() + "Z",
        purpose="Diagnostic pilot only; not production UCM inputs",
        assumptions=dict(rural_station="Kenley Airfield 00726",
            urban_statistic="daily maximum across Kew, St James's Park and Battersea",
            minimum_hourly_observations_per_day=18, lapse_rate_c_per_m=LAPSE_C_PER_M,
            comparison_elevation_m=TARGET_ELEVATION_M,
            uhi_choice="90th percentile of elevation-adjusted daily Tmax contrast",
            future_method="observed rural anchor plus matched WorldClim rural-point anomaly",
            future_periods=PERIODS),
        inputs=[dict(path=str(path.resolve()), sha256=sha256(path)) for path in sorted(set(inputs))],
        outputs={path.name:sha256(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()})
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(stations=len(station_inventory), common_paired_days=len(common_dates),
        observed_t_ref_c=observed_t_ref, pilot_uhi_p90_c=pilot_uhi,
        ensemble_candidates={row["ssp"]:row["t_ref_c"] for row in candidates}), indent=2))


if __name__ == "__main__":
    main()
