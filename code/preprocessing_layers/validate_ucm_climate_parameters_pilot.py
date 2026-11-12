"""Validate the baseline-only UCM climate-parameter pilot.

Compares the observed-period model at four MIDAS stations and summarizes future
candidate London means. This does not calculate energy, health or work capacity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, osr


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def transform_point(lon: float, lat: float, projection: str) -> tuple[float, float]:
    source = osr.SpatialReference(); source.ImportFromEPSG(4326)
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target = osr.SpatialReference(wkt=projection)
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(source, target).TransformPoint(lon, lat)[:2]


def sample(path: Path, lon: float, lat: float) -> float:
    dataset = gdal.Open(str(path)); x, y = transform_point(lon, lat, dataset.GetProjection())
    transform = dataset.GetGeoTransform()
    col = int((x-transform[0])/transform[1]); row = int((y-transform[3])/transform[5])
    return float(dataset.GetRasterBand(1).ReadAsArray(col, row, 1, 1)[0, 0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--parameter-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root/"code/post_processing_layers"))
    from valuation_accounting import london_grid, masked_means

    inventory = read_csv(args.parameter_dir/"station_inventory.csv")
    station_summary = {r["station_id"]:r for r in read_csv(args.parameter_dir/"station_summary.csv")}
    parameters = read_csv(args.parameter_dir/"ucm_climate_parameters_pilot.csv")
    expected = {
        round(float(r["t_ref_c"]), 4):(r["role"], r["ssp"], r["period"])
        for r in parameters if r["role"] in {"observed_pilot_baseline", "ensemble_mean_candidate"}
    }
    rasters, manifests = {}, {}
    for manifest_path in (args.run_root/"baseline").glob("run_manifest_*.json"):
        manifest = json.loads(manifest_path.read_text())
        t_ref = float(manifest["model_arguments"]["t_ref"])
        key = min(expected, key=lambda value:abs(value-t_ref))
        if abs(key-t_ref) > 1e-3:
            continue
        raster = manifest_path.with_name(manifest_path.name.replace("run_manifest_", "T_air_")).with_suffix(".tif")
        if not raster.exists():
            raise FileNotFoundError(raster)
        if expected[key][0] == "ensemble_mean_candidate":
            year = "2050" if expected[key][2] == "2041-2060" else "2070"
            label = f"{expected[key][1]}_{year}"
        else:
            label = "observed_2021"
        rasters[label] = raster; manifests[label] = manifest_path
    expected_labels = {"observed_2021", "ssp370_2050", "ssp585_2050",
                       "ssp370_2070", "ssp585_2070"}
    if set(rasters) != expected_labels:
        raise ValueError(f"Missing pilot runs: {rasters}")

    rows, zones, counts = london_grid(args.boundary, list(rasters.values()))
    london = {label:masked_means(path, zones, counts)[0] for label,path in rasters.items()}
    station_checks = []
    for station in inventory:
        station_id = station["station_id"]
        observed = float(station_summary[station_id]["mean_daily_max_c"])
        elevation = float(station["elevation_m"])
        adjusted = observed + 0.0065*(elevation-5)
        modelled = sample(rasters["observed_2021"], float(station["longitude"]), float(station["latitude"]))
        station_checks.append(dict(station_id=station_id, station_name=station["station_name"],
            role=station["role"], observed_raw_c=observed,
            observed_adjusted_to_5m_c=adjusted, modelled_c=modelled,
            residual_c=modelled-adjusted))
    residuals = np.array([r["residual_c"] for r in station_checks])
    validation = [dict(stations=len(residuals), bias_c=float(residuals.mean()),
        mae_c=float(np.abs(residuals).mean()), rmse_c=float(np.sqrt(np.mean(residuals**2))),
        correlation=float(np.corrcoef(
            [r["observed_adjusted_to_5m_c"] for r in station_checks],
            [r["modelled_c"] for r in station_checks])[0,1]))]
    observed_adjusted = np.array([r["observed_adjusted_to_5m_c"] for r in station_checks])
    modelled_h = (np.array([r["modelled_c"] for r in station_checks])-22.18862903225807)/2.1275
    design = np.column_stack([np.ones(len(modelled_h)), modelled_h])
    fitted_t_ref, fitted_uhi = np.linalg.lstsq(design, observed_adjusted, rcond=None)[0]
    calibration = [dict(method="unconstrained_four_station_diagnostic",
        fitted_t_ref_c=float(fitted_t_ref), fitted_uhi_c=float(fitted_uhi),
        physically_admissible_uhi=bool(fitted_uhi >= 0),
        interpretation="Negative fitted UHI indicates station/model spatial ordering mismatch")]
    targets = {"observed_2021":21.85052135,
               "ssp370_2050":25.71846010, "ssp585_2050":26.06487723,
               "ssp370_2070":26.64127868, "ssp585_2070":27.31449131}
    london_checks = [dict(configuration=label, ucm_london_area_weighted_mean_c=value,
        figure1_equal_borough_mean_c=targets[label], difference_c=value-targets[label])
        for label,value in london.items()]
    write_csv(args.parameter_dir/"station_model_validation.csv", station_checks)
    write_csv(args.parameter_dir/"validation_metrics.csv", validation)
    write_csv(args.parameter_dir/"station_calibration_diagnostic.csv", calibration)
    write_csv(args.parameter_dir/"london_mean_comparison.csv", london_checks)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), constrained_layout=True)
    observed = np.array([r["observed_adjusted_to_5m_c"] for r in station_checks])
    modelled = np.array([r["modelled_c"] for r in station_checks])
    axes[0].scatter(observed, modelled, s=45, color="#4E79A7", edgecolor="white", linewidth=.7)
    low, high = min(observed.min(),modelled.min())-.3, max(observed.max(),modelled.max())+.3
    axes[0].plot([low,high],[low,high],color="#777777",linewidth=.8,linestyle="--")
    label_offsets = {"00726":(5,5), "00723":(5,5), "62084":(5,-13), "00697":(5,7)}
    for r in station_checks:
        axes[0].annotate(r["station_name"].replace("LONDON, ", "").title(),
            (r["observed_adjusted_to_5m_c"],r["modelled_c"]),
            xytext=label_offsets[r["station_id"]],
            textcoords="offset points",fontsize=6.5)
    axes[0].set(xlim=(low,high),ylim=(low,high),xlabel="Observed July 2021 Tmax (°C)\nadjusted to 5 m",
                ylabel="Modelled temperature (°C)")
    axes[0].set_title("a  Station check",loc="left",fontweight="bold")
    order=["observed_2021","ssp370_2050","ssp585_2050","ssp370_2070","ssp585_2070"]
    labels=["Observed\n2021","SSP370\n2050","SSP585\n2050","SSP370\n2070","SSP585\n2070"]
    ucm=[london[k] for k in order]
    fig1=[targets[k] for k in order]
    x=np.arange(len(order)); width=.36
    axes[1].bar(x-width/2,fig1,width,label="Figure 1",color="#B9B9B9",alpha=.8,edgecolor="#444")
    axes[1].bar(x+width/2,ucm,width,label="Pilot UCM",color="#F28E2B",alpha=.7,edgecolor="#444")
    axes[1].set_xticks(x,labels); axes[1].set_ylabel("London temperature (°C)")
    axes[1].set_title("b  London means",loc="left",fontweight="bold")
    axes[1].legend(frameon=False,fontsize=8)
    for axis in axes:
        axis.spines[["top","right"]].set_visible(False); axis.grid(axis="y",color="#DDDDDD",linewidth=.5)
    figure = args.parameter_dir/"ucm_climate_parameter_validation.png"
    fig.savefig(figure,dpi=300,facecolor="white"); plt.close(fig)
    output_paths = [args.parameter_dir/name for name in ["station_model_validation.csv",
        "validation_metrics.csv","station_calibration_diagnostic.csv",
        "london_mean_comparison.csv","ucm_climate_parameter_validation.png"]]
    manifest = dict(schema_version=1, purpose="Baseline-only temperature validation pilot",
        generator=str(Path(__file__).resolve()), generator_sha256=sha256(Path(__file__)),
        run_manifests={k:dict(path=str(v),sha256=sha256(v)) for k,v in manifests.items()},
        common_london_cells=int(counts.sum()), outputs={p.name:sha256(p) for p in output_paths},
        warning="Four stations; Kenley requires elevation adjustment; diagnostic only")
    (args.parameter_dir/"validation_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(validation=validation[0],calibration=calibration[0],
        london=london_checks),indent=2))


if __name__ == "__main__":
    main()
