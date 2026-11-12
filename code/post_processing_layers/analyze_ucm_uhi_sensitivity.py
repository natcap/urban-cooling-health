"""Compare temperature-only UCM results across UHI_max = 2, 3 and 5 C.

The analysis holds reference temperature (25 C), land-cover rasters and all
other InVEST inputs fixed.  UHI=2 and UHI=3 are diagnostic runs; UHI=5 reuses
the existing production temperature rasters.  No health, energy, WBGT or work-
capacity outputs are calculated here.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

from valuation_accounting import london_grid, masked_means, sha256


SCENARIOS = [
    "baseline", "allbuilt", "treerisk", "treeopp", "green10", "green20",
    "green30", "target10", "target20", "target30",
]
LABELS = {
    "baseline": "Baseline", "allbuilt": "AllBuilt", "treerisk": "TreeRisk",
    "treeopp": "TreeOpp", "green10": "Green10", "green20": "Green20",
    "green30": "Green30", "target10": "Target10", "target20": "Target20",
    "target30": "Target30",
}
COLORS = {
    "allbuilt": "#d95f0e", "treerisk": "#fec44f", "treeopp": "#d9f0a3",
    "green10": "#78c679", "green20": "#238443", "green30": "#004529",
    "target10": "#8c96c6", "target20": "#8856a7", "target30": "#810f7c",
}
CORE_ARGUMENTS = [
    "aoi_vector_path", "avg_rel_humidity", "biophysical_table_path",
    "cc_method", "cc_weight_albedo", "cc_weight_eti", "cc_weight_shade",
    "green_area_cooling_distance", "lulc_raster_path", "ref_eto_raster_path",
    "t_air_average_radius", "t_ref",
]


def raster_path(diagnostic_root, production_root, scenario, uhi):
    root = diagnostic_root / f"uhi{uhi}" if uhi in (2, 3) else production_root
    candidates = list((root / scenario).glob(
        f"T_air_london_{scenario}_25deg_{uhi}uhi_45hum_revised_equal_area.tif"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one raster for {scenario}, UHI={uhi}: {candidates}")
    return candidates[0]


def manifest_path(diagnostic_root, production_root, scenario, uhi):
    root = diagnostic_root / f"uhi{uhi}" if uhi in (2, 3) else production_root
    candidates = list((root / scenario).glob(
        f"run_manifest_london_{scenario}_25deg_{uhi}uhi_45hum_revised_equal_area.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one manifest for {scenario}, UHI={uhi}: {candidates}")
    return candidates[0]


def write_csv(path, rows):
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def rank(values):
    order = np.argsort(-np.asarray(values))
    out = np.empty(len(order), dtype=int)
    out[order] = np.arange(1, len(order) + 1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    diagnostic = (args.data_root /
        "2_postprocess_intermediate/UCM_official_runs/diagnostics/"
        "uhi_sensitivity_25c_20261111_v1")
    production = (args.data_root /
        "2_postprocess_intermediate/UCM_official_runs/production/"
        "equal_area_2026-09/03_energy_work_capacity")
    boundary = (args.data_root /
        "1_preprocess/UrbanCoolingModel/OfficialWorkingInputs/AOIs/"
        "London_Borough_aoi.shp")
    rasters = {(uhi, scenario): raster_path(diagnostic, production, scenario, uhi)
               for uhi in (2, 3, 5) for scenario in SCENARIOS}
    manifests = {(uhi, scenario): manifest_path(diagnostic, production, scenario, uhi)
                 for uhi in (2, 3, 5) for scenario in SCENARIOS}

    # Verify that UHI_max is the only climate/model argument varied. Production
    # UHI=5 includes valuation inputs, so compare the temperature-model core only.
    for scenario in SCENARIOS:
        records = [json.loads(manifests[uhi, scenario].read_text()) for uhi in (2, 3, 5)]
        for uhi, record in zip((2, 3, 5), records):
            assert record["software"]["invest"] == "3.20.2"
            assert float(record["model_arguments"]["uhi_max"]) == uhi
        for key in CORE_ARGUMENTS:
            values = [record["model_arguments"][key] for record in records]
            if not values[0] == values[1] == values[2]:
                raise ValueError(f"Core argument differs for {scenario}, {key}: {values}")

    borough_rows, zones, counts = london_grid(boundary, list(rasters.values()))
    city_rows, borough_output, spatial_rows = [], [], []
    means = {}
    for uhi in (2, 3, 5):
        for scenario in SCENARIOS:
            city, borough = masked_means(rasters[uhi, scenario], zones, counts)
            means[uhi, scenario] = (city, borough)
            values = gdal.Open(str(rasters[uhi, scenario])).ReadAsArray()[zones > 0]
            spatial_rows.append({
                "uhi_max_c": uhi, "scenario": scenario,
                "p10_temperature_c": float(np.quantile(values, 0.10)),
                "median_temperature_c": float(np.median(values)),
                "p90_temperature_c": float(np.quantile(values, 0.90)),
                "minimum_temperature_c": float(np.min(values)),
                "maximum_temperature_c": float(np.max(values)),
            })
        baseline_city, baseline_borough = means[uhi, "baseline"]
        cooling = [baseline_city - means[uhi, scenario][0] for scenario in SCENARIOS[1:]]
        ranks = rank(cooling)
        for scenario, cooling_c, scenario_rank in zip(SCENARIOS[1:], cooling, ranks):
            scenario_city, scenario_borough = means[uhi, scenario]
            city_rows.append({
                "uhi_max_c": uhi, "scenario": scenario,
                "scenario_label": LABELS[scenario],
                "area_weighted_temperature_c": scenario_city,
                "area_weighted_cooling_vs_baseline_c": cooling_c,
                "cooling_rank": int(scenario_rank),
            })
            for (code, name, _), base_value, scenario_value in zip(
                    borough_rows, baseline_borough, scenario_borough):
                borough_output.append({
                    "uhi_max_c": uhi, "scenario": scenario,
                    "borough_code": code, "borough": name,
                    "borough_temperature_c": float(scenario_value),
                    "borough_cooling_vs_baseline_c": float(base_value - scenario_value),
                })

    # The UCM temperature equation is linear in UHI_max, so the sensitivity
    # checks whether absolute effects scale without changing scenario ordering.
    rank_rows = []
    for left, right in ((2, 3), (2, 5), (3, 5)):
        a = np.array([next(r["cooling_rank"] for r in city_rows
                           if r["uhi_max_c"] == left and r["scenario"] == s)
                      for s in SCENARIOS[1:]])
        b = np.array([next(r["cooling_rank"] for r in city_rows
                           if r["uhi_max_c"] == right and r["scenario"] == s)
                      for s in SCENARIOS[1:]])
        rank_rows.append({
            "uhi_pair": f"{left}_vs_{right}",
            "spearman_rank_correlation": float(np.corrcoef(a, b)[0, 1]),
            "identical_rank_order": bool(np.array_equal(a, b)),
        })

    pair_rows = []
    for uhi in (2, 3, 5):
        for tier in (10, 20, 30):
            green = next(r for r in city_rows if r["uhi_max_c"] == uhi
                         and r["scenario"] == f"green{tier}")
            target = next(r for r in city_rows if r["uhi_max_c"] == uhi
                          and r["scenario"] == f"target{tier}")
            borough_green = np.array([r["borough_cooling_vs_baseline_c"]
                                      for r in borough_output
                                      if r["uhi_max_c"] == uhi and r["scenario"] == f"green{tier}"])
            borough_target = np.array([r["borough_cooling_vs_baseline_c"]
                                       for r in borough_output
                                       if r["uhi_max_c"] == uhi and r["scenario"] == f"target{tier}"])
            pair_rows.append({
                "uhi_max_c": uhi, "tier_pct": tier,
                "target_minus_green_citywide_c":
                    target["area_weighted_cooling_vs_baseline_c"] -
                    green["area_weighted_cooling_vs_baseline_c"],
                "boroughs_where_target_cools_more_n": int(np.sum(borough_target > borough_green)),
                "boroughs_where_green_cools_more_n": int(np.sum(borough_green > borough_target)),
                "boroughs_tied_n": int(np.sum(np.isclose(borough_green, borough_target, atol=1e-8))),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "uhi_sensitivity_citywide.csv", city_rows)
    write_csv(args.output_dir / "uhi_sensitivity_borough.csv", borough_output)
    write_csv(args.output_dir / "uhi_sensitivity_spatial_distribution.csv", spatial_rows)
    write_csv(args.output_dir / "uhi_sensitivity_rank_stability.csv", rank_rows)
    write_csv(args.output_dir / "uhi_sensitivity_green_target_pairs.csv", pair_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for scenario in SCENARIOS[1:]:
        rows = [r for r in city_rows if r["scenario"] == scenario]
        axes[0].plot([r["uhi_max_c"] for r in rows],
                     [r["area_weighted_cooling_vs_baseline_c"] for r in rows],
                     marker="o", lw=1.5, ms=4, color=COLORS[scenario], label=LABELS[scenario])
    axes[0].axhline(0, color="#777777", lw=0.7)
    axes[0].set(xlabel="Maximum UHI magnitude (°C)",
                ylabel="London-wide cooling versus baseline (°C)",
                title="a  Land-use effects across UHI settings", xticks=[2, 3, 5])
    axes[0].legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    for tier, color in zip((10, 20, 30), ("#8c96c6", "#8856a7", "#810f7c")):
        rows = [r for r in pair_rows if r["tier_pct"] == tier]
        axes[1].plot([r["uhi_max_c"] for r in rows],
                     [r["target_minus_green_citywide_c"] for r in rows],
                     marker="o", lw=1.8, ms=4, color=color, label=f"{tier}% tier")
    axes[1].axhline(0, color="#777777", lw=0.7)
    axes[1].set(xlabel="Maximum UHI magnitude (°C)",
                ylabel="Target minus Green cooling (°C)",
                title="b  Green–Target contrast", xticks=[2, 3, 5])
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e5e5e5", lw=0.6)
    figure_path = args.repo_root / "figures/climate_parameters/ucm_uhi_sensitivity_25c_20261111_v1.png"
    fig.savefig(figure_path, dpi=320, facecolor="white")
    plt.close(fig)

    manifest = {
        "analysis_scope": "Temperature only; health, energy, WBGT and work capacity not run",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(__file__),
        "reference_temperature_c": 25,
        "uhi_max_values_c": [2, 3, 5],
        "common_london_cells": int(counts.sum()),
        "boundary": str(boundary),
        "raster_sources": {f"uhi{u}_{s}": {"path": str(p), "sha256": sha256(p)}
                           for (u, s), p in rasters.items()},
        "run_manifests": {f"uhi{u}_{s}": {"path": str(p), "sha256": sha256(p)}
                          for (u, s), p in manifests.items()},
        "core_argument_consistency": {
            "status": "passed",
            "compared_arguments": CORE_ARGUMENTS,
            "intentional_difference": "uhi_max only; valuation flags in production UHI=5 are outside the temperature-model core",
            "invest_version": "3.20.2",
        },
        "figure": str(figure_path),
    }
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "rank_stability": rank_rows,
        "green_target_pairs": pair_rows,
        "figure": str(figure_path),
    }, indent=2))


if __name__ == "__main__":
    main()
