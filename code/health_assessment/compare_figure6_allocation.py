"""Isolate area versus population allocation on the current Green30 grid.

Run with the urban-cooling-health Python environment and --data-root.
V0-style area allocation is reconstructed, not the archived V0 raster:
equal-area grid cells receive equal deaths within each borough, preserving
observed totals and avoiding historical resampling losses as a confounder.
"""
import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("figures/figure6/allocation_normalization_comparison"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory to preserve previous comparisons")
    protected = {str(p): digest(p) for p in Path("figures").rglob("*") if p.is_file()}
    run = args.data_root / "2_postprocess_intermediate/UCM_official_runs/health_v3_revised_equal_area_population_weighted_2021_2026-09-10/green30_25c"
    manifest = json.loads((run / "run_manifest.json").read_text())
    mortality = Path(manifest["inputs"]["mortality_all_cause"]["path"])
    am = json.loads((mortality.parent / "manifest.json").read_text())
    borough_path = Path(am["inputs"]["boroughs"]["path"])
    mortality_csv = Path(am["inputs"]["mortality_csv"]["path"])
    source_paths = [run / "run_manifest.json", mortality.parent / "manifest.json", mortality_csv,
                    *borough_path.parent.glob(borough_path.stem + ".*"),
                    Path("data/derived/fig7_vulnerability_lsoa11_fig7_revised_equal_area.gpkg")]
    arrays = {}
    grid = None
    for key in ("t_baseline", "t_scenario", "mortality_all_cause", "population_2021"):
        path = Path(manifest["inputs"][key]["path"])
        source_paths.append(path)
        with rasterio.open(path) as ds:
            current = (ds.crs, ds.transform, ds.width, ds.height)
            if grid is not None and current != grid:
                raise ValueError(f"Grid mismatch: {key}")
            grid = current
            arrays[key] = ds.read(1, masked=True).astype("float64").filled(np.nan)
    crs, transform, width, height = grid
    boroughs = gpd.read_file(borough_path).to_crs(crs)
    assert len(boroughs) == 33 and boroughs.GSS_CODE.is_unique
    zones = rasterize(zip(boroughs.geometry, range(1, 34)), out_shape=(height, width), transform=transform, dtype="int16")
    counts = np.bincount(zones.ravel(), minlength=34)
    deaths = pd.read_csv(mortality_csv).query("year == 2021 and cause == 'all_cause'")
    assert deaths.borough_id.is_unique
    totals = np.r_[0., deaths.set_index("borough_id").loc[boroughs.GSS_CODE, "deaths"].to_numpy()]
    rates = np.zeros(34)
    rates[1:] = totals[1:] / counts[1:]
    area = rates[zones]
    population = np.nan_to_num(arrays["mortality_all_cause"], nan=0.)
    allocated = np.bincount(zones.ravel(), weights=population.ravel(), minlength=34)
    assert np.allclose(allocated[1:], totals[1:], rtol=0, atol=1e-3)
    assert np.allclose(np.bincount(zones.ravel(), weights=area.ravel())[1:], totals[1:], rtol=0, atol=1e-6)
    valid = (zones > 0) & np.isfinite(arrays["t_baseline"]) & np.isfinite(arrays["t_scenario"])
    rr = next(c["rr_1c"] for c in manifest["causes"] if c["name"] == "all_cause")
    # Negative excess mortality becomes positive deaths averted.
    benefit = np.zeros(zones.shape)
    benefit[valid] = np.expm1(-np.log(rr) * (arrays["t_scenario"][valid] - arrays["t_baseline"][valid]))
    v = gpd.read_file("data/derived/fig7_vulnerability_lsoa11_fig7_revised_equal_area.gpkg").to_crs(crs)
    lsoa = rasterize(zip(v.geometry, v.id.astype(int)), out_shape=(height,width), transform=transform, dtype="int32")
    rows, cols = np.nonzero(valid & (lsoa == 0))
    max_distance = 0.
    if len(rows):
        xx, yy = rasterio.transform.xy(transform, rows, cols)
        pts = gpd.GeoDataFrame({"point": np.arange(len(rows))},geometry=gpd.points_from_xy(xx,yy),crs=crs)
        joined = gpd.sjoin_nearest(pts,v[["id","geometry"]],distance_col="distance")
        joined = joined.sort_values(["point","distance","id"]).drop_duplicates("point").sort_values("point")
        assert not joined.id.isna().any()
        lsoa[rows,cols] = joined.id.astype(int)
        max_distance = float(joined["distance"].max())
    result = v[["id","LSOA11CD"]].copy()
    summaries = []
    for name, allocation in (("area",area),("population",population)):
        values = allocation * benefit
        sums = np.bincount(lsoa[valid],weights=values[valid],minlength=int(v.id.max())+1)
        result[name] = sums[v.id.astype(int)]
        assert np.isclose(result[name].sum(),values[valid].sum(),rtol=0,atol=1e-7)
        zero_pop = ~np.isfinite(arrays["population_2021"]) | (arrays["population_2021"]<=0)
        summaries.append({"allocation":name,"baseline_deaths":float(allocation.sum()),
          "deaths_with_temperature":float(allocation[valid].sum()),"deaths_without_temperature":float(allocation[~valid].sum()),
          "baseline_deaths_on_zero_population_cells":float(allocation[zero_pop].sum()),
          "deaths_averted":float(values.sum())})
    reference = pd.read_csv("data/derived/health_lsoa_fig7_revised_equal_area_population_weighted_2021.csv").query(
        "lc_scenario == 'Green30' and indicator == 'All_cause' and temperature_setting_c == 25")
    check = result.merge(reference[["LSOA11CD","deaths_averted"]],on="LSOA11CD",validate="one_to_one")
    discrepancy = float(np.max(np.abs(check.population-check.deaths_averted)))
    assert len(check)==4835 and discrepancy < 1e-6, discrepancy
    args.output.mkdir(parents=True)
    result.to_csv(args.output / "allocation_lsoa.csv",index=False)
    pd.DataFrame(summaries).to_csv(args.output / "allocation_city_summary.csv",index=False)
    pd.DataFrame({"path":[str(p) for p in source_paths],"sha256":[digest(p) for p in source_paths]}).to_csv(args.output/"input_checksums.csv",index=False)
    (args.output/"allocation_qa.json").write_text(json.dumps({"reference_max_lsoa_difference":discrepancy,
      "edge_cells_assigned":len(rows),"edge_max_distance_m":max_distance,"rr_1c":rr,
      "definition":"V0-style uniform area allocation reconstructed on V1 grid; same borough totals, Green30 cooling, RR and aggregation. Not exact historical V0 output.",
      "temperature_coverage":"Area allocation includes zero-population borough cells; neither method is renormalized after temperature masking."},indent=2))
    assert all(digest(p)==h for p,h in protected.items())
    (args.output/"protected_figures.json").write_text(json.dumps(protected,indent=2))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    print("Maximum LSOA discrepancy from V1:",discrepancy,flush=True)


if __name__ == "__main__":
    main()
