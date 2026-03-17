"""
tree_equity_scenario.py — optimized for large datasets
"""

import os
import sys
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from shapely.geometry import MultiPoint
from shapely import buffer as shapely_buffer  # vectorized Shapely 2.0 API

# Optional: uncomment if using Dask for out-of-core processing
# import dask_geopandas as dgpd

# ============================================================
# 1. DIRECTORY SETUP
# ============================================================

DIR_G       = Path("G:/Shared drives/Wellcome Trust Project Data")
DIR_G_IN    = DIR_G / "1_preprocess/UrbanCoolingModel/OfficialWorkingInputs"
DIR_AOI     = DIR_G_IN / "AOIs"
DIR_LULC    = DIR_G_IN / "LULC"
DIR_RAW     = Path("G:/My Drive/NatCap/projects/KCL_Welcome/london-equity-tree-scenario/Results")
DIR_FIGURES = DIR_LULC / "figures"
DIR_FIGURES.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LOAD AOI
# ============================================================

aoi = gpd.read_file(DIR_AOI / "London_Borough_aoi.shp")[["NAME", "GSS_CODE", "geometry"]]
print(f"AOI loaded: {len(aoi)} boroughs | CRS: {aoi.crs}")

# ============================================================
# 3. PLOT FUNCTION
# ============================================================

# def plot_map(buffered_gdf: gpd.GeoDataFrame, title_pre: str = "data2",
#              n_tree: int = 0, diameter: float = 10):

#     aoi_plot = aoi.copy()
#     if aoi_plot.crs != buffered_gdf.crs:
#         aoi_plot = aoi_plot.to_crs(buffered_gdf.crs)

#     fig, ax = plt.subplots(figsize=(10, 10))

#     aoi_plot.plot(ax=ax, facecolor="whitesmoke", edgecolor="black", linewidth=0.5)
#     buffered_gdf.plot(ax=ax, facecolor="darkgreen", edgecolor="none", alpha=0.6)

#     epsg_code = buffered_gdf.crs.to_epsg()
#     ax.set_title(f"{title_pre} - tree planting (n = {n_tree:,})", fontsize=13, fontweight="bold")
#     ax.annotate(f"{diameter}m Diameter (EPSG:{epsg_code})", xy=(0.5, 0.01),
#                 xycoords="axes fraction", ha="center", fontsize=10, color="grey")
#     ax.axis("off")

#     patch = mpatches.Patch(facecolor="darkgreen", alpha=0.6, label="Tree planting zones")
#     ax.legend(handles=[patch], loc="lower right", fontsize=9)

#     out_path = DIR_FIGURES / f"{title_pre}_map.png"
#     fig.savefig(out_path, dpi=600, bbox_inches="tight")
#     plt.close(fig)
#     print(f"✓ Map saved: {out_path}")


# ============================================================
# 4. FAST DATA LOADING — only read columns we need
# ============================================================

import time
t0 = time.time()

# ⚡ OPTIMIZATION 1: Read only required columns — skips loading unused attributes
# For a 3GB file this alone can halve read time
points_gdf = gpd.read_file(
    DIR_RAW / "Potential_Tree_Points_Ranked.shp",
    columns=["rank", "geometry"],   # only load what we need
    engine="pyogrio"                # ⚡ pyogrio is 5-10x faster than fiona for large files
)
print(f"Loaded {len(points_gdf):,} points in {time.time()-t0:.1f}s | CRS: {points_gdf.crs}")

# # ============================================================
# # 5. SCENARIO CONFIG
# # ============================================================

# N_TREE   = 842_100
# DIAMETER = 10
# RADIUS   = DIAMETER / 2

# # ============================================================
# # 6. FAST SELECTION + BUFFER
# # ============================================================

# t1 = time.time()

# # ⚡ OPTIMIZATION 2: Use nsmallest instead of full sort — only sorts what's needed
# pt_plant = (
#     points_gdf
#     .nsmallest(N_TREE, "rank")     # faster than sort_values().head() on large data
#     .reset_index(drop=True)
# )
# print(f"Selected {len(pt_plant):,} trees in {time.time()-t1:.1f}s")

# t2 = time.time()

# # ⚡ OPTIMIZATION 3: Shapely 2.0 vectorized buffer — operates on numpy arrays directly
# # Much faster than gdf.buffer() which loops internally on older Shapely
# buffered_geoms = shapely_buffer(
#     pt_plant.geometry.values,      # passes GeometryArray directly to C layer
#     distance=RADIUS,
#     quad_segs=8                    # ⚡ reduce quad_segs 16→8: halves vertex count
#                                    # 8 is visually indistinguishable at this scale
# )
# buffered_gdf = gpd.GeoDataFrame(
#     pt_plant.drop(columns="geometry"),
#     geometry=buffered_geoms,
#     crs=pt_plant.crs
# )
# print(f"Buffered {len(buffered_gdf):,} polygons in {time.time()-t2:.1f}s")

# # Rename reserved fid column if present
# if "fid" in buffered_gdf.columns:
#     buffered_gdf = buffered_gdf.rename(columns={"fid": "original_fid"})

# # ============================================================
# # 7. FAST WRITE
# # ============================================================

# t3 = time.time()

# out_gpkg = DIR_RAW / "tree_equity_scenario710.gpkg"

# # ⚡ OPTIMIZATION 4: pyogrio write is significantly faster than fiona for large files
# buffered_gdf.to_file(
#     out_gpkg,
#     driver="GPKG",
#     layer="tree_equity_scenario710",
#     engine="pyogrio"               # ⚡ pyogrio writer is 3-5x faster than fiona
# )
# print(f"✓ Saved GeoPackage in {time.time()-t3:.1f}s → {out_gpkg}")

# # ============================================================
# # 8. PLOT
# # ============================================================

# # plot_map(buffered_gdf, title_pre="data7", n_tree=N_TREE, diameter=DIAMETER)

# print(f"\nTotal time: {time.time()-t0:.1f}s")





# ============================================================
# 5. SCENARIO CONFIG
# ============================================================

DIAMETER = 10
RADIUS   = DIAMETER / 2

SCENARIOS = {
    # "tree_equity_scenario710":  842_100,    # 10%
    "tree_equity_scenario715":  1_263_150,  # 15%
    # "tree_equity_scenario720":  1_684_200,  # 20%
    # "tree_equity_scenario730":  2_526_300,  # 30%
}

# ============================================================
# 6. RUN ALL SCENARIOS
# ============================================================

for scenario_name, n_tree in SCENARIOS.items():

    print(f"\n====== Processing: {scenario_name} (n = {n_tree:,}) ======")

    t1 = time.time()

    # ⚡ OPTIMIZATION 2: nsmallest — only sorts what's needed
    pt_plant = (
        points_gdf
        .nsmallest(n_tree, "rank")
        .reset_index(drop=True)
    )
    print(f"Selected {len(pt_plant):,} trees in {time.time()-t1:.1f}s")

    t2 = time.time()

    # ⚡ OPTIMIZATION 3: Shapely 2.0 vectorized buffer
    buffered_geoms = shapely_buffer(
        pt_plant.geometry.values,
        distance=RADIUS,
        quad_segs=8
    )
    buffered_gdf = gpd.GeoDataFrame(
        pt_plant.drop(columns="geometry"),
        geometry=buffered_geoms,
        crs=pt_plant.crs
    )
    print(f"Buffered {len(buffered_gdf):,} polygons in {time.time()-t2:.1f}s")

    # Rename reserved fid column if present
    if "fid" in buffered_gdf.columns:
        buffered_gdf = buffered_gdf.rename(columns={"fid": "original_fid"})

    # ============================================================
    # 7. FAST WRITE
    # ============================================================

    t3 = time.time()

    out_gpkg = DIR_LULC / f"{scenario_name}.gpkg"

    # ⚡ OPTIMIZATION 4: pyogrio writer
    buffered_gdf.to_file(
        out_gpkg,
        driver="GPKG",
        layer=scenario_name,
        engine="pyogrio"
    )
    print(f"✓ Saved in {time.time()-t3:.1f}s → {out_gpkg}")

    # ============================================================
    # 8. PLOT
    # ============================================================

    # plot_map(buffered_gdf, title_pre=scenario_name, n_tree=n_tree, diameter=DIAMETER)

print(f"\nAll scenarios complete. Total time: {time.time()-t0:.1f}s")