



from pathlib import Path
import geopandas as gpd

# -------------------------
# 1. Read data
# -------------------------
path_main = Path(r"G:\Shared drives\Wellcome Trust Project Data\1_preprocess\UrbanCoolingModel\OfficialWorkingInputs")
path_counties = path_main / "AOIs" / "London_Borough_aoi.shp"
path_buildings = path_main / "energy_buildings" /  "bld_with_attr_compact_ucm2.gpkg"

counties = gpd.read_file(path_counties)
buildings = gpd.read_file(path_buildings)

# -------------------------
# 2. Reproject to equal-area CRS (meters)
# -------------------------
target_crs = "EPSG:27700"  # British National Grid

counties = counties.to_crs(target_crs)
buildings = buildings.to_crs(target_crs)

# (Optional) Fix invalid geometries if needed
counties["geometry"] = counties.buffer(0)
buildings["geometry"] = buildings.buffer(0)

# -------------------------
# 3. Intersect buildings with counties
#    This splits buildings that cross county boundaries
# -------------------------
# Keep only the ID + name from counties to avoid column clutter
counties_min = counties[["GSS_CODE", "NAME", "geometry"]]

intersect = gpd.overlay(buildings, counties_min, how="intersection")

# -------------------------
# 4. Compute area per intersected piece
# -------------------------
intersect["area_m2"] = intersect.geometry.area  # units: m²

# -------------------------
# 5. Group by county + building type
# -------------------------
# Replace 'bldg_type' with your actual building-type field
group_cols = ["GSS_CODE", "NAME", "type"]

zonal_stats = (
    intersect
    .groupby(group_cols, as_index=False)["area_m2"]
    .sum()
    .rename(columns={"area_m2": "bldg_area_m2"})
)

# Optionally convert to km² or hectares
zonal_stats["bldg_area_km2"] = zonal_stats["bldg_area_m2"] / 1e6
zonal_stats["bldg_area_ha"]  = zonal_stats["bldg_area_m2"] / 1e4

# -------------------------
# 6. Save to CSV / GeoPackage etc.
# -------------------------
path_building_stats = path_main / "energy_buildings" /  "bld_with_attr_compact_ucm2_stats.csv"

zonal_stats.to_csv(path_building_stats, index=False)