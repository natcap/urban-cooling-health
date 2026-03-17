import geopandas as gpd
import os
import sys
import numpy as np
from pathlib import Path

# --- 1. SETUP PATHS & IMPORTS ---
# Add .../code (parent of lc_scenarios) to Python path
CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

# Import your custom function
try:
    from function_clip_raster_to_aoi import clip_raster_to_aoi
except ImportError:
    print(f"Error: Could not find 'function_clip_raster_to_aoi.py' in {CODE_DIR}")
    sys.exit(1)

# Paths
wd_shp = r'G:\Shared drives\Wellcome Trust Project Data\1_preprocess\UrbanCoolingModel\OfficialWorkingInputs\AOIs'
aoi_shapefile = os.path.join(wd_shp, "London_Borough_aoi.shp") 
BASE_DIR = r"E:/London/jlg_tree_planting"

# Ensure output directory exists
os.makedirs(BASE_DIR, exist_ok=True)

lc_path = r"G:\Shared drives\Wellcome Trust Project Data\1_preprocess\UrbanCoolingModel\EP_preliminary_tests\clipped_lulc\UKECH\LCM2023_London_10m_clip2aoi_tcc24.tif"

lc_code_new_tcc = 100 # New code for Tree canopy in the reclassified data
tree_cover_code = 1  # UK data: Tree cover
shrubland_code = 2   # UK data: Shrubland
pavement_code = 20    # UK data: Urban (pavement)
pavement_name = 'Urban'

# list of source codes you want to convert to Built-up
codes_to_built = [1, 2, 4, 5, 6, 7, 9, 10, 100]  # add any others

# Define labels
land_cover_labels = {
    1: 'Deciduous woodland',
    2: 'Coniferous woodland',
    3: 'Arable',
    4: 'Improved Grassland',
    5: 'Neutral Grassland',
    6: 'Calcareous Grassland',
    7: 'Acid grassland',
    8: 'Fen, Marsh, and Swamp',
    9: 'Heather',
    10: 'Heather grassland',
    11: 'Bog',
    12: 'Inland Rock',
    13: 'Saltwater',
    14: 'Freshwater',
    15: 'Supralittoral Rock',
    16: 'Supralittoral Sediment',
    17: 'Littoral Rock',
    18: 'Littoral Sediment',
    19: 'Saltmarsh',
    20: 'Urban',
    21: 'Suburban',
    100: 'Tree canopy'
}



# --- 2. PREPARE AOI (CAMDEN) ---

# Read and Filter
# Note: Ensure "NAME" and "Camden" match exactly (case-sensitive) in your shapefile
full_aoi = gpd.read_file(aoi_shapefile)
aoi = full_aoi.loc[:, ["NAME", "GSS_CODE", "geometry"]].query("NAME == 'Camden'")

# SAFETY CHECK: Did we actually find Camden?
if aoi.empty:
    raise ValueError("Error: Filter returned empty AOI. Check spelling of 'Camden' or column 'NAME'.")

# SAFETY CHECK 2: Does it have a CRS?
# If missing, we FORCE it to British National Grid (EPSG:27700)
if aoi.crs is None:
    print("Warning: Input shapefile is missing CRS. Setting to EPSG:27700 (British National Grid).")
    aoi.set_crs(epsg=27700, allow_override=True, inplace=True)
else:
    # If it exists but might be different (e.g., Lat/Lon), ensure it matches the projected system
    # This helps avoid pyproj errors if the definition is slightly off
    aoi = aoi.to_crs(epsg=27700)

# Save the filtered AOI (Camden only)
aoi_filtered_shp = os.path.join(BASE_DIR, "aoi_Camden.gpkg")
aoi.to_file(aoi_filtered_shp, driver="GPKG")
print(f"Filtered AOI saved to: {aoi_filtered_shp} with CRS: {aoi.crs}")




import rasterio
import geopandas as gpd

print("\n--- DEBUG CRS CHECK ---")
# 1. Check Raster
try:
    with rasterio.open(lc_path) as src:
        print(f"\n Raster CRS: {src.crs}")
        if src.crs is None:
            print(">>> PROBLEM: Raster has NO CRS!")
except Exception as e:
    print(f"Raster Read Error: {e}")

# 2. Check Vector (AOI)
try:
    gdf_test = gpd.read_file(aoi_shapefile)
    print(f"\n Original Vector CRS: {gdf_test.crs}")
    if gdf_test.crs is None:
        print(">>> PROBLEM: Vector has NO CRS!")
except Exception as e:
    print(f"Vector Read Error: {e}")
print("-----------------------")


# -----------------------------
import os
import glob
import sys
import rasterio

# --- STEP 1: FIND PROJ.DB ---
# We look in common locations based on your previous error paths
possible_paths = [
    # Check pyproj's folder (most likely)
    r"C:\Users\pc\AppData\Roaming\Python\Python311\site-packages\pyproj\proj_dir\share",
    # Check rasterio's folder
    r"C:\Users\pc\AppData\Roaming\Python\Python311\site-packages\rasterio\proj_data",
    # Check generic Conda/Python paths
    os.path.join(sys.prefix, "share", "proj"),
    os.path.join(sys.prefix, "Library", "share", "proj"),
]

proj_lib_path = None

for path in possible_paths:
    if os.path.exists(os.path.join(path, "proj.db")):
        proj_lib_path = path
        break

# Fallback: If not found, try to ask pyproj directly (if installed)
if proj_lib_path is None:
    try:
        import pyproj
        path = pyproj.datadir.get_data_dir()
        if os.path.exists(os.path.join(path, "proj.db")):
            proj_lib_path = path
    except ImportError:
        pass

if proj_lib_path:
    print(f"FOUND proj.db at: {proj_lib_path}")
    print("Setting PROJ_LIB environment variable...")
    os.environ['PROJ_LIB'] = proj_lib_path
else:
    print("CRITICAL WARNING: Could not find proj.db automatically.")
    print("You may need to search your C: drive for 'proj.db' and set os.environ['PROJ_LIB'] manually.")



# --- 3. CLIP RASTER TO AOI ---

lc_input = lc_path
lc_clipped_tif = os.path.basename(lc_path).replace(".tif", "_clip2aoi_Camden.tif") # Updated name to reflect Camden
lc_clipped_path = os.path.join(BASE_DIR, lc_clipped_tif)

try:
    # Open in 'r+' mode (read/write)
    with rasterio.open(lc_path, 'r+') as src:
        print(f"Current CRS type: {src.crs.wkt[:10]}...") # Likely LOCAL_CS
        
        # Force the correct EPSG code
        print("Overwriting CRS to EPSG:27700 (British National Grid)...")
        src.crs = rasterio.crs.CRS.from_epsg(27700)
        
    print("Success! The raster CRS has been fixed.")

except Exception as e:
    print(f"Error: {e}")



print(f"Clipping raster to {aoi_filtered_shp}...")

# *** THE FIX IS HERE ***
# We pass 'aoi_filtered_shp' (the path to the Camden file) 
# instead of 'aoi_shapefile' (the path to the whole London file).

filled_arr, filled_transform, filled_profile = clip_raster_to_aoi(
    raster_path=lc_input,
    aoi=aoi_filtered_shp,       # <--- CHANGED THIS LINE
    out_path=lc_clipped_path,
    replace_nodata_with=0,      
    keep_nodata_tag=False       
)

# --- 4. CHECK OUTPUT ---
print("Done. Saved to:\n\t", lc_clipped_path)

# Check valid classes (ignoring 0/NoData if that's what 0 represents)
valid_values = np.unique(filled_arr[~np.isnan(filled_arr)])
print(f"Unique values in output: {valid_values}")