
import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np
from shapely.geometry import box
import os
import sys
from pathlib import Path

# --- 1. IMPORT AND RUN SETUP ---
# Add .../code (parent of lc_scenarios) to Python path
CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

# Import your custom function
try:
    import gis_setup  # Import the script we just made
except ImportError:
    print(f"Error: Could not find 'function_clip_raster_to_aoi.py' in {CODE_DIR}")
    sys.exit(1)

# Run the environment fix immediately
gis_setup.configure_proj_environment()


# ==========================================
# PART 1: FUNCTION DEFINITIONS (Keep these)
# ==========================================

def calculate_opportunity_area(gdf, target_crs=None):
    """
    Calculates the total area of the opportunity polygons.
    
    Args:
        gdf (GeoDataFrame): The input polygons.
        target_crs (str/int): EPSG code for an Equal Area projection (e.g., 27700 for UK, 5070 for US).
                              If None, attempts to use current CRS if it is projected.
    """
    # Create a copy to avoid modifying the original outside this function
    gdf_calc = gdf.copy()
    
    # Ensure we are working with a projected CRS for accurate area calculation
    if target_crs:
        gdf_calc = gdf_calc.to_crs(target_crs)
    elif not gdf_calc.crs.is_projected:
        print("Warning: GDF is in geographic CRS. Reprojecting to Pseudo-Mercator for estimation, but specify an Equal Area CRS for accuracy.")
        gdf_calc = gdf_calc.to_crs(epsg=3857)

    total_area_m2 = gdf_calc.geometry.area.sum()
    total_area_km2 = total_area_m2 / 1e6
    
    print(f"--- Statistics ---")
    # print(f"Total Opportunity Area: {total_area_m2:,.2f} m²")
    print(f"Total Opportunity Area: {total_area_km2:,.4f} km²")
    
    return total_area_m2

def create_scenario_raster(baseline_raster_path, opportunity_vector_path, output_path, new_class_value, all_touched=True):
    """
    Generates a new LULC raster by burning vector polygons into a baseline raster.
    
    Args:
        baseline_raster_path (str): Path to the original LULC .tif
        opportunity_vector_path (str): Path to the buffered tree points/polygons .shp/.gpkg
        output_path (str): Where to save the result.
        new_class_value (int): The pixel value for the new tree/forest class.
        all_touched (bool): If True, pixels touching the polygon edges are updated. 
                            If False, only pixels whose center is within the polygon are updated.
    """
    
    # 1. READ BASELINE RASTER
    with rasterio.open(baseline_raster_path) as src:
        # Read metadata and data
        meta = src.meta.copy()
        base_data = src.read(1)
        
        # Get Raster properties for alignment
        raster_crs = src.crs
        raster_transform = src.transform
        raster_shape = src.shape
        nodata_val = src.nodata

    print(f"Baseline loaded. Shape: {raster_shape}, CRS: {raster_crs}")

    # 2. LOAD AND PREPARE VECTOR DATA
    gdf = gpd.read_file(opportunity_vector_path)
    
    # Filter empty or null geometries
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    
    if len(gdf) == 0:
        raise ValueError("Vector file contains no valid geometries.")

    # Reproject Vector to match Raster exactly
    if gdf.crs != raster_crs:
        print(f"Reprojecting vector from {gdf.crs} to {raster_crs}...")
        gdf = gdf.to_crs(raster_crs)
    
    # Optional: Filter vectors to only those that overlap the raster bounds
    # (Speeds up processing for large datasets)
    raster_bounds = box(*src.bounds)
    gdf = gdf[gdf.geometry.intersects(raster_bounds)]
    
    if len(gdf) == 0:
        print("Warning: No opportunity polygons overlap with the raster extent.")
        return

    # 3. RASTERIZE VECTORS
    # Create a generator of (geometry, value) pairs
    # We burn a temporary value of 1 where polygons exist
    shapes = ((geom, 1) for geom in gdf.geometry)
    
    print("Rasterizing opportunity layer...")
    mask_image = features.rasterize(
        shapes=shapes,
        out_shape=raster_shape,
        transform=raster_transform,
        fill=0,            # Background value
        all_touched=all_touched, 
        dtype='uint8'
    )

    # 4. APPLY UPDATE LOGIC
    # Create a copy of the base data
    scenario_data = base_data.copy()

    # Define where to update:
    # 1. Where the mask_image is 1 (where polygons are)
    # 2. AND where the original data is NOT NoData (don't plant trees in the void)
    if nodata_val is not None:
        valid_pixels = (base_data != nodata_val)
        update_locs = (mask_image == 1) & valid_pixels
    else:
        update_locs = (mask_image == 1)

    # Apply the new class value
    scenario_data[update_locs] = new_class_value
    
    # Calculate how many pixels changed
    pixels_changed = np.count_nonzero(update_locs)
    print(f"Updated {pixels_changed} pixels to class '{new_class_value}'.")

    # 5. SAVE RESULT
    # Ensure dtype matches the data (often uint8 or uint16 for LULC)
    meta.update(compress='lzw') 
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(scenario_data, 1)
        
    print(f"Scenario saved successfully: {output_path}")



# ==========================================
# PART 2: MAIN EXECUTION (Replace the old one with this)
# ==========================================

# if __name__ == "__main__":
    
#     # --- CONFIGURATION ---
#     # Paths
#     BASE_RASTER = r"E:/London/Data/baseline_lulc.tif"
#     OPP_VECTORS = r"E:/London/Data/GiGL_OpenSpace_Sites_opportunityLC.shp"
#     OUTPUT_RASTER = r"E:/London/Data/scenario_lulc_tree_planting.tif"
    
#     # Parameters
#     NEW_FOREST_CODE = 1      # The pixel integer value for "Forest/Trees" in your standard
#     CALC_CRS = 27700         # EPSG for Area Calc (British National Grid)
    
#     # --- RUN WORKFLOW ---
    
#     # 1. Check stats
#     polys = gpd.read_file(OPP_VECTORS)
#     calculate_opportunity_area(polys, target_crs=CALC_CRS)
    
#     # 2. Generate Raster
#     create_scenario_raster(
#         baseline_raster_path=BASE_RASTER,
#         opportunity_vector_path=OPP_VECTORS,
#         output_path=OUTPUT_RASTER,
#         new_class_value=NEW_FOREST_CODE,
#         all_touched=True # True ensures small buffers are caught even if they don't hit pixel center
#     )



## ------------------------------------------------------------------------

if __name__ == "__main__":
    
    # --- 1. SETUP PATHS ---
    BASE_DIR = r"E:/London/jlg_tree_planting"
    BASELINE_RASTER = os.path.join(BASE_DIR, "LCM2023_London_10m_clip2aoi_tcc24_clip2aoi_Camden.tif")
    OUTPUT_DIR = os.path.join(BASE_DIR, "lc_scenarios_output")
    

    # Ensure the baseline raster is fixed before we start processing scenarios
    gis_setup.ensure_raster_crs(BASELINE_RASTER, target_epsg=27700)


    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- 2. DEFINE SCENARIOS ---
    # Dictionary format: "Name": {"file": "filename.shp", "code": pixel_value}
    SCENARIO_CONFIG = {
        "Scenario_2": {
            "file": "data2.gpkg",
            "lc_code": 100 
        },
        "Scenario_3": {
            "file": "data3.gpkg",
            "lc_code": 100
        },
        "Scenario_4": {
            "file": "data4.gpkg",
            "lc_code": 100
        },
        "Scenario_5": {
            "file": "data5.gpkg",
            "lc_code": 100
        },
        "Scenario_7": {
            "file": "data7.gpkg",
            "lc_code": 100
        }
    }

    # --- 3. RUN THE LOOP ---
    print(f"Starting batch processing for {len(SCENARIO_CONFIG)} scenarios...\n")

    for scenario_name, params in SCENARIO_CONFIG.items():
        
        print(f"\n====== Processing: {scenario_name} ======")
        
        # Construct full file paths
        vec_path = os.path.join(BASE_DIR, params["file"])
        out_path = os.path.join(OUTPUT_DIR, f"LULC_{scenario_name}.tif")
        
        # Check if input file exists before running
        if not os.path.exists(vec_path):
            print(f"ERROR: Input file not found: {vec_path}")
            continue # Skip to next scenario

        try:
            # 1. Optional: Calculate stats
            gdf = gpd.read_file(vec_path)
            calculate_opportunity_area(gdf, target_crs=27700)
            
            # 2. Generate Raster
            create_scenario_raster(
                baseline_raster_path=BASELINE_RASTER,
                opportunity_vector_path=vec_path,
                output_path=out_path,
                new_class_value=params["lc_code"],
                all_touched=True
            )
            
        except Exception as e:
            print(f"CRITICAL ERROR on {scenario_name}: {e}")

    print("\nAll scenarios processed.")