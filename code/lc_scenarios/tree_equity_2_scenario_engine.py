"""
tree_equity_2_scenario_engine.py
Generates LULC scenario rasters by burning tree planting polygons into a baseline raster.
"""

import os
import sys
from pathlib import Path

import sqlite3
import subprocess
import shutil

# ============================================================
# STEP 1: FIX PROJ BEFORE RASTERIO IMPORT
# ============================================================

def _configure_proj():
    candidates = [
        Path(sys.prefix) / "Library" / "share" / "proj",  # Windows conda-forge
        Path(sys.prefix) / "share" / "proj",               # Linux/Mac conda-forge
    ]
    for proj_dir in candidates:
        db = proj_dir / "proj.db"
        if not db.exists():
            continue
        try:
            con   = sqlite3.connect(str(db))
            minor = int(con.execute(
                "SELECT value FROM metadata WHERE key='DATABASE.LAYOUT.VERSION.MINOR'"
            ).fetchone()[0])
            con.close()
            if minor < 5:
                continue
        except Exception:
            continue
        os.environ["PROJ_DATA"]    = str(proj_dir)
        os.environ["PROJ_LIB"]     = str(proj_dir)
        os.environ["PROJ_NETWORK"] = "OFF"
        print(f"[proj] PROJ_DATA → {proj_dir} (v{minor} ✓)")
        return True
    print("[proj] WARNING: valid proj.db not found")
    return False

_configure_proj()

# ============================================================
# STEP 2: GIS IMPORTS — safe after PROJ is configured
# ============================================================

# --- 1. IMPORT AND RUN SETUP ---

import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np
from shapely.geometry import box



# ============================================================
# STEP 3: GDALWARP HELPER
# ============================================================

def _find_gdalwarp() -> str:
    path = shutil.which("gdalwarp")
    if path:
        return path
    candidates = [
        Path(sys.prefix) / "Library" / "bin" / "gdalwarp.exe",
        Path(sys.prefix) / "bin" / "gdalwarp",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError("gdalwarp not found. Run: conda install -c conda-forge gdal")

# ============================================================
# PART 1: FUNCTIONS
# ============================================================

def ensure_raster_crs(raster_path: str, target_epsg: int, overwrite: bool = False) -> str:
    """
    Checks a raster CRS and reprojects via gdalwarp if needed.

    Args:
        raster_path  (str):  Path to the raster file.
        target_epsg  (int):  Desired EPSG code.
        overwrite   (bool):  If True, replaces original. If False, saves alongside.

    Returns:
        Path to the (possibly reprojected) raster.
    """
    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        current_epsg = src.crs.to_epsg() if src.crs else None

    if current_epsg == target_epsg:
        print(f"[ensure_crs] Already EPSG:{target_epsg} — no action needed.")
        return str(raster_path)

    print(f"[ensure_crs] Reprojecting {raster_path.name}: EPSG:{current_epsg} → EPSG:{target_epsg}")

    out_path = (raster_path.with_suffix(".tmp.tif") if overwrite
                else raster_path.with_name(f"{raster_path.stem}_EPSG{target_epsg}.tif"))

    cmd = [
        _find_gdalwarp(),
        "-t_srs",  f"EPSG:{target_epsg}",
        "-r",      "bilinear",
        "-co",     "COMPRESS=LZW",
        "-co",     "TILED=YES",
        "-overwrite",
        str(raster_path),
        str(out_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gdalwarp failed:\n{result.stderr}")

    if overwrite:
        raster_path.unlink()
        out_path.rename(raster_path)
        print(f"[ensure_crs] Saved (overwritten): {raster_path}")
        return str(raster_path)

    print(f"[ensure_crs] Saved alongside original: {out_path}")
    return str(out_path)

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

def create_scenario_raster(baseline_raster_path, opportunity_vector_path, output_path, new_class_value, all_touched=False):
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
        raster_bounds = box(*src.bounds)

    print(f"Baseline loaded. Shape: {raster_shape}, CRS: {raster_crs}")

    # 2. LOAD AND PREPARE VECTOR DATA
    gdf = gpd.read_file(
        opportunity_vector_path,
        columns=["geometry"],   # ⚡ skip unused attribute columns
        engine="pyogrio"        # ⚡ 5-10x faster than fiona
    )
    
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
    meta.update(compress='lzw', nodata=nodata_val)
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(scenario_data, 1)
        
    print(f"Scenario saved successfully: {output_path}")




# ============================================================
# PART 2: MAIN
# ============================================================

if __name__ == "__main__":
    
    # --- 1. SETUP PATHS ---
    BASE_DIR = Path(r"G:\Shared drives\Wellcome Trust Project Data\1_preprocess\UrbanCoolingModel\OfficialWorkingInputs\LULC")
    VEC_DIR  = BASE_DIR / "lc_tree_equity_scenarios_output"
    BASELINE_RASTER = BASE_DIR / "LCM2023_London_10m_clip2aoi_tcc24.tif"
    # OUTPUT_DIR = os.path.join(BASE_DIR, "lc_scenarios_output")
    OUTPUT_DIR = BASE_DIR / "lc_tree_equity_scenarios_output"
    
    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- VALIDATE BASELINE ---
    if not BASELINE_RASTER.exists():
        sys.exit(f"CRITICAL: Baseline raster not found: {BASELINE_RASTER}")
    
    # --- ENSURE CORRECT CRS (overwrite=False saves alongside original) ---
    BASELINE_RASTER = Path(ensure_raster_crs(
        str(BASELINE_RASTER), target_epsg=27700, overwrite=False
    ))

    # --- 2. DEFINE SCENARIOS ---
    # Dictionary format: "Name": {"file": "filename.shp", "code": pixel_value}
    SCENARIO_CONFIG = {
        # "Scenario715": {"file": "tree_equity_scenario715.gpkg", "lc_code": 100},
        # "Scenario710": {"file": "tree_equity_scenario710.gpkg", "lc_code": 100},
        # "Scenario720": {"file": "tree_equity_scenario720.gpkg", "lc_code": 100},
        # "Scenario730": {"file": "tree_equity_scenario730.gpkg", "lc_code": 100},

        # "Scenario710v2": {"file": "tree_equity_scenario710v2.gpkg", "lc_code": 100},
        # "Scenario720v2": {"file": "tree_equity_scenario720v2.gpkg", "lc_code": 100},
        # "Scenario730v2": {"file": "tree_equity_scenario730v2.gpkg", "lc_code": 100},

        "Scenario710v3": {"file": "tree_equity_scenario710v3.gpkg", "lc_code": 100},
        "Scenario720v3": {"file": "tree_equity_scenario720v3.gpkg", "lc_code": 100},
        "Scenario730v3": {"file": "tree_equity_scenario730v3.gpkg", "lc_code": 100},
    }

    # --- 3. RUN THE LOOP ---
    print(f"\tStarting batch processing for {len(SCENARIO_CONFIG)} scenarios...\n")

    for scenario_name, params in SCENARIO_CONFIG.items():
        
        print(f"\n====== Processing: {scenario_name} ======")
        
        # Construct full file paths
        vec_path = VEC_DIR   / params["file"]
        out_path = OUTPUT_DIR / f"LULC_{scenario_name}.tif"
        
        # Check if input file exists before running
        if not vec_path.exists():
            print(f"ERROR: Input file not found: {vec_path}")
            continue # Skip to next scenario

        try:
            # 1. Optional: Calculate stats
            gdf = gpd.read_file(vec_path, columns=["geometry"], engine="pyogrio")
            calculate_opportunity_area(gdf, target_crs=27700)
            
            # 2. Generate Raster
            create_scenario_raster(
                baseline_raster_path=str(BASELINE_RASTER),
                opportunity_vector_path=str(vec_path),
                output_path=str(out_path),
                new_class_value=params["lc_code"],
                all_touched=False # Use False for more conservative planting (only pixels whose center is within the polygon)
            )
            
        except Exception as e:
            print(f"CRITICAL ERROR on {scenario_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll scenarios processed.")