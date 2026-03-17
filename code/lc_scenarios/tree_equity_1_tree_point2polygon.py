"""
tree_equity_2_scenario_engine.py
Generates LULC scenario rasters by burning tree planting polygons into a baseline raster.
"""

import os
import sys
import subprocess
import shutil
import sqlite3
import time
import numpy as np
from pathlib import Path

# ============================================================
# STEP 1: FIX PROJ ENVIRONMENT — must be before rasterio import
# ============================================================

def _configure_proj():
    """
    Sets PROJ_DATA to the conda env's validated proj.db.
    Skips pyproj entirely to avoid stale system-level paths.
    """
    candidates = [
        Path(sys.prefix) / "Library" / "share" / "proj",  # Windows conda-forge
        Path(sys.prefix) / "share" / "proj",               # Linux/Mac conda-forge
    ]
    for proj_dir in candidates:
        db = proj_dir / "proj.db"
        if not db.exists():
            continue
        # Validate version
        try:
            con = sqlite3.connect(str(db))
            cur = con.cursor()
            cur.execute("SELECT value FROM metadata WHERE key = 'DATABASE.LAYOUT.VERSION.MINOR'")
            minor = int(cur.fetchone()[0])
            con.close()
            if minor < 5:
                print(f"[proj] Skipping {db} — version {minor} too old (need ≥ 5)")
                continue
        except Exception:
            continue

        os.environ["PROJ_DATA"]    = str(proj_dir)
        os.environ["PROJ_LIB"]     = str(proj_dir)
        os.environ["PROJ_NETWORK"] = "OFF"
        print(f"[proj] PROJ_DATA → {proj_dir} (version {minor} ✓)")
        return True

    print("[proj] WARNING: Could not find a valid proj.db — CRS operations may fail.")
    return False

_configure_proj()

# ============================================================
# STEP 2: GIS IMPORTS — safe after PROJ is configured
# ============================================================

import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import box

# ============================================================
# STEP 3: FIND GDALWARP — needed for CRS fix
# ============================================================

def _find_gdalwarp() -> str:
    """Locates gdalwarp binary — checks PATH then conda env."""
    path = shutil.which("gdalwarp")
    if path:
        return path
    candidates = [
        Path(sys.prefix) / "Library" / "bin" / "gdalwarp.exe",  # Windows
        Path(sys.prefix) / "bin" / "gdalwarp",                   # Linux/Mac
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        "gdalwarp not found. Run: conda install -c conda-forge gdal"
    )

# ============================================================
# PART 1: FUNCTION DEFINITIONS
# ============================================================

def ensure_raster_crs(raster_path: str, target_epsg: int, overwrite: bool = False) -> str:
    """
    Checks a raster's CRS and reprojects via gdalwarp if needed.

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


def calculate_opportunity_area(gdf, target_crs=None):
    """
    Calculates total area of opportunity polygons.

    Args:
        gdf        (GeoDataFrame): Input polygons.
        target_crs (int):          EPSG code for equal-area projection.
    Returns:
        Total area in m².
    """
    gdf_calc = gdf.copy()

    if target_crs:
        gdf_calc = gdf_calc.to_crs(target_crs)
    elif not gdf_calc.crs.is_projected:
        print("Warning: Geographic CRS detected. Reprojecting to Pseudo-Mercator — specify Equal Area CRS for accuracy.")
        gdf_calc = gdf_calc.to_crs(epsg=3857)

    total_area_m2  = gdf_calc.geometry.area.sum()
    total_area_km2 = total_area_m2 / 1e6

    print(f"--- Statistics ---")
    print(f"Total Opportunity Area: {total_area_km2:,.4f} km²")

    return total_area_m2


def create_scenario_raster(baseline_raster_path, opportunity_vector_path,
                           output_path, new_class_value, all_touched=False):
    """
    Generates a new LULC raster by burning vector polygons into a baseline raster.

    Args:
        baseline_raster_path    (str):  Path to the original LULC .tif
        opportunity_vector_path (str):  Path to the buffered tree polygons .gpkg
        output_path             (str):  Where to save the result.
        new_class_value         (int):  Pixel value for the new tree/forest class.
        all_touched             (bool): If True, pixels touching polygon edges are updated.
    """

    t0 = time.time()

    # 1. READ BASELINE RASTER
    with rasterio.open(baseline_raster_path) as src:
        meta           = src.meta.copy()
        base_data      = src.read(1)
        raster_crs     = src.crs
        raster_transform = src.transform
        raster_shape   = src.shape
        nodata_val     = src.nodata
        raster_bounds  = box(*src.bounds)   # ✅ captured inside with block

    print(f"Baseline loaded. Shape: {raster_shape}, CRS: {raster_crs}")

    # 2. LOAD VECTOR — geometry only for speed
    gdf = gpd.read_file(
        opportunity_vector_path,
        columns=["geometry"],   # ⚡ skip unused attribute columns
        engine="pyogrio"        # ⚡ 5-10x faster than fiona
    )

    # Filter empty/null geometries
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if len(gdf) == 0:
        raise ValueError("Vector file contains no valid geometries.")

    # Reproject vector to match raster CRS
    if gdf.crs != raster_crs:
        print(f"Reprojecting vector from {gdf.crs} to {raster_crs}...")
        gdf = gdf.to_crs(raster_crs)

    # Spatial filter — keep only features overlapping raster extent
    hits = gdf.sindex.query(raster_bounds, predicate="intersects")
    gdf  = gdf.iloc[hits]                  # ⚡ spatial index, not full intersects loop

    if len(gdf) == 0:
        print("Warning: No opportunity polygons overlap with the raster extent.")
        return

    print(f"Features within raster extent: {len(gdf):,}")

    # 3. RASTERIZE VECTORS
    shapes = ((geom, 1) for geom in gdf.geometry)
    print("Rasterizing opportunity layer...")

    mask_image = features.rasterize(
        shapes=shapes,
        out_shape=raster_shape,
        transform=raster_transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8"
    )

    # 4. APPLY UPDATE LOGIC
    scenario_data = base_data.copy()

    if nodata_val is not None:
        update_locs = (mask_image == 1) & (base_data != nodata_val)
    else:
        update_locs = (mask_image == 1)

    scenario_data[update_locs] = new_class_value
    pixels_changed = np.count_nonzero(update_locs)
    print(f"Updated {pixels_changed:,} pixels to class '{new_class_value}'.")

    # 5. SAVE RESULT
    meta.update(compress="lzw", nodata=nodata_val)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(scenario_data, 1)

    print(f"Scenario saved in {time.time()-t0:.1f}s → {output_path}")


# ============================================================
# PART 2: MAIN
# ============================================================

if __name__ == "__main__":

    # --- PATHS ---
    BASE_DIR        = Path(r"G:\Shared drives\Wellcome Trust Project Data\1_preprocess\UrbanCoolingModel\OfficialWorkingInputs\LULC")
    VEC_DIR         = BASE_DIR
    BASELINE_RASTER = BASE_DIR / "LCM2023_London_10m_clip2aoi_tcc24.tif"
    OUTPUT_DIR      = BASE_DIR / "lc_scenarios_output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- VALIDATE BASELINE ---
    if not BASELINE_RASTER.exists():
        sys.exit(f"CRITICAL: Baseline raster not found: {BASELINE_RASTER}")

    # --- ENSURE CORRECT CRS ---
    # overwrite=False saves alongside original as LCM2023_..._EPSG27700.tif
    BASELINE_RASTER = Path(ensure_raster_crs(str(BASELINE_RASTER), target_epsg=27700, overwrite=False))

    # --- SCENARIO CONFIG ---
    # lc_code 100 = Broadleaved woodland in LCM2023
    SCENARIO_CONFIG = {
        "Scenario710": {"file": "tree_equity_scenario710.gpkg", "lc_code": 100},
        "Scenario720": {"file": "tree_equity_scenario720.gpkg", "lc_code": 100},
        "Scenario730": {"file": "tree_equity_scenario730.gpkg", "lc_code": 100},
    }

    # --- BATCH LOOP ---
    print(f"\nStarting batch processing for {len(SCENARIO_CONFIG)} scenarios...\n")
    t_total = time.time()

    for scenario_name, params in SCENARIO_CONFIG.items():

        print(f"\n====== Processing: {scenario_name} ======")

        vec_path = VEC_DIR  / params["file"]
        out_path = OUTPUT_DIR / f"LULC_{scenario_name}.tif"

        if not vec_path.exists():
            print(f"ERROR: Input file not found: {vec_path}")
            continue

        try:
            # Stats — geometry only for speed
            gdf = gpd.read_file(vec_path, columns=["geometry"], engine="pyogrio")
            calculate_opportunity_area(gdf, target_crs=27700)

            create_scenario_raster(
                baseline_raster_path=str(BASELINE_RASTER),
                opportunity_vector_path=str(vec_path),
                output_path=str(out_path),
                new_class_value=params["lc_code"],
                all_touched=False
            )

        except Exception as e:
            print(f"CRITICAL ERROR on {scenario_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll scenarios processed in {time.time()-t_total:.1f}s")